package node

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/2tbmz9y2xt-lang/rubin-protocol/clients/go/consensus"
)

const (
	DefaultMempoolMaxTransactions = 300
	DefaultMempoolMaxBytes        = consensus.MAX_RELAY_MSG_BYTES
	DefaultMempoolMinFeeRate      = uint64(1)

	// DefaultMinDaFeeRate is the spec-side per-byte DA fee floor from
	// POLICY_MEMPOOL_ADMISSION_GENESIS.md Stage C (`min_da_fee_rate`).
	DefaultMinDaFeeRate = uint64(1)

	mempoolLowWaterNumerator   = 9
	mempoolLowWaterDenominator = 10

	// mempoolSigCacheCapacity is the fixed process-memory capacity of the one
	// positive signature cache each Mempool owns. It is not configurable: no
	// CLI, RPC, config, metric, or persistence surface exposes it.
	mempoolSigCacheCapacity = 50_000
)

type mempoolTxSource string

const (
	mempoolTxSourceRemote mempoolTxSource = "remote"
	mempoolTxSourceLocal  mempoolTxSource = "local"
	mempoolTxSourceReorg  mempoolTxSource = "reorg"
)

type mempoolEntry struct {
	raw    []byte
	txid   [32]byte
	wtxid  [32]byte
	inputs []consensus.Outpoint
	// token is the exact pending-outpoint reservation this entry holds. It is
	// the zero token exactly when the entry spends no outpoint and therefore
	// claims nothing (the pre-owner validateEntryInputsLocked was likewise a
	// no-op for an empty input set).
	token PendingOutpointToken
	// fee is the authoritative admitted fee: the exact u128 scalar
	// consensus derived. weight stays u64 per the policy contract.
	fee          consensus.Uint128
	weight       uint64
	size         int
	admissionSeq uint64
	source       mempoolTxSource
}

type Mempool struct {
	mu                sync.RWMutex
	chainState        *ChainState
	blockStore        *BlockStore
	chainID           [32]byte
	policy            MempoolConfig
	maxTxs            int
	maxBytes          int
	lowWaterBytes     int
	usedBytes         int
	lastAdmissionSeq  uint64
	currentMinFeeRate uint64
	txs               map[[32]byte]*mempoolEntry
	wtxids            map[[32]byte][32]byte
	// pendingOutpoints is the sole conflict-admission authority. It replaces
	// the former spenders index: outpoint ownership now lives with the claim
	// and its token, so no second spender map can drift from the records.
	pendingOutpoints *PendingOutpointOwner
	// sigCache is the one positive-only signature cache this Mempool owns.
	sigCache *consensus.SigCache
	// Admission counters are bumped exactly once for each AddTx call on a
	// non-nil Mempool that reaches the final outcome accounting path.
	// Nil-receiver calls return before that defer is registered and are
	// therefore intentionally excluded from these counters. Lock-free via
	// atomic.Uint64 — no impact on the admissionMu / mu ordering. Buckets
	// are the closed enum {accepted, conflict, rejected, unavailable};
	// any non-TxAdmitError reachable from AddTx falls into the rejected
	// bucket so no unbounded label class can grow from this surface. P2P
	// disconnect metrics are intentionally not tracked here; they are
	// scoped to issue #1307 because the disconnect boundary needs a
	// separate semantic audit (no double-count, normal shutdown is not a
	// peer fault).
	admitAccepted    atomic.Uint64
	admitConflict    atomic.Uint64
	admitRejected    atomic.Uint64
	admitUnavailable atomic.Uint64

	// evictedResidentTotal counts cumulative resident-entry capacity
	// evictions since process start. It is bumped exactly once per
	// already-admitted entry that is removed by capacity pressure
	// (the per-victim counter loop in addEntryLockedProbed, the one
	// implementation of the locked admission path, after
	// validateCapacityAdmissionLocked classifies the candidate as
	// admitted-and-evicting and commitStandardDeltaLocked removes the
	// victims with their exact tokens). Candidate-worst rejection —
	// where the incoming candidate is rejected at capacity and no
	// resident is evicted — does not increment this counter; that path
	// returns txAdmitUnavailable with an empty victim list. Fee-floor
	// rejection of an incoming transaction never reaches this counter
	// either, because the fee-floor check happens before
	// validateCapacityAdmissionLocked. Public terminal removals and canonical
	// M/O plan publication are not policy capacity eviction, and also do not
	// increment this counter.
	evictedResidentTotal atomic.Uint64
}

// PendingOutpointOwner returns the single pending-outpoint owner bound to this
// mempool at construction. The pointer is stable for the mempool's lifetime;
// a nil receiver returns nil so callers can wire unconditionally.
func (m *Mempool) PendingOutpointOwner() *PendingOutpointOwner {
	if m == nil {
		return nil
	}
	return m.pendingOutpoints
}

// checkNeverUsedForBindingLocked proves the pool is not merely EMPTY but never
// used. SyncEngine.SetMempool is initialization-only, so a candidate that
// already admitted anything cannot be adopted: its records were validated
// against a canonical tip that engine never guarded, and emptiness alone cannot
// see that — a pool admitted at tip A and drained back to zero is
// index-identical to a fresh one. The history-bearing state is therefore the
// closed boundary here: resident records, the admission sequence high-water, the
// rolling fee floor (which congestion control raises and only ever decays back
// TOWARD the default), and the cumulative admission and eviction counters.
//
// STATIC policy configuration is deliberately NOT part of this boundary: limits,
// providers and the rest of MempoolConfig are an operator's construction-time
// choice, not evidence of past admission, so a custom-configured fresh pool
// still binds. The caller holds m.mu.
func (m *Mempool) checkNeverUsedForBindingLocked() error {
	if len(m.txs) != 0 || len(m.wtxids) != 0 || m.usedBytes != 0 {
		return fmt.Errorf("mempool candidate is not empty: entries=%d wtxids=%d used_bytes=%d", len(m.txs), len(m.wtxids), m.usedBytes)
	}
	if m.lastAdmissionSeq != 0 {
		return fmt.Errorf("mempool candidate already admitted: last_admission_seq=%d", m.lastAdmissionSeq)
	}
	// The raw field, never currentMinFeeRateLocked. The accessor normalizes ONLY
	// below-default values: a partially decayed raised floor (say 4 against the
	// default 1) is visible through either read, but a raw value BELOW the
	// default is reported AS the default and would be waved through as untouched.
	// The raw read is load-bearing for exactly that state.
	if m.currentMinFeeRate != DefaultMempoolMinFeeRate {
		return fmt.Errorf("mempool candidate carries a non-default rolling fee floor: min_fee_rate=%d want=%d", m.currentMinFeeRate, DefaultMempoolMinFeeRate)
	}
	counts := m.AdmissionCounts()
	evicted := m.evictedResidentTotal.Load()
	if counts != (MempoolAdmissionCounts{}) || evicted != 0 {
		return fmt.Errorf("mempool candidate carries admission history: accepted=%d conflict=%d rejected=%d unavailable=%d evicted_resident=%d",
			counts.Accepted, counts.Conflict, counts.Rejected, counts.Unavailable, evicted)
	}
	return nil
}

// AllTxIDs returns the txids of every transaction currently in the mempool.
// The slice ordering is not guaranteed to be stable between calls.
func (m *Mempool) AllTxIDs() [][32]byte {
	return m.txIDsLimit(0)
}

// TxIDsLimit returns at most limit txids from the current mempool snapshot.
// It returns nil when limit <= 0; use AllTxIDs for an unbounded snapshot.
// The slice ordering is not guaranteed to be stable between calls.
func (m *Mempool) TxIDsLimit(limit int) [][32]byte {
	if limit <= 0 {
		return nil
	}
	return m.txIDsLimit(limit)
}

func (m *Mempool) txIDsLimit(limit int) [][32]byte {
	if m == nil {
		return nil
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	capHint := len(m.txs)
	if limit > 0 && limit < capHint {
		capHint = limit
	}
	ids := make([][32]byte, 0, capHint)
	for txid := range m.txs {
		ids = append(ids, txid)
		if limit > 0 && len(ids) >= limit {
			break
		}
	}
	return ids
}

// TxByID returns the raw transaction bytes of a mempool entry with the given
// txid. The returned slice is a defensive copy and safe for the caller to
// retain or mutate. Returns (nil, false) if no matching entry is present.
func (m *Mempool) TxByID(txid [32]byte) ([]byte, bool) {
	if m == nil {
		return nil, false
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	entry, ok := m.txs[txid]
	if !ok {
		return nil, false
	}
	raw := make([]byte, len(entry.raw))
	copy(raw, entry.raw)
	return raw, true
}

// RetainedTxSnapshot is one indivisible read of a retained mempool row: the txid
// it is indexed under, the wtxid stored at its admission, and a defensive copy of
// its exact retained bytes, all read under one continuously held mempool read lock
// so that they describe one row incarnation and never a mixed tuple.
type RetainedTxSnapshot struct {
	IndexedTxID    [32]byte
	AdmissionWTxID [32]byte
	Raw            []byte
}

// RetainedTxByID snapshots the retained row indexed under txid.
//
// The bool reports ONLY whether the primary txid index held that key under the
// read lock — never validity, availability, standard-domain, or relay authority.
// The method parses, hashes, validates and repairs nothing, and never consults
// the separate wtxid index. Raw is a defensive exact-length copy: mutating it
// cannot alter the retained entry or any later snapshot.
//
// A present nil row is CORRUPTION, not absence, and stays distinguishable from
// it: it returns (IndexedTxID = the requested key, zero AdmissionWTxID, nil Raw,
// true), where a nil receiver or a missing key returns (zero snapshot, false).
// Classifying that corruption belongs to the caller.
func (m *Mempool) RetainedTxByID(txid [32]byte) (RetainedTxSnapshot, bool) {
	if m == nil {
		return RetainedTxSnapshot{}, false
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	entry, ok := m.txs[txid]
	if !ok {
		return RetainedTxSnapshot{}, false
	}
	// The indexed key, never entry.txid: a [32]byte map hit means the argument IS the key.
	snapshot := RetainedTxSnapshot{IndexedTxID: txid}
	if entry == nil {
		return snapshot, true
	}
	snapshot.AdmissionWTxID = entry.wtxid
	snapshot.Raw = make([]byte, len(entry.raw))
	copy(snapshot.Raw, entry.raw)
	return snapshot, true
}

// Contains reports whether a transaction with the given txid is currently
// present in the mempool.
func (m *Mempool) Contains(txid [32]byte) bool {
	if m == nil {
		return false
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.txs[txid]
	return ok
}

func (m *Mempool) AddTx(txBytes []byte) (retErr error) {
	return m.addTxWithSource(txBytes, mempoolTxSourceLocal, nil)
}

// AddRemoteTx admits a transaction received from a peer while preserving the
// same validation and admission policy as AddTx. The source is metadata only.
//
// It is the compatibility wrapper over the SAME implementation
// AddRemoteTxForRelay drives: a nil relay probe is the only difference, and it
// changes no validation order, no mutation, no error, and no counter.
func (m *Mempool) AddRemoteTx(txBytes []byte) (retErr error) {
	return m.addTxWithSource(txBytes, mempoolTxSourceRemote, nil)
}

// AddReorgTx admits a transaction requeued from a disconnected canonical block
// while preserving the same validation and admission policy as AddTx.
func (m *Mempool) AddReorgTx(txBytes []byte) (retErr error) {
	return m.addTxWithSource(txBytes, mempoolTxSourceReorg, nil)
}

// addTxWithSource validates and admits a transaction while recording the
// caller-declared origin in the mempool entry. Source provenance does not
// grant admission priority or bypass; invalid source values reject.
//
// Its ChainState admission read returns unavailable once a canonical transition
// has entered the fail-closed terminal state described on canonicalTransition.end.
//
// probe is the per-call relay sink AddRemoteTxForRelay supplies and every
// legacy entry point leaves nil. It carries identity, the proven admission
// context and the success selection only; a failed admission's disposition
// rides on the admission error itself, selected at its originating branch.
func (m *Mempool) addTxWithSource(txBytes []byte, source mempoolTxSource, probe *relayAdmissionProbe) (retErr error) {
	if m == nil {
		return selectRelayDisposition(txAdmitUnavailable("nil mempool"), RelayAdmissionUnavailable)
	}
	// Exactly one admission counter increment per non-nil-receiver call. The
	// nil-ChainState and terminal guards count at their return sites; ordinary
	// outcomes remain counted inside the admission guard. A nil receiver counts
	// nothing.
	if m.chainState == nil {
		err := selectRelayDisposition(txAdmitUnavailable("nil chainstate"), RelayAdmissionUnavailable)
		m.noteAdmissionResult(err)
		return err
	}

	if !m.chainState.admissionMu.RLockUnlessTerminal() {
		retErr = selectRelayDisposition(txAdmitUnavailable("pending-outpoint owner admission context unavailable"), RelayAdmissionUnavailable)
		m.noteAdmissionResult(retErr)
		return retErr
	}
	defer m.chainState.admissionMu.RUnlock()
	// Registered AFTER the guard, so LIFO runs the count BEFORE the RUnlock
	// above: the whole lifecycle — validation, insertion, outcome count — is
	// contained by ONE guard acquisition, which is what SetMempool's exclusive
	// acquisition must be able to exclude.
	defer func() { m.noteAdmissionResult(retErr) }()

	// Pure read, inside the guard that pins both the owner context and the
	// chainstate tip for this whole call. It decides nothing and mutates
	// nothing; a context it cannot prove simply stays unproven.
	m.bindRelayAdmissionContext(probe)

	// Inside the guard, so its rejection and count belong to that contained
	// lifecycle: caller metadata never reports an outcome the binding cannot see.
	//
	// Every entry point pins the source constant itself, so an invalid source is
	// an impossible invariant rather than a candidate property.
	if !validMempoolTxSource(source) {
		return selectRelayDisposition(txAdmitRejected(fmt.Sprintf("invalid mempool tx source %q", source)), RelayAdmissionInternal)
	}

	snapshot := m.chainState.admissionSnapshot()
	policy := m.policySnapshot()
	// Wave-6/8 (PR #1422): snap currentMinFeeRate ONCE so the cheap
	// precheck has a stable floor input for its accept/reject decision.
	// admissionMu excludes canonical publication. The locked path enforces max(snappedFloor, live currentMinFeeRate) so admission cannot miss a concurrent raise:
	//   - raise race: if raiseMinFeeRateAfterEvictionLocked fires
	//     between snap and lock, live currentMinFeeRate (higher) wins →
	//     tx correctly rejected against the current rolling floor;
	//     never admits below the live congestion-control level.
	snappedFloor := m.CurrentMinFeeRateSnapshot()
	checked, inputs, err := m.checkTransactionWithSnapshot(txBytes, snapshot, policy, snappedFloor, probe)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	entry := newMempoolEntry(checked, inputs, source)
	if err := m.rejectNonStandardKindLocked(entry, checked.Tx.TxKind); err != nil {
		return err
	}
	if err := m.addEntryLockedProbed(entry, snappedFloor, probe); err != nil {
		return err
	}
	m.noteRetainedLocked(entry, probe)
	return nil
}

// rejectNonStandardKindLocked is the standard-candidate kind slot of
// RUBIN_MEMPOOL_POLICY.md sections 6.1 and 6.2. It returns nil exactly for kind
// 0x00; every other kind gets the identity slot's txid-then-wtxid duplicate
// error when one applies, its third arm — the zero-txid INTERNAL refusal —
// being unreachable for a checked candidate, and otherwise the standard-domain
// rejection, tagged STABLE_TERMINAL_REJECT because that verdict rests on the kind.
//
// It writes nothing and takes no lock: two index reads and one error, so a
// candidate it refuses is left with no token, no sequence and no index row.
//
// kind is the parsed tx_kind of the completed checked transaction, whose
// reachable domain the canonical parser already closed to {0x00, 0x01, 0x02}.
// The caller holds m.mu.
func (m *Mempool) rejectNonStandardKindLocked(entry *mempoolEntry, kind uint8) error {
	if kind == 0x00 {
		return nil
	}
	if err := m.validateEntryIdentityLocked(entry); err != nil {
		return err
	}
	return selectRelayDisposition(txAdmitRejected("standard mempool accepts only tx_kind=0x00"), RelayAdmissionStableTerminalReject)
}

// RelayMetadata returns the metadata a relay peer needs to forward the
// transaction (fee + serialized size). It runs full structural +
// chainstate validation via checkParsedTransactionWithSnapshot and then
// enforces the rolling-relay-fee floor read-only. Below-floor otherwise-valid
// txs return the same TxAdmitUnavailable class/message family as admit-path
// validateFeeFloorLockedWithFloor.
//
// RelayMetadata is not full mempool admission: it does not insert, does not
// record source/admission_seq, and does not check duplicate, conflict, or
// capacity state. Those remain owned by addTxWithSource/addEntryLockedProbed.
func (m *Mempool) RelayMetadata(txBytes []byte) (RelayTxMetadata, error) {
	if m == nil {
		return RelayTxMetadata{}, txAdmitUnavailable("nil mempool")
	}
	if m.chainState == nil {
		return RelayTxMetadata{}, txAdmitUnavailable("nil chainstate")
	}
	tx, txid, wtxid, err := parseRelayMetadataTx(txBytes)
	if err != nil {
		return RelayTxMetadata{}, err
	}
	if !m.chainState.admissionMu.RLockUnlessTerminal() {
		return RelayTxMetadata{}, txAdmitUnavailable("pending-outpoint owner admission context unavailable")
	}
	defer m.chainState.admissionMu.RUnlock()
	snapshot := m.chainState.admissionSnapshotForInputs(relayMetadataInputs(tx))
	policy := m.policySnapshot()
	snappedFloor := m.CurrentMinFeeRateSnapshot()
	checked, _, err := m.checkParsedTransactionWithSnapshot(txBytes, tx, txid, wtxid, snapshot, policy)
	if err != nil {
		return RelayTxMetadata{}, err
	}
	if err := m.validateRelayMetadataFeeFloor(checked, snappedFloor); err != nil {
		return RelayTxMetadata{}, err
	}
	return RelayTxMetadata{
		Fee:  checked.Fee,
		Size: checked.SerializedSize,
	}, nil
}

func (m *Mempool) checkParsedTransactionWithSnapshot(
	txBytes []byte,
	tx *consensus.Tx,
	txid [32]byte,
	wtxid [32]byte,
	snapshot *chainStateAdmissionSnapshot,
	policy MempoolConfig,
) (*consensus.CheckedTransaction, []consensus.Outpoint, error) {
	// Validate chain snapshot and extract next height
	nextHeight, err := validateChainSnapshot(snapshot)
	if err != nil {
		return nil, nil, err
	}

	// Get block MTP
	blockMTP, err := m.nextBlockMTP(nextHeight)
	if err != nil {
		return nil, nil, selectRelayDisposition(txAdmitUnavailable(err.Error()), RelayAdmissionUnavailable)
	}

	// Prepare policy UTXOs if needed
	policyUtxos, err := buildPolicyInputSnapshotIfNeeded(tx, snapshot, policy)
	if err != nil {
		return nil, nil, err
	}
	if err := m.rejectSimplicityPreActivationLane(tx, policyUtxos, nextHeight, policy); err != nil {
		return nil, nil, err
	}

	// Perform consensus validation
	checked, err := m.validateTransactionWithConsensus(txBytes, tx, txid, wtxid, snapshot, nextHeight, blockMTP, policy)
	if err != nil {
		return nil, nil, err
	}

	// Apply policy validation
	if err := m.applyPolicyAgainstState(checked, nextHeight, policyUtxos, policy); err != nil {
		// Classified from the ORIGINAL fan-out error, read by type before the public
		// wrapper renders its message: the impossible invariant is the one wrapper
		// reachable here, the Simplicity wrapper being preempted by the lane above.
		return nil, nil, selectRelayDisposition(txAdmitRejected(err.Error()), relayDispositionForPolicyError(err))
	}

	// Extract inputs and return
	inputs := extractTxInputs(checked)
	return checked, inputs, nil
}

// rejectSimplicityPreActivationLane is checkParsedTransactionWithSnapshot's
// CORE_SIMPLICITY pre-activation lane. The (reject, err) tuple separates the
// deployment-set-dependent half from the context-complete well-formedness
// verdict, and the returned error carries the disposition that tuple selects,
// as at the standard producer's own site.
func (m *Mempool) rejectSimplicityPreActivationLane(tx *consensus.Tx, policyUtxos map[consensus.Outpoint]consensus.UtxoEntry, nextHeight uint64, policy MempoolConfig) error {
	if !policy.PolicyRejectSimplicityPreActivation {
		return nil
	}
	reject, reason, err := rejectCoreSimplicityPreActivation(tx, policyUtxos, m.chainID, nextHeight, policy.RotationProvider)
	if err != nil {
		return selectRelayDisposition(txAdmitRejected(err.Error()), relayDispositionForSimplicityPreActivationOutcome(reject, policy.RotationProvider))
	}
	if reject {
		return selectRelayDisposition(txAdmitRejected(reason), relayDispositionForSimplicityPreActivationOutcome(true, policy.RotationProvider))
	}
	return nil
}

func (m *Mempool) applyPolicyAgainstState(checked *consensus.CheckedTransaction, nextHeight uint64, utxos map[consensus.Outpoint]consensus.UtxoEntry, policy MempoolConfig) error {
	if checked == nil || checked.Tx == nil {
		// Both call sites reach here only with the CheckedTransaction a completed
		// consensus validation returned, so this is an impossible invariant and
		// never a candidate property. It is TYPED — never message-matched — so the
		// relay classifier publishes INTERNAL for it instead of the
		// cache-authorizing stable terminal rejection its default would give. The
		// message is unchanged, so every public error built from it is unchanged.
		return &policyImpossibleInvariantError{err: errors.New("nil checked transaction")}
	}
	// Apply non-coinbase anchor output policy
	if err := applyPolicyAgainstStateAnchor(checked, policy); err != nil {
		return err
	}

	// Apply DA fee policy
	if err := applyPolicyAgainstStateDA(checked, policy, utxos); err != nil {
		return err
	}

	// Apply Simplicity policy
	if err := applyPolicyAgainstStateSimplicity(checked, utxos, m.chainID, nextHeight, policy); err != nil {
		return err
	}

	return nil
}

func prevTimestampsFromStore(store *BlockStore, nextHeight uint64) ([]uint64, error) {
	if store == nil || nextHeight == 0 {
		return nil, nil
	}
	k := uint64(11)
	if nextHeight < k {
		k = nextHeight
	}
	out := make([]uint64, 0, k)
	for i := uint64(0); i < k; i++ {
		height := nextHeight - 1 - i
		timestamp, err := getBlockTimestamp(store, height, nextHeight)
		if err != nil {
			return nil, err
		}
		out = append(out, timestamp)
	}
	return out, nil
}

// getBlockTimestamp retrieves the timestamp from a block at the given height
func getBlockTimestamp(store *BlockStore, height, nextHeight uint64) (uint64, error) {
	hash, ok, err := store.CanonicalHash(height)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, fmt.Errorf("missing canonical hash at height %d for timestamp context (next_height=%d)", height, nextHeight)
	}
	headerBytes, err := store.GetHeaderByHash(hash)
	if err != nil {
		return 0, err
	}
	header, err := consensus.ParseBlockHeaderBytes(headerBytes)
	if err != nil {
		return 0, err
	}
	return header.Timestamp, nil
}
