package node

import (
	"bytes"
	"context"
	"crypto/sha3"
	"errors"
	"math"
	"time"

	"github.com/2tbmz9y2xt-lang/rubin-protocol/clients/go/consensus"
)

var unixNow = func() int64 { return time.Now().Unix() }

type MinerConfig struct {
	TimestampSource func() uint64
	MaxTxPerBlock   int
	Target          [32]byte
	// MineAddress is canonical CORE_P2PK covenant_data (suite_id || key_id)
	// used for the subsidy-bearing coinbase output.
	MineAddress []byte

	// PolicyDaAnchorAntiAbuse is the master switch for the whole DA/anchor
	// anti-abuse miner-template policy package. When false,
	// PolicyRejectNonCoinbaseAnchorOutputs is ignored. This is policy-only
	// and does not change consensus validity.
	PolicyDaAnchorAntiAbuse bool

	// PolicyRejectNonCoinbaseAnchorOutputs rejects transactions that create
	// CORE_ANCHOR outputs when PolicyDaAnchorAntiAbuse is enabled.
	// (Non-coinbase CORE_ANCHOR is treated as non-standard by policy.)
	PolicyRejectNonCoinbaseAnchorOutputs bool

	// PolicyMaxDaBytesPerBlock caps total DA payload bytes included in a block template (policy-only).
	// This is independent from the consensus DA byte cap.
	PolicyMaxDaBytesPerBlock uint64

	// PolicyDaSurchargePerByte is the operator-tunable DA per-byte surcharge
	// added on top of the spec-side DA floor in
	// POLICY_MEMPOOL_ADMISSION_GENESIS.md Stage C:
	//   da_required_fee(tx) = da_payload_len(tx) * (MinDaFeeRate + PolicyDaSurchargePerByte)
	// Setting it to 0 disables only the surcharge term, not the DA floor;
	// the DA floor still uses MinDaFeeRate.
	PolicyDaSurchargePerByte uint64

	// MinDaFeeRate is the spec-side per-byte DA fee floor
	// (POLICY_MEMPOOL_ADMISSION_GENESIS.md Stage C `min_da_fee_rate`,
	// default 1). Together with PolicyDaSurchargePerByte it composes the
	// DA half of the Stage C admission contract:
	//   da_required_fee(tx) = da_payload_len(tx) * (MinDaFeeRate + PolicyDaSurchargePerByte)
	// Setting it to 0 disables only the spec floor; the surcharge term
	// still applies when PolicyDaSurchargePerByte > 0.
	MinDaFeeRate uint64

	// CurrentMempoolMinFeeRate is the static fallback for the rolling
	// local floor when CurrentMempoolMinFeeRateFn is nil. Standalone
	// miner builds with no live mempool keep this at
	// DefaultMempoolMinFeeRate (the #1336 baseline). Live nodes SHOULD
	// set CurrentMempoolMinFeeRateFn instead so the miner template
	// reflects the rolling floor as it rises and decays.
	CurrentMempoolMinFeeRate uint64

	// CurrentMempoolMinFeeRateFn returns the live rolling local floor
	// for the relay-fee half of the Stage C admission contract:
	//   relay_fee_floor(tx) = weight(tx) * <floor returned by Fn>
	// Production wiring sets this to Mempool.CurrentMinFeeRateSnapshot
	// so the miner template stays aligned with the admission predicate
	// as the rolling floor changes. When nil the miner falls back to
	// CurrentMempoolMinFeeRate above.
	CurrentMempoolMinFeeRateFn func() uint64

	// PolicyRejectSimplicityPreActivation controls non-consensus
	// pre-activation guardrails for CORE_SIMPLICITY. When enabled, the
	// miner excludes CORE_SIMPLICITY transactions until the miner rotation
	// provider reports Simplicity active at the candidate height.
	PolicyRejectSimplicityPreActivation bool

	CompleteDASetProvider CompleteDASetProvider
}

type MinedBlock struct {
	Height    uint64
	Hash      [32]byte
	Timestamp uint64
	Nonce     uint64
	TxCount   int
}

// CanonicalCommitState is the RUBIN_MEMPOOL_POLICY.md Section 6.4.1 commit truth
// of the canonical apply that selects one mining result, in the exact wire form
// RUBIN_NODE_RPC_DEVNET.md Section 3.4 requires of its mandatory commit_state
// field. The three constants below are the CLOSED set; no other value is ever
// produced, and "unknown" MUST NOT be flattened onto "not_committed".
type CanonicalCommitState string

const (
	// CanonicalCommitStateNotCommitted is truth OLD, NOT_APPLICABLE, or any
	// outcome reached before a commit could cross: the exact old image stands.
	CanonicalCommitStateNotCommitted CanonicalCommitState = "not_committed"
	// CanonicalCommitStateCommitted is truth NEW: the complete planned new
	// image was published, whether or not the transition then latched.
	CanonicalCommitStateCommitted CanonicalCommitState = "committed"
	// CanonicalCommitStateUnknown is TERMINAL_PERSISTENCE(neither_or_unreadable):
	// neither image may be guessed and no identity may be exposed.
	CanonicalCommitStateUnknown CanonicalCommitState = "unknown"
)

// MineOneDisposition is the closed public-result class of one mining attempt.
// It is what a transport maps to its own status vocabulary — Section 3.4 maps
// these three to 200, 422 and 503 — so no transport has to classify an error
// string, a latch, a tip or a storage reread to answer.
type MineOneDisposition uint8

const (
	// MineOneDispositionServiceUnavailable is a LOCAL condition rather than a
	// verdict on the candidate: an unavailable runtime, a latched engine, and
	// every failure at or after the durable commit stage — bounded resource,
	// stale plan, noncanonical store, precommit persistence, and every terminal
	// class including a terminal NEW that did publish.
	//
	// It is deliberately the ZERO value, so a MineOneOutcome whose Disposition
	// some later path leaves unset fails CLOSED on 503 instead of claiming a
	// success. That covers the STATUS only: CommitState has no safe zero — its
	// own zero is the empty string, which is outside the closed set — so every
	// MineOneOutcome built anywhere assigns it a constant or the state
	// canonicalMineOneOutcome selected, never a default.
	MineOneDispositionServiceUnavailable MineOneDisposition = iota
	// MineOneDispositionUnprocessable is the candidate's OWN failure: a
	// consensus or policy rejection, a nonce or build failure, a request
	// cancellation, or any other ordinary candidate outcome.
	MineOneDispositionUnprocessable
	// MineOneDispositionSuccess is a mined candidate that committed.
	MineOneDispositionSuccess
)

// MineOneOutcome is the complete typed result of ONE MineOneWithOutcome
// invocation, and the only thing a caller needs to render Section 3.4's
// projection.
//
// Block is nonnil exactly when the attempt published a candidate — including a
// terminal candidate NEW, which returns both the block identity and the terminal
// error. CommitState and Disposition are independent: a committed attempt can be
// unavailable (terminal NEW), and a not_committed one can be unprocessable
// (ordinary rejection) or unavailable (local failure).
type MineOneOutcome struct {
	Block       *MinedBlock
	CommitState CanonicalCommitState
	Disposition MineOneDisposition
}

// canonicalMineOneOutcome projects ONE canonical apply's per-invocation typed
// projection onto the closed (commit state, disposition) pair.
//
// It reads the projection and the storage-persistence-fault sentinel and NOTHING
// else: no error text, no latch inspection, no current tip, no published summary,
// no storage reread. err is the error the apply returned, used only for its
// presence and for that one sentinel identity.
func canonicalMineOneOutcome(projection canonicalApplyProjection, err error) (CanonicalCommitState, MineOneDisposition) {
	switch projection.truth {
	case canonicalTruthNew:
		if err != nil {
			return CanonicalCommitStateCommitted, MineOneDispositionServiceUnavailable
		}
		return CanonicalCommitStateCommitted, MineOneDispositionSuccess
	case canonicalTruthUnknown:
		return CanonicalCommitStateUnknown, MineOneDispositionServiceUnavailable
	default:
		if err == nil {
			return CanonicalCommitStateNotCommitted, MineOneDispositionSuccess
		}
		if projection.commitStageEntered || errors.Is(err, errStoragePersistenceFault) {
			return CanonicalCommitStateNotCommitted, MineOneDispositionServiceUnavailable
		}
		return CanonicalCommitStateNotCommitted, MineOneDispositionUnprocessable
	}
}

// unavailableMineOneOutcome is the outcome of a failure that never reached a
// canonical apply at all, so no truth was classified and the old image stands.
func unavailableMineOneOutcome() MineOneOutcome {
	return MineOneOutcome{CommitState: CanonicalCommitStateNotCommitted, Disposition: MineOneDispositionServiceUnavailable}
}

// unprocessableMineOneOutcome is the candidate's own pre-apply failure.
func unprocessableMineOneOutcome() MineOneOutcome {
	return MineOneOutcome{CommitState: CanonicalCommitStateNotCommitted, Disposition: MineOneDispositionUnprocessable}
}

type Miner struct {
	chainState *ChainState
	blockStore *BlockStore
	sync       *SyncEngine
	cfg        MinerConfig
}

type minedCandidate struct {
	raw    []byte
	txid   [32]byte
	wtxid  [32]byte
	weight uint64
	nonce  uint64
}

type miningBuildContext struct {
	prevHash         [32]byte
	remainingWeight  uint64
	nextHeight       uint64
	alreadyGenerated consensus.Uint128
	utxos            map[consensus.Outpoint]consensus.UtxoEntry
	candidateTxs     [][]byte
}

type miningConsensusContext struct {
	blockMTP uint64
	chainID  [32]byte
	rotation consensus.RotationProvider
	registry *consensus.SuiteRegistry
}

type miningChainStateSnapshot struct {
	hasTip           bool
	height           uint64
	tipHash          [32]byte
	alreadyGenerated consensus.Uint128
	utxos            map[consensus.Outpoint]consensus.UtxoEntry
}

func DefaultMinerConfig() MinerConfig {
	return MinerConfig{
		Target: consensus.POW_LIMIT,
		TimestampSource: func() uint64 {
			return unixNowU64()
		},
		MineAddress:                          defaultMineAddress(),
		MaxTxPerBlock:                        1024,
		PolicyDaAnchorAntiAbuse:              true,
		PolicyRejectNonCoinbaseAnchorOutputs: true,
		PolicyMaxDaBytesPerBlock:             consensus.MAX_DA_BYTES_PER_BLOCK / 4, // 25% policy budget (issue #353 draft)
		PolicyDaSurchargePerByte:             0,                                    // controller-tunable; disabled by default
		MinDaFeeRate:                         DefaultMinDaFeeRate,                  // Stage C spec-side per-byte DA floor
		CurrentMempoolMinFeeRate:             DefaultMempoolMinFeeRate,             // baseline for the rolling floor when no live mempool is bound
		PolicyRejectSimplicityPreActivation:  true,
	}
}

func NewMiner(chainState *ChainState, blockStore *BlockStore, sync *SyncEngine, cfg MinerConfig) (*Miner, error) {
	// Validate inputs
	if err := validateNewMinerInputs(chainState, blockStore, sync); err != nil {
		return nil, err
	}

	// Validate alias requirements
	if err := validateMinerAliasRequirements(chainState, blockStore, sync); err != nil {
		return nil, err
	}

	// Normalize configuration
	if err := normalizeMinerConfig(&cfg); err != nil {
		return nil, err
	}

	return &Miner{
		chainState: chainState,
		blockStore: blockStore,
		sync:       sync,
		cfg:        cfg,
	}, nil
}

// MineN mines blocks sequentially via MineOne, inheriting its cancellation
// postconditions; cancellation between blocks surfaces from the next call.
func (m *Miner) MineN(ctx context.Context, blocks int, txs [][]byte) ([]MinedBlock, error) {
	if blocks < 0 {
		return nil, errors.New("blocks must be >= 0")
	}
	out := make([]MinedBlock, 0, blocks)
	for i := 0; i < blocks; i++ {
		mb, err := m.MineOne(ctx, txs)
		if err != nil {
			return nil, err
		}
		out = append(out, *mb)
	}
	return out, nil
}

// MineOne observes ctx at entry (pre-bootstrap), every nonce attempt, and
// immediately before canonical apply; observing cancellation returns
// ctx.Err() with no write past the check. A nil ctx disables observation.
//
// Callee-side ctx contract: every checkpoint is a NON-BLOCKING poll (a select
// with a default). MineOne never blocks on ctx.Done(), never retains ctx or
// its Done channel across checkpoints, and never shares ctx with another
// goroutine — so a caller may pass a single-goroutine context implementation
// whose Done/Err are only valid on the calling goroutine.
func (m *Miner) MineOne(ctx context.Context, txs [][]byte) (*MinedBlock, error) {
	outcome, err := m.MineOneWithOutcome(ctx, txs)
	if err != nil {
		// Unchanged contract: ANY error returns a nil block, including the
		// terminal candidate NEW whose block the typed outcome does carry.
		return nil, err
	}
	return outcome.Block, nil
}

// MineOneWithOutcome is MineOne plus the typed per-invocation commit state and
// public-result class of the attempt. It observes ctx at exactly the same
// checkpoints, returns exactly the same errors, and mutates exactly as much.
//
// A caller that only needs the block should keep using MineOne; a caller that
// must PROJECT the attempt — Section 3.4's /mine_next is the one in tree — needs
// the outcome, because the block alone cannot distinguish a terminal candidate
// NEW (published, unavailable) from an ordinary success, nor an UNKNOWN truth
// from an ordinary refusal.
func (m *Miner) MineOneWithOutcome(ctx context.Context, txs [][]byte) (MineOneOutcome, error) {
	// An uninitialized miner is an unavailable runtime, not a verdict on the
	// candidate. /mine_next cannot reach it — it answers 503 on a nil miner and
	// NewMiner rejects a partially built one — so this classification is
	// defensive.
	if err := m.validateMineOneInput(); err != nil {
		return unavailableMineOneOutcome(), err
	}

	// Entry cancellation. A request-local cancellation alone keeps the ordinary
	// candidate class; a caller that ALSO observed its lifecycle source canceled
	// at this checkpoint overrides that classification itself, which is where
	// that authority already lived.
	if err := mineOneContextErr(ctx); err != nil {
		return unprocessableMineOneOutcome(), err
	}

	projection, err := m.sync.bootstrapCanonicalGenesisIfEmpty()
	if err != nil {
		state, disposition := canonicalMineOneOutcome(projection, err)
		return MineOneOutcome{CommitState: state, Disposition: disposition}, err
	}

	return m.executeMineOne(ctx, txs)
}

// mineOneContextErr is the ONE non-blocking cancellation poll shape every
// checkpoint uses: a select with a default, never a blocking receive, and never
// a retained Done channel. A nil ctx disables observation.
func mineOneContextErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}

func (m *Miner) buildBlock(ctx context.Context, txs [][]byte) ([]byte, []uint64, uint64, uint64, int, error) {
	buildCtx, err := m.buildContext(txs)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}

	parsed, err := m.selectCandidateTransactions(buildCtx.candidateTxs, buildCtx.utxos, buildCtx.nextHeight, buildCtx.remainingWeight)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	witnessCommitment, err := buildWitnessCommitment(parsed)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	coinbase, merkleRoot, err := m.buildCoinbaseAndMerkleRoot(buildCtx.nextHeight, buildCtx.alreadyGenerated, witnessCommitment, parsed)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	prevTimestamps, timestamp, headerBytes, nonce, err := m.mineHeader(ctx, buildCtx.nextHeight, buildCtx.prevHash, merkleRoot)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	blockBytes := assembleBlockBytes(headerBytes, coinbase, parsed)
	return blockBytes, prevTimestamps, timestamp, nonce, 1 + len(parsed), nil
}

func (m *Miner) buildContext(txs [][]byte) (miningBuildContext, error) {
	copyPolicyUtxos := len(txs) != 0 || m.cfg.CompleteDASetProvider != nil ||
		(m.maxSelectedTransactions() > 0 && m.sync != nil && m.sync.mempool != nil && m.sync.mempool.Len() > 0)
	state, err := m.snapshotBuildContextState(copyPolicyUtxos)
	if err != nil {
		return miningBuildContext{}, err
	}
	candidateTxs := m.candidateTransactions(txs)
	nextHeight, expectedPrev, err := nextBlockContextFromFields(state.hasTip, state.height, state.tipHash)
	if err != nil {
		return miningBuildContext{}, err
	}
	var prevHash [32]byte
	if expectedPrev != nil {
		prevHash = *expectedPrev
	}
	remainingWeight, err := m.remainingWeightBudget(nextHeight, state.alreadyGenerated)
	if err != nil {
		return miningBuildContext{}, err
	}
	return miningBuildContext{
		nextHeight:       nextHeight,
		prevHash:         prevHash,
		remainingWeight:  remainingWeight,
		alreadyGenerated: state.alreadyGenerated,
		utxos:            state.utxos,
		candidateTxs:     candidateTxs,
	}, nil
}

func (m *Miner) snapshotBuildContextState(copyPolicyUtxos bool) (miningChainStateSnapshot, error) {
	if m == nil || m.chainState == nil {
		return miningChainStateSnapshot{}, errors.New("nil chainstate")
	}
	state := m.chainState
	state.mu.RLock()
	defer state.mu.RUnlock()

	snapshot := miningChainStateSnapshot{
		hasTip:           state.HasTip,
		height:           state.Height,
		tipHash:          state.TipHash,
		alreadyGenerated: state.AlreadyGenerated,
	}
	if copyPolicyUtxos {
		snapshot.utxos = copyUtxoSet(state.Utxos)
	}
	return snapshot, nil
}

func (m *Miner) candidateTransactions(txs [][]byte) [][]byte {
	maxSelected := m.maxSelectedTransactions()
	if maxSelected == 0 {
		return nil
	}
	if len(txs) != 0 {
		return pickFlatCandidateRaw(txs, maxSelected)
	}
	if m.sync == nil || m.sync.mempool == nil {
		return nil
	}
	return m.mempoolCandidateTransactions(maxSelected)
}

func (m *Miner) maxSelectedTransactions() int {
	maxSelected := m.cfg.MaxTxPerBlock - 1
	if maxSelected < 0 {
		return 0
	}
	return maxSelected
}

func (m *Miner) remainingWeightBudget(nextHeight uint64, alreadyGenerated consensus.Uint128) (uint64, error) {
	coinbaseWeight, err := canonicalCoinbaseWeight(nextHeight, alreadyGenerated, m.cfg.MineAddress)
	if err != nil {
		return 0, err
	}
	return remainingWeightFromCoinbase(coinbaseWeight)
}

func canonicalCoinbaseWeight(height uint64, alreadyGenerated consensus.Uint128, mineAddress []byte) (uint64, error) {
	if height > math.MaxUint32 {
		return 0, errors.New("block height exceeds coinbase locktime range")
	}
	subsidy := consensus.BlockSubsidyBig(height, alreadyGenerated.Big())
	if subsidy > 0 {
		if err := validateMineAddress(mineAddress); err != nil {
			return 0, err
		}
	}

	outputCount := uint64(1)
	strippedSize := uint64(4 + 1 + 8) // version + tx_kind + tx_nonce
	parts := []uint64{
		compactSizeLenForMiner(1),
		32 + 4 + compactSizeLenForMiner(0) + 4,
	}
	if subsidy > 0 {
		outputCount++
		addrLen := uint64(len(mineAddress))
		parts = append(parts, 8+2+compactSizeLenForMiner(addrLen)+addrLen)
	}
	parts = append(
		parts,
		compactSizeLenForMiner(outputCount),
		8+2+compactSizeLenForMiner(32)+32,
		4,
	)
	if err := addCoinbaseBaseSize(&strippedSize, parts...); err != nil {
		return 0, err
	}

	// Canonical coinbase carries zero witness items and no DA payload, so the
	// discounted trailer still contributes one CompactSize byte for each field.
	witnessSize := compactSizeLenForMiner(0)
	daSize := compactSizeLenForMiner(0)
	return finalizeCoinbaseWeight(strippedSize, witnessSize, daSize)
}

func addCoinbaseBaseSize(dst *uint64, values ...uint64) error {
	for _, value := range values {
		if err := addU64NoOverflow(dst, value); err != nil {
			return errors.New("coinbase weight overflow")
		}
	}
	return nil
}

func finalizeCoinbaseWeight(strippedSize uint64, witnessSize uint64, daSize uint64) (uint64, error) {
	extraWeight, err := addU64NoOverflowValue(witnessSize, daSize)
	if err != nil {
		return 0, errors.New("coinbase weight overflow")
	}
	if strippedSize > (math.MaxUint64-extraWeight)/consensus.WITNESS_DISCOUNT_DIVISOR {
		return 0, errors.New("coinbase weight overflow")
	}
	return uint64(consensus.WITNESS_DISCOUNT_DIVISOR)*strippedSize + extraWeight, nil
}

func remainingWeightFromCoinbase(coinbaseWeight uint64) (uint64, error) {
	if coinbaseWeight > consensus.MAX_BLOCK_WEIGHT {
		return 0, errors.New("coinbase weight exceeds max block weight")
	}
	return consensus.MAX_BLOCK_WEIGHT - coinbaseWeight, nil
}

func addU64NoOverflowValue(left uint64, right uint64) (uint64, error) {
	if right > math.MaxUint64-left {
		return 0, errors.New("u64 overflow")
	}
	return left + right, nil
}

func compactSizeLenForMiner(n uint64) uint64 {
	switch {
	case n < 0xfd:
		return 1
	case n <= 0xffff:
		return 3
	case n <= 0xffff_ffff:
		return 5
	default:
		return 9
	}
}

func (m *Miner) selectCandidateTransactions(candidateTxs [][]byte, utxos map[consensus.Outpoint]consensus.UtxoEntry, nextHeight uint64, remainingWeight uint64) ([]minedCandidate, error) {
	maxSelected := m.maxSelectedTransactions()
	validationCtx, err := m.providerConsensusContext(nextHeight)
	if err != nil {
		return nil, err
	}
	parsed := make([]minedCandidate, 0, min(len(candidateTxs), maxSelected))
	var selectedWeight uint64
	var policyDaIncluded uint64
	selectedNonces := make(map[uint64]struct{}, maxSelected)
	selectedInputs := make(map[consensus.Outpoint]struct{}, maxSelected)
flatCandidates:
	for _, raw := range candidateTxs {
		if len(parsed) >= maxSelected {
			break
		}
		candidate, nextDaIncluded, ok, err := m.trySelectFlatCandidate(raw, utxos, nextHeight, selectedWeight, remainingWeight, policyDaIncluded, validationCtx)
		if err != nil {
			return nil, err
		}
		if ok {
			if candidate.minedCandidate.nonce == 0 {
				continue
			}
			if _, exists := selectedNonces[candidate.minedCandidate.nonce]; exists {
				continue
			}
			for _, input := range candidate.tx.Inputs {
				if _, exists := selectedInputs[consensus.Outpoint{Txid: input.PrevTxid, Vout: input.PrevVout}]; exists {
					continue flatCandidates
				}
			}
			selectedWeight += candidate.minedCandidate.weight
			policyDaIncluded = nextDaIncluded
			selectedNonces[candidate.minedCandidate.nonce] = struct{}{}
			for _, input := range candidate.tx.Inputs {
				selectedInputs[consensus.Outpoint{Txid: input.PrevTxid, Vout: input.PrevVout}] = struct{}{}
			}
			parsed = append(parsed, candidate.minedCandidate)
		}
	}
	if m.cfg.CompleteDASetProvider == nil || maxSelected == 0 ||
		len(parsed) >= maxSelected || selectedWeight >= remainingWeight {
		return parsed, nil
	}
	providerDaIncluded := uint64(0)
	selectedDAIDs := make(map[[32]byte]struct{})
	providerBudget := uint64(consensus.MAX_DA_BYTES_PER_BLOCK)
	if m.cfg.PolicyDaAnchorAntiAbuse && m.cfg.PolicyMaxDaBytesPerBlock > 0 && m.cfg.PolicyMaxDaBytesPerBlock < providerBudget {
		providerBudget = m.cfg.PolicyMaxDaBytesPerBlock
	}
	for _, set := range m.cfg.CompleteDASetProvider.CompleteDASetCandidates(providerBudget) {
		if len(parsed) >= maxSelected || len(selectedDAIDs) >= consensus.MAX_DA_BATCHES_PER_BLOCK {
			break
		}
		remainingSlots := maxSelected - len(parsed)
		if len(set.Chunks) >= remainingSlots {
			continue
		}
		group, ok, err := m.parseCompleteDASetCandidate(set)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		if _, exists := selectedDAIDs[group.daID]; exists || len(parsed)+len(group.txs) > maxSelected {
			continue
		}
		nextProviderDaIncluded, ok := updatedPolicyDaBytes(providerDaIncluded, group.daBytes, providerBudget)
		if !ok {
			continue
		}
		groupWeight, nextDaIncluded, ok := m.projectCompleteDASetGroup(group.txs, selectedNonces, selectedInputs, utxos, nextHeight, validationCtx, selectedWeight, remainingWeight, policyDaIncluded)
		if !ok {
			continue
		}
		selectedWeight += groupWeight
		policyDaIncluded = nextDaIncluded
		providerDaIncluded = nextProviderDaIncluded
		selectedDAIDs[group.daID] = struct{}{}
		for _, candidate := range group.txs {
			selectedNonces[candidate.minedCandidate.nonce] = struct{}{}
			for _, input := range candidate.tx.Inputs {
				selectedInputs[consensus.Outpoint{Txid: input.PrevTxid, Vout: input.PrevVout}] = struct{}{}
			}
			parsed = append(parsed, candidate.minedCandidate)
		}
	}
	return parsed, nil
}

func (m *Miner) providerConsensusContext(nextHeight uint64) (miningConsensusContext, error) {
	var ctx miningConsensusContext
	if m == nil {
		return ctx, nil
	}
	if m.blockStore != nil && nextHeight != 0 {
		prevTimestamps, err := m.prevTimestamps(nextHeight)
		if err != nil {
			return miningConsensusContext{}, err
		}
		if len(prevTimestamps) != 0 {
			ctx.blockMTP = mtpMedian(nextHeight, prevTimestamps)
		}
	}
	if m.sync != nil {
		ctx.chainID = m.sync.cfg.ChainID
		ctx.rotation = m.sync.cfg.RotationProvider
		ctx.registry = m.sync.cfg.SuiteRegistry
	}
	return ctx, nil
}

type completeDASetMiningCandidate struct {
	daID    [32]byte
	txs     []miningCandidate
	daBytes uint64
}

func (m *Miner) parseCompleteDASetCandidate(set CompleteDASetCandidate) (completeDASetMiningCandidate, bool, error) {
	commit, err := m.parseMiningCandidate(set.CommitTx)
	if err != nil {
		return completeDASetMiningCandidate{}, false, err
	}
	core := commit.tx.DaCommitCore
	if core == nil || core.DaID != set.DAID || core.ChunkCount == 0 || uint64(core.ChunkCount) > consensus.MAX_DA_CHUNK_COUNT ||
		len(set.Chunks) != int(core.ChunkCount) || len(commit.tx.Inputs) == 0 {
		return completeDASetMiningCandidate{}, false, nil
	}

	group := completeDASetMiningCandidate{daID: set.DAID, txs: []miningCandidate{commit}}
	group.daBytes = uint64(len(commit.tx.DaPayload))
	hasher := sha3.New256()
	for i, chunk := range set.Chunks {
		wantIndex := uint16(i)
		if chunk.Index != wantIndex {
			return completeDASetMiningCandidate{}, false, nil
		}
		candidate, err := m.parseMiningCandidate(chunk.Tx)
		if err != nil {
			return completeDASetMiningCandidate{}, false, err
		}
		tx := candidate.tx
		chunkCore := tx.DaChunkCore
		if chunkCore == nil || chunkCore.DaID != set.DAID || chunkCore.ChunkIndex != wantIndex ||
			len(tx.Inputs) == 0 || sha3.Sum256(tx.DaPayload) != chunkCore.ChunkHash {
			return completeDASetMiningCandidate{}, false, nil
		}
		nextDaBytes, err := addU64NoOverflowValue(group.daBytes, uint64(len(candidate.tx.DaPayload)))
		if err != nil {
			return completeDASetMiningCandidate{}, false, nil
		}
		group.daBytes = nextDaBytes
		_, _ = hasher.Write(candidate.tx.DaPayload)
		group.txs = append(group.txs, candidate)
	}
	if !completeDACommitmentMatches(commit.tx, hasher.Sum(nil)) {
		return completeDASetMiningCandidate{}, false, nil
	}
	return group, true, nil
}

func completeDACommitmentMatches(tx *consensus.Tx, commitment []byte) bool {
	count := 0
	for _, output := range tx.Outputs {
		if output.CovenantType != consensus.COV_TYPE_DA_COMMIT {
			continue
		}
		if count != 0 || len(output.CovenantData) != 32 || !bytes.Equal(output.CovenantData, commitment) {
			return false
		}
		count++
	}
	return count == 1
}

func (m *Miner) projectCompleteDASetGroup(group []miningCandidate, selectedNonces map[uint64]struct{}, selectedInputs map[consensus.Outpoint]struct{}, utxos map[consensus.Outpoint]consensus.UtxoEntry, nextHeight uint64, validationCtx miningConsensusContext, selectedWeight uint64, remainingWeight uint64, policyDaIncluded uint64) (uint64, uint64, bool) {
	if len(group) == 0 || selectedWeight > remainingWeight {
		return 0, policyDaIncluded, false
	}
	availableWeight := remainingWeight - selectedWeight
	var groupWeight uint64
	nextDaIncluded := policyDaIncluded

	groupInputOutpoints, ok := collectCompleteDASetGroupInputs(group, selectedNonces, selectedInputs)
	if !ok {
		return 0, policyDaIncluded, false
	}
	if !validateCompleteDASetGroupConsensus(group, utxos, groupInputOutpoints, nextHeight, validationCtx) {
		return 0, policyDaIncluded, false
	}
	for _, candidate := range group {
		reject, daIncluded, err := m.rejectCandidate(candidate.tx, utxos, nextHeight, nextDaIncluded)
		if err != nil || reject || groupWeight > availableWeight || candidate.minedCandidate.weight > availableWeight-groupWeight {
			return 0, policyDaIncluded, false
		}
		groupWeight += candidate.minedCandidate.weight
		nextDaIncluded = daIncluded
	}
	return groupWeight, nextDaIncluded, true
}

func collectCompleteDASetGroupInputs(group []miningCandidate, selectedNonces map[uint64]struct{}, selectedInputs map[consensus.Outpoint]struct{}) ([]consensus.Outpoint, bool) {
	groupNonces := make(map[uint64]struct{}, len(group))
	groupInputs := make(map[consensus.Outpoint]struct{}, len(group))
	groupInputOutpoints := make([]consensus.Outpoint, 0, len(group))
	for _, candidate := range group {
		nonce := candidate.minedCandidate.nonce
		if nonce == 0 {
			return nil, false
		}
		if _, exists := selectedNonces[nonce]; exists {
			return nil, false
		}
		if _, exists := groupNonces[nonce]; exists {
			return nil, false
		}
		for _, input := range candidate.tx.Inputs {
			op := consensus.Outpoint{Txid: input.PrevTxid, Vout: input.PrevVout}
			if _, exists := selectedInputs[op]; exists {
				return nil, false
			}
			if _, exists := groupInputs[op]; exists {
				return nil, false
			}
			groupInputs[op] = struct{}{}
			groupInputOutpoints = append(groupInputOutpoints, op)
		}
		groupNonces[nonce] = struct{}{}
	}
	return groupInputOutpoints, true
}

func validateCompleteDASetGroupConsensus(group []miningCandidate, utxos map[consensus.Outpoint]consensus.UtxoEntry, groupInputOutpoints []consensus.Outpoint, nextHeight uint64, validationCtx miningConsensusContext) bool {
	workUtxos := copySelectedUtxoSet(utxos, groupInputOutpoints)
	for _, candidate := range group {
		if _, err := consensus.CheckParsedTransactionWithOwnedUtxoSetAndSuiteContext(
			candidate.minedCandidate.raw,
			candidate.tx,
			consensus.ParsedTxIDs{TxID: candidate.minedCandidate.txid, WTxID: candidate.minedCandidate.wtxid},
			workUtxos,
			nextHeight,
			validationCtx.blockMTP,
			validationCtx.chainID,
			consensus.SuiteValidationContext{Rotation: validationCtx.rotation, Registry: validationCtx.registry},
		); err != nil {
			return false
		}
	}
	return true
}

func (m *Miner) trySelectFlatCandidate(raw []byte, utxos map[consensus.Outpoint]consensus.UtxoEntry, nextHeight uint64, selectedWeight uint64, remainingWeight uint64, policyDaIncluded uint64, validationCtx miningConsensusContext) (miningCandidate, uint64, bool, error) {
	candidate, err := m.parseMiningCandidate(raw)
	if err != nil {
		return miningCandidate{}, policyDaIncluded, false, err
	}
	if isMiningDATx(candidate.tx) {
		return miningCandidate{}, policyDaIncluded, false, nil
	}
	reject, nextDaIncluded, err := m.rejectCandidate(candidate.tx, utxos, nextHeight, policyDaIncluded)
	if err != nil || reject {
		return miningCandidate{}, policyDaIncluded, false, nil
	}
	if selectedWeight >= remainingWeight {
		return miningCandidate{}, policyDaIncluded, false, nil
	}
	availableWeight := remainingWeight - selectedWeight
	if candidate.minedCandidate.weight > availableWeight {
		return miningCandidate{}, policyDaIncluded, false, nil
	}
	workUtxos, err := policyInputSnapshot(candidate.tx, utxos)
	if err != nil {
		return miningCandidate{}, policyDaIncluded, false, nil
	}
	if _, err := consensus.CheckParsedTransactionWithOwnedUtxoSetAndSuiteContext(candidate.minedCandidate.raw, candidate.tx, consensus.ParsedTxIDs{TxID: candidate.minedCandidate.txid, WTxID: candidate.minedCandidate.wtxid}, workUtxos, nextHeight, validationCtx.blockMTP, validationCtx.chainID, consensus.SuiteValidationContext{Rotation: validationCtx.rotation, Registry: validationCtx.registry}); err != nil {
		return miningCandidate{}, policyDaIncluded, false, nil
	}
	return candidate, nextDaIncluded, true, nil
}

type miningCandidate struct {
	tx             *consensus.Tx
	minedCandidate minedCandidate
}

func (m *Miner) parseMiningCandidate(raw []byte) (miningCandidate, error) {
	tx, txid, wtxid, err := parseCanonicalTx(raw, "non-canonical tx bytes in miner input")
	if err != nil {
		return miningCandidate{}, err
	}
	txWeight, _, _, err := consensus.TxWeightAndStats(tx)
	if err != nil {
		return miningCandidate{}, err
	}
	return miningCandidate{
		tx: tx,
		minedCandidate: minedCandidate{
			raw:    append([]byte(nil), raw...),
			txid:   txid,
			wtxid:  wtxid,
			weight: txWeight,
			nonce:  tx.TxNonce,
		},
	}, nil
}

func (m *Miner) rejectCandidate(tx *consensus.Tx, utxos map[consensus.Outpoint]consensus.UtxoEntry, nextHeight uint64, policyDaIncluded uint64) (bool, uint64, error) {
	var reject bool
	var err error

	// PolicyDaAnchorAntiAbuse gates the full DA/anchor policy package,
	// including the non-coinbase CORE_ANCHOR sub-flag.
	if m.cfg.PolicyDaAnchorAntiAbuse {
		reject, policyDaIncluded, err = m.rejectCandidateDAPolicy(tx, utxos, policyDaIncluded)
		if err != nil {
			return false, policyDaIncluded, err
		}
		if reject {
			return true, policyDaIncluded, nil
		}

		reject, err = m.rejectCandidateAnchorPolicy(tx)
		if err != nil {
			return false, policyDaIncluded, err
		}
		if reject {
			return true, policyDaIncluded, nil
		}
	}

	// Apply Simplicity policy
	reject, err = m.rejectCandidateSimplicityPolicy(tx, utxos, nextHeight)
	if err != nil {
		return false, policyDaIncluded, err
	}
	if reject {
		return true, policyDaIncluded, nil
	}

	return false, policyDaIncluded, nil
}

func updatedPolicyDaBytes(current uint64, daBytes uint64, maxPerBlock uint64) (uint64, bool) {
	if daBytes == 0 || maxPerBlock == 0 {
		return current, true
	}
	next := current + daBytes
	if next < current || next > maxPerBlock {
		return current, false
	}
	return next, true
}

func buildWitnessCommitment(parsed []minedCandidate) ([32]byte, error) {
	wtxids := make([][32]byte, 1, 1+len(parsed))
	for _, p := range parsed {
		wtxids = append(wtxids, p.wtxid)
	}
	witnessRoot, err := consensus.WitnessMerkleRootWtxids(wtxids)
	if err != nil {
		return [32]byte{}, err
	}
	return consensus.WitnessCommitmentHash(witnessRoot), nil
}

func (m *Miner) buildCoinbaseAndMerkleRoot(nextHeight uint64, alreadyGenerated consensus.Uint128, witnessCommitment [32]byte, parsed []minedCandidate) ([]byte, [32]byte, error) {
	coinbase, err := buildCoinbaseTx(nextHeight, alreadyGenerated, m.cfg.MineAddress, witnessCommitment)
	if err != nil {
		return nil, [32]byte{}, err
	}
	_, coinbaseTxid, _, err := parseCanonicalTx(coinbase, "coinbase serialization is non-canonical")
	if err != nil {
		return nil, [32]byte{}, err
	}
	txids := make([][32]byte, 0, 1+len(parsed))
	txids = append(txids, coinbaseTxid)
	for _, p := range parsed {
		txids = append(txids, p.txid)
	}
	merkleRoot, err := consensus.MerkleRootTxids(txids)
	if err != nil {
		return nil, [32]byte{}, err
	}
	return coinbase, merkleRoot, nil
}

func (m *Miner) mineHeader(ctx context.Context, nextHeight uint64, prevHash [32]byte, merkleRoot [32]byte) ([]uint64, uint64, []byte, uint64, error) {
	target, err := m.templateTarget(prevHash, nextHeight)
	if err != nil {
		return nil, 0, nil, 0, err
	}
	prevTimestamps, err := m.prevTimestamps(nextHeight)
	if err != nil {
		return nil, 0, nil, 0, err
	}
	now := m.cfg.TimestampSource()
	timestamp := chooseValidTimestamp(nextHeight, prevTimestamps, now)
	blockWithoutNonce := makeHeaderPrefix(prevHash, merkleRoot, timestamp, target)
	headerBytes, nonce, err := mineHeaderNonce(ctx, blockWithoutNonce, target)
	if err != nil {
		return nil, 0, nil, 0, err
	}
	return prevTimestamps, timestamp, headerBytes, nonce, nil
}

// templateTarget returns the target the post-genesis template commits to, for
// BOTH header construction and the nonce search — the miner must never mine
// against a target the apply path will then reject.
//
// prevHash is the canonical tip snapshotted by buildContext under the chainstate
// read lock, i.e. this template's authoritative selected parent. On a stock
// devnet chain the derived target replaces MinerConfig.Target, so that field
// carries no static authority there; everywhere else the configured target is
// returned unchanged, including nextHeight == 0 (the empty-chain path, already
// redirected to the published genesis by BootstrapCanonicalGenesisIfEmpty).
func (m *Miner) templateTarget(prevHash [32]byte, nextHeight uint64) ([32]byte, error) {
	if nextHeight == 0 || m.sync == nil || m.blockStore == nil ||
		!StockDevnetTargetSchedule(m.sync.cfg.Network, m.sync.cfg.ChainID) {
		return m.cfg.Target, nil
	}
	return deriveExpectedTargetFn(m.blockStore, prevHash, nextHeight)
}

func assembleBlockBytes(headerBytes []byte, coinbase []byte, parsed []minedCandidate) []byte {
	blockBytes := make([]byte, 0, len(headerBytes)+4+len(coinbase))
	blockBytes = append(blockBytes, headerBytes...)
	blockBytes = consensus.AppendCompactSize(blockBytes, uint64(1+len(parsed)))
	blockBytes = append(blockBytes, coinbase...)
	for _, p := range parsed {
		blockBytes = append(blockBytes, p.raw...)
	}
	return blockBytes
}

func mineHeaderNonce(ctx context.Context, blockWithoutNonce []byte, target [32]byte) ([]byte, uint64, error) {
	var nonce uint64
	for {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return nil, 0, ctx.Err()
			default:
			}
		}
		headerBytes := consensus.AppendU64le(append([]byte(nil), blockWithoutNonce...), nonce)
		if err := consensus.PowCheck(headerBytes, target); err == nil {
			return headerBytes, nonce, nil
		}
		nonce++
	}
}
