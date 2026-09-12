package node

import (
	"bytes"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/2tbmz9y2xt-lang/rubin-protocol/clients/go/consensus"
)

// relayHarness is one signed-P2PK admission fixture: a spendable chainstate, a
// mempool bound to it, and the keys needed to build candidates against it.
type relayHarness struct {
	t         *testing.T
	fromKey   *consensus.MLDSA87Keypair
	fromAddr  []byte
	toAddr    []byte
	st        *ChainState
	outpoints []consensus.Outpoint
	mp        *Mempool
}

func newRelayHarness(t *testing.T, cfg *MempoolConfig, values ...uint64) *relayHarness {
	t.Helper()
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	h := &relayHarness{
		t:        t,
		fromKey:  fromKey,
		fromAddr: consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes()),
		toAddr:   consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes()),
	}
	h.st, h.outpoints = testSpendableChainState(h.fromAddr, values)
	var err error
	if cfg == nil {
		h.mp, err = NewMempool(h.st, nil, devnetGenesisChainID)
	} else {
		h.mp, err = NewMempoolWithConfig(h.st, nil, devnetGenesisChainID, *cfg)
	}
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	return h
}

func (h *relayHarness) tx(input int, amount, fee, nonce uint64) []byte {
	h.t.Helper()
	return mustBuildSignedTransferTx(h.t, h.st.Utxos, []consensus.Outpoint{h.outpoints[input]}, amount, fee, nonce, h.fromKey, h.fromAddr, h.toAddr)
}

// context returns the owner's current exact admission context.
func (h *relayHarness) context() *PendingOutpointAdmissionContext {
	h.t.Helper()
	ctx, ok := h.mp.PendingOutpointOwner().AdmissionContext()
	if !ok {
		h.t.Fatal("owner admission context unavailable")
	}
	return &ctx
}

// advanceOwnerGeneration commits a transition back onto the SAME stable tip, so
// the tip compares equal while the generation the owner is bound to has moved
// past every context observed before this call.
func (h *relayHarness) advanceOwnerGeneration() {
	h.t.Helper()
	owner := h.mp.PendingOutpointOwner()
	if _, err := owner.beginTransition(); err != nil {
		h.t.Fatalf("beginTransition: %v", err)
	}
	if err := owner.commitStableTip(pendingOutpointTipOf(h.st)); err != nil {
		h.t.Fatalf("commitStableTip: %v", err)
	}
}

func mustSnapshot(t *testing.T, mp *Mempool) mempoolSnapshot {
	t.Helper()
	snap, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshotMempool: %v", err)
	}
	return snap
}

func admitKind(t *testing.T, err error) TxAdmitErrorKind {
	t.Helper()
	var admitErr *TxAdmitError
	if !errors.As(err, &admitErr) {
		t.Fatalf("expected *TxAdmitError, got %T: %v", err, err)
	}
	return admitErr.Kind
}

// TestRelayAdmissionDispositionCorpus drives ONE candidate per branch of the
// closed classifier through the live public relay entry point and pins the
// disposition its originating branch selected, the unchanged public admission
// kind, and whether cache-authorizing context evidence was published.
//
// The rows deliberately include inputs whose public TxAdmitErrorKind is
// ambiguous — TxAdmitConflict carries both DUPLICATE and CONFLICT,
// TxAdmitUnavailable carries ROLLING_FLOOR, CAPACITY, ADMISSION_SEQUENCE and
// UNAVAILABLE — so a classifier that read the kind instead of the branch would
// fail here.
func TestRelayAdmissionDispositionCorpus(t *testing.T) {
	cases := []struct {
		name string
		// build returns the mempool, the candidate bytes, and the caller's
		// expected admission context (nil for the legacy-equivalent shape).
		build       func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext)
		want        RelayAdmissionDisposition
		wantKind    TxAdmitErrorKind
		wantNilErr  bool
		wantContext bool
		wantTxID    bool
	}{
		{
			name: "retained without expected context",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				return h.mp, h.tx(0, 100_000, 100_000, 1), nil
			},
			want:       RelayAdmissionRetained,
			wantNilErr: true,
			wantTxID:   true,
		},
		{
			name: "retained with the exact stable context carries no cache authority",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				return h.mp, h.tx(0, 100_000, 100_000, 1), h.context()
			},
			want:       RelayAdmissionRetained,
			wantNilErr: true,
			wantTxID:   true,
		},
		{
			name: "stable terminal reject: unparseable bytes",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				return h.mp, []byte{0xFF, 0x00, 0x13, 0x37}, h.context()
			},
			want:        RelayAdmissionStableTerminalReject,
			wantKind:    TxAdmitRejected,
			wantContext: true,
		},
		{
			name: "stable terminal reject: trailing bytes after canonical tx",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				return h.mp, append(h.tx(0, 100_000, 100_000, 1), 0x00), h.context()
			},
			want:        RelayAdmissionStableTerminalReject,
			wantKind:    TxAdmitRejected,
			wantContext: true,
		},
		{
			name: "stable terminal reject: consensus-invalid signature",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				return h.mp, corruptFirstWitnessSignature(t, h.tx(0, 100_000, 100_000, 1)), h.context()
			},
			want:        RelayAdmissionStableTerminalReject,
			wantKind:    TxAdmitRejected,
			wantContext: true,
			wantTxID:    true,
		},
		{
			name: "stable terminal reject: static anchor-output policy",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, &MempoolConfig{
					MaxTransactions:                      10,
					MaxBytes:                             1 << 20,
					PolicyRejectNonCoinbaseAnchorOutputs: true,
				}, 1_000_000)
				// A CORE_ANCHOR output of value 0 is CONSENSUS-valid
				// (clients/go/consensus/covenant_genesis.go
				// validateAnchorGenesisOutput), so the candidate reaches the
				// static-policy lane this row is named for and is rejected by
				// applyPolicyAgainstState — the branch that must select the tag.
				raw := mustBuildSignedAnchorOutputTx(t, h.st.Utxos, h.outpoints[0], 0, 100_000, 1, h.fromKey, h.fromAddr)
				return h.mp, raw, h.context()
			},
			want:        RelayAdmissionStableTerminalReject,
			wantKind:    TxAdmitRejected,
			wantContext: true,
			wantTxID:    true,
		},
		{
			name: "missing dependency never carries cache authority",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000, 1_000_000)
				raw := h.tx(1, 100_000, 100_000, 1)
				delete(h.st.Utxos, h.outpoints[1])
				return h.mp, raw, h.context()
			},
			want:     RelayAdmissionMissingDependency,
			wantKind: TxAdmitRejected,
			wantTxID: true,
		},
		{
			name: "duplicate resident txid keeps the conflict kind",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				raw := h.tx(0, 100_000, 100_000, 1)
				if err := h.mp.AddRemoteTx(raw); err != nil {
					t.Fatalf("seed AddRemoteTx: %v", err)
				}
				return h.mp, raw, h.context()
			},
			want:     RelayAdmissionDuplicate,
			wantKind: TxAdmitConflict,
			wantTxID: true,
		},
		{
			name: "pending-outpoint owner conflict keeps the conflict kind",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				if err := h.mp.AddRemoteTx(h.tx(0, 100_000, 100_000, 1)); err != nil {
					t.Fatalf("seed AddRemoteTx: %v", err)
				}
				return h.mp, h.tx(0, 100_000, 200_000, 2), h.context()
			},
			want:     RelayAdmissionConflict,
			wantKind: TxAdmitConflict,
			wantTxID: true,
		},
		{
			// The cheap-precheck floor authority: a single-input plain P2PK
			// shape is fast-rejected at mempool_precheck.go's precheck wrap
			// before any signature verification.
			name: "rolling floor: cheap precheck authority",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				raw := h.tx(0, 100_000, 100_000, 1)
				h.mp.SetCurrentMinFeeRateForTest(1 << 40)
				return h.mp, raw, h.context()
			},
			want:     RelayAdmissionRollingFloor,
			wantKind: TxAdmitUnavailable,
			wantTxID: true,
		},
		{
			// The LOCKED floor authority (validateFeeFloorLockedWithFloor, the
			// max(snappedFloor, live) re-check). Two inputs mean two witness
			// items, which disqualifies feePrecheckP2PKInputValue, so the cheap
			// precheck defers and the locked check owns the outcome.
			name: "rolling floor: locked max(snapped,live) authority",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000, 1_000_000)
				raw := mustBuildSignedTransferTx(t, h.st.Utxos, h.outpoints, 100_000, 100_000, 1, h.fromKey, h.fromAddr, h.toAddr)
				h.mp.SetCurrentMinFeeRateForTest(1 << 40)
				return h.mp, raw, h.context()
			},
			want:     RelayAdmissionRollingFloor,
			wantKind: TxAdmitUnavailable,
			wantTxID: true,
		},
		{
			name: "capacity: candidate is the worst under byte pressure",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000, 1_000_000)
				rich := h.tx(0, 100_000, 200_000, 1)
				poor := h.tx(1, 100_000, 100_000, 2)
				mp, err := NewMempoolWithConfig(h.st, nil, devnetGenesisChainID, MempoolConfig{
					MaxTransactions: 10,
					MaxBytes:        len(rich) + len(poor) - 1,
				})
				if err != nil {
					t.Fatalf("new mempool: %v", err)
				}
				if err := mp.AddRemoteTx(rich); err != nil {
					t.Fatalf("seed AddRemoteTx: %v", err)
				}
				ctx, ok := mp.PendingOutpointOwner().AdmissionContext()
				if !ok {
					t.Fatal("owner admission context unavailable")
				}
				return mp, poor, &ctx
			},
			want:     RelayAdmissionCapacity,
			wantKind: TxAdmitUnavailable,
			wantTxID: true,
		},
		{
			name: "admission sequence exhausted",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				raw := h.tx(0, 100_000, 100_000, 1)
				h.mp.mu.Lock()
				h.mp.lastAdmissionSeq = ^uint64(0)
				h.mp.mu.Unlock()
				return h.mp, raw, h.context()
			},
			want:     RelayAdmissionAdmissionSequence,
			wantKind: TxAdmitUnavailable,
			wantTxID: true,
		},
		{
			name: "unavailable: absent chainstate",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				return &Mempool{}, []byte{0x01}, nil
			},
			want:     RelayAdmissionUnavailable,
			wantKind: TxAdmitUnavailable,
		},
		{
			name: "unavailable: superseded expected context at the owner seam",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				raw := h.tx(0, 100_000, 100_000, 1)
				stale := h.context()
				h.advanceOwnerGeneration()
				return h.mp, raw, stale
			},
			want:     RelayAdmissionUnavailable,
			wantKind: TxAdmitUnavailable,
			wantTxID: true,
		},
		{
			name: "unavailable: owner transition in progress",
			build: func(t *testing.T) (*Mempool, []byte, *PendingOutpointAdmissionContext) {
				h := newRelayHarness(t, nil, 1_000_000)
				raw := h.tx(0, 100_000, 100_000, 1)
				expected := h.context()
				if _, err := h.mp.PendingOutpointOwner().beginTransition(); err != nil {
					t.Fatalf("beginTransition: %v", err)
				}
				return h.mp, raw, expected
			},
			want:     RelayAdmissionUnavailable,
			wantKind: TxAdmitUnavailable,
			wantTxID: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mp, raw, expected := tc.build(t)
			got := mp.AddRemoteTxForRelay(raw, expected)

			if got.Disposition != tc.want {
				t.Fatalf("disposition=%v, want %v (err=%v)", got.Disposition, tc.want, got.Err)
			}
			if got.Disposition == RelayAdmissionCancelled {
				t.Fatal("no production branch may select CANCELLED")
			}
			if tc.wantNilErr {
				if got.Err != nil {
					t.Fatalf("err=%v, want nil", got.Err)
				}
			} else {
				if got.Err == nil {
					t.Fatal("err=nil, want a public admission error")
				}
				if kind := admitKind(t, got.Err); kind != tc.wantKind {
					t.Fatalf("TxAdmitErrorKind=%q, want %q", kind, tc.wantKind)
				}
			}
			if got.HasAdmissionContext != tc.wantContext {
				t.Fatalf("HasAdmissionContext=%v, want %v", got.HasAdmissionContext, tc.wantContext)
			}
			if got.HasAdmissionContext {
				if got.Disposition != RelayAdmissionStableTerminalReject {
					t.Fatalf("disposition %v published cache-authorizing context", got.Disposition)
				}
				if got.AdmissionContext != *expected {
					t.Fatalf("AdmissionContext=%+v, want the exact expected context %+v", got.AdmissionContext, *expected)
				}
			} else if got.AdmissionContext != (PendingOutpointAdmissionContext{}) {
				t.Fatalf("AdmissionContext=%+v published without HasAdmissionContext", got.AdmissionContext)
			}
			if hasTxID := got.TxID != ([32]byte{}); hasTxID != tc.wantTxID {
				t.Fatalf("TxID present=%v (%x), want present=%v", hasTxID, got.TxID, tc.wantTxID)
			}
			if tc.wantTxID {
				if want := txID(t, raw); got.TxID != want {
					t.Fatalf("TxID=%x, want the producer-parsed %x", got.TxID, want)
				}
				if got.WTxID == ([32]byte{}) {
					t.Fatalf("WTxID is zero for a parsed candidate")
				}
			}
		})
	}
}

// TestRelayAdmissionDispositionNeverInfersFromTheErrorSurface proves the
// classification is READ from the producing branch's own selection and is never
// reconstructed from the compatibility kind, the message text, or a default.
func TestRelayAdmissionDispositionNeverInfersFromTheErrorSurface(t *testing.T) {
	// An error no branch classified is INTERNAL, never stable-terminal — even
	// when its kind and message look exactly like a stable rejection.
	untagged := []error{
		txAdmitRejected("TX_ERR_SIG_INVALID: signature invalid"),
		txAdmitConflict("tx already in mempool"),
		txAdmitUnavailable("mempool fee below rolling minimum: fee=0 weight=1 min_fee_rate=9"),
		errors.New("not an admission error"),
	}
	for _, err := range untagged {
		if got := relayDispositionOf(err); got != RelayAdmissionInternal {
			t.Fatalf("relayDispositionOf(%v)=%v, want INTERNAL", err, got)
		}
	}

	// The FIRST selection wins: an outer class tag cannot overwrite the precise
	// disposition an inner branch already selected.
	inner := txAdmitConflict("tx already in mempool")
	tagged := selectRelayDisposition(inner, RelayAdmissionDuplicate)
	tagged = selectRelayDisposition(tagged, RelayAdmissionInternal)
	if got := relayDispositionOf(tagged); got != RelayAdmissionDuplicate {
		t.Fatalf("disposition=%v after an outer tag, want the inner DUPLICATE", got)
	}

	// Tagging changes no public surface.
	if inner.Kind != TxAdmitConflict || inner.Error() != "tx already in mempool" {
		t.Fatalf("tagging mutated the public error surface: kind=%q message=%q", inner.Kind, inner.Error())
	}

	// Pass-through shapes.
	if selectRelayDisposition(nil, RelayAdmissionRetained) != nil {
		t.Fatal("selectRelayDisposition(nil) must stay nil")
	}
	plain := errors.New("plain")
	if selectRelayDisposition(plain, RelayAdmissionCapacity) != plain {
		t.Fatal("selectRelayDisposition must return a non-admission error unchanged")
	}
}

// TestRelayAdmissionDispositionEnumIsClosed pins the eleven-value taxonomy and
// the deliberately-unselected zero value.
func TestRelayAdmissionDispositionEnumIsClosed(t *testing.T) {
	want := map[RelayAdmissionDisposition]string{
		RelayAdmissionRetained:             "RETAINED",
		RelayAdmissionStableTerminalReject: "STABLE_TERMINAL_REJECT",
		RelayAdmissionDuplicate:            "DUPLICATE",
		RelayAdmissionConflict:             "CONFLICT",
		RelayAdmissionMissingDependency:    "MISSING_DEPENDENCY",
		RelayAdmissionRollingFloor:         "ROLLING_FLOOR",
		RelayAdmissionCapacity:             "CAPACITY",
		RelayAdmissionAdmissionSequence:    "ADMISSION_SEQUENCE",
		RelayAdmissionUnavailable:          "UNAVAILABLE",
		RelayAdmissionInternal:             "INTERNAL",
		RelayAdmissionCancelled:            "CANCELLED",
	}
	if len(want) != 11 {
		t.Fatalf("taxonomy has %d values, want exactly 11", len(want))
	}
	seen := make(map[RelayAdmissionDisposition]struct{}, len(want))
	for value, name := range want {
		if value == 0 {
			t.Fatalf("%s must not be the zero value", name)
		}
		if _, dup := seen[value]; dup {
			t.Fatalf("%s duplicates another enum value", name)
		}
		seen[value] = struct{}{}
		if got := value.String(); got != name {
			t.Fatalf("String()=%q, want %q", got, name)
		}
	}
	for _, outside := range []RelayAdmissionDisposition{0, 12, 255} {
		if got := outside.String(); got != "UNSELECTED" {
			t.Fatalf("String() for %d = %q, want UNSELECTED", outside, got)
		}
	}
}

// TestRelayAdmissionDispositionTypedProducerMappings pins the two producer
// mappings that read a TYPED code from the failing producer rather than any
// rendered text, including the retryable-dependency class that must never be
// published as a stable terminal rejection.
func TestRelayAdmissionDispositionTypedProducerMappings(t *testing.T) {
	ownerCases := []struct {
		err  error
		want RelayAdmissionDisposition
	}{
		{&PendingOutpointError{Kind: PendingOutpointConflict, Msg: "mempool double-spend conflict with ab"}, RelayAdmissionConflict},
		{&PendingOutpointError{Kind: PendingOutpointUnavailable, Msg: "pending-outpoint expected tip mismatch"}, RelayAdmissionUnavailable},
		{&PendingOutpointError{Kind: PendingOutpointInternal, Msg: "zero or foreign pending-outpoint token"}, RelayAdmissionInternal},
		{errors.New("not an owner error"), RelayAdmissionInternal},
	}
	for _, tc := range ownerCases {
		if got := relayDispositionForOwnerError(tc.err); got != tc.want {
			t.Fatalf("relayDispositionForOwnerError(%v)=%v, want %v", tc.err, got, tc.want)
		}
	}

	dependency := []consensus.ErrorCode{
		consensus.TX_ERR_MISSING_UTXO,
		consensus.TX_ERR_COINBASE_IMMATURE,
		consensus.TX_ERR_TIMELOCK_NOT_MET,
	}
	for _, code := range dependency {
		err := &consensus.TxError{Code: code, Msg: "input"}
		for _, nonDependency := range []RelayAdmissionDisposition{RelayAdmissionStableTerminalReject, RelayAdmissionInternal} {
			if got := relayDispositionForInputError(err, nonDependency); got != RelayAdmissionMissingDependency {
				t.Fatalf("relayDispositionForInputError(%s)=%v, want MISSING_DEPENDENCY", code, got)
			}
		}
	}
	stable := &consensus.TxError{Code: consensus.TX_ERR_SIG_INVALID, Msg: "bad signature"}
	if got := relayDispositionForInputError(stable, RelayAdmissionStableTerminalReject); got != RelayAdmissionStableTerminalReject {
		t.Fatalf("non-dependency consensus failure=%v, want STABLE_TERMINAL_REJECT", got)
	}
	if got := relayDispositionForInputError(errors.New("nil utxo set"), RelayAdmissionInternal); got != RelayAdmissionInternal {
		t.Fatalf("non-typed input failure=%v, want the branch's own INTERNAL", got)
	}
}

// TestRelayAdmissionDispositionRetainedProvesResidency pins RETAINED against the
// live index rather than against a nil error: a published record that is not the
// exact candidate is the retained-identity-mismatch invariant, i.e. INTERNAL.
func TestRelayAdmissionDispositionRetainedProvesResidency(t *testing.T) {
	entry := &mempoolEntry{txid: [32]byte{0x01}, wtxid: [32]byte{0x02}, weight: 1, size: 1}
	mp := &Mempool{txs: map[[32]byte]*mempoolEntry{entry.txid: entry}}

	probe := &relayAdmissionProbe{}
	mp.noteRetainedLocked(entry, probe)
	if probe.success != RelayAdmissionRetained {
		t.Fatalf("resident exact candidate selected %v, want RETAINED", probe.success)
	}
	if probe.txid != entry.txid || probe.wtxid != entry.wtxid {
		t.Fatalf("identity=(%x,%x), want the candidate's own (%x,%x)", probe.txid, probe.wtxid, entry.txid, entry.wtxid)
	}

	impostor := &mempoolEntry{txid: entry.txid, wtxid: entry.wtxid, weight: 1, size: 1}
	mismatch := &relayAdmissionProbe{}
	mp.noteRetainedLocked(impostor, mismatch)
	if mismatch.success != RelayAdmissionInternal {
		t.Fatalf("retained identity mismatch selected %v, want INTERNAL", mismatch.success)
	}

	missing := &relayAdmissionProbe{}
	(&Mempool{}).noteRetainedLocked(entry, missing)
	if missing.success != RelayAdmissionInternal {
		t.Fatalf("absent record selected %v, want INTERNAL", missing.success)
	}

	// A nil probe is the legacy path and must stay a no-op — including result,
	// which the type documents as nil-safe like every other method here.
	mp.noteRetainedLocked(entry, nil)
	var legacy *relayAdmissionProbe
	cause := txAdmitUnavailable("nil mempool")
	if got := legacy.result(selectRelayDisposition(cause, RelayAdmissionUnavailable)); got.Disposition != RelayAdmissionUnavailable || got.Err != error(cause) {
		t.Fatalf("nil probe result=%+v, want the producer's UNAVAILABLE and its error", got)
	}
	if got := legacy.result(nil); got.Disposition != RelayAdmissionInternal {
		t.Fatalf("nil probe with an unselected outcome=%v, want INTERNAL fail closed", got.Disposition)
	}
}

// TestRelayAdmissionDispositionDuplicateWtxidBranch covers the second duplicate
// branch: a candidate whose wtxid is already indexed under another txid. It is
// unreachable through two distinct signed candidates, so the index is seeded
// directly, exactly as the baseline wtxid test does.
func TestRelayAdmissionDispositionDuplicateWtxidBranch(t *testing.T) {
	h := newRelayHarness(t, nil, 1_000_000, 1_000_000)
	tx1 := h.tx(0, 100_000, 100_000, 1)
	if err := h.mp.AddRemoteTx(tx1); err != nil {
		t.Fatalf("seed AddRemoteTx: %v", err)
	}
	tx2 := h.tx(1, 100_000, 100_000, 2)
	_, tx2ID, tx2Wtxid, _, err := consensus.ParseTx(tx2)
	if err != nil {
		t.Fatalf("ParseTx(tx2): %v", err)
	}
	h.mp.mu.Lock()
	h.mp.wtxids[tx2Wtxid] = txID(t, tx1)
	h.mp.mu.Unlock()

	got := h.mp.AddRemoteTxForRelay(tx2, h.context())
	if got.Disposition != RelayAdmissionDuplicate {
		t.Fatalf("disposition=%v, want DUPLICATE (err=%v)", got.Disposition, got.Err)
	}
	if kind := admitKind(t, got.Err); kind != TxAdmitConflict {
		t.Fatalf("TxAdmitErrorKind=%q, want %q", kind, TxAdmitConflict)
	}
	if got.HasAdmissionContext {
		t.Fatal("a duplicate must never publish cache-authorizing context")
	}
	if got.TxID != tx2ID || got.WTxID != tx2Wtxid {
		t.Fatalf("identity=(%x,%x), want the producer-parsed (%x,%x)", got.TxID, got.WTxID, tx2ID, tx2Wtxid)
	}
	if h.mp.Contains(tx2ID) {
		t.Fatal("a duplicate candidate entered the mempool")
	}
}

// TestRelayAdmissionDispositionDuplicateTxidWithAlternateWtxid covers the mirror
// hostile row: a re-signed candidate carries the SAME txid under a different
// wtxid, and the resident-txid branch must keep winning over the wtxid branch.
func TestRelayAdmissionDispositionDuplicateTxidWithAlternateWtxid(t *testing.T) {
	h := newRelayHarness(t, nil, 1_000_000)
	first := h.tx(0, 100_000, 100_000, 1)
	second := h.tx(0, 100_000, 100_000, 1)
	if bytes.Equal(first, second) {
		t.Skip("signature backend is deterministic: no alternate witness representation available")
	}
	_, firstTxid, firstWtxid, _, err := consensus.ParseTx(first)
	if err != nil {
		t.Fatalf("ParseTx(first): %v", err)
	}
	_, secondTxid, secondWtxid, _, err := consensus.ParseTx(second)
	if err != nil {
		t.Fatalf("ParseTx(second): %v", err)
	}
	if firstTxid != secondTxid || firstWtxid == secondWtxid {
		t.Fatalf("fixture is not same-txid/alternate-wtxid: txids (%x,%x) wtxids (%x,%x)", firstTxid, secondTxid, firstWtxid, secondWtxid)
	}
	if err := h.mp.AddRemoteTx(first); err != nil {
		t.Fatalf("seed AddRemoteTx: %v", err)
	}

	got := h.mp.AddRemoteTxForRelay(second, h.context())
	if got.Disposition != RelayAdmissionDuplicate {
		t.Fatalf("disposition=%v, want DUPLICATE (err=%v)", got.Disposition, got.Err)
	}
	// The resident-txid branch owns this outcome, not the wtxid branch.
	if got.Err.Error() != "tx already in mempool" {
		t.Fatalf("message=%q, want the resident-txid branch's %q", got.Err.Error(), "tx already in mempool")
	}
	if kind := admitKind(t, got.Err); kind != TxAdmitConflict {
		t.Fatalf("TxAdmitErrorKind=%q, want %q", kind, TxAdmitConflict)
	}
	if got.WTxID != secondWtxid {
		t.Fatalf("WTxID=%x, want the producer-parsed %x", got.WTxID, secondWtxid)
	}
	if got.HasAdmissionContext {
		t.Fatal("a duplicate must never publish cache-authorizing context")
	}
}

// TestRelayAdmissionDispositionMixedViolations pins the disposition of
// candidates that violate TWO rules at once. The winner must be the branch the
// baseline validation order reaches first, unchanged by this surface.
func TestRelayAdmissionDispositionMixedViolations(t *testing.T) {
	t.Run("missing input plus malformed witness stays a dependency gap", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000, 1_000_000)
		raw := corruptFirstWitnessSignature(t, h.tx(1, 100_000, 100_000, 1))
		delete(h.st.Utxos, h.outpoints[1])
		got := h.mp.AddRemoteTxForRelay(raw, h.context())
		if got.Disposition != RelayAdmissionMissingDependency {
			t.Fatalf("disposition=%v, want MISSING_DEPENDENCY (err=%v)", got.Disposition, got.Err)
		}
		if got.HasAdmissionContext {
			t.Fatal("a dependency gap must never publish cache-authorizing context")
		}
	})

	t.Run("below floor plus invalid signature keeps the cheap-floor precedence", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		raw := corruptFirstWitnessSignature(t, h.tx(0, 100_000, 100_000, 1))
		h.mp.SetCurrentMinFeeRateForTest(1 << 40)
		got := h.mp.AddRemoteTxForRelay(raw, h.context())
		if got.Disposition != RelayAdmissionRollingFloor {
			t.Fatalf("disposition=%v, want ROLLING_FLOOR (err=%v)", got.Disposition, got.Err)
		}
		if got.HasAdmissionContext {
			t.Fatal("a floor rejection must never publish cache-authorizing context")
		}
	})

	t.Run("below floor plus a shape the precheck defers on is stable terminal", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		// tx_nonce == 0 is a shape the cheap precheck refuses to decide, so the
		// slow path owns the outcome and reports terminal invalidity.
		raw := h.tx(0, 100_000, 100_000, 0)
		h.mp.SetCurrentMinFeeRateForTest(1 << 40)
		got := h.mp.AddRemoteTxForRelay(raw, h.context())
		if got.Disposition != RelayAdmissionStableTerminalReject {
			t.Fatalf("disposition=%v, want STABLE_TERMINAL_REJECT (err=%v)", got.Disposition, got.Err)
		}
		if !got.HasAdmissionContext {
			t.Fatal("a proven stable terminal rejection must publish its exact context")
		}
	})
}

// TestRelayAdmissionDispositionContextTransitionsNeverAuthorizeCaching walks the
// owner generations a tip comparison alone cannot see. An A-to-B-to-A reorg and
// an aborted transition both leave the stable tip equal and the generation past
// every earlier context, so a cached expected context must lose.
func TestRelayAdmissionDispositionContextTransitionsNeverAuthorizeCaching(t *testing.T) {
	t.Run("A to B to A with the same tip", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		raw := h.tx(0, 100_000, 100_000, 1)
		stale := h.context()

		owner := h.mp.PendingOutpointOwner()
		otherTip := PendingOutpointTip{HasTip: true, Height: 101, Hash: [32]byte{0x22}}
		if _, err := owner.beginTransition(); err != nil {
			t.Fatalf("beginTransition(B): %v", err)
		}
		if err := owner.commitStableTip(otherTip); err != nil {
			t.Fatalf("commitStableTip(B): %v", err)
		}
		if _, err := owner.beginTransition(); err != nil {
			t.Fatalf("beginTransition(A): %v", err)
		}
		if err := owner.commitStableTip(pendingOutpointTipOf(h.st)); err != nil {
			t.Fatalf("commitStableTip(A): %v", err)
		}

		current, ok := owner.AdmissionContext()
		if !ok {
			t.Fatal("owner admission context unavailable")
		}
		if current.StableTip != stale.StableTip {
			t.Fatalf("stable tip moved: got %+v, want the original %+v", current.StableTip, stale.StableTip)
		}
		if current.Generation == stale.Generation {
			t.Fatal("generation did not advance across the reorg")
		}

		before := mustSnapshot(t, h.mp)
		got := h.mp.AddRemoteTxForRelay(raw, stale)
		if got.Disposition != RelayAdmissionUnavailable {
			t.Fatalf("disposition=%v, want UNAVAILABLE (err=%v)", got.Disposition, got.Err)
		}
		if got.HasAdmissionContext {
			t.Fatal("a superseded context must never be published as cache authority")
		}
		// The owner refuses an unavailable binding before it scans for a
		// conflict and before it consumes a sequence: nothing moved at all.
		if after := mustSnapshot(t, h.mp); !reflect.DeepEqual(after, before) {
			t.Fatalf("superseded-context admission mutated state: before=%+v after=%+v", before, after)
		}
	})

	t.Run("superseded context loses to nothing at the conflict slot", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		if err := h.mp.AddRemoteTx(h.tx(0, 100_000, 100_000, 1)); err != nil {
			t.Fatalf("seed AddRemoteTx: %v", err)
		}
		conflicting := h.tx(0, 100_000, 200_000, 2)
		stale := h.context()
		h.advanceOwnerGeneration()

		// The SAME candidate is a live owner conflict. Availability is checked
		// first, so the superseded context wins over the conflict.
		got := h.mp.AddRemoteTxForRelay(conflicting, stale)
		if got.Disposition != RelayAdmissionUnavailable {
			t.Fatalf("disposition=%v, want UNAVAILABLE ahead of the conflict (err=%v)", got.Disposition, got.Err)
		}
		if got.HasAdmissionContext {
			t.Fatal("a superseded context must never be published as cache authority")
		}
		// With a current context the same candidate reaches the conflict.
		current := h.context()
		if again := h.mp.AddRemoteTxForRelay(conflicting, current); again.Disposition != RelayAdmissionConflict {
			t.Fatalf("disposition=%v with the current context, want CONFLICT (err=%v)", again.Disposition, again.Err)
		}
	})
}

// TestAddRemoteTxForRelayMatchesAddRemoteTxExactly proves the new entry point is
// a compatibility-preserving twin of AddRemoteTx: same public error kind and
// message, same residency, same admission counters, same rolling floor, same
// byte accounting, for accepted and rejected candidates alike.
func TestAddRemoteTxForRelayMatchesAddRemoteTxExactly(t *testing.T) {
	relay := newRelayHarness(t, nil, 1_000_000, 1_000_000)
	// The SAME candidate bytes drive both sides, so every public message —
	// including the ones that embed a conflicting txid — must compare equal.
	// testSpendableChainState is deterministic for a given address, so the
	// legacy twin starts from an identical, separately owned chainstate.
	legacySt, _ := testSpendableChainState(relay.fromAddr, []uint64{1_000_000, 1_000_000})
	legacyMp, err := NewMempool(legacySt, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new legacy mempool: %v", err)
	}

	accepted := relay.tx(0, 100_000, 100_000, 1)
	candidates := [][]byte{
		accepted,                         // accepted
		accepted,                         // duplicate txid
		relay.tx(0, 100_000, 200_000, 2), // owner conflict
		{0xFF, 0x00},                     // unparseable
		append(relay.tx(1, 100_000, 100_000, 3), 0x00),                    // trailing bytes
		corruptFirstWitnessSignature(t, relay.tx(1, 100_000, 100_000, 4)), // consensus-invalid
	}

	for i := range candidates {
		legacyErr := legacyMp.AddRemoteTx(candidates[i])
		result := relay.mp.AddRemoteTxForRelay(candidates[i], nil)

		switch {
		case legacyErr == nil && result.Err != nil:
			t.Fatalf("candidate %d: legacy accepted, relay returned %v", i, result.Err)
		case legacyErr != nil && result.Err == nil:
			t.Fatalf("candidate %d: legacy returned %v, relay accepted", i, legacyErr)
		case legacyErr != nil:
			if got, want := admitKind(t, result.Err), admitKind(t, legacyErr); got != want {
				t.Fatalf("candidate %d: relay kind=%q, want the legacy %q", i, got, want)
			}
			if got, want := result.Err.Error(), legacyErr.Error(); got != want {
				t.Fatalf("candidate %d: relay message=%q, want the legacy %q", i, got, want)
			}
		}
		if result.HasAdmissionContext {
			t.Fatalf("candidate %d: a nil expected context authorized caching", i)
		}
		if result.AdmissionContext != (PendingOutpointAdmissionContext{}) {
			t.Fatalf("candidate %d: a nil expected context published a context", i)
		}
	}

	if got, want := relay.mp.AdmissionCounts(), legacyMp.AdmissionCounts(); got != want {
		t.Fatalf("admission counters=%+v, want the legacy %+v", got, want)
	}
	if got, want := relay.mp.Stats(), legacyMp.Stats(); got != want {
		t.Fatalf("mempool stats=%+v, want the legacy %+v", got, want)
	}
	if got, want := relay.mp.Len(), legacyMp.Len(); got != want {
		t.Fatalf("mempool len=%d, want the legacy %d", got, want)
	}
	relay.mp.mu.RLock()
	relaySeq := relay.mp.lastAdmissionSeq
	relay.mp.mu.RUnlock()
	legacyMp.mu.RLock()
	legacySeq := legacyMp.lastAdmissionSeq
	legacyMp.mu.RUnlock()
	if relaySeq != legacySeq {
		t.Fatalf("lastAdmissionSeq=%d, want the legacy %d", relaySeq, legacySeq)
	}
}

// TestAddRemoteTxForRelayValidatesExpectedContextAtTheOwnerSeam pins the
// ordering clause: a supplied expected context is validated AT the owner seam
// and never earlier, so it cannot preempt the baseline parse or duplicate
// precedence — and validating it mutates nothing.
func TestAddRemoteTxForRelayValidatesExpectedContextAtTheOwnerSeam(t *testing.T) {
	t.Run("parse precedes the context check", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		stale := h.context()
		h.advanceOwnerGeneration()
		before := mustSnapshot(t, h.mp)

		got := h.mp.AddRemoteTxForRelay([]byte{0xFF, 0x00, 0x13, 0x37}, stale)
		if got.Disposition != RelayAdmissionStableTerminalReject {
			t.Fatalf("disposition=%v, want STABLE_TERMINAL_REJECT ahead of the context check (err=%v)", got.Disposition, got.Err)
		}
		if got.HasAdmissionContext {
			t.Fatal("a superseded context must not be published for the parse rejection")
		}
		if after := mustSnapshot(t, h.mp); !reflect.DeepEqual(after, before) {
			t.Fatalf("classification mutated state: before=%+v after=%+v", before, after)
		}
	})

	t.Run("the duplicate slot precedes the context check", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		raw := h.tx(0, 100_000, 100_000, 1)
		if err := h.mp.AddRemoteTx(raw); err != nil {
			t.Fatalf("seed AddRemoteTx: %v", err)
		}
		stale := h.context()
		h.advanceOwnerGeneration()

		got := h.mp.AddRemoteTxForRelay(raw, stale)
		if got.Disposition != RelayAdmissionDuplicate {
			t.Fatalf("disposition=%v, want DUPLICATE ahead of the context check (err=%v)", got.Disposition, got.Err)
		}
	})

	t.Run("an unavailable owner context is not proven and admits nothing", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		raw := h.tx(0, 100_000, 100_000, 1)
		expected := h.context()
		if _, err := h.mp.PendingOutpointOwner().beginTransition(); err != nil {
			t.Fatalf("beginTransition: %v", err)
		}
		got := h.mp.AddRemoteTxForRelay(raw, expected)
		if got.Disposition != RelayAdmissionUnavailable {
			t.Fatalf("disposition=%v, want UNAVAILABLE (err=%v)", got.Disposition, got.Err)
		}
		if got.HasAdmissionContext {
			t.Fatal("an active transition must never publish cache authority")
		}
		if h.mp.Len() != 0 {
			t.Fatal("a candidate entered the mempool during an active owner transition")
		}
	})

	t.Run("an input-less candidate still validates the supplied context", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		stale := h.context()
		h.advanceOwnerGeneration()

		entry := &mempoolEntry{txid: [32]byte{0x71}, wtxid: [32]byte{0x72}, weight: 1, size: 1}
		probe := newRelayAdmissionProbe(stale)
		h.mp.mu.Lock()
		err := h.mp.reserveEntryInputsLocked(entry, probe)
		h.mp.mu.Unlock()
		if err == nil {
			t.Fatal("a superseded context must not pass the input-less seam")
		}
		if got := relayDispositionOf(err); got != RelayAdmissionUnavailable {
			t.Fatalf("disposition=%v, want UNAVAILABLE", got)
		}
		if entry.token != (PendingOutpointToken{}) {
			t.Fatal("the input-less seam issued a token")
		}

		// The current context passes, and a nil probe keeps the baseline no-op.
		probe = newRelayAdmissionProbe(h.context())
		h.mp.mu.Lock()
		err = h.mp.reserveEntryInputsLocked(entry, probe)
		if err == nil {
			err = h.mp.reserveEntryInputsLocked(entry, nil)
		}
		h.mp.mu.Unlock()
		if err != nil {
			t.Fatalf("input-less seam with the current context: %v", err)
		}
	})

	t.Run("the input-less seam refuses a nil owner exactly as the sibling does", func(t *testing.T) {
		// A bare Mempool value has no owner at all, so the owner's admission
		// context is unavailable — the same first check the input-bearing
		// sibling runs, and the shape a nil owner can only take.
		h := newRelayHarness(t, nil, 1_000_000)
		expected := h.context()
		bare := &Mempool{}
		entry := &mempoolEntry{txid: [32]byte{0x81}, wtxid: [32]byte{0x82}, weight: 1, size: 1}
		probe := newRelayAdmissionProbe(expected)
		bare.mu.Lock()
		err := bare.reserveEntryInputsLocked(entry, probe)
		bare.mu.Unlock()
		if err == nil {
			t.Fatal("a nil owner must not pass the input-less seam")
		}
		if got := relayDispositionOf(err); got != RelayAdmissionUnavailable {
			t.Fatalf("disposition=%v, want UNAVAILABLE", got)
		}
		if got, want := err.Error(), "pending-outpoint owner admission context unavailable"; got != want {
			t.Fatalf("message=%q, want the sibling's %q", got, want)
		}
		if bare.pendingOutpoints != nil {
			t.Fatal("classifying created an owner")
		}
	})

	t.Run("the input-less seam refuses an owner tip that left the guarded chainstate tip", func(t *testing.T) {
		// The input-bearing sibling refuses this shape (mempool_policy.go's
		// reserveEntryInputsLocked tip check); the input-less seam must refuse
		// it identically, or a candidate that claims no outpoint would pass
		// what every other candidate is held to.
		h := newRelayHarness(t, nil, 1_000_000)
		expected := h.context()
		h.st.Height = 101

		entry := &mempoolEntry{txid: [32]byte{0x91}, wtxid: [32]byte{0x92}, weight: 1, size: 1}
		probe := newRelayAdmissionProbe(expected)
		h.mp.mu.Lock()
		err := h.mp.reserveEntryInputsLocked(entry, probe)
		h.mp.mu.Unlock()
		if err == nil {
			t.Fatal("an owner tip behind the guarded chainstate tip must not pass the input-less seam")
		}
		if got := relayDispositionOf(err); got != RelayAdmissionUnavailable {
			t.Fatalf("disposition=%v, want UNAVAILABLE", got)
		}
		if got, want := err.Error(), "pending-outpoint owner tip does not match the guarded chainstate tip"; got != want {
			t.Fatalf("message=%q, want the sibling's %q", got, want)
		}
	})
}

// TestAddRemoteTxForRelaySnapshotsExpectedContextAtEntry pins the API-boundary
// contract: the caller-supplied expected context is COPIED once at entry, and no
// later seam reads the caller's memory again. Every row takes the snapshot, then
// mutates the caller's variable to a context the owner never issued, and asserts
// the call still behaves as the entry-time value — each row fails if the probe
// goes back to holding the caller's pointer, because it would then read the
// mutated value and refuse.
func TestAddRemoteTxForRelaySnapshotsExpectedContextAtEntry(t *testing.T) {
	// superseded is the same stable tip with a generation the owner has never
	// issued: it compares unequal to the current context at every seam.
	superseded := func(current PendingOutpointAdmissionContext) PendingOutpointAdmissionContext {
		return PendingOutpointAdmissionContext{StableTip: current.StableTip, Generation: current.Generation + 1}
	}

	t.Run("the proving read binds the entry-time context", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		expected := h.context()
		entryTime := *expected
		probe := newRelayAdmissionProbe(expected)
		*expected = superseded(entryTime)

		h.mp.chainState.admissionMu.RLock()
		h.mp.bindRelayAdmissionContext(probe)
		h.mp.chainState.admissionMu.RUnlock()
		if !probe.contextProven || probe.observed != entryTime {
			t.Fatalf("proven=%v observed=%+v, want the entry-time %+v proven", probe.contextProven, probe.observed, entryTime)
		}
	})

	t.Run("the input-less seam checks the entry-time context", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		expected := h.context()
		probe := newRelayAdmissionProbe(expected)
		*expected = superseded(*expected)

		entry := &mempoolEntry{txid: [32]byte{0xA1}, wtxid: [32]byte{0xA2}, weight: 1, size: 1}
		h.mp.mu.Lock()
		err := h.mp.reserveEntryInputsLocked(entry, probe)
		h.mp.mu.Unlock()
		if err != nil {
			t.Fatalf("input-less seam refused the entry-time context: %v", err)
		}
	})

	t.Run("the owner seam reserves against the entry-time context", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		expected := h.context()
		probe := newRelayAdmissionProbe(expected)
		*expected = superseded(*expected)

		entry := &mempoolEntry{
			txid: [32]byte{0xB1}, wtxid: [32]byte{0xB2}, weight: 1, size: 1,
			inputs: []consensus.Outpoint{h.outpoints[0]},
		}
		h.mp.mu.Lock()
		err := h.mp.reserveEntryInputsLocked(entry, probe)
		h.mp.mu.Unlock()
		if err != nil {
			t.Fatalf("owner seam refused the entry-time context: %v", err)
		}
		if entry.token == (PendingOutpointToken{}) {
			t.Fatal("owner seam issued no token for the entry-time context")
		}
	})

	t.Run("the published evidence is the entry-time context", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		expected := h.context()
		entryTime := *expected

		got := h.mp.AddRemoteTxForRelay([]byte{0xFF, 0x00, 0x13, 0x37}, expected)
		*expected = superseded(entryTime)

		if got.Disposition != RelayAdmissionStableTerminalReject || !got.HasAdmissionContext {
			t.Fatalf("disposition=%v hasContext=%v, want a STABLE_TERMINAL_REJECT carrying context (err=%v)",
				got.Disposition, got.HasAdmissionContext, got.Err)
		}
		if got.AdmissionContext != entryTime {
			t.Fatalf("published context=%+v, want the entry-time %+v", got.AdmissionContext, entryTime)
		}
	})
}

// TestAddRemoteTxForRelayNilReceiverIsUnavailable pins the nil-receiver contract
// shared with every other exported Mempool accessor: a typed unavailable result,
// no panic, and no counter movement.
func TestAddRemoteTxForRelayNilReceiverIsUnavailable(t *testing.T) {
	var mp *Mempool
	got := mp.AddRemoteTxForRelay([]byte{0x01}, nil)
	if got.Disposition != RelayAdmissionUnavailable {
		t.Fatalf("disposition=%v, want UNAVAILABLE", got.Disposition)
	}
	if kind := admitKind(t, got.Err); kind != TxAdmitUnavailable {
		t.Fatalf("TxAdmitErrorKind=%q, want %q", kind, TxAdmitUnavailable)
	}
	if got.Err.Error() != "nil mempool" {
		t.Fatalf("message=%q, want the legacy %q", got.Err.Error(), "nil mempool")
	}
	if got.TxID != ([32]byte{}) || got.HasAdmissionContext {
		t.Fatalf("nil receiver published identity or context: %+v", got)
	}
	if counts := mp.AdmissionCounts(); counts != (MempoolAdmissionCounts{}) {
		t.Fatalf("nil receiver moved counters: %+v", counts)
	}
}

// relaySimplicityProvider publishes exactly ONE of the deployment states that
// consensus.SimplicityActiveAtHeight collapses into its single (false, nil)
// "inactive" answer.
type relaySimplicityProvider struct {
	consensus.DefaultRotationProvider
	descriptors []consensus.SimplicityDeploymentDescriptor
	anchor      [32]byte
	ok          bool
	err         error
}

func (p relaySimplicityProvider) PublishedSimplicityDeployments() ([]consensus.SimplicityDeploymentDescriptor, [32]byte, bool, error) {
	return p.descriptors, p.anchor, p.ok, p.err
}

// knownInactiveSimplicityProvider publishes a COMPLETE anchor-verified set whose
// only descriptor activates far above the candidate height: state known, and
// known inactive here.
func knownInactiveSimplicityProvider(t *testing.T) relaySimplicityProvider {
	t.Helper()
	descriptor := consensus.LiveSimplicityDeploymentDescriptor(devnetGenesisChainID)
	descriptor.ActivationHeight = 1 << 40
	set := []consensus.SimplicityDeploymentDescriptor{descriptor}
	anchor, err := consensus.SimplicityDeploymentSetAnchor(devnetGenesisChainID, set)
	if err != nil {
		t.Fatalf("SimplicityDeploymentSetAnchor: %v", err)
	}
	return relaySimplicityProvider{descriptors: set, anchor: anchor, ok: true}
}

// simplicityPolicyCandidate defers the cheap P2PK floor precheck and reaches the
// CORE_SIMPLICITY pre-activation policy lane, the branch under test.
func simplicityPolicyCandidate(h *relayHarness) []byte {
	return txWithOneInputOneOutput(h.outpoints[0].Txid, h.outpoints[0].Vout, 1,
		consensus.COV_TYPE_CORE_SIMPLICITY, simplicityCovenantDataForNodeTest([32]byte{0x53}, nil), nil)
}

// TestRelayAdmissionDispositionSimplicityDeploymentState proves the
// CORE_SIMPLICITY pre-activation lane publishes cache authority ONLY when no
// deployment provider governs the verdict. With a provider present the outcome
// depends on a published deployment set that a {StableTip, Generation}
// admission context does not pin — a lookup failure, an ok=false "unobtainable
// / only partially known" answer, a set failing its own anchor and a complete
// set with no governing surface alike — so all of them are UNAVAILABLE, never
// the stable terminal tag that would let a relay cache blacklist bytes which
// admit as soon as that set answers differently. Every row pins the EXACT
// public error and message — only the published classification changes.
func TestRelayAdmissionDispositionSimplicityDeploymentState(t *testing.T) {
	known := knownInactiveSimplicityProvider(t)
	mismatchedAnchor := known
	mismatchedAnchor.anchor[0] ^= 0xFF

	cases := []struct {
		name        string
		rotation    consensus.RotationProvider
		want        RelayAdmissionDisposition
		wantMessage string
	}{
		{"provider I/O failure cannot determine the state", relaySimplicityProvider{err: errors.New("deployment store offline")}, RelayAdmissionUnavailable, "CORE_SIMPLICITY deployment lookup failure: deployment store offline"},
		{"ok=false publishes an unobtainable or partial set", relaySimplicityProvider{ok: false}, RelayAdmissionUnavailable, "CORE_SIMPLICITY output pre-ACTIVE"},
		{"a set failing its own anchor proves nothing", mismatchedAnchor, RelayAdmissionUnavailable, "CORE_SIMPLICITY output pre-ACTIVE"},
		// A complete anchor-verified set is still a set the context does not pin:
		// the same bytes admit at this exact tip and generation once the provider
		// publishes a descriptor governing this height, and a descriptor's
		// ActivationHeight makes the rejection flip with height regardless.
		{"known complete set with no governing surface is still not pinned", known, RelayAdmissionUnavailable, "CORE_SIMPLICITY output pre-ACTIVE"},
		// The regression guard for the legitimate case, and the default node
		// wiring: the rotation implements no deployment surface at all, so the
		// verdict rests on the constructor-frozen configuration.
		{"no deployment provider configured stays terminal", nil, RelayAdmissionStableTerminalReject, "CORE_SIMPLICITY output pre-ACTIVE"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newRelayHarness(t, &MempoolConfig{
				MaxTransactions:                     10,
				MaxBytes:                            1 << 20,
				PolicyRejectSimplicityPreActivation: true,
				RotationProvider:                    tc.rotation,
			}, 1_000_000)
			raw := simplicityPolicyCandidate(h)

			got := h.mp.AddRemoteTxForRelay(raw, h.context())
			if got.Disposition != tc.want {
				t.Fatalf("disposition=%v, want %v", got.Disposition, tc.want)
			}
			if kind := admitKind(t, got.Err); kind != TxAdmitRejected {
				t.Fatalf("kind=%v, want the unchanged TxAdmitRejected", kind)
			}
			if got.Err.Error() != tc.wantMessage {
				t.Fatalf("message=%q, want the unchanged %q", got.Err.Error(), tc.wantMessage)
			}
			if wantContext := tc.want == RelayAdmissionStableTerminalReject; got.HasAdmissionContext != wantContext {
				t.Fatalf("HasAdmissionContext=%v, want %v", got.HasAdmissionContext, wantContext)
			}
			if h.mp.Len() != 0 {
				t.Fatalf("a rejected candidate became resident: len=%d", h.mp.Len())
			}
			// The legacy wrapper observes the byte-identical failure for the same
			// bytes: the classification is published beside the error, not in it.
			legacy := h.mp.AddRemoteTx(raw)
			if legacy == nil || legacy.Error() != tc.wantMessage || admitKind(t, legacy) != TxAdmitRejected {
				t.Fatalf("legacy AddRemoteTx err=%v, want the identical %q", legacy, tc.wantMessage)
			}
		})
	}
}

// TestRelayAdmissionDispositionSimplicityPolicyFanOut covers the SECOND
// CORE_SIMPLICITY site — the applyPolicyAgainstState fan-out reached from
// mempool_precheck.go — whose plain error is re-wrapped at the seam. The
// precheck lane returns first for every production candidate, so this site is
// unreachable end-to-end today and is driven directly; it must still separate
// the provider-present outcomes from the no-provider one, because the seam
// consuming it tags the whole fan-out.
func TestRelayAdmissionDispositionSimplicityPolicyFanOut(t *testing.T) {
	h := newRelayHarness(t, &MempoolConfig{MaxTransactions: 10, MaxBytes: 1 << 20}, 1_000_000)
	checked := &consensus.CheckedTransaction{Tx: mustParseTx(t, simplicityPolicyCandidate(h))}

	for _, tc := range []struct {
		name        string
		rotation    consensus.RotationProvider
		want        RelayAdmissionDisposition
		wantMessage string
	}{
		{"provider I/O failure", relaySimplicityProvider{err: errors.New("offline")}, RelayAdmissionUnavailable, "CORE_SIMPLICITY deployment lookup failure: offline"},
		{"unknown deployment state", relaySimplicityProvider{ok: false}, RelayAdmissionUnavailable, "CORE_SIMPLICITY output pre-ACTIVE"},
		{"no deployment provider configured", nil, RelayAdmissionStableTerminalReject, "CORE_SIMPLICITY output pre-ACTIVE"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := applyPolicyAgainstStateSimplicity(checked, nil, devnetGenesisChainID, 1, MempoolConfig{
				PolicyRejectSimplicityPreActivation: true,
				RotationProvider:                    tc.rotation,
			})
			if err == nil || err.Error() != tc.wantMessage {
				t.Fatalf("err=%v, want the unchanged %q", err, tc.wantMessage)
			}
			if got := relayDispositionForPolicyError(err); got != tc.want {
				t.Fatalf("disposition=%v, want %v", got, tc.want)
			}
		})
	}

	// The same carve-out the precheck call site applies: the well-formedness
	// tuple (reject == false) reads no deployment set, so it must arrive
	// UNWRAPPED — terminal — even with a provider governing the lane.
	malformed := &consensus.CheckedTransaction{Tx: mustParseTx(t, simplicityMalformedSiblingCandidate(t, h))}
	for _, rotation := range []consensus.RotationProvider{nil, knownInactiveSimplicityProvider(t), relaySimplicityProvider{ok: false}} {
		err := applyPolicyAgainstStateSimplicity(malformed, nil, devnetGenesisChainID, 1, MempoolConfig{
			PolicyRejectSimplicityPreActivation: true,
			RotationProvider:                    rotation,
		})
		wantMessage := "TX_ERR_COVENANT_TYPE_INVALID: invalid CORE_P2PK covenant_data length"
		if err == nil || err.Error() != wantMessage {
			t.Fatalf("rotation=%T err=%v, want the unchanged %q", rotation, err, wantMessage)
		}
		if got := relayDispositionForPolicyError(err); got != RelayAdmissionStableTerminalReject {
			t.Fatalf("rotation=%T disposition=%v, want STABLE_TERMINAL_REJECT for the well-formedness verdict", rotation, got)
		}
	}

	// Every other static-policy term is a decision over pinned inputs.
	if got := relayDispositionForPolicyError(errors.New("non-coinbase CORE_ANCHOR output")); got != RelayAdmissionStableTerminalReject {
		t.Fatalf("disposition=%v, want STABLE_TERMINAL_REJECT for a pinned-input term", got)
	}
}

// TestRelayAdmissionDispositionUnassignedCovenantReject proves the canonical
// consensus pipeline rejects a covenant_type 0x0102 output before that
// pipeline's input-resolution, witness-slot, or signature-verification stages.
func TestRelayAdmissionDispositionUnassignedCovenantReject(t *testing.T) {
	h := newRelayHarness(t, nil, 1_000_000)
	raw := txWithOneInputOneOutput(h.outpoints[0].Txid, h.outpoints[0].Vout, 1,
		consensus.COV_TYPE_CORE_EXT, nil, nil)

	got := h.mp.AddRemoteTxForRelay(raw, h.context())
	if got.Disposition != RelayAdmissionStableTerminalReject {
		t.Fatalf("disposition=%v, want STABLE_TERMINAL_REJECT (err=%v)", got.Disposition, got.Err)
	}
	wantMessage := "TX_ERR_COVENANT_TYPE_INVALID: unknown covenant_type"
	if got.Err == nil || got.Err.Error() != wantMessage {
		t.Fatalf("err=%v, want %q", got.Err, wantMessage)
	}
	if kind := admitKind(t, got.Err); kind != TxAdmitRejected {
		t.Fatalf("TxAdmitErrorKind=%q, want %q", kind, TxAdmitRejected)
	}
	if !got.HasAdmissionContext {
		t.Fatal("a proven stable terminal rejection must publish its exact context")
	}
	if h.mp.Len() != 0 {
		t.Fatalf("an unassigned covenant candidate became resident: len=%d", h.mp.Len())
	}
	// The legacy wrapper observes the byte-identical rejection for the same bytes.
	legacy := h.mp.AddRemoteTx(raw)
	if legacy == nil || legacy.Error() != wantMessage || admitKind(t, legacy) != TxAdmitRejected {
		t.Fatalf("legacy AddRemoteTx err=%v, want the identical %q", legacy, wantMessage)
	}
}

func TestRelayAdmissionDispositionUnassignedCovenantSpendReject(t *testing.T) {
	h := newRelayHarness(t, nil, 1_000_000)
	entry := h.st.Utxos[h.outpoints[0]]
	entry.CovenantType = consensus.COV_TYPE_CORE_EXT
	h.st.Utxos[h.outpoints[0]] = entry
	raw := txWithOneInputOneOutput(h.outpoints[0].Txid, h.outpoints[0].Vout, 1,
		consensus.COV_TYPE_P2PK, h.toAddr, nil)

	expected := h.context()
	got := h.mp.AddRemoteTxForRelay(raw, expected)
	if got.Disposition != RelayAdmissionStableTerminalReject {
		t.Fatalf("disposition=%v, want STABLE_TERMINAL_REJECT (err=%v)", got.Disposition, got.Err)
	}
	wantMessage := "TX_ERR_COVENANT_TYPE_INVALID: unsupported covenant in basic apply"
	if got.Err == nil || got.Err.Error() != wantMessage {
		t.Fatalf("err=%v, want %q", got.Err, wantMessage)
	}
	if kind := admitKind(t, got.Err); kind != TxAdmitRejected {
		t.Fatalf("TxAdmitErrorKind=%q, want %q", kind, TxAdmitRejected)
	}
	if !got.HasAdmissionContext || got.AdmissionContext != *expected {
		t.Fatalf("published context=%+v hasContext=%v, want %+v", got.AdmissionContext, got.HasAdmissionContext, *expected)
	}
	if h.mp.Len() != 0 {
		t.Fatalf("an unassigned covenant spend became resident: len=%d", h.mp.Len())
	}
}

func TestRelayAdmissionUnassignedCovenantOutputOrder(t *testing.T) {
	cases := []struct {
		outputs []consensus.TxOutput
		want    string
	}{
		{[]consensus.TxOutput{{Value: 1, CovenantType: 0x0102, CovenantData: []byte{0x01}}, {Value: 1, CovenantType: consensus.COV_TYPE_P2PK}}, "TX_ERR_COVENANT_TYPE_INVALID: unknown covenant_type"},
		{[]consensus.TxOutput{{Value: 1, CovenantType: consensus.COV_TYPE_P2PK}, {Value: 1, CovenantType: 0x0102, CovenantData: []byte{0x01}}}, "TX_ERR_COVENANT_TYPE_INVALID: invalid CORE_P2PK covenant_data length"},
	}
	for _, tc := range cases {
		h := newRelayHarness(t, nil, 1_000_000)
		expected := h.context()
		got := h.mp.AddRemoteTxForRelay(mustMarshalTxForNodeTest(t, &consensus.Tx{
			Version: 1, TxNonce: 1,
			Inputs:  []consensus.TxInput{{PrevTxid: h.outpoints[0].Txid, PrevVout: h.outpoints[0].Vout}},
			Outputs: tc.outputs,
		}), expected)
		if got.Disposition != RelayAdmissionStableTerminalReject || got.Err == nil || got.Err.Error() != tc.want || admitKind(t, got.Err) != TxAdmitRejected || !got.HasAdmissionContext || got.AdmissionContext != *expected || h.mp.Len() != 0 {
			t.Fatalf("result=%+v len=%d, want terminal %q with published context", got, h.mp.Len(), tc.want)
		}
	}
}

// simplicityMalformedSiblingCandidate is a CORE_SIMPLICITY output alongside a
// malformed CORE_P2PK sibling: it reaches the ValidateTxCovenantsGenesis error
// branch inside rejectCoreSimplicityPreActivation, whose (false, "", err) tuple
// is the well-formedness half of the lane.
func simplicityMalformedSiblingCandidate(t *testing.T, h *relayHarness) []byte {
	t.Helper()
	return mustMarshalTxForNodeTest(t, &consensus.Tx{
		Version: 1,
		TxKind:  0x00,
		TxNonce: 1,
		Inputs:  []consensus.TxInput{{PrevTxid: h.outpoints[0].Txid, PrevVout: h.outpoints[0].Vout}},
		Outputs: []consensus.TxOutput{
			{Value: 1, CovenantType: consensus.COV_TYPE_CORE_SIMPLICITY, CovenantData: simplicityCovenantDataForNodeTest([32]byte{0x59}, nil)},
			{Value: 1, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: nil},
		},
	})
}

// TestRelayAdmissionDispositionSimplicityGenesisWellFormednessReject covers
// the ValidateTxCovenantsGenesis error branch inside
// rejectCoreSimplicityPreActivation — a malformed non-Simplicity
// output alongside a CORE_SIMPLICITY output, the same shape
// TestMempoolPolicyDoesNotMaskMalformedNonSimplicityOutput (mempool_test.go)
// pins the message for without asserting a disposition. That test is not
// itself a relay-path test and mempool_test.go is outside this issue's
// allowed surface, so the disposition is proven here instead, driving the
// identical branch through the live relay entry point.
//
// The rows prove the CARVE-OUT on ONE candidate shape: the well-formedness
// verdict is evaluated against a rotation shim that forces Simplicity active, so
// it reads no deployment set and stays STABLE_TERMINAL_REJECT with published
// cache authority whether or not a provider is configured. Only the sibling
// exit of the same helper — the deployment lookup, which returns
// (true, reason, err) — depends on the provider and is UNAVAILABLE with no cache
// authority.
func TestRelayAdmissionDispositionSimplicityGenesisWellFormednessReject(t *testing.T) {
	const wellFormedness = "TX_ERR_COVENANT_TYPE_INVALID: invalid CORE_P2PK covenant_data length"

	for _, tc := range []struct {
		name        string
		rotation    consensus.RotationProvider
		want        RelayAdmissionDisposition
		wantMessage string
	}{
		{"no deployment provider configured", nil, RelayAdmissionStableTerminalReject, wellFormedness},
		// The regression this carve-out exists for: a provider present must not
		// turn a verdict that never consulted its set into a retryable one.
		{"complete inactive set present stays terminal", knownInactiveSimplicityProvider(t), RelayAdmissionStableTerminalReject, wellFormedness},
		{"unobtainable set present stays terminal", relaySimplicityProvider{ok: false}, RelayAdmissionStableTerminalReject, wellFormedness},
		// The sibling exit on the SAME bytes: the lookup fails before the
		// well-formedness check runs, and that verdict does depend on the provider.
		{"lookup failure stays unavailable", relaySimplicityProvider{err: errors.New("deployment store offline")}, RelayAdmissionUnavailable, "CORE_SIMPLICITY deployment lookup failure: deployment store offline"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newRelayHarness(t, &MempoolConfig{
				PolicyRejectSimplicityPreActivation: true,
				RotationProvider:                    tc.rotation,
			}, 100)
			raw := simplicityMalformedSiblingCandidate(t, h)

			got := h.mp.AddRemoteTxForRelay(raw, h.context())
			if got.Disposition != tc.want {
				t.Fatalf("disposition=%v, want %v (err=%v)", got.Disposition, tc.want, got.Err)
			}
			if got.Err == nil || got.Err.Error() != tc.wantMessage {
				t.Fatalf("err=%v, want the unchanged %q", got.Err, tc.wantMessage)
			}
			if kind := admitKind(t, got.Err); kind != TxAdmitRejected {
				t.Fatalf("TxAdmitErrorKind=%q, want %q", kind, TxAdmitRejected)
			}
			if wantContext := tc.want == RelayAdmissionStableTerminalReject; got.HasAdmissionContext != wantContext {
				t.Fatalf("HasAdmissionContext=%v, want %v", got.HasAdmissionContext, wantContext)
			}
			if h.mp.Len() != 0 {
				t.Fatalf("a rejected candidate became resident: len=%d", h.mp.Len())
			}
			// The legacy wrapper observes the byte-identical rejection for the same bytes.
			legacy := h.mp.AddRemoteTx(raw)
			if legacy == nil || legacy.Error() != tc.wantMessage || admitKind(t, legacy) != TxAdmitRejected {
				t.Fatalf("legacy AddRemoteTx err=%v, want the identical %q", legacy, tc.wantMessage)
			}
		})
	}
}

// cleanupRejectionMempool is a bare mempool whose admission-sequence space is
// exhausted, so a candidate that clears the owner seam is rejected by the LATER
// admission-sequence check — the exact branch whose post-decision cleanup
// releases the candidate reservation.
func cleanupRejectionMempool() *Mempool {
	return &Mempool{maxTxs: 10, maxBytes: 100, lastAdmissionSeq: ^uint64(0)}
}

// drivePostDecisionCleanup runs the ONE live locked admission implementation
// with a relay probe and returns the published result.
func drivePostDecisionCleanup(t *testing.T, mp *Mempool, entry *mempoolEntry) RelayAdmissionResult {
	t.Helper()
	probe := &relayAdmissionProbe{}
	mp.mu.Lock()
	err := mp.addEntryLockedProbed(entry, 0, probe)
	mp.mu.Unlock()
	return probe.result(err)
}

// corruptedClaimToken reserves a REAL claim on mp's owner and then deletes the
// by-outpoint row that claim installed, so the owner's own index disagrees with
// the token. That is exactly the accounting corruption Release detects and
// reports as an internal fault.
//
// The corruption is installed before the drive because the live path leaves no
// seam between Reserve and the later check: Reserve installs a self-consistent
// claim and nothing between it and the rejection touches the owner.
func corruptedClaimToken(t *testing.T, mp *Mempool, op consensus.Outpoint) PendingOutpointToken {
	t.Helper()
	mp.mu.Lock()
	owner := mp.pendingOutpointOwnerLocked()
	mp.mu.Unlock()
	token := mustReserve(t, owner, [32]byte{0x3f}, op)
	owner.mu.Lock()
	delete(owner.byOutpoint, op)
	owner.mu.Unlock()
	return token
}

// assertSameAdmissionError proves the public compatibility channel is untouched:
// same concrete type, same TxAdmitErrorKind, same message bytes.
func assertSameAdmissionError(t *testing.T, got, want error) {
	t.Helper()
	if reflect.TypeOf(got) != reflect.TypeOf(want) {
		t.Fatalf("error type %T, want the identical %T", got, want)
	}
	if gotKind, wantKind := admitKind(t, got), admitKind(t, want); gotKind != wantKind {
		t.Fatalf("TxAdmitErrorKind=%q, want the identical %q", gotKind, wantKind)
	}
	if got.Error() != want.Error() {
		t.Fatalf("message=%q, want the byte-identical %q", got.Error(), want.Error())
	}
}

// admissionCounters is every counter and accounting total this surface can move.
func admissionCounters(mp *Mempool) [7]uint64 {
	return [7]uint64{
		mp.admitAccepted.Load(),
		mp.admitConflict.Load(),
		mp.admitRejected.Load(),
		mp.admitUnavailable.Load(),
		mp.evictedResidentTotal.Load(),
		uint64(len(mp.txs)),
		uint64(mp.usedBytes),
	}
}

// TestRelayAdmissionDispositionCleanupFailureOutranksTheBranch pins the
// fail-closed override: a REQUIRED post-decision cleanup that reports owner
// accounting corruption publishes INTERNAL even though the deciding branch
// already tagged the error ADMISSION_SEQUENCE and selectRelayDisposition is
// first-selection-wins.
//
// The control row proves the override cannot swallow an ordinary rejection: an
// identical rejection whose cleanup succeeds still publishes the branch's own
// disposition, and both rows return the byte-identical public error.
func TestRelayAdmissionDispositionCleanupFailureOutranksTheBranch(t *testing.T) {
	const wantMsg = "mempool admission sequence exhausted"

	// Row 1 — ordinary cleanup: the live conflict slot issues the candidate's
	// own reservation and the post-decision release succeeds.
	ordinaryMp := cleanupRejectionMempool()
	ordinaryEntry := &mempoolEntry{
		txid: [32]byte{0x2c}, wtxid: [32]byte{0x2d},
		inputs: []consensus.Outpoint{testOutpoint(11)},
		fee:    consensus.Uint128FromU64(1), weight: 1, size: 1,
	}
	ordinary := drivePostDecisionCleanup(t, ordinaryMp, ordinaryEntry)
	if ordinary.Disposition != RelayAdmissionAdmissionSequence {
		t.Fatalf("ordinary disposition=%v, want ADMISSION_SEQUENCE (err=%v)", ordinary.Disposition, ordinary.Err)
	}
	assertCandidateExactlyReleased(t, ordinaryMp, ordinaryEntry, ordinary.Err, TxAdmitUnavailable, wantMsg)

	// Row 2 — the same rejection, but the cleanup release hits a claim whose
	// by-outpoint row disagrees with the token.
	corruptMp := cleanupRejectionMempool()
	op := testOutpoint(12)
	token := corruptedClaimToken(t, corruptMp, op)
	corruptEntry := &mempoolEntry{
		txid: [32]byte{0x3c}, wtxid: [32]byte{0x3d}, token: token,
		fee: consensus.Uint128FromU64(1), weight: 1, size: 1,
	}
	corrupted := drivePostDecisionCleanup(t, corruptMp, corruptEntry)

	if corrupted.Disposition != RelayAdmissionInternal {
		t.Fatalf("corrupted disposition=%v, want INTERNAL (err=%v)", corrupted.Disposition, corrupted.Err)
	}
	if corrupted.HasAdmissionContext {
		t.Fatal("an internal cleanup failure must publish no cache-authorizing context")
	}
	// The public compatibility channel is byte-identical to the same rejection
	// whose cleanup succeeded.
	assertSameAdmissionError(t, corrupted.Err, ordinary.Err)
	if got := admissionCounters(corruptMp); got != admissionCounters(ordinaryMp) {
		t.Fatalf("counters=%v, want the ordinary row's %v", got, admissionCounters(ordinaryMp))
	}
	if corruptEntry.token != (PendingOutpointToken{}) {
		t.Fatalf("rejected candidate token=%+v, want the zero token", corruptEntry.token)
	}
	// The release really failed: the corrupted claim is still live and the owner
	// still reports the exact index-mismatch fault the cleanup step hit.
	releaseErr := corruptMp.pendingOutpoints.Release(token)
	if testOwnerKind(t, releaseErr) != PendingOutpointInternal {
		t.Fatalf("Release after cleanup: %v, want an internal owner fault", releaseErr)
	}
	wantRelease := fmt.Sprintf("pending-outpoint index mismatch for txid=%x vout=%d", op.Txid, op.Vout)
	if releaseErr.Error() != wantRelease {
		t.Fatalf("Release error=%q, want %q", releaseErr.Error(), wantRelease)
	}
}

// unboundAlgSuiteRegistry is the node-side configuration that makes THIS
// process unable to resolve a live verifier binding for a suite it otherwise
// authorizes: the registry entry keeps the canonical suite id, lengths and
// verify cost — so the candidate passes every suite-authorization and
// canonical-length check — and only its algorithm identity has no entry in the
// local live-binding policy artifact. It is reached through
// MempoolConfig.SuiteRegistry, the exported node configuration seam
// NewMempoolWithConfig carries into consensus validation; no unexported
// consensus hook, linkname or production test knob is involved.
func unboundAlgSuiteRegistry() *consensus.SuiteRegistry {
	return consensus.NewSuiteRegistryFromParams([]consensus.SuiteParams{{
		SuiteID:    consensus.SUITE_ID_ML_DSA_87,
		PubkeyLen:  consensus.ML_DSA_87_PUBKEY_BYTES,
		SigLen:     consensus.ML_DSA_87_SIG_BYTES,
		VerifyCost: consensus.VERIFY_COST_ML_DSA_87,
		AlgName:    "ML-DSA-87-NOT-LIVE-BOUND",
	}})
}

// TestRelayAdmissionLocalCryptoBackendFaultPublishesInternal is the END-TO-END
// row of the classification matrix: a PROCESS-LOCAL crypto-backend fault driven
// through the real relay-admission entry point publishes INTERNAL and no
// cache-authorizing context, while the public admission error — kind and
// message alike — stays byte-identical to what the same fault produced before
// the typed cause existed.
//
// The consumer decides from the typed cause alone. The public code carrying it
// here, TX_ERR_SIG_ALG_INVALID, is the SAME code an unauthorized candidate suite
// produces, and that row still classifies as a stable terminal rejection below,
// so a classifier reading the code — or the message — could not tell them apart.
func TestRelayAdmissionLocalCryptoBackendFaultPublishesInternal(t *testing.T) {
	h := newRelayHarness(t, &MempoolConfig{SuiteRegistry: unboundAlgSuiteRegistry()}, 1_000_000)
	raw := h.tx(0, 100_000, 100_000, 1)

	got := h.mp.AddRemoteTxForRelay(raw, h.context())

	if got.Disposition != RelayAdmissionInternal {
		t.Fatalf("disposition=%v, want INTERNAL (err=%v)", got.Disposition, got.Err)
	}
	if got.HasAdmissionContext {
		t.Fatal("a local backend fault must publish no cache-authorizing context")
	}
	if got.AdmissionContext != (PendingOutpointAdmissionContext{}) {
		t.Fatalf("published context=%+v, want the zero context", got.AdmissionContext)
	}
	if kind := admitKind(t, got.Err); kind != TxAdmitRejected {
		t.Fatalf("public admission kind=%s, want %s (unchanged)", kind, TxAdmitRejected)
	}
	wantMsg := fmt.Sprintf(
		"%s: suite_id=0x%02x resolveSuiteVerifierBinding: unsupported alg=%q pubkey_len=%d sig_len=%d",
		consensus.TX_ERR_SIG_ALG_INVALID,
		consensus.SUITE_ID_ML_DSA_87,
		"ML-DSA-87-NOT-LIVE-BOUND",
		consensus.ML_DSA_87_PUBKEY_BYTES,
		consensus.ML_DSA_87_SIG_BYTES,
	)
	if got.Err.Error() != wantMsg {
		t.Fatalf("public message=%q, want %q (unchanged)", got.Err.Error(), wantMsg)
	}
	if h.mp.Len() != 0 {
		t.Fatalf("mempool len=%d, want 0", h.mp.Len())
	}
}

// TestRelayAdmissionCandidateFailuresKeepBaselineClassification pins the three
// NEGATIVE rows of the matrix: with a healthy backend, invalidity that belongs
// to the candidate bytes carries no local-fault cause, so each one keeps its
// baseline STABLE_TERMINAL_REJECT and its cache-authorizing context evidence.
func TestRelayAdmissionCandidateFailuresKeepBaselineClassification(t *testing.T) {
	cases := []struct {
		name string
		// wantMsg pins the exact public message, so a row can never silently
		// become a different rejection than the one it is named for.
		wantMsg string
		build   func(t *testing.T, h *relayHarness) []byte
	}{
		{
			name:    "cryptographically invalid signature",
			wantMsg: "TX_ERR_SIG_INVALID: CORE_P2PK signature invalid",
			build: func(t *testing.T, h *relayHarness) []byte {
				return corruptFirstWitnessSignature(t, h.tx(0, 100_000, 100_000, 1))
			},
		},
		{
			name:    "structurally malformed candidate",
			wantMsg: "TX_ERR_PARSE: unsupported tx version",
			build: func(t *testing.T, h *relayHarness) []byte {
				return []byte{0xFF, 0x00, 0x13, 0x37}
			},
		},
		{
			name:    "candidate selected an unauthorized suite",
			wantMsg: "TX_ERR_SIG_ALG_INVALID: CORE_P2PK suite not in native spend set",
			build: func(t *testing.T, h *relayHarness) []byte {
				return retargetFirstWitnessSuite(t, h.tx(0, 100_000, 100_000, 1), 0x02)
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newRelayHarness(t, nil, 1_000_000)
			got := h.mp.AddRemoteTxForRelay(tc.build(t, h), h.context())
			if got.Disposition != RelayAdmissionStableTerminalReject {
				t.Fatalf("disposition=%v, want STABLE_TERMINAL_REJECT (err=%v)", got.Disposition, got.Err)
			}
			if !got.HasAdmissionContext {
				t.Fatal("a proven candidate-intrinsic rejection keeps its cache-authorizing context")
			}
			if kind := admitKind(t, got.Err); kind != TxAdmitRejected {
				t.Fatalf("public admission kind=%s, want %s", kind, TxAdmitRejected)
			}
			if got.Err.Error() != tc.wantMsg {
				t.Fatalf("public message=%q, want %q (unchanged)", got.Err.Error(), tc.wantMsg)
			}
		})
	}
}

// retargetFirstWitnessSuite rewrites the suite_id the CANDIDATE selected in its
// first witness item, leaving every other byte the producer signed in place.
func retargetFirstWitnessSuite(t *testing.T, txBytes []byte, suiteID uint8) []byte {
	t.Helper()
	tx, _, _, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx before retarget: %v", err)
	}
	if len(tx.Witness) == 0 {
		t.Fatal("expected a first witness item")
	}
	tx.Witness[0].SuiteID = suiteID
	out, err := consensus.MarshalTx(tx)
	if err != nil {
		t.Fatalf("MarshalTx after retarget: %v", err)
	}
	return out
}

func mutateDAAdmissionPolicy(f *daNonReplayFixture, mutate func(*MempoolConfig)) {
	f.mp.mu.Lock()
	defer f.mp.mu.Unlock()
	mutate(&f.mp.policy)
}

// TestAdmitDACandidateFailuresCarryOriginatingDisposition drives signed DA candidates through AdmitDA; each row pins its exact public kind, message, and Section 6.5 disposition.
func TestAdmitDACandidateFailuresCarryOriginatingDisposition(t *testing.T) {
	simplicityOutput := consensus.TxOutput{Value: 1, CovenantType: consensus.COV_TYPE_CORE_SIMPLICITY, CovenantData: simplicityCovenantDataForNodeTest([32]byte{0x53}, nil)}
	for _, row := range []struct {
		name        string
		build       func(*testing.T, *daNonReplayFixture) []byte
		kind        TxAdmitErrorKind
		message     string
		disposition RelayAdmissionDisposition
	}{
		{
			name: "unavailable chain context",
			build: func(_ *testing.T, f *daNonReplayFixture) []byte {
				raw := f.signed(daNonReplayTxSpec{kind: 0x02, daID: [32]byte{0xe1}, payload: []byte("context")}).raw
				f.mp.pendingOutpoints.mu.Lock()
				f.mp.pendingOutpoints.stableTip = PendingOutpointTip{HasTip: true, Height: ^uint64(0), Hash: f.state.TipHash}
				f.mp.pendingOutpoints.mu.Unlock()
				f.state.Height = ^uint64(0)
				return raw
			},
			kind: TxAdmitUnavailable, message: "height overflow", disposition: RelayAdmissionUnavailable,
		},
		{
			name: "unavailable block MTP",
			build: func(t *testing.T, f *daNonReplayFixture) []byte {
				f.mp.blockStore = mustCreateBlockStore(t, filepath.Join(t.TempDir(), "blockstore"))
				return f.signed(daNonReplayTxSpec{kind: 0x02, daID: [32]byte{0xe2}, payload: []byte("mtp")}).raw
			},
			kind: TxAdmitUnavailable, message: "missing canonical hash at height 100 for timestamp context (next_height=101)", disposition: RelayAdmissionUnavailable,
		},
		{
			name: "missing input dependency",
			build: func(_ *testing.T, f *daNonReplayFixture) []byte {
				tx := f.signed(daNonReplayTxSpec{kind: 0x02, daID: [32]byte{0xe3}, payload: []byte("dependency")})
				delete(f.state.Utxos, tx.inputs[0])
				return tx.raw
			},
			kind: TxAdmitRejected, message: "TX_ERR_MISSING_UTXO: utxo not found", disposition: RelayAdmissionMissingDependency,
		},
		{
			name: "canonical unassigned covenant",
			build: func(_ *testing.T, f *daNonReplayFixture) []byte {
				return f.signed(daNonReplayTxSpec{
					kind: 0x02, daID: [32]byte{0xe4}, payload: []byte("core-ext"),
					extraOutputs: []consensus.TxOutput{{Value: 1, CovenantType: consensus.COV_TYPE_CORE_EXT}},
				}).raw
			},
			kind: TxAdmitRejected, message: "TX_ERR_COVENANT_TYPE_INVALID: unknown covenant_type", disposition: RelayAdmissionStableTerminalReject,
		},
		{
			name: "simplicity pre-activation without a deployment provider",
			build: func(_ *testing.T, f *daNonReplayFixture) []byte {
				mutateDAAdmissionPolicy(f, func(c *MempoolConfig) { c.PolicyRejectSimplicityPreActivation = true })
				return f.signed(daNonReplayTxSpec{
					kind: 0x02, daID: [32]byte{0xe5}, payload: []byte("frozen"),
					extraOutputs: []consensus.TxOutput{simplicityOutput},
				}).raw
			},
			kind: TxAdmitRejected, message: "CORE_SIMPLICITY output pre-ACTIVE", disposition: RelayAdmissionStableTerminalReject,
		},
		{
			name: "simplicity pre-activation decided from a deployment provider",
			build: func(t *testing.T, f *daNonReplayFixture) []byte {
				mutateDAAdmissionPolicy(f, func(c *MempoolConfig) {
					c.PolicyRejectSimplicityPreActivation, c.RotationProvider = true, knownInactiveSimplicityProvider(t)
				})
				return f.signed(daNonReplayTxSpec{
					kind: 0x02, daID: [32]byte{0xe6}, payload: []byte("provider"),
					extraOutputs: []consensus.TxOutput{simplicityOutput},
				}).raw
			},
			kind: TxAdmitRejected, message: "CORE_SIMPLICITY output pre-ACTIVE", disposition: RelayAdmissionUnavailable,
		},
		{
			name: "simplicity deployment lookup failure",
			build: func(_ *testing.T, f *daNonReplayFixture) []byte {
				mutateDAAdmissionPolicy(f, func(c *MempoolConfig) {
					c.PolicyRejectSimplicityPreActivation, c.RotationProvider = true, relaySimplicityProvider{err: errors.New("deployment store offline")}
				})
				return f.signed(daNonReplayTxSpec{
					kind: 0x02, daID: [32]byte{0xed}, payload: []byte("lookup"),
					extraOutputs: []consensus.TxOutput{simplicityOutput},
				}).raw
			},
			kind: TxAdmitRejected, message: "CORE_SIMPLICITY deployment lookup failure: deployment store offline", disposition: RelayAdmissionUnavailable,
		},
		{
			name: "simplicity well-formedness verdict reads no deployment set",
			build: func(t *testing.T, f *daNonReplayFixture) []byte {
				mutateDAAdmissionPolicy(f, func(c *MempoolConfig) {
					c.PolicyRejectSimplicityPreActivation, c.RotationProvider = true, knownInactiveSimplicityProvider(t)
				})
				return f.signed(daNonReplayTxSpec{
					kind: 0x02, daID: [32]byte{0xe7}, payload: []byte("malformed"),
					extraOutputs: []consensus.TxOutput{simplicityOutput, {Value: 1, CovenantType: consensus.COV_TYPE_P2PK}},
				}).raw
			},
			kind: TxAdmitRejected, message: "TX_ERR_COVENANT_TYPE_INVALID: invalid CORE_P2PK covenant_data length", disposition: RelayAdmissionStableTerminalReject,
		},
		{
			name: "consensus rejection",
			build: func(t *testing.T, f *daNonReplayFixture) []byte {
				tx := f.signed(daNonReplayTxSpec{kind: 0x02, daID: [32]byte{0xe8}, payload: []byte("consensus")})
				parsed, _, _, consumed, err := consensus.ParseTx(tx.raw)
				if err != nil || consumed != len(tx.raw) {
					t.Fatalf("independent ParseTx=(%d,%v)", consumed, err)
				}
				parsed.Outputs[0].Value++
				return mustMarshalTxForNodeTest(t, parsed)
			},
			kind: TxAdmitRejected, message: "TX_ERR_SIG_INVALID: CORE_P2PK signature invalid", disposition: RelayAdmissionStableTerminalReject,
		},
		{
			name: "consensus rejection with a retryable typed cause",
			build: func(_ *testing.T, f *daNonReplayFixture) []byte {
				// Present in the snapshot, so the input-dependency precheck passes and
				// consensus itself refuses the one-block-short coinbase maturity.
				entry := f.state.Utxos[f.outpoints[0]]
				entry.CreationHeight = f.state.Height + 2 - consensus.COINBASE_MATURITY
				f.state.Utxos[f.outpoints[0]] = entry
				return f.signed(daNonReplayTxSpec{kind: 0x02, daID: [32]byte{0xeb}, payload: []byte("immature")}).raw
			},
			kind: TxAdmitRejected, message: "TX_ERR_COINBASE_IMMATURE: coinbase immature", disposition: RelayAdmissionMissingDependency,
		},
		{
			name: "consensus uncaused suite error under an untrusted native-suite context",
			build: func(t *testing.T, f *daNonReplayFixture) []byte {
				// The Simplicity lane is switched off here so its pass-through arm is
				// also driven; the consensus verdict does not depend on that lane.
				mutateDAAdmissionPolicy(f, func(c *MempoolConfig) {
					c.PolicyRejectSimplicityPreActivation, c.RotationProvider = false, &deploymentCauseRotation{}
				})
				return retargetFirstWitnessSuite(t, f.signed(daNonReplayTxSpec{kind: 0x02, daID: [32]byte{0xec}, payload: []byte("suite")}).raw, 0x02)
			},
			kind: TxAdmitRejected, message: "TX_ERR_SIG_ALG_INVALID: CORE_P2PK suite not in native spend set", disposition: RelayAdmissionUnavailable,
		},
		{
			name: "policy rejection",
			build: func(_ *testing.T, f *daNonReplayFixture) []byte {
				mutateDAAdmissionPolicy(f, func(c *MempoolConfig) { c.PolicyMaxDaBytesPerBlock = 1 })
				return f.signed(daNonReplayTxSpec{kind: 0x01, daID: [32]byte{0xe9}, chunkCount: 1, commitment: [32]byte{0xa8}, commitmentOutputs: 1}).raw
			},
			kind: TxAdmitRejected, message: "DA declared chunk budget exceeded (declared_da_bytes=524288 max_da_bytes=1 chunk_count=1 chunk_bytes=524288)", disposition: RelayAdmissionStableTerminalReject,
		},
		{
			name: "DA payload hash mismatch",
			build: func(_ *testing.T, f *daNonReplayFixture) []byte {
				return f.signed(daNonReplayTxSpec{kind: 0x02, daID: [32]byte{0xea}, payload: []byte("hash"), literalChunkHash: true, chunkHash: [32]byte{0xff}}).raw
			},
			kind: TxAdmitRejected, message: "DA chunk payload hash mismatch", disposition: RelayAdmissionStableTerminalReject,
		},
	} {
		t.Run(row.name, func(t *testing.T) {
			f := newDANonReplayFixture(t, 2)
			raw := row.build(t, f)
			relayBefore, ownerBefore := daRelayStateSnapshot(f.relay), cloneDAAdmissionOwner(f.mp.pendingOutpoints)
			got, err := f.relay.AdmitDA(raw, publicPeer(t, row.name))
			requirePublicDAFailure(t, got, err, row.kind, row.message, row.disposition)
			requireDANonReplayUnchanged(t, f.relay, f.mp.pendingOutpoints, relayBefore, ownerBefore)
			if !f.state.admissionMu.TryLock() {
				t.Fatalf("candidate failure leaked the admission guard: %v", err)
			}
			f.state.admissionMu.Unlock()
		})
	}
	// Structurally unreachable through AdmitDA, so listed instead of executed:
	//
	//   - validateChainSnapshot's nil-snapshot arm: validateDACandidate refuses a
	//     nil held snapshot with errDARelayImageIncompatible before the shared
	//     path runs, so the branch cannot see a nil snapshot here. The arm still
	//     stamps its own disposition, asserted directly below.
	//   - buildPolicyInputSnapshotIfNeeded's two impossible-invariant arms: the
	//     nil parsed transaction (parseDAAdmissionCandidate hands AdmitDA a
	//     non-nil tx or an error) and the nil UTXO set (admissionSnapshotForInputs
	//     always allocates the selected set for a non-nil ChainState and returns
	//     nil only for the nil ChainState acquireDAAdmissionHold refuses).
	//   - applyPolicyAgainstState's impossible invariant: the fan-out only ever
	//     receives the CheckedTransaction a completed consensus validation
	//     returned, so no candidate property can produce it. Its typed local
	//     fault must still never become a cache-authorizing stable terminal
	//     rejection, which is asserted directly below.
	if _, err := validateChainSnapshot(nil); relayDispositionOf(err) != RelayAdmissionUnavailable {
		t.Fatalf("nil-snapshot arm disposition=%v, want UNAVAILABLE", relayDispositionOf(err))
	}
	if got := relayDispositionForPolicyError(&policyImpossibleInvariantError{err: errors.New("nil checked transaction")}); got != RelayAdmissionInternal {
		t.Fatalf("typed impossible-invariant disposition=%v, want INTERNAL", got)
	}
}
