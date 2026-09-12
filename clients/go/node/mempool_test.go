package node

import (
	"bytes"
	"context"
	"crypto/sha3"
	"encoding/hex"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/2tbmz9y2xt-lang/rubin-protocol/clients/go/consensus"
)

// assertPostSlotRejectionSnapshot pins the shape of an admission failure that
// happened AFTER the conflict slot issued a reservation: no entry is published,
// every record and live claim is byte-identical, and the issued token sequence
// stays consumed — the owner high-water never decreases, so no sequence can be
// handed out twice.
func assertPostSlotRejectionSnapshot(t *testing.T, before, after mempoolSnapshot) {
	t.Helper()
	if after.pending.tokenHighWater != before.pending.tokenHighWater+1 {
		t.Fatalf("token high-water=%d, want exactly one consumed sequence above %d", after.pending.tokenHighWater, before.pending.tokenHighWater)
	}
	after.pending.tokenHighWater = before.pending.tokenHighWater
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("rejected candidate mutated mempool: before=%+v after=%+v", before, after)
	}
}

func TestMempoolAdd(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}
	if got := mp.Len(); got != 1 {
		t.Fatalf("mempool len=%d, want 1", got)
	}
}

func TestMempoolAcceptedEntryMetadataAndIndexes(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 300_000, 1, fromKey, fromAddress, toAddress)
	tx, txid, wtxid, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx: %v", err)
	}
	weight, _, _, err := consensus.TxWeightAndStats(tx)
	if err != nil {
		t.Fatalf("TxWeightAndStats: %v", err)
	}

	if err := mp.addTxWithSource(txBytes, mempoolTxSourceRemote, nil); err != nil {
		t.Fatalf("addTxWithSource: %v", err)
	}

	mp.mu.RLock()
	defer mp.mu.RUnlock()
	entry, ok := mp.txs[txid]
	if !ok {
		t.Fatalf("entry for txid %x missing", txid)
	}
	if !bytes.Equal(entry.raw, txBytes) {
		t.Fatal("entry raw bytes mismatch")
	}
	if entry.txid != txid {
		t.Fatalf("entry txid=%x, want %x", entry.txid, txid)
	}
	if entry.wtxid != wtxid {
		t.Fatalf("entry wtxid=%x, want %x", entry.wtxid, wtxid)
	}
	if entry.fee.Cmp(consensus.Uint128FromU64(300_000)) != 0 {
		t.Fatalf("entry fee=%d, want 300000", entry.fee)
	}
	if entry.weight != weight {
		t.Fatalf("entry weight=%d, want %d", entry.weight, weight)
	}
	if entry.size != len(txBytes) {
		t.Fatalf("entry wire bytes=%d, want %d", entry.size, len(txBytes))
	}
	if entry.admissionSeq != 1 {
		t.Fatalf("entry admission_seq=%d, want 1", entry.admissionSeq)
	}
	if entry.source != mempoolTxSourceRemote {
		t.Fatalf("entry source=%q, want %q", entry.source, mempoolTxSourceRemote)
	}
	if got, ok := mp.wtxids[wtxid]; !ok || got != txid {
		t.Fatalf("wtxid index got %x ok=%v, want txid %x", got, ok, txid)
	}
	if got, ok := mp.pendingOutpoints.txidForOutpoint(outpoints[0]); !ok || got != txid {
		t.Fatalf("pending-outpoint claim got %x ok=%v, want txid %x", got, ok, txid)
	}
}

func TestMempoolAdmissionSourceWrappersRecordOrigin(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	cases := []struct {
		name      string
		outpoint  consensus.Outpoint
		nonce     uint64
		source    mempoolTxSource
		admitFunc func([]byte) error
	}{
		{
			name:      "local",
			outpoint:  outpoints[0],
			nonce:     1,
			source:    mempoolTxSourceLocal,
			admitFunc: mp.AddTx,
		},
		{
			name:      "remote",
			outpoint:  outpoints[1],
			nonce:     2,
			source:    mempoolTxSourceRemote,
			admitFunc: mp.AddRemoteTx,
		},
		{
			name:      "reorg",
			outpoint:  outpoints[2],
			nonce:     3,
			source:    mempoolTxSourceReorg,
			admitFunc: mp.AddReorgTx,
		},
	}

	for _, tc := range cases {
		txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{tc.outpoint}, 100_000, 300_000, tc.nonce, fromKey, fromAddress, toAddress)
		if err := tc.admitFunc(txBytes); err != nil {
			t.Fatalf("%s admit: %v", tc.name, err)
		}
		txid := txID(t, txBytes)
		mp.mu.RLock()
		entry := mp.txs[txid]
		mp.mu.RUnlock()
		if entry == nil {
			t.Fatalf("%s entry for txid %x missing", tc.name, txid)
		}
		if entry.source != tc.source {
			t.Fatalf("%s source=%q, want %q", tc.name, entry.source, tc.source)
		}
	}
}

func TestMempoolRejectsInvalidEntrySource(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	err = mp.addTxWithSource(txBytes, "sidecar", nil)
	if err == nil || !strings.Contains(err.Error(), "invalid mempool tx source") {
		t.Fatalf("expected invalid source rejection, got %v", err)
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) {
		t.Fatalf("expected TxAdmitError, got %T: %v", err, err)
	}
	if txErr.Kind != TxAdmitRejected {
		t.Fatalf("expected TxAdmitRejected, got %v", txErr.Kind)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
	if mp.lastAdmissionSeq != 0 {
		t.Fatalf("lastAdmissionSeq after invalid source=%d, want 0", mp.lastAdmissionSeq)
	}
}

func TestMempoolAddEntryLockedInitializesMetadataIndexes(t *testing.T) {
	op := consensus.Outpoint{Txid: [32]byte{0x01}, Vout: 2}
	entry := &mempoolEntry{
		txid:         [32]byte{0x02},
		wtxid:        [32]byte{0x03},
		inputs:       []consensus.Outpoint{op},
		fee:          consensus.Uint128FromU64(5),
		weight:       5,
		size:         7,
		admissionSeq: 9,
		source:       mempoolTxSourceReorg,
	}

	mp := &Mempool{maxTxs: 10, maxBytes: 100}
	if err := mp.addEntryLocked(entry); err != nil {
		t.Fatalf("addEntryLocked: %v", err)
	}

	if mp.txs == nil || mp.wtxids == nil {
		t.Fatalf("indexes were not initialized: txs=%v wtxids=%v", mp.txs != nil, mp.wtxids != nil)
	}
	if got := mp.txs[entry.txid]; got != entry {
		t.Fatalf("tx index got %p, want entry %p", got, entry)
	}
	if got := mp.wtxids[entry.wtxid]; got != entry.txid {
		t.Fatalf("wtxid index got %x, want txid %x", got, entry.txid)
	}
	if got, ok := mp.pendingOutpoints.txidForOutpoint(op); !ok || got != entry.txid {
		t.Fatalf("pending-outpoint claim got %x ok=%v, want txid %x", got, ok, entry.txid)
	}
	if mp.lastAdmissionSeq != entry.admissionSeq {
		t.Fatalf("lastAdmissionSeq=%d, want %d", mp.lastAdmissionSeq, entry.admissionSeq)
	}
	if mp.usedBytes != entry.size {
		t.Fatalf("usedBytes=%d, want %d", mp.usedBytes, entry.size)
	}
}

func TestMempoolAddEntryLockedDefaultsUnsetWtxid(t *testing.T) {
	entry := &mempoolEntry{
		txid:   [32]byte{0x0a},
		fee:    consensus.Uint128FromU64(1),
		weight: 1,
		size:   1,
	}

	mp := &Mempool{
		maxTxs:   1,
		maxBytes: 10,
	}
	if err := mp.addEntryLocked(entry); err != nil {
		t.Fatalf("addEntryLocked: %v", err)
	}

	if entry.wtxid != entry.txid {
		t.Fatalf("entry wtxid=%x, want txid %x", entry.wtxid, entry.txid)
	}
	if got, ok := mp.wtxids[entry.txid]; !ok || got != entry.txid {
		t.Fatalf("wtxid index got %x ok=%v, want txid %x", got, ok, entry.txid)
	}
	if got, ok := mp.wtxids[[32]byte{}]; ok {
		t.Fatalf("zero wtxid key unexpectedly indexed txid %x", got)
	}
	err := mp.addEntryLocked(&mempoolEntry{txid: [32]byte{0x0b}, fee: consensus.Uint128FromU64(1), weight: 1, size: 1})
	if err == nil || !strings.Contains(err.Error(), "mempool capacity candidate rejected by eviction ordering") {
		t.Fatalf("expected candidate-worst rejection after zero-wtxid default, got %v", err)
	}
}

func TestMempoolAddEntryLockedRejectsZeroTxidWithoutMutation(t *testing.T) {
	mp := &Mempool{}

	err := mp.addEntryLocked(&mempoolEntry{weight: 1, size: 1})
	if err == nil || !strings.Contains(err.Error(), "invalid mempool entry txid") {
		t.Fatalf("expected invalid txid rejection, got %v", err)
	}
	if mp.txs != nil || mp.wtxids != nil {
		t.Fatalf("indexes initialized after zero txid reject: txs=%v wtxids=%v", mp.txs != nil, mp.wtxids != nil)
	}
	if mp.usedBytes != 0 {
		t.Fatalf("usedBytes=%d, want 0 after zero txid reject", mp.usedBytes)
	}
	if mp.lastAdmissionSeq != 0 {
		t.Fatalf("lastAdmissionSeq=%d, want 0 after zero txid reject", mp.lastAdmissionSeq)
	}

	err = mp.validateNonCapacityAdmissionLocked(&mempoolEntry{weight: 1, size: 1})
	if err == nil || !strings.Contains(err.Error(), "invalid mempool entry txid") {
		t.Fatalf("expected validate invalid txid rejection, got %v", err)
	}
}

func TestMempoolEvictionComparatorTiers(t *testing.T) {
	lowerRate := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x01}, fee: consensus.Uint128FromU64(1), weight: 2, size: 1, admissionSeq: 1}}
	higherRate := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x02}, fee: consensus.Uint128FromU64(1), weight: 1, size: 1, admissionSeq: 2}}
	if !evictionPlanEntryWorse(lowerRate, higherRate) {
		t.Fatal("lower fee/weight entry was not worse")
	}

	lowerAbsoluteFee := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x03}, fee: consensus.Uint128FromU64(1), weight: 1, size: 1000, admissionSeq: 3}}
	higherAbsoluteFee := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x04}, fee: consensus.Uint128FromU64(2), weight: 2, size: 1, admissionSeq: 4}}
	if !evictionPlanEntryWorse(lowerAbsoluteFee, higherAbsoluteFee) {
		t.Fatal("lower absolute fee tie-break was not worse before admission_seq")
	}

	older := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x05}, fee: consensus.Uint128FromU64(3), weight: 3, size: 1, admissionSeq: 5}}
	newer := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x06}, fee: consensus.Uint128FromU64(3), weight: 3, size: 1, admissionSeq: 6}}
	if !evictionPlanEntryWorse(older, newer) {
		t.Fatal("older admission_seq tie-break was not worse")
	}

	candidate := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x07}, fee: consensus.Uint128FromU64(3), weight: 3, size: 1}, candidate: true}
	if !evictionPlanEntryWorse(candidate, older) {
		t.Fatal("capacity candidate did not compare as virtual admission_seq=0")
	}

	local := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x09}, fee: consensus.Uint128FromU64(3), weight: 3, size: 1, admissionSeq: 7, source: mempoolTxSourceLocal}}
	remote := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x08}, fee: consensus.Uint128FromU64(3), weight: 3, size: 1, admissionSeq: 7, source: mempoolTxSourceRemote}}
	if !evictionPlanEntryWorse(local, remote) {
		t.Fatal("source provenance unexpectedly affected eviction ordering before deterministic txid tie-break")
	}
	reorg := mempoolEvictionPlanEntry{entry: &mempoolEntry{txid: [32]byte{0x0a}, fee: consensus.Uint128FromU64(3), weight: 3, size: 1, admissionSeq: 7, source: mempoolTxSourceReorg}}
	if !evictionPlanEntryWorse(reorg, remote) {
		t.Fatal("reorg source provenance unexpectedly affected eviction ordering before deterministic txid tie-break")
	}
}

func TestMempoolFeeRateComparatorUsesWeightAndDoesNotOverflow(t *testing.T) {
	if got := compareFeeRateWeightValues(consensus.Uint128FromU64(^uint64(0)), ^uint64(0)-1, consensus.Uint128FromU64(^uint64(0)-1), ^uint64(0)); got <= 0 {
		t.Fatalf("overflow-sensitive fee-rate compare=%d, want first greater", got)
	}
	lowWeightFeeRate := &mempoolEntry{txid: [32]byte{0x01}, fee: consensus.Uint128FromU64(10), weight: 5, size: 10_000, admissionSeq: 1}
	highWeightFeeRate := &mempoolEntry{txid: [32]byte{0x02}, fee: consensus.Uint128FromU64(10), weight: 10, size: 1, admissionSeq: 2}
	if !evictionPlanEntryWorse(mempoolEvictionPlanEntry{entry: highWeightFeeRate}, mempoolEvictionPlanEntry{entry: lowWeightFeeRate}) {
		t.Fatal("eviction comparator used wire bytes instead of weight")
	}
}

func TestMempoolAddEntryLockedRejectsInvalidSourceAndDuplicateAdmissionSeq(t *testing.T) {
	mp := &Mempool{maxTxs: 10, maxBytes: 100}
	first := &mempoolEntry{
		txid:         [32]byte{0x11},
		fee:          consensus.Uint128FromU64(1),
		weight:       1,
		size:         1,
		admissionSeq: 7,
		source:       mempoolTxSourceLocal,
	}
	if err := mp.addEntryLocked(first); err != nil {
		t.Fatalf("addEntryLocked(first): %v", err)
	}
	err := mp.addEntryLocked(&mempoolEntry{
		txid:         [32]byte{0x12},
		fee:          consensus.Uint128FromU64(1),
		weight:       1,
		size:         1,
		admissionSeq: 7,
		source:       mempoolTxSourceRemote,
	})
	if err == nil || !strings.Contains(err.Error(), "mempool admission sequence conflict") {
		t.Fatalf("expected admission sequence conflict, got %v", err)
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) {
		t.Fatalf("expected TxAdmitError for admission sequence conflict, got %T: %v", err, err)
	}
	if txErr.Kind != TxAdmitRejected {
		t.Fatalf("admission sequence conflict kind=%v, want %v", txErr.Kind, TxAdmitRejected)
	}
	if err := mp.addEntryLocked(&mempoolEntry{
		txid:   [32]byte{0x13},
		fee:    consensus.Uint128FromU64(1),
		weight: 1,
		size:   1,
		source: "sidecar",
	}); err == nil || !strings.Contains(err.Error(), "invalid mempool tx source") {
		t.Fatalf("expected invalid source rejection, got %v", err)
	}
	if got := mp.Len(); got != 1 {
		t.Fatalf("mempool len=%d, want 1 after helper rejects", got)
	}
}

func TestDefaultMempoolLowWaterBytes(t *testing.T) {
	for _, tc := range []struct {
		name     string
		maxBytes int
		want     int
	}{
		{name: "zero", maxBytes: 0, want: 0},
		{name: "negative", maxBytes: -1, want: 0},
		{name: "one", maxBytes: 1, want: 1},
		{name: "small", maxBytes: 9, want: 8},
		{name: "ten", maxBytes: 10, want: 9},
		{name: "remainder", maxBytes: 11, want: 9},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := defaultMempoolLowWaterBytes(tc.maxBytes); got != tc.want {
				t.Fatalf("defaultMempoolLowWaterBytes(%d)=%d, want %d", tc.maxBytes, got, tc.want)
			}
		})
	}
}

func TestMempoolCapacityPlanRejectsInvalidDryRunInputs(t *testing.T) {
	validCandidate := func() *mempoolEntry {
		return &mempoolEntry{
			txid:   [32]byte{0x21},
			fee:    consensus.Uint128FromU64(1),
			weight: 1,
			size:   1,
		}
	}
	for _, tc := range []struct {
		name      string
		mp        *Mempool
		candidate *mempoolEntry
		want      string
	}{
		{
			name:      "nil_candidate",
			mp:        &Mempool{maxTxs: 10, maxBytes: 100},
			candidate: nil,
			want:      "nil mempool entry",
		},
		{
			name:      "negative_max_bytes",
			mp:        &Mempool{maxTxs: 10, maxBytes: -1},
			candidate: validCandidate(),
			want:      "invalid mempool max_bytes",
		},
		{
			name:      "zero_capacity",
			mp:        &Mempool{maxTxs: 0, maxBytes: 100},
			candidate: validCandidate(),
			want:      "invalid mempool capacity limits",
		},
		{
			name: "negative_candidate_size",
			mp:   &Mempool{maxTxs: 10, maxBytes: 100},
			candidate: &mempoolEntry{
				txid:   [32]byte{0x22},
				fee:    consensus.Uint128FromU64(1),
				weight: 1,
				size:   -1,
			},
			want: "invalid mempool candidate_size",
		},
		{
			name:      "negative_used_bytes",
			mp:        &Mempool{maxTxs: 10, maxBytes: 100, usedBytes: -1},
			candidate: validCandidate(),
			want:      "invalid mempool used_bytes",
		},
		{
			name: "candidate_over_max_bytes",
			mp:   &Mempool{maxTxs: 10, maxBytes: 1},
			candidate: &mempoolEntry{
				txid:   [32]byte{0x23},
				fee:    consensus.Uint128FromU64(1),
				weight: 1,
				size:   2,
			},
			want: "mempool byte limit exceeded",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := tc.mp.capacityEvictionPlanLocked(tc.candidate)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected %q rejection, got %v", tc.want, err)
			}
		})
	}
}

func TestMempoolCapacityPlanRejectsInvalidExistingMetadata(t *testing.T) {
	validExisting := func(txid [32]byte, seq uint64) *mempoolEntry {
		return &mempoolEntry{
			txid:         txid,
			fee:          consensus.Uint128FromU64(10),
			weight:       1,
			size:         1,
			admissionSeq: seq,
			source:       mempoolTxSourceLocal,
		}
	}
	validCandidate := &mempoolEntry{
		txid:   [32]byte{0xaa},
		fee:    consensus.Uint128FromU64(10),
		weight: 1,
		size:   1,
	}
	for _, tc := range []struct {
		name    string
		entries map[[32]byte]*mempoolEntry
		want    string
	}{
		{
			name:    "nil_existing",
			entries: map[[32]byte]*mempoolEntry{{0x01}: nil},
			want:    "nil mempool entry",
		},
		{
			name:    "zero_txid",
			entries: map[[32]byte]*mempoolEntry{{0x02}: {fee: consensus.Uint128FromU64(10), weight: 1, size: 1, admissionSeq: 1}},
			want:    "invalid mempool entry txid",
		},
		{
			name: "zero_size",
			entries: map[[32]byte]*mempoolEntry{
				{0x03}: {txid: [32]byte{0x03}, fee: consensus.Uint128FromU64(10), weight: 1, admissionSeq: 1},
			},
			want: "invalid mempool entry size",
		},
		{
			name: "zero_weight",
			entries: map[[32]byte]*mempoolEntry{
				{0x04}: {txid: [32]byte{0x04}, fee: consensus.Uint128FromU64(10), size: 1, admissionSeq: 1},
			},
			want: "invalid mempool entry weight",
		},
		{
			name: "zero_admission_seq",
			entries: map[[32]byte]*mempoolEntry{
				{0x05}: {txid: [32]byte{0x05}, fee: consensus.Uint128FromU64(10), weight: 1, size: 1},
			},
			want: "invalid mempool entry admission_seq",
		},
		{
			name: "duplicate_admission_seq",
			entries: map[[32]byte]*mempoolEntry{
				{0x06}: validExisting([32]byte{0x06}, 1),
				{0x07}: validExisting([32]byte{0x07}, 1),
			},
			want: "duplicate mempool entry admission_seq",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mp := &Mempool{
				maxTxs:    1,
				maxBytes:  100,
				usedBytes: len(tc.entries),
				txs:       tc.entries,
			}
			_, _, err := mp.capacityEvictionPlanLocked(validCandidate)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected %q rejection, got %v", tc.want, err)
			}
		})
	}
}

func TestMempoolEntryFloorRateUsesSatisfiableFloor(t *testing.T) {
	for _, tc := range []struct {
		name   string
		fee    uint64
		weight uint64
		want   uint64
	}{
		{name: "C1_non_div_even_3_over_2", fee: 3, weight: 2, want: 1},
		{name: "C2_non_div_odd_5_over_3", fee: 5, weight: 3, want: 1},
		{name: "C3_non_div_big_7_over_4", fee: 7, weight: 4, want: 1},
		{name: "C4_divisible_4_over_2", fee: 4, weight: 2, want: 2},
		{name: "C5_divisible_9_over_3", fee: 9, weight: 3, want: 3},
		{name: "C6_fee_equals_weight", fee: 1, weight: 1, want: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			entry := &mempoolEntry{
				txid:   [32]byte{0x71},
				fee:    consensus.Uint128FromU64(tc.fee),
				weight: tc.weight,
				size:   1,
			}

			floor, ok := entryFloorRate(entry)
			if !ok {
				t.Fatal("entryFloorRate returned !ok for valid entry")
			}
			if floor != tc.want {
				t.Fatalf("entryFloorRate=%d, want %d for fee=%d weight=%d", floor, tc.want, tc.fee, tc.weight)
			}
			if feeRateBelowFloor(entry.fee, entry.weight, floor) {
				t.Fatalf("entry is below its satisfiable floor: fee=%d weight=%d floor=%d", entry.fee, entry.weight, floor)
			}
			if !feeRateBelowFloor(entry.fee, entry.weight, floor+DefaultMempoolMinFeeRate) {
				t.Fatalf("entry unexpectedly satisfies raised floor: fee=%d weight=%d floor=%d", entry.fee, entry.weight, floor+DefaultMempoolMinFeeRate)
			}
		})
	}
	if floor, ok := entryFloorRate(&mempoolEntry{txid: [32]byte{0x72}, fee: consensus.Uint128FromU64(3)}); ok || floor != 0 {
		t.Fatalf("zero-weight entryFloorRate=(%d,%v), want (0,false)", floor, ok)
	}
}

// RUB-1127: entryFloorRate's two high-limb branches. Both are unreachable
// with a fee at or below u64 — fee.Hi is zero there, so the clamp never fires
// and bits.Div64 degenerates to a single-limb divide. A fee narrowed to u64
// yields 0 on both rows.
func TestMempoolEntryFloorRateHighLimbBranches(t *testing.T) {
	// fee.Hi >= weight: the exact quotient is at or above 2^64, so the u64
	// policy RATE clamps to its maximum. The entry's own fee is not narrowed.
	clamped, ok := entryFloorRate(&mempoolEntry{txid: [32]byte{0x81}, fee: consensus.Uint128{Hi: 1}, weight: 1, size: 1})
	if !ok || clamped != ^uint64(0) {
		t.Fatalf("clamp branch = (%d,%v), want (%d,true)", clamped, ok, ^uint64(0))
	}

	// fee.Hi != 0 and fee.Hi < weight: the 128-by-64 division runs and keeps
	// every bit the high limb contributes. 2^64/4 = 2^62.
	divided, ok := entryFloorRate(&mempoolEntry{txid: [32]byte{0x82}, fee: consensus.Uint128{Hi: 1}, weight: 4, size: 1})
	if !ok || divided != uint64(1)<<62 {
		t.Fatalf("high-limb division = (%d,%v), want (%d,true)", divided, ok, uint64(1)<<62)
	}
	// The derived floor must be satisfiable by the entry it came from, and one
	// step above it must not be — the same contract the u64 rows assert.
	if feeRateBelowFloor(consensus.Uint128{Hi: 1}, 4, divided) {
		t.Fatalf("entry is below its own derived floor %d", divided)
	}
	if !feeRateBelowFloor(consensus.Uint128{Hi: 1}, 4, divided+DefaultMempoolMinFeeRate) {
		t.Fatalf("entry unexpectedly satisfies raised floor %d", divided+DefaultMempoolMinFeeRate)
	}
}

// RUB-1127: the policy layer ranks the exact u128 fee end to end — admission,
// miner ordering, eviction priority, the capacity victim choice, and live
// selection. Every row straddles u64: 2^64 truncates to 0 under a u64
// narrowing while 2^64-1 truncates to itself, so a narrowed comparison
// inverts each verdict.
func TestMempoolPolicyAboveU64FeesAdmitOrderAndEvictExactly(t *testing.T) {
	aboveU64 := consensus.Uint128{Hi: 1}             // exactly 2^64
	belowU64 := consensus.Uint128FromU64(^uint64(0)) // 2^64-1
	if aboveU64.Cmp(belowU64) <= 0 {
		t.Fatal("row must straddle u64")
	}

	// Admission: weight*floor is 1000, which the exact fee clears and a fee
	// truncated to its low limb (0) does not.
	admitting := &Mempool{maxTxs: 10, maxBytes: 100, currentMinFeeRate: 1000}
	if err := admitting.addEntryLocked(&mempoolEntry{txid: [32]byte{0x91}, fee: aboveU64, weight: 1, size: 1}); err != nil {
		t.Fatalf("fee above u64 must clear a weight*floor of 1000: %v", err)
	}

	wide := &mempoolEntry{txid: [32]byte{0x92}, fee: aboveU64, weight: 1, size: 1, raw: []byte{0x92}, admissionSeq: 1}
	narrow := &mempoolEntry{txid: [32]byte{0x93}, fee: belowU64, weight: 1, size: 1, raw: []byte{0x93}, admissionSeq: 2}

	// Miner ordering.
	entries := []*mempoolEntry{narrow, wide}
	sortMempoolEntries(entries)
	if entries[0] != wide {
		t.Fatalf("sortMempoolEntries ranked %x first, want the above-u64 fee %x", entries[0].txid, wide.txid)
	}

	// Eviction priority.
	if !evictionPlanEntryWorse(mempoolEvictionPlanEntry{entry: narrow}, mempoolEvictionPlanEntry{entry: wide}) {
		t.Fatal("eviction priority did not rank the below-u64 fee as worse")
	}

	// Capacity victim: the competing resident fees straddle u64, so the plan
	// must remove the below-u64 entry and keep the above-u64 one.
	full := &Mempool{
		maxTxs:    2,
		maxBytes:  100,
		usedBytes: 2,
		txs:       map[[32]byte]*mempoolEntry{wide.txid: wide, narrow.txid: narrow},
	}
	evicted, candidateEvicted, err := full.capacityEvictionPlanLocked(&mempoolEntry{
		txid: [32]byte{0x94}, fee: consensus.Uint128{Hi: 2}, weight: 1, size: 1,
	})
	if err != nil || candidateEvicted {
		t.Fatalf("capacity plan err=%v candidateEvicted=%v", err, candidateEvicted)
	}
	if len(evicted) != 1 || evicted[0] != narrow {
		t.Fatalf("capacity evicted %d entries (first %x), want exactly the below-u64 entry %x", len(evicted), evicted[0].txid, narrow.txid)
	}

	// Live miner selection over the same pair.
	selected := full.SelectTransactions(2, 100)
	if len(selected) != 2 || !bytes.Equal(selected[0], wide.raw) {
		t.Fatalf("SelectTransactions=%v, want the above-u64 raw %x first", selected, wide.raw)
	}
}

func TestMempoolRaiseMinFeeRateUsesHighestSatisfiableEvictedFloor(t *testing.T) {
	mp := &Mempool{maxTxs: 10, maxBytes: 100, currentMinFeeRate: DefaultMempoolMinFeeRate}
	mp.raiseMinFeeRateAfterEvictionLocked([]*mempoolEntry{
		{txid: [32]byte{0x81}, fee: consensus.Uint128FromU64(3), weight: 2, size: 1},
		{txid: [32]byte{0x82}, fee: consensus.Uint128FromU64(10), weight: 2, size: 1},
		{txid: [32]byte{0x83}, fee: consensus.Uint128FromU64(7), weight: 4, size: 1},
	})

	wantFloor := uint64(5) + DefaultMempoolMinFeeRate
	if got := mp.currentMinFeeRate; got != wantFloor {
		t.Fatalf("currentMinFeeRate=%d, want %d after mixed non-divisible/divisible eviction", got, wantFloor)
	}
	if err := mp.validateFeeFloorLocked(&mempoolEntry{txid: [32]byte{0x84}, fee: consensus.Uint128FromU64(12), weight: 2, size: 1}); err != nil {
		t.Fatalf("candidate at raised floor was rejected: %v", err)
	}
	for _, entry := range []*mempoolEntry{
		{txid: [32]byte{0x85}, fee: consensus.Uint128FromU64(10), weight: 2, size: 1},
		{txid: [32]byte{0x86}, fee: consensus.Uint128FromU64(8), weight: 2, size: 1},
	} {
		if err := mp.validateFeeFloorLocked(entry); err == nil || !strings.Contains(err.Error(), "mempool fee below rolling minimum") {
			t.Fatalf("candidate below raised floor was not rejected as below-floor: fee=%d weight=%d err=%v", entry.fee, entry.weight, err)
		}
	}
}

func TestMempoolRaiseMinFeeRateUsesHighestNonDivisibleEvictedFloor(t *testing.T) {
	mp := &Mempool{maxTxs: 10, maxBytes: 100, currentMinFeeRate: DefaultMempoolMinFeeRate}
	mp.raiseMinFeeRateAfterEvictionLocked([]*mempoolEntry{
		{txid: [32]byte{0xa1}, fee: consensus.Uint128FromU64(3), weight: 2, size: 1},
		{txid: [32]byte{0xa2}, fee: consensus.Uint128FromU64(7), weight: 4, size: 1},
	})

	wantFloor := uint64(1) + DefaultMempoolMinFeeRate
	if got := mp.currentMinFeeRate; got != wantFloor {
		t.Fatalf("currentMinFeeRate=%d, want %d after all-non-divisible eviction", got, wantFloor)
	}
	if err := mp.validateFeeFloorLocked(&mempoolEntry{txid: [32]byte{0xa3}, fee: consensus.Uint128FromU64(4), weight: 2, size: 1}); err != nil {
		t.Fatalf("candidate at non-divisible raised floor was rejected: %v", err)
	}
}

func TestMempoolRaiseMinFeeRatePreservesDivisibleEvictedFloor(t *testing.T) {
	mp := &Mempool{maxTxs: 10, maxBytes: 100, currentMinFeeRate: DefaultMempoolMinFeeRate}
	mp.raiseMinFeeRateAfterEvictionLocked([]*mempoolEntry{
		{txid: [32]byte{0x91}, fee: consensus.Uint128FromU64(4), weight: 2, size: 1},
	})

	wantFloor := uint64(2) + DefaultMempoolMinFeeRate
	if got := mp.currentMinFeeRate; got != wantFloor {
		t.Fatalf("currentMinFeeRate=%d, want %d after divisible eviction", got, wantFloor)
	}
	if err := mp.validateFeeFloorLocked(&mempoolEntry{txid: [32]byte{0x92}, fee: consensus.Uint128FromU64(6), weight: 2, size: 1}); err != nil {
		t.Fatalf("candidate at divisible raised floor was rejected: %v", err)
	}
	if err := mp.validateFeeFloorLocked(&mempoolEntry{txid: [32]byte{0x93}, fee: consensus.Uint128FromU64(4), weight: 2, size: 1}); err == nil || !strings.Contains(err.Error(), "mempool fee below rolling minimum") {
		t.Fatalf("candidate below divisible raised floor was not rejected as below-floor: %v", err)
	}
}

func TestMempoolSortAndEvictionUseWeightFeeRate(t *testing.T) {
	feeSizeWinner := &mempoolEntry{txid: [32]byte{0xb1}, fee: consensus.Uint128FromU64(4), weight: 4, size: 1}
	feeWeightWinner := &mempoolEntry{txid: [32]byte{0xb2}, fee: consensus.Uint128FromU64(2), weight: 1, size: 1}

	entries := []*mempoolEntry{feeSizeWinner, feeWeightWinner}
	sortMempoolEntries(entries)
	if entries[0] != feeWeightWinner {
		t.Fatalf("sortMempoolEntries picked txid %x first, want fee/weight winner %x", entries[0].txid, feeWeightWinner.txid)
	}

	if !evictionPlanEntryWorse(
		mempoolEvictionPlanEntry{entry: feeSizeWinner},
		mempoolEvictionPlanEntry{entry: feeWeightWinner},
	) {
		t.Fatal("eviction priority did not mark lower fee/weight entry as worse")
	}
}

func TestMempoolAddEntryLockedCapacityPlanRejectsWithoutMutation(t *testing.T) {
	badResidentID := [32]byte{0x30}
	mp := &Mempool{
		maxTxs:    1,
		maxBytes:  100,
		usedBytes: 1,
		txs: map[[32]byte]*mempoolEntry{
			badResidentID: {
				txid:         badResidentID,
				fee:          consensus.Uint128FromU64(10),
				size:         1,
				admissionSeq: 1,
			},
		},
	}
	err := mp.addEntryLocked(&mempoolEntry{
		txid:   [32]byte{0x31},
		fee:    consensus.Uint128FromU64(10),
		weight: 1,
		size:   1,
	})
	if err == nil || !strings.Contains(err.Error(), "invalid mempool entry weight") {
		t.Fatalf("expected capacity plan metadata rejection, got %v", err)
	}
	if len(mp.txs) != 1 || mp.txs[badResidentID] == nil || mp.wtxids != nil || mp.lastAdmissionSeq != 0 || mp.currentMinFeeRate != 0 || mp.usedBytes != 1 {
		t.Fatalf("capacity-plan error mutated mempool: len=%d wtxids=%v seq=%d floor=%d used=%d", len(mp.txs), mp.wtxids != nil, mp.lastAdmissionSeq, mp.currentMinFeeRate, mp.usedBytes)
	}
}

func TestMempoolAddEntryLockedCandidateWorstRejectsWithoutMutation(t *testing.T) {
	mp := &Mempool{maxTxs: 1, maxBytes: 100}
	resident := &mempoolEntry{
		txid:   [32]byte{0x41},
		fee:    consensus.Uint128FromU64(100),
		weight: 1,
		size:   1,
	}
	if err := mp.addEntryLocked(resident); err != nil {
		t.Fatalf("addEntryLocked(resident): %v", err)
	}
	before, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot before direct candidate-worst: %v", err)
	}
	err = mp.addEntryLocked(&mempoolEntry{
		txid:   [32]byte{0x42},
		fee:    consensus.Uint128FromU64(1),
		weight: 1,
		size:   1,
	})
	if err == nil || !strings.Contains(err.Error(), "mempool capacity candidate rejected by eviction ordering") {
		t.Fatalf("expected direct candidate-worst rejection, got %v", err)
	}
	after, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot after direct candidate-worst: %v", err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("direct candidate-worst mutated mempool: before=%+v after=%+v", before, after)
	}
}

func TestMempoolRejectsZeroWeightMetadata(t *testing.T) {
	mp := &Mempool{maxTxs: 10, maxBytes: 100}
	err := mp.validateNonCapacityAdmissionLocked(&mempoolEntry{
		txid: [32]byte{0x21},
		fee:  consensus.Uint128FromU64(1),
		size: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "invalid mempool entry weight") {
		t.Fatalf("expected zero weight rejection, got %v", err)
	}
}

func TestMempoolEntryIndexesRemovedWithEntry(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 300_000, 1, fromKey, fromAddress, toAddress)
	_, txid, wtxid, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx: %v", err)
	}
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}

	mp.mu.Lock()
	if err := mp.removeTxLocked(txid); err != nil {
		t.Fatalf("removeTxLocked: %v", err)
	}
	if _, ok := mp.txs[txid]; ok {
		t.Fatalf("removed txid %x still present", txid)
	}
	if _, ok := mp.wtxids[wtxid]; ok {
		t.Fatalf("removed wtxid %x still indexed", wtxid)
	}
	if _, ok := mp.pendingOutpoints.txidForOutpoint(outpoints[0]); ok {
		t.Fatalf("removed claim %x:%d still indexed", outpoints[0].Txid, outpoints[0].Vout)
	}
	mp.mu.Unlock()
}

func TestMempoolAdmissionSeqOnlyAcceptedTxs(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if err := mp.AddTx([]byte{0xde, 0xad}); err == nil {
		t.Fatal("malformed tx unexpectedly accepted")
	}
	if mp.lastAdmissionSeq != 0 {
		t.Fatalf("lastAdmissionSeq after malformed=%d, want 0", mp.lastAdmissionSeq)
	}

	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 100_000, 2, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(tx1); err != nil {
		t.Fatalf("AddTx(tx1): %v", err)
	}
	if got := mp.txs[txID(t, tx1)].admissionSeq; got != 1 {
		t.Fatalf("tx1 admission_seq=%d, want 1", got)
	}
	if err := mp.AddTx(tx1); err == nil {
		t.Fatal("duplicate tx unexpectedly accepted")
	}
	if mp.lastAdmissionSeq != 1 {
		t.Fatalf("lastAdmissionSeq after duplicate=%d, want 1", mp.lastAdmissionSeq)
	}
	if err := mp.AddTx(tx2); err != nil {
		t.Fatalf("AddTx(tx2): %v", err)
	}
	if got := mp.txs[txID(t, tx2)].admissionSeq; got != 2 {
		t.Fatalf("tx2 admission_seq=%d, want 2", got)
	}
}

func TestMempoolAdmissionSeqDoesNotWrap(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	mp.lastAdmissionSeq = ^uint64(0)

	err = mp.AddTx(txBytes)
	if err == nil || !strings.Contains(err.Error(), "mempool admission sequence exhausted") {
		t.Fatalf("expected sequence exhaustion rejection, got %v", err)
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) {
		t.Fatalf("expected TxAdmitError, got %T: %v", err, err)
	}
	if txErr.Kind != TxAdmitUnavailable {
		t.Fatalf("expected TxAdmitUnavailable, got %v", txErr.Kind)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
	if mp.lastAdmissionSeq != ^uint64(0) {
		t.Fatalf("lastAdmissionSeq mutated to %d", mp.lastAdmissionSeq)
	}
}

func TestMempoolRejectsDuplicateWtxidIndexWithoutMutation(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(tx1); err != nil {
		t.Fatalf("AddTx(tx1): %v", err)
	}
	tx1ID := txID(t, tx1)
	tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 100_000, 2, fromKey, fromAddress, toAddress)
	_, tx2ID, tx2Wtxid, _, err := consensus.ParseTx(tx2)
	if err != nil {
		t.Fatalf("ParseTx(tx2): %v", err)
	}

	mp.mu.Lock()
	mp.wtxids[tx2Wtxid] = tx1ID
	usedBytes := mp.usedBytes
	lastAdmissionSeq := mp.lastAdmissionSeq
	mp.mu.Unlock()

	err = mp.AddTx(tx2)
	if err == nil || !strings.Contains(err.Error(), "mempool wtxid conflict") {
		t.Fatalf("expected wtxid conflict rejection, got %v", err)
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) {
		t.Fatalf("expected TxAdmitError, got %T: %v", err, err)
	}
	if txErr.Kind != TxAdmitConflict {
		t.Fatalf("expected TxAdmitConflict, got %v", txErr.Kind)
	}
	if got := mp.Len(); got != 1 {
		t.Fatalf("mempool len=%d, want 1 after wtxid conflict", got)
	}
	if mp.Contains(tx2ID) {
		t.Fatalf("wtxid conflict admitted tx2 %x", tx2ID)
	}
	if mp.usedBytes != usedBytes {
		t.Fatalf("usedBytes=%d, want %d after wtxid conflict", mp.usedBytes, usedBytes)
	}
	if mp.lastAdmissionSeq != lastAdmissionSeq {
		t.Fatalf("lastAdmissionSeq=%d, want %d after wtxid conflict", mp.lastAdmissionSeq, lastAdmissionSeq)
	}
	if got := mp.wtxids[tx2Wtxid]; got != tx1ID {
		t.Fatalf("wtxid index overwritten with %x, want existing %x", got, tx1ID)
	}
}

func TestMempoolAddTxWaitsForChainStateWriter(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)

	st.admissionMu.Lock()
	done := make(chan error, 1)
	started := make(chan struct{})
	go func() {
		close(started)
		done <- mp.AddTx(txBytes)
	}()
	<-started

	select {
	case err := <-done:
		t.Fatalf("AddTx returned while chainstate writer lock held: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	st.admissionMu.Unlock()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("AddTx after writer unlock: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AddTx remained blocked after chainstate writer unlock")
	}
}

func TestMempoolAddTxRejectsWhenWriterInvalidatesSnapshotBeforeAdmission(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)

	st.admissionMu.Lock()
	st.mu.Lock()
	delete(st.Utxos, outpoints[0])
	st.mu.Unlock()

	done := make(chan error, 1)
	go func() {
		done <- mp.AddTx(txBytes)
	}()

	select {
	case err := <-done:
		t.Fatalf("AddTx returned while writer gate held: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	st.admissionMu.Unlock()

	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), string(consensus.TX_ERR_MISSING_UTXO)) {
			t.Fatalf("expected missing utxo after writer mutation, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AddTx remained blocked after writer gate unlock")
	}

	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
}

func TestMempoolAddTxWaitsForPolicyWriterBeforeSnapshot(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{100})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedAnchorOutputTx(t, st.Utxos, outpoints[0], 0, 1, 1, fromKey, toAddress)

	mp.mu.Lock()
	mp.policy.PolicyRejectNonCoinbaseAnchorOutputs = true

	done := make(chan error, 1)
	go func() {
		done <- mp.AddTx(txBytes)
	}()

	select {
	case err := <-done:
		t.Fatalf("AddTx returned while policy writer lock held: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	mp.mu.Unlock()

	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), "non-coinbase CORE_ANCHOR") {
			t.Fatalf("expected policy rejection after writer unlock, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AddTx remained blocked after policy writer unlock")
	}

	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
}

func TestMempoolRelayMetadata(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	mp.SetCurrentMinFeeRateForTest(8)
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 300_000, 5, fromKey, fromAddress, toAddress)
	meta, err := mp.RelayMetadata(txBytes)
	if err != nil {
		t.Fatalf("RelayMetadata: %v", err)
	}
	if meta.Fee.Cmp(consensus.Uint128FromU64(300_000)) != 0 {
		t.Fatalf("fee=%d, want 300000", meta.Fee)
	}
	if meta.Size != len(txBytes) {
		t.Fatalf("size=%d, want %d", meta.Size, len(txBytes))
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("RelayMetadata inserted tx; mempool len=%d, want 0", got)
	}
	if mp.Contains(txID(t, txBytes)) {
		t.Fatalf("RelayMetadata inserted txid %x", txID(t, txBytes))
	}
	if got := mp.BytesUsed(); got != 0 {
		t.Fatalf("RelayMetadata mutated bytes used=%d, want 0", got)
	}
	if mp.lastAdmissionSeq != 0 {
		t.Fatalf("RelayMetadata consumed admission_seq=%d, want 0", mp.lastAdmissionSeq)
	}
}

func TestMempoolRelayMetadataBelowRollingFloorUnavailable(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	mp.SetCurrentMinFeeRateForTest(8)
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 1, 5, fromKey, fromAddress, toAddress)

	_, err = mp.RelayMetadata(txBytes)
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitUnavailable {
		t.Fatalf("RelayMetadata below-floor err=%T %v, want TxAdmitUnavailable", err, err)
	}
	if !strings.Contains(txErr.Message, "mempool fee below rolling minimum") {
		t.Fatalf("below-floor reason %q does not match admit-path fee-floor wording", txErr.Message)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("RelayMetadata below-floor inserted tx; mempool len=%d, want 0", got)
	}
	if mp.Contains(txID(t, txBytes)) {
		t.Fatalf("RelayMetadata below-floor inserted txid %x", txID(t, txBytes))
	}
	if got := mp.BytesUsed(); got != 0 {
		t.Fatalf("RelayMetadata below-floor mutated bytes used=%d, want 0", got)
	}
	if mp.lastAdmissionSeq != 0 {
		t.Fatalf("RelayMetadata below-floor consumed admission_seq=%d, want 0", mp.lastAdmissionSeq)
	}
}

func TestMempoolRelayMetadataTrailingBytes(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	mp.SetCurrentMinFeeRateForTest(8)
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 1, 5, fromKey, fromAddress, toAddress)
	txBytes = append(txBytes, 0x00)
	_, err = mp.RelayMetadata(txBytes)
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitRejected {
		t.Fatalf("RelayMetadata trailing bytes err=%T %v, want TxAdmitRejected", err, err)
	}
	if !strings.Contains(txErr.Message, "trailing bytes after canonical tx") {
		t.Fatalf("expected trailing-bytes rejection, got %v", err)
	}
	if strings.Contains(txErr.Message, "mempool fee below rolling minimum") {
		t.Fatalf("trailing-bytes path was stolen by rolling-floor check: %q", txErr.Message)
	}
}

func TestMempoolRelayMetadataNil(t *testing.T) {
	var mp *Mempool
	if _, err := mp.RelayMetadata([]byte{0x01}); err == nil {
		t.Fatal("nil mempool should reject RelayMetadata")
	}
}

func TestMempoolPolicyRejectsNonCoinbaseAnchorOutputs(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{100})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		PolicyRejectNonCoinbaseAnchorOutputs: true,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	txBytes := mustBuildSignedAnchorOutputTx(t, st.Utxos, outpoints[0], 0, 1, 1, fromKey, toAddress)
	if err := mp.AddTx(txBytes); err == nil || !strings.Contains(err.Error(), "non-coinbase CORE_ANCHOR") {
		t.Fatalf("expected non-coinbase anchor policy rejection, got %v", err)
	}
	if _, err := mp.RelayMetadata(txBytes); err == nil || !strings.Contains(err.Error(), "non-coinbase CORE_ANCHOR") {
		t.Fatalf("expected relay metadata anchor policy rejection, got %v", err)
	}
}

func TestMempoolPolicyRejectsLowFeeDaCommit(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{100})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		PolicyDaSurchargePerByte: 1,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	txBytes := mustBuildSignedDaCommitTx(t, st.Utxos, outpoints[0], 99, 1, 1, fromKey, toAddress, []byte("0123456789"))
	if err := mp.AddTx(txBytes); err == nil || !strings.Contains(err.Error(), "DA fee below Stage C floor") {
		t.Fatalf("expected DA Stage C floor rejection, got %v", err)
	}
	_, err = mp.RelayMetadata(txBytes)
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitRejected {
		t.Fatalf("RelayMetadata low-fee DA err=%T %v, want TxAdmitRejected", err, err)
	}
	if !strings.Contains(txErr.Message, "DA fee below Stage C floor") {
		t.Fatalf("expected relay metadata DA Stage C floor rejection, got %v", err)
	}
	if strings.Contains(txErr.Message, "mempool fee below rolling minimum") {
		t.Fatalf("DA floor path was stolen by rolling-floor check: %q", txErr.Message)
	}
}

// TestMempoolPolicySufficientFeeDaCommitRejectedAtKindGuard is the A6 row: a DA
// commit that clears the Stage C fee terms is still policy-valid — RelayMetadata
// returns its exact fee and serialized size and inserts nothing — while standard
// admission refuses the same bytes at the tx_kind guard.
func TestMempoolPolicySufficientFeeDaCommitRejectedAtKindGuard(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		PolicyDaSurchargePerByte: 1,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	txBytes := mustBuildSignedDaCommitTx(t, st.Utxos, outpoints[0], 100_000, 900_000, 1, fromKey, toAddress, []byte("0123456789"))
	metadata, err := mp.RelayMetadata(txBytes)
	if err != nil || metadata.Fee != consensus.Uint128FromU64(900_000) || metadata.Size != len(txBytes) {
		t.Fatalf("RelayMetadata=(%+v,%v), want the exact fee 900000 and size %d", metadata, err, len(txBytes))
	}
	requireDAKindReject(t, mp.AddTx(txBytes))
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
}

func TestMempoolPolicyRejectsDaCommitDeclaredChunkBudget(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		PolicyMaxDaBytesPerBlock: consensus.CHUNK_BYTES - 1,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	txBytes := mustBuildSignedDaCommitTxWithChunkCount(t, st.Utxos, outpoints[0], 50_000, 950_000, 1, fromKey, toAddress, 1, []byte("0123456789"))
	err = mp.AddTx(txBytes)
	if err == nil || !strings.Contains(err.Error(), "DA declared chunk budget exceeded") {
		t.Fatalf("expected DA declared chunk budget rejection, got %v", err)
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitRejected {
		t.Fatalf("over-budget DA err=%v, want TxAdmitRejected", err)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d after rejected AddTx, want 0", got)
	}
	if got := mp.BytesUsed(); got != 0 {
		t.Fatalf("mempool bytes=%d after rejected AddTx, want 0", got)
	}
	if mp.lastAdmissionSeq != 0 {
		t.Fatalf("rejected AddTx consumed admission_seq=%d, want 0", mp.lastAdmissionSeq)
	}
	counts := mp.AdmissionCounts()
	if counts.Rejected != 1 || counts.Accepted != 0 || counts.Conflict != 0 || counts.Unavailable != 0 {
		t.Fatalf("admission counts=%+v, want one rejected AddTx", counts)
	}

	_, err = mp.RelayMetadata(txBytes)
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitRejected {
		t.Fatalf("RelayMetadata over-budget err=%T %v, want TxAdmitRejected", err, err)
	}
	if !strings.Contains(txErr.Message, "DA declared chunk budget exceeded") {
		t.Fatalf("RelayMetadata reason=%q, want declared chunk budget rejection", txErr.Message)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("RelayMetadata mutated mempool len=%d, want 0", got)
	}
	if got := mp.BytesUsed(); got != 0 {
		t.Fatalf("RelayMetadata mutated mempool bytes=%d, want 0", got)
	}
	if mp.lastAdmissionSeq != 0 {
		t.Fatalf("RelayMetadata consumed admission_seq=%d, want 0", mp.lastAdmissionSeq)
	}
}

// TestMempoolPolicyDaCommitAtDeclaredChunkBudgetRejectedAtKindGuard keeps the
// at-budget boundary: a DA commit whose declared chunk budget exactly fits still
// passes rejectDaCommitDeclaredBudget, so RelayMetadata succeeds and inserts
// nothing, and standard admission refuses it at the tx_kind guard instead.
func TestMempoolPolicyDaCommitAtDeclaredChunkBudgetRejectedAtKindGuard(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		PolicyMaxDaBytesPerBlock: consensus.CHUNK_BYTES,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	txBytes := mustBuildSignedDaCommitTxWithChunkCount(t, st.Utxos, outpoints[0], 50_000, 950_000, 1, fromKey, toAddress, 1, []byte("0123456789"))
	if _, err := mp.RelayMetadata(txBytes); err != nil {
		t.Fatalf("RelayMetadata at budget: %v", err)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("RelayMetadata inserted tx; mempool len=%d, want 0", got)
	}
	requireDAKindReject(t, mp.AddTx(txBytes))
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
}

func TestMempoolAdmissionSourceWrappersRejectDaCommitDeclaredBudget(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())

	cases := []struct {
		name  string
		admit func(*Mempool, []byte) error
	}{
		{name: "local", admit: func(mp *Mempool, tx []byte) error { return mp.AddTx(tx) }},
		{name: "remote", admit: func(mp *Mempool, tx []byte) error { return mp.AddRemoteTx(tx) }},
		{name: "reorg", admit: func(mp *Mempool, tx []byte) error { return mp.AddReorgTx(tx) }},
	}
	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
			mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
				PolicyMaxDaBytesPerBlock: consensus.CHUNK_BYTES - 1,
			})
			if err != nil {
				t.Fatalf("new mempool: %v", err)
			}
			txBytes := mustBuildSignedDaCommitTxWithChunkCount(t, st.Utxos, outpoints[0], 50_000, 950_000, uint64(i+1), fromKey, toAddress, 1, []byte("0123456789"))
			err = tc.admit(mp, txBytes)
			if err == nil || !strings.Contains(err.Error(), "DA declared chunk budget exceeded") {
				t.Fatalf("expected DA declared chunk budget rejection, got %v", err)
			}
			if got := mp.Len(); got != 0 {
				t.Fatalf("mempool len=%d, want 0", got)
			}
		})
	}
}

// TestMempoolPartialConfigBackfillsMinDaFeeRateAndRejectsDaTxAtKindGuard
// pins the default-config path for PR #1368: a partial MempoolConfig
// literal is interpreted as defaults plus overrides, so an omitted
// MinDaFeeRate backfills to DefaultMinDaFeeRate and a DA-bearing tx that
// pays both the DA-side floor and relay floor clears every fee term.
//
// Proof assertion: mp.policy.MinDaFeeRate is the backfilled default, so partial
// configs cannot silently fall back to the old surcharge-only behavior, and the
// fee-clearing DA candidate is refused at the tx_kind guard rather than by any
// fee term.
func TestMempoolPartialConfigBackfillsMinDaFeeRateAndRejectsDaTxAtKindGuard(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MaxTransactions:          10,
		MaxBytes:                 1 << 20,
		PolicyDaSurchargePerByte: 0,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if got := mp.policy.MinDaFeeRate; got != DefaultMinDaFeeRate {
		t.Fatalf("MinDaFeeRate=%d, want DefaultMinDaFeeRate=%d", got, DefaultMinDaFeeRate)
	}

	txBytes := mustBuildSignedDaCommitTx(t, st.Utxos, outpoints[0], 50_000, 950_000, 1, fromKey, toAddress, []byte("0123456789"))
	requireDAKindReject(t, mp.AddTx(txBytes))
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
}

func TestMempoolPartialConfigBackfillsMinDaFeeRateForLowFeeDaCommit(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{100})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MaxTransactions: 10,
		MaxBytes:        1 << 20,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if got := mp.policy.MinDaFeeRate; got != DefaultMinDaFeeRate {
		t.Fatalf("MinDaFeeRate=%d, want DefaultMinDaFeeRate=%d", got, DefaultMinDaFeeRate)
	}

	txBytes := mustBuildSignedDaCommitTx(t, st.Utxos, outpoints[0], 99, 1, 1, fromKey, toAddress, []byte("0123456789"))
	err = mp.AddTx(txBytes)
	if err == nil || !strings.Contains(err.Error(), "DA fee below Stage C floor") {
		t.Fatalf("expected Stage C DA floor rejection from default MinDaFeeRate, got %v", err)
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitRejected {
		t.Fatalf("low-fee DA err=%v, want TxAdmitRejected", err)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
}

func TestMempoolConfigZeroMinDaFeeRateMeansDefault(t *testing.T) {
	st, _ := testSpendableChainState(nil, nil)

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MinDaFeeRate:             0,
		PolicyDaSurchargePerByte: 2,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if got := mp.policy.MinDaFeeRate; got != DefaultMinDaFeeRate {
		t.Fatalf("MinDaFeeRate=%d, want DefaultMinDaFeeRate=%d", got, DefaultMinDaFeeRate)
	}
	if got, want := mp.policy.PolicyMaxDaBytesPerBlock, DefaultMinerConfig().PolicyMaxDaBytesPerBlock; got != want {
		t.Fatalf("PolicyMaxDaBytesPerBlock=%d, want default %d", got, want)
	}
	if got := mp.policy.PolicyDaSurchargePerByte; got != 2 {
		t.Fatalf("PolicyDaSurchargePerByte=%d, want 2", got)
	}
}

// TestMempoolSetCurrentMinFeeRateForTestRoundTrips pins the test-only
// rolling-floor setter contract: values at or above
// DefaultMempoolMinFeeRate are observed exactly by
// CurrentMinFeeRateSnapshot, while below-default values still pass
// through the production baseline clamp. The same setter is consumed by
// the cmd/rubin-node sentinel-floor wiring tests
// (TestRunMineBlocksPassesMineAddressToMiner /
// TestRunDevnetWithRPCBindLiveMinerHasCurrentMempoolMinFeeRateFn) but
// those tests live in package main and do not contribute to per-package
// node coverage; this same-package test pins the helper for the diff
// coverage gate.
//
// Proof assertion: a fresh mempool with a sentinel injected via
// SetCurrentMinFeeRateForTest returns that exact sentinel from
// CurrentMinFeeRateSnapshot, a second sentinel overrides the first,
// below-default values are clamped to DefaultMempoolMinFeeRate, and a nil
// receiver is a no-op.
func TestMempoolSetCurrentMinFeeRateForTestRoundTrips(t *testing.T) {
	st := NewChainState()
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	const sentinelA uint64 = 0x1234_5678_9ABC_DEF0
	const sentinelB uint64 = 0xFEDC_BA98_7654_3210
	mp.SetCurrentMinFeeRateForTest(sentinelA)
	if got := mp.CurrentMinFeeRateSnapshot(); got != sentinelA {
		t.Fatalf("after first set: got=%#x want=%#x", got, sentinelA)
	}
	mp.SetCurrentMinFeeRateForTest(sentinelB)
	if got := mp.CurrentMinFeeRateSnapshot(); got != sentinelB {
		t.Fatalf("after second set: got=%#x want=%#x", got, sentinelB)
	}
	mp.SetCurrentMinFeeRateForTest(0)
	if got := mp.CurrentMinFeeRateSnapshot(); got != DefaultMempoolMinFeeRate {
		t.Fatalf("after below-default set: got=%d want DefaultMempoolMinFeeRate=%d", got, DefaultMempoolMinFeeRate)
	}

	var nilMempool *Mempool
	nilMempool.SetCurrentMinFeeRateForTest(sentinelA) // must not panic
	if got := nilMempool.CurrentMinFeeRateSnapshot(); got != DefaultMempoolMinFeeRate {
		t.Fatalf("nil receiver after no-op set: got=%d want=%d", got, DefaultMempoolMinFeeRate)
	}
}

// TestMempoolNilReceiverCurrentMinFeeRateSnapshotReturnsBaseline pins the
// nil-receiver guard on `(*Mempool).CurrentMinFeeRateSnapshot()`. Other
// exported accessors (BytesUsed, AdmissionCounts, Contains) are
// nil-safe; this method MUST be too because it is also exported and
// used as a production callback (MinerConfig.CurrentMempoolMinFeeRateFn
// = mempool.CurrentMinFeeRateSnapshot). A nil receiver path would
// otherwise panic at m.mu.RLock() during unusual fail-closed wiring or
// test-time fixtures that hold a nil mempool reference.
//
// Proof assertion: a typed-nil call returns DefaultMempoolMinFeeRate
// without panicking.
func TestMempoolNilReceiverCurrentMinFeeRateSnapshotReturnsBaseline(t *testing.T) {
	var nilMempool *Mempool
	if got := nilMempool.CurrentMinFeeRateSnapshot(); got != DefaultMempoolMinFeeRate {
		t.Fatalf("nil receiver returned %d, want DefaultMempoolMinFeeRate=%d", got, DefaultMempoolMinFeeRate)
	}
}

// TestMempoolDAKindGuardPrecedesStandardRollingFloor pins the tx_kind guard
// against the standard rolling relay floor: a DA candidate that pays at or
// above the DA-side floor but far below an inflated rolling floor is refused by
// the kind guard, which sits before the locked floor check, and the floor state
// it never consulted is left exactly as arranged.
//
// It also keeps the wave-6 proof for PR #1368 alive on the surface that still
// classifies a DA relay floor: the admit caller passes
// currentMempoolMinFeeRate=0 to RejectDaAnchorTxPolicy so the Stage C helper
// enforces only the DA-side terms, and RelayMetadata therefore still reports
// the same input as the transient TxAdmitUnavailable non-DA transactions get,
// never as a Stage C "DA fee below Stage C floor ... relay_fee_floor=..."
// rejection.
func TestMempoolDAKindGuardPrecedesStandardRollingFloor(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000_000})

	// Config has DA-side floor (so the Stage C helper runs) but no
	// surcharge. Inflate the rolling local floor to a level the test tx
	// cannot match so validateFeeFloorLocked fires; the helper itself
	// passes (DA floor = daBytes * 1 is tiny).
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MinDaFeeRate: 1,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	mp.mu.Lock()
	mp.currentMinFeeRate = 1_000_000 // inflated rolling relay floor
	mp.mu.Unlock()

	// Build a DA tx that pays well above the DA-side floor (daBytes*1) but far
	// below weight*1_000_000, so the rolling floor is the only later check that
	// could have decided this candidate.
	txBytes := mustBuildSignedDaCommitTx(t, st.Utxos, outpoints[0], 50_000, 999_950_000, 1, fromKey, toAddress, []byte("0123456789"))
	requireDAKindReject(t, mp.AddTx(txBytes))
	mp.mu.RLock()
	floor, used, seq := mp.currentMinFeeRate, mp.usedBytes, mp.lastAdmissionSeq
	mp.mu.RUnlock()
	if floor != 1_000_000 || used != 0 || seq != 0 || mp.Len() != 0 {
		t.Fatalf("floor=%d used=%d seq=%d len=%d, want the arranged floor and an untouched pool", floor, used, seq, mp.Len())
	}

	// RelayMetadata still owns the relay-floor classification for the same
	// bytes, and it is still the transient TxAdmitUnavailable, not Stage C.
	_, err = mp.RelayMetadata(txBytes)
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitUnavailable || !strings.Contains(txErr.Message, "mempool fee below rolling minimum") {
		t.Fatalf("RelayMetadata err=%v, want TxAdmitUnavailable carrying the rolling-floor wording", err)
	}
}

func TestMempoolCheapFeeFloorPrecheckRejectsBeforeSignatureValidationForAllSources(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())

	cases := []struct {
		name  string
		admit func(*Mempool, []byte) error
	}{
		{name: "local", admit: func(mp *Mempool, tx []byte) error { return mp.AddTx(tx) }},
		{name: "remote", admit: func(mp *Mempool, tx []byte) error { return mp.AddRemoteTx(tx) }},
		{name: "reorg", admit: func(mp *Mempool, tx []byte) error { return mp.AddReorgTx(tx) }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
			mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 10, MaxBytes: 1 << 20})
			if err != nil {
				t.Fatalf("new mempool: %v", err)
			}
			mp.currentMinFeeRate = 8
			txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 1, 1, fromKey, fromAddress, toAddress)
			txBytes = corruptFirstWitnessSignature(t, txBytes)

			err = tc.admit(mp, txBytes)
			var txErr *TxAdmitError
			if !errors.As(err, &txErr) || txErr.Kind != TxAdmitUnavailable {
				t.Fatalf("below-floor invalid-signature admit err=%T %v, want TxAdmitUnavailable", err, err)
			}
			if !strings.Contains(txErr.Message, "mempool fee below rolling minimum") {
				t.Fatalf("reason %q does not match rolling floor precheck", txErr.Message)
			}
			if strings.Contains(txErr.Message, string(consensus.TX_ERR_SIG_INVALID)) {
				t.Fatalf("precheck reached signature validation: %q", txErr.Message)
			}
			if got := mp.Len(); got != 0 {
				t.Fatalf("mempool len after below-floor reject=%d, want 0", got)
			}
			if mp.lastAdmissionSeq != 0 {
				t.Fatalf("lastAdmissionSeq after below-floor reject=%d, want 0", mp.lastAdmissionSeq)
			}
		})
	}

	t.Run("default_policy_plain_p2pk", func(t *testing.T) {
		st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
		mp, err := NewMempool(st, nil, devnetGenesisChainID)
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		mp.currentMinFeeRate = 8
		txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 1, 1, fromKey, fromAddress, toAddress)
		txBytes = corruptFirstWitnessSignature(t, txBytes)

		err = mp.AddTx(txBytes)
		var txErr *TxAdmitError
		if !errors.As(err, &txErr) || txErr.Kind != TxAdmitUnavailable {
			t.Fatalf("default-policy below-floor err=%T %v, want TxAdmitUnavailable", err, err)
		}
		if !strings.Contains(txErr.Message, "mempool fee below rolling minimum") {
			t.Fatalf("default-policy reason %q does not match rolling floor precheck", txErr.Message)
		}
	})
}

func TestMempoolCheapFeeFloorPrecheckPreservesMissingUTXOReject(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 1, 1, fromKey, fromAddress, toAddress)
	delete(st.Utxos, outpoints[0])

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 10, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	mp.currentMinFeeRate = 8

	err = mp.AddTx(txBytes)
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitRejected {
		t.Fatalf("missing-utxo admit err=%T %v, want TxAdmitRejected", err, err)
	}
	if !strings.Contains(txErr.Message, string(consensus.TX_ERR_MISSING_UTXO)) {
		t.Fatalf("missing-utxo error=%q, want %s", txErr.Message, consensus.TX_ERR_MISSING_UTXO)
	}
	if strings.Contains(txErr.Message, "mempool fee below rolling minimum") {
		t.Fatalf("missing-utxo path was stolen by fee precheck: %q", txErr.Message)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len after missing-utxo reject=%d, want 0", got)
	}
	if mp.lastAdmissionSeq != 0 {
		t.Fatalf("lastAdmissionSeq after missing-utxo reject=%d, want 0", mp.lastAdmissionSeq)
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenTxNonceIsZero pins the
// wave-4 class-closure guard: tx_nonce == 0 for non-coinbase is
// permanently rejected by the slow path
// (clients/go/consensus/connect_block_parallel.go +
// clients/go/consensus/utxo_basic.go (`applyNonCoinbaseTxBasic*`) with TX_ERR_TX_NONCE_INVALID).
// Without the wave-4 early-defer guard, a below-floor tx with
// TxNonce==0 would be fast-rejected as transient Unavailable("mempool
// fee below rolling minimum"), masking the permanent reject class.
func TestMempoolCheapFeeFloorPrecheckDefersWhenTxNonceIsZero(t *testing.T) {
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 0, // wave-4 defer trigger
	}
	snapshot := &chainStateAdmissionSnapshot{utxos: map[consensus.Outpoint]consensus.UtxoEntry{}}
	if err := cheapFeeFloorPrecheck(tx, snapshot, 1, 1, nil, nil); err != nil {
		t.Fatalf("tx_nonce==0 must defer (nil) so the slow path returns the permanent TxErrTxNonceInvalid; got %v", err)
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenP2PKOutputValueIsZero pins
// the wave-4 class-closure guard: a P2PK output with Value==0 is
// permanently rejected by ValidateTxCovenantsGenesis at
// clients/go/consensus/covenant_genesis.go (`ValidateTxCovenantsGenesis`) with "CORE_P2PK value
// must be > 0". Without the wave-4 defer guard, a below-floor tx with
// a zero-value P2PK output would be fast-rejected as transient
// Unavailable, masking the permanent reject class.
func TestMempoolCheapFeeFloorPrecheckDefersWhenP2PKOutputValueIsZero(t *testing.T) {
	covData := make([]byte, consensus.MAX_P2PK_COVENANT_DATA)
	covData[0] = consensus.SUITE_ID_ML_DSA_87
	outputs := []consensus.TxOutput{{
		Value:        0, // wave-4 defer trigger
		CovenantType: consensus.COV_TYPE_P2PK,
		CovenantData: covData,
	}}
	if _, ok := feePrecheckP2PKOutputValue(outputs, 1, nil); ok {
		t.Fatalf("P2PK output Value==0 must return ok=false so precheck defers")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenP2PKCovenantDataLengthInvalid
// pins the wave-4 class-closure guard: a P2PK output whose
// CovenantData length is not exactly MAX_P2PK_COVENANT_DATA (33 =
// 1-byte suite_id + 32-byte payload) is permanently rejected by
// ValidateTxCovenantsGenesis at
// clients/go/consensus/covenant_genesis.go (`ValidateTxCovenantsGenesis`) with "invalid CORE_P2PK
// covenant_data length". Both len < 33 (empty) and len > 33
// (oversized) branches pinned.
func TestMempoolCheapFeeFloorPrecheckDefersWhenP2PKCovenantDataLengthInvalid(t *testing.T) {
	emptyOutputs := []consensus.TxOutput{{
		Value:        100,
		CovenantType: consensus.COV_TYPE_P2PK,
		CovenantData: nil,
	}}
	if _, ok := feePrecheckP2PKOutputValue(emptyOutputs, 1, nil); ok {
		t.Fatalf("empty CovenantData (len=0) must return ok=false so precheck defers")
	}
	oversizedOutputs := []consensus.TxOutput{{
		Value:        100,
		CovenantType: consensus.COV_TYPE_P2PK,
		CovenantData: make([]byte, 64),
	}}
	if _, ok := feePrecheckP2PKOutputValue(oversizedOutputs, 1, nil); ok {
		t.Fatalf("oversized CovenantData (len=64) must return ok=false so precheck defers")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenP2PKSuiteNotInNativeCreateSet
// pins the wave-4 class-closure guard: a P2PK output whose
// CovenantData[0] (suite_id) is not in the active
// NativeCreateSuites(nextHeight) set is permanently rejected by
// ValidateTxCovenantsGenesis at
// clients/go/consensus/covenant_genesis.go (`ValidateTxCovenantsGenesis`) with "CORE_P2PK suite
// not in native create set". DefaultRotationProvider accepts only
// SUITE_ID_ML_DSA_87 (0x01); any other byte triggers the defer.
func TestMempoolCheapFeeFloorPrecheckDefersWhenP2PKSuiteNotInNativeCreateSet(t *testing.T) {
	if 0xFE == consensus.SUITE_ID_ML_DSA_87 {
		t.Fatalf("test fixture sanity: 0xFE must differ from SUITE_ID_ML_DSA_87")
	}
	covData := make([]byte, consensus.MAX_P2PK_COVENANT_DATA)
	covData[0] = 0xFE // wave-4 defer trigger: non-native suite
	outputs := []consensus.TxOutput{{
		Value:        100,
		CovenantType: consensus.COV_TYPE_P2PK,
		CovenantData: covData,
	}}
	if _, ok := feePrecheckP2PKOutputValue(outputs, 1, nil); ok {
		t.Fatalf("non-native suite_id must return ok=false so precheck defers")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenInputScriptSigNonEmpty pins
// the wave-4 input-side class-closure guard: a non-empty ScriptSig on
// a P2PK input is rejected by the slow path
// (clients/go/consensus/utxo_basic.go (`applyNonCoinbaseTxBasic*`) with TX_ERR_PARSE
// "script_sig must be empty under genesis covenant set"). Without the
// wave-4 defer guard, a below-floor tx with non-empty ScriptSig would
// be misclassified as transient Unavailable instead of Rejected
// (terminal).
func TestMempoolCheapFeeFloorPrecheckDefersWhenInputScriptSigNonEmpty(t *testing.T) {
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs: []consensus.TxInput{{
			PrevTxid:  [32]byte{0x11},
			PrevVout:  0,
			ScriptSig: []byte{0x01}, // wave-4 defer trigger
			Sequence:  0,
		}},
		Witness: []consensus.WitnessItem{{}},
	}
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {Value: 100, CovenantType: consensus.COV_TYPE_P2PK},
	}
	if _, ok := feePrecheckP2PKInputValue(tx, utxos /* nextHeight */, 1, nil, nil); ok {
		t.Fatalf("non-empty ScriptSig must return ok=false so precheck defers")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenInputSequenceOutOfRange
// pins the wave-4 input-side class-closure guard: a Sequence >
// 0x7fffffff on a P2PK input is rejected by the slow path at
// clients/go/consensus/utxo_basic.go (`applyNonCoinbaseTxBasic*`) with
// TX_ERR_SEQUENCE_INVALID "sequence exceeds 0x7fffffff".
func TestMempoolCheapFeeFloorPrecheckDefersWhenInputSequenceOutOfRange(t *testing.T) {
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs: []consensus.TxInput{{
			PrevTxid: [32]byte{0x11},
			PrevVout: 0,
			Sequence: 0x80000000, // wave-4 defer trigger
		}},
		Witness: []consensus.WitnessItem{{}},
	}
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {Value: 100, CovenantType: consensus.COV_TYPE_P2PK},
	}
	if _, ok := feePrecheckP2PKInputValue(tx, utxos /* nextHeight */, 1, nil, nil); ok {
		t.Fatalf("Sequence > 0x7fffffff must return ok=false so precheck defers")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenWitnessCountNotExactlyOne
// pins the wave-4 input-side class-closure guard: a tx with
// len(Witness) != 1 on a single-P2PK-input tx is rejected by the slow
// path (`applyNonCoinbaseTxBasic*` witness-slots check, mirror of
// Rust utxo_basic.rs `apply_non_coinbase_tx_basic_update_*`. Both len == 0 and len == 2 branches
// pinned.
func TestMempoolCheapFeeFloorPrecheckDefersWhenWitnessCountNotExactlyOne(t *testing.T) {
	makeTx := func(witnessCount int) *consensus.Tx {
		w := make([]consensus.WitnessItem, witnessCount)
		return &consensus.Tx{
			TxKind:  0x00,
			TxNonce: 1,
			Inputs: []consensus.TxInput{{
				PrevTxid: [32]byte{0x11},
				PrevVout: 0,
				Sequence: 0,
			}},
			Witness: w,
		}
	}
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {Value: 100, CovenantType: consensus.COV_TYPE_P2PK},
	}
	// Branch 1: zero witness slots.
	if _, ok := feePrecheckP2PKInputValue(makeTx(0), utxos /* nextHeight */, 1, nil, nil); ok {
		t.Fatalf("len(Witness) == 0 must return ok=false so precheck defers")
	}
	// Branch 2: two witness slots.
	if _, ok := feePrecheckP2PKInputValue(makeTx(2), utxos /* nextHeight */, 1, nil, nil); ok {
		t.Fatalf("len(Witness) == 2 must return ok=false so precheck defers")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenInputUsesCoinbasePrevoutMarker
// pins the wave-4 input-side class-closure guard: a non-coinbase tx
// whose input uses the coinbase-prevout marker (PrevTxid == zero AND
// PrevVout == 0xffffffff) is rejected by the slow path at
// clients/go/consensus/utxo_basic.go (`applyNonCoinbaseTxBasic*`) with TX_ERR_PARSE
// "coinbase prevout encoding forbidden in non-coinbase".
func TestMempoolCheapFeeFloorPrecheckDefersWhenInputUsesCoinbasePrevoutMarker(t *testing.T) {
	var zeroTxid [32]byte
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs: []consensus.TxInput{{
			PrevTxid: zeroTxid,   // wave-4 defer trigger
			PrevVout: 0xffffffff, // wave-4 defer trigger
			Sequence: 0,
		}},
		Witness: []consensus.WitnessItem{{}},
	}
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: zeroTxid, Vout: 0xffffffff}: {Value: 100, CovenantType: consensus.COV_TYPE_P2PK},
	}
	if _, ok := feePrecheckP2PKInputValue(tx, utxos /* nextHeight */, 1, nil, nil); ok {
		t.Fatalf("coinbase-prevout marker on non-coinbase input must return ok=false so precheck defers")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenP2PKInputIsImmatureCoinbase
// pins the wave-5 input-side class-closure guard: an immature
// coinbase P2PK spend (CreatedByCoinbase && nextHeight -
// CreationHeight < COINBASE_MATURITY) is rejected by the slow path at
// clients/go/consensus/utxo_basic.go (`applyNonCoinbaseTxBasic*`) with
// TX_ERR_COINBASE_IMMATURE "coinbase immature". Without the wave-5
// defer guard, a below-floor immature-coinbase spend would be
// misclassified as transient Unavailable("mempool fee below rolling
// minimum"), signalling caller to retry-with-higher-fee when the
// actual remedy is to wait for COINBASE_MATURITY blocks. Different
// caller action means real class-leak (P1).
func TestMempoolCheapFeeFloorPrecheckDefersWhenP2PKInputIsImmatureCoinbase(t *testing.T) {
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs: []consensus.TxInput{{
			PrevTxid: [32]byte{0x11},
			PrevVout: 0,
			Sequence: 0,
		}},
		Witness: []consensus.WitnessItem{{}},
	}
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:             100,
			CovenantType:      consensus.COV_TYPE_P2PK,
			CreatedByCoinbase: true,
			CreationHeight:    0,
		},
	}
	// nextHeight < COINBASE_MATURITY threshold => immature.
	immatureHeight := uint64(consensus.COINBASE_MATURITY) - 1
	if _, ok := feePrecheckP2PKInputValue(tx, utxos, immatureHeight, nil, nil); ok {
		t.Fatalf("immature coinbase spend must return ok=false so precheck defers")
	}
	// At maturity threshold the defer no longer fires (sanity-pin
	// requires a registered ML-DSA-87 witness item — wave-14 added
	// witness-item structural validation, wave-15 added SHA3
	// key-binding + SIGHASH_ALL trailer; synthesize the minimum-valid
	// witness shape so this branch reaches the (value, true) return).
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	covData := append([]byte{consensus.SUITE_ID_ML_DSA_87}, pubkeyHash[:]...)
	utxos[consensus.Outpoint{Txid: [32]byte{0x11}, Vout: 0}] = consensus.UtxoEntry{
		Value:             100,
		CovenantType:      consensus.COV_TYPE_P2PK,
		CovenantData:      covData,
		CreatedByCoinbase: true,
		CreationHeight:    0,
	}
	signature := make([]byte, consensus.ML_DSA_87_SIG_BYTES+1)
	signature[len(signature)-1] = consensus.SIGHASH_ALL
	tx.Witness = []consensus.WitnessItem{{
		SuiteID:   consensus.SUITE_ID_ML_DSA_87,
		Pubkey:    pubkey,
		Signature: signature,
	}}
	matureHeight := uint64(consensus.COINBASE_MATURITY)
	if _, ok := feePrecheckP2PKInputValue(tx, utxos, matureHeight, nil, nil); !ok {
		t.Fatalf("mature coinbase spend must NOT defer (precheck returns ok=true)")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenInputUTXOCovenantDataLengthInvalid
// pins the wave-15 panic-safety guard: defer when entry.CovenantData
// length is not exactly MAX_P2PK_COVENANT_DATA (33). Without this
// guard a corrupted on-disk UTXO entry (chainStateFromDisk accepts
// arbitrary CovenantData bytes) would panic the admission loop on
// the [0] / [1:33] indexing. Mirrors slow-path covenant_data length
// check at clients/go/consensus/spend_verify.go counterpart.
func TestMempoolCheapFeeFloorPrecheckDefersWhenInputUTXOCovenantDataLengthInvalid(t *testing.T) {
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	signature := make([]byte, consensus.ML_DSA_87_SIG_BYTES+1)
	signature[len(signature)-1] = consensus.SIGHASH_ALL
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs:  []consensus.TxInput{{PrevTxid: [32]byte{0x11}, PrevVout: 0, Sequence: 0}},
		Witness: []consensus.WitnessItem{{
			SuiteID:   consensus.SUITE_ID_ML_DSA_87,
			Pubkey:    pubkey,
			Signature: signature,
		}},
	}
	// Branch 1: empty CovenantData (len 0 < 33) — panic-safety case.
	emptyUtxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:        100,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: nil,
		},
	}
	if _, ok := feePrecheckP2PKInputValue(tx, emptyUtxos, 1, nil, nil); ok {
		t.Fatalf("empty CovenantData must defer (ok=false) — guards [0] index against panic")
	}
	// Branch 2: oversized CovenantData (len 64 > 33).
	oversizedUtxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:        100,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: make([]byte, 64),
		},
	}
	if _, ok := feePrecheckP2PKInputValue(tx, oversizedUtxos, 1, nil, nil); ok {
		t.Fatalf("oversized CovenantData must defer (ok=false)")
	}
	// Sanity: correct 33-byte CovenantData with valid SHA3 binding accepts.
	covData := append([]byte{consensus.SUITE_ID_ML_DSA_87}, pubkeyHash[:]...)
	goodUtxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:        100,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: covData,
		},
	}
	if _, ok := feePrecheckP2PKInputValue(tx, goodUtxos, 1, nil, nil); !ok {
		t.Fatalf("valid CovenantData (len=33, valid SHA3) must NOT defer")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersOnlyOnInvalidSighashTrailer
// pins the wave-16 sighash trailer guard: defer when last sig byte
// is NOT one of the six valid sighash types accepted by
// IsValidSighashType (clients/go/consensus/sighash.go (`IsValidSighashType`)). Verifies
// (a) invalid trailer 0x05 defers and (b) valid non-ALL trailer
// SIGHASH_NONE (0x02) does NOT defer (closes wave-15 over-defer P1).
func TestMempoolCheapFeeFloorPrecheckDefersOnlyOnInvalidSighashTrailer(t *testing.T) {
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	covData := append([]byte{consensus.SUITE_ID_ML_DSA_87}, pubkeyHash[:]...)
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:        100,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: covData,
		},
	}
	makeTx := func(trailer byte) *consensus.Tx {
		signature := make([]byte, consensus.ML_DSA_87_SIG_BYTES+1)
		signature[len(signature)-1] = trailer
		return &consensus.Tx{
			TxKind:  0x00,
			TxNonce: 1,
			Inputs:  []consensus.TxInput{{PrevTxid: [32]byte{0x11}, PrevVout: 0, Sequence: 0}},
			Witness: []consensus.WitnessItem{{
				SuiteID:   consensus.SUITE_ID_ML_DSA_87,
				Pubkey:    pubkey,
				Signature: signature,
			}},
		}
	}
	// Branch 1: invalid trailer 0x05 (not in 6-element accept set) defers.
	if _, ok := feePrecheckP2PKInputValue(makeTx(0x05), utxos, 1, nil, nil); ok {
		t.Fatalf("invalid sighash trailer 0x05 must defer (ok=false)")
	}
	// Branch 2: valid SIGHASH_NONE (0x02) trailer does NOT defer (wave-15 over-defer fix).
	if _, ok := feePrecheckP2PKInputValue(makeTx(consensus.SIGHASH_NONE), utxos, 1, nil, nil); !ok {
		t.Fatalf("valid SIGHASH_NONE trailer must NOT defer")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenPubkeyKeyBindingMismatch
// pins the wave-15 SHA3 key-binding guard: defer when SHA3(pubkey)
// does not match entry.CovenantData[1:33]. Mirrors slow-path
// validate_p2pk_spend_q key-binding check.
func TestMempoolCheapFeeFloorPrecheckDefersWhenPubkeyKeyBindingMismatch(t *testing.T) {
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	covData := append([]byte{consensus.SUITE_ID_ML_DSA_87}, pubkeyHash[:]...)
	signature := make([]byte, consensus.ML_DSA_87_SIG_BYTES+1)
	signature[len(signature)-1] = consensus.SIGHASH_ALL
	// Mutate first byte of pubkey so SHA3(pubkey) diverges from
	// covenant_data[1:33] (which still binds the original all-zero pubkey).
	mutatedPubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	mutatedPubkey[0] = 0xFF
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs:  []consensus.TxInput{{PrevTxid: [32]byte{0x11}, PrevVout: 0, Sequence: 0}},
		Witness: []consensus.WitnessItem{{
			SuiteID:   consensus.SUITE_ID_ML_DSA_87,
			Pubkey:    mutatedPubkey,
			Signature: signature,
		}},
	}
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:        100,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: covData,
		},
	}
	if _, ok := feePrecheckP2PKInputValue(tx, utxos, 1, nil, nil); ok {
		t.Fatalf("pubkey key-binding mismatch must defer (ok=false)")
	}
}

// emptySpendRotationProvider returns DefaultRotationProvider's create
// set (so output-side checks still see a populated set), but reports an
// EMPTY native-spend set, which forces the wave-14 input-side rotation
// guard to defer.
type emptySpendRotationProvider struct{}

func (emptySpendRotationProvider) NativeCreateSuites(uint64) *consensus.NativeSuiteSet {
	return consensus.NewNativeSuiteSet(consensus.SUITE_ID_ML_DSA_87)
}

func (emptySpendRotationProvider) NativeSpendSuites(uint64) *consensus.NativeSuiteSet {
	// Empty set rejects ALL spend suites.
	return consensus.NewNativeSuiteSet()
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenInputSuiteNotInNativeSpendSet
// pins the wave-14 input-side class-closure rotation guard: a P2PK
// input whose witness suite_id is not in the active
// NativeSpendSuites(nextHeight) set is rejected by the slow path at
// clients/go/consensus/spend_verify.go (SigAlgInvalid). Without this
// guard a below-floor tx with an out-of-rotation suite would be
// misclassified as transient Unavailable instead of terminal Rejected
// (different caller action: rotation does not retry vs floor does).
// Closes Copilot wave-17 P1 (TestMempoolCheap* regression coverage gap).
func TestMempoolCheapFeeFloorPrecheckDefersWhenInputSuiteNotInNativeSpendSet(t *testing.T) {
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	covData := append([]byte{consensus.SUITE_ID_ML_DSA_87}, pubkeyHash[:]...)
	signature := make([]byte, consensus.ML_DSA_87_SIG_BYTES+1)
	signature[len(signature)-1] = consensus.SIGHASH_ALL
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs:  []consensus.TxInput{{PrevTxid: [32]byte{0x11}, PrevVout: 0, Sequence: 0}},
		Witness: []consensus.WitnessItem{{
			SuiteID:   consensus.SUITE_ID_ML_DSA_87,
			Pubkey:    pubkey,
			Signature: signature,
		}},
	}
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:        100,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: covData,
		},
	}
	if _, ok := feePrecheckP2PKInputValue(tx, utxos, 1, emptySpendRotationProvider{}, nil); ok {
		t.Fatalf("rotation with empty NativeSpendSuites must defer (ok=false)")
	}
	// Sanity (negative-branch pin): default rotation accepts
	// SUITE_ID_ML_DSA_87 in the spend set, so the same fixture with
	// rotation=nil must NOT defer. Confirms the rotation provider is
	// the only difference exercised by this test.
	if _, ok := feePrecheckP2PKInputValue(tx, utxos, 1, nil, nil); !ok {
		t.Fatalf("default rotation must NOT defer on valid P2PK fixture (ok=true)")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenInputSuiteRegistryLookupMisses
// pins the wave-14 input-side class-closure registry guard: a P2PK
// input whose witness suite_id has no entry in the active SuiteRegistry
// (Lookup returns ok=false) is rejected by the slow path at
// clients/go/consensus/spend_verify.go (SigAlgInvalid). Without this
// guard a below-floor tx with an unregistered suite would be
// misclassified as transient Unavailable instead of terminal Rejected.
// Closes Copilot wave-17 P1 (TestMempoolCheap* regression coverage gap).
func TestMempoolCheapFeeFloorPrecheckDefersWhenInputSuiteRegistryLookupMisses(t *testing.T) {
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	covData := append([]byte{consensus.SUITE_ID_ML_DSA_87}, pubkeyHash[:]...)
	signature := make([]byte, consensus.ML_DSA_87_SIG_BYTES+1)
	signature[len(signature)-1] = consensus.SIGHASH_ALL
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs:  []consensus.TxInput{{PrevTxid: [32]byte{0x11}, PrevVout: 0, Sequence: 0}},
		Witness: []consensus.WitnessItem{{
			SuiteID:   consensus.SUITE_ID_ML_DSA_87,
			Pubkey:    pubkey,
			Signature: signature,
		}},
	}
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:        100,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: covData,
		},
	}
	// Empty registry: Lookup of any suite_id (including the witness
	// SUITE_ID_ML_DSA_87) returns ok=false, so the wave-14 registry
	// guard fires.
	emptyRegistry := consensus.NewSuiteRegistryFromParams(nil)
	if _, ok := feePrecheckP2PKInputValue(tx, utxos, 1, nil, emptyRegistry); ok {
		t.Fatalf("empty registry must defer (Lookup miss returns ok=false)")
	}
	// Sanity (negative-branch pin): default registry has
	// SUITE_ID_ML_DSA_87 with canonical params, so the same fixture
	// with registry=nil must NOT defer. Confirms the registry is the
	// only difference exercised by this test.
	if _, ok := feePrecheckP2PKInputValue(tx, utxos, 1, nil, nil); !ok {
		t.Fatalf("default registry must NOT defer on valid P2PK fixture (ok=true)")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenWitnessPubkeyOrSignatureLengthNoncanonical
// pins the wave-14 input-side witness-length guard: a P2PK input
// whose witness Pubkey or Signature length does not match the
// SuiteParams (PubkeyLen, SigLen+1) is rejected by the slow path at
// clients/go/consensus/spend_verify.go (TX_ERR_SIG_NONCANONICAL).
// Without this guard a below-floor tx with a malformed witness would
// be misclassified as transient Unavailable instead of terminal
// Rejected. Closes Copilot wave-19 P1 (TestMempoolCheap* coverage gap).
func TestMempoolCheapFeeFloorPrecheckDefersWhenWitnessPubkeyOrSignatureLengthNoncanonical(t *testing.T) {
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	covData := append([]byte{consensus.SUITE_ID_ML_DSA_87}, pubkeyHash[:]...)
	signature := make([]byte, consensus.ML_DSA_87_SIG_BYTES+1)
	signature[len(signature)-1] = consensus.SIGHASH_ALL
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:        100,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: covData,
		},
	}
	makeTx := func(pk, sig []byte) *consensus.Tx {
		return &consensus.Tx{
			TxKind:  0x00,
			TxNonce: 1,
			Inputs:  []consensus.TxInput{{PrevTxid: [32]byte{0x11}, PrevVout: 0, Sequence: 0}},
			Witness: []consensus.WitnessItem{{
				SuiteID:   consensus.SUITE_ID_ML_DSA_87,
				Pubkey:    pk,
				Signature: sig,
			}},
		}
	}
	// Branch 1: pubkey too short (truncated by 1 byte).
	if _, ok := feePrecheckP2PKInputValue(makeTx(pubkey[:len(pubkey)-1], signature), utxos, 1, nil, nil); ok {
		t.Fatalf("truncated pubkey must defer (ok=false)")
	}
	// Branch 2: signature too long (extended by 1 byte beyond SigLen+1).
	{
		oversize := append([]byte{}, signature...)
		oversize = append(oversize, 0)
		if _, ok := feePrecheckP2PKInputValue(makeTx(pubkey, oversize), utxos, 1, nil, nil); ok {
			t.Fatalf("oversized signature must defer (ok=false)")
		}
	}
	// Sanity (negative-branch pin): canonical lengths must NOT defer.
	if _, ok := feePrecheckP2PKInputValue(makeTx(pubkey, signature), utxos, 1, nil, nil); !ok {
		t.Fatalf("canonical pubkey/signature lengths must NOT defer (ok=true)")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersOnOutputSumOverflow pins the
// wave-4 output-side overflow guard: when the sum of P2PK output
// values overflows uint64 (bits.Add64 carry != 0), defer to the
// expensive admission path instead of returning a spurious accept on
// the wrapped value. Mirrors Rust
// rub166_precheck_defers_on_output_sum_overflow. Closes Copilot
// wave-19 P1 (TestMempoolCheap* coverage gap).
func TestMempoolCheapFeeFloorPrecheckDefersOnOutputSumOverflow(t *testing.T) {
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	covData := append([]byte{consensus.SUITE_ID_ML_DSA_87}, pubkeyHash[:]...)
	// Two outputs whose values sum to > uint64.MaxValue.
	outputs := []consensus.TxOutput{
		{Value: ^uint64(0) - 5, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: covData},
		{Value: 100, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: covData},
	}
	if _, ok := feePrecheckP2PKOutputValue(outputs, 1, nil); ok {
		t.Fatalf("output sum overflow must defer (ok=false)")
	}
	// Sanity (negative-branch pin): non-overflowing two-output sum
	// must NOT defer. Confirms the overflow branch is the only
	// difference exercised by this test.
	okOutputs := []consensus.TxOutput{
		{Value: 100, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: covData},
		{Value: 200, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: covData},
	}
	if total, ok := feePrecheckP2PKOutputValue(okOutputs, 1, nil); !ok || total != 300 {
		t.Fatalf("non-overflowing sum must NOT defer; got total=%d ok=%v", total, ok)
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenInputUTXOCovenantDataSuiteMismatch
// pins the wave-15 panic-safety + suite-consistency guard
// (mempool_precheck_input.go (`feePrecheckP2PKInputValue`)): defer when entry.CovenantData[0]
// != witness.SuiteID even when length is canonical 33 bytes. Mirrors
// slow-path TX_ERR_COVENANT_TYPE_INVALID at clients/go/consensus/spend_verify.go.
// Closes pre-push-reviewer wave-20 P1 finding #1 (Go counterpart of
// rub166_precheck_defers_when_input_utxo_covenant_data_suite_mismatch).
func TestMempoolCheapFeeFloorPrecheckDefersWhenInputUTXOCovenantDataSuiteMismatch(t *testing.T) {
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	signature := make([]byte, consensus.ML_DSA_87_SIG_BYTES+1)
	signature[len(signature)-1] = consensus.SIGHASH_ALL
	// covenant_data[0] = 0xFE (mismatched), bytes [1:33] = SHA3(pubkey).
	covData := append([]byte{0xFE}, pubkeyHash[:]...)
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs:  []consensus.TxInput{{PrevTxid: [32]byte{0x11}, PrevVout: 0, Sequence: 0}},
		Witness: []consensus.WitnessItem{{
			SuiteID:   consensus.SUITE_ID_ML_DSA_87,
			Pubkey:    pubkey,
			Signature: signature,
		}},
	}
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value:        100,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: covData,
		},
	}
	if _, ok := feePrecheckP2PKInputValue(tx, utxos, 1, nil, nil); ok {
		t.Fatalf("covenant_data[0] != witness.SuiteID must defer (ok=false)")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenInputCountNotExactlyOne
// pins the input-count == 1 guard (mempool_precheck_input.go (`feePrecheckP2PKInputValue`)).
// Mirrors Rust rub166_precheck_defers_when_input_count_not_exactly_one.
// Closes pre-push-reviewer wave-20 P1 finding #2 (Go parity gap).
func TestMempoolCheapFeeFloorPrecheckDefersWhenInputCountNotExactlyOne(t *testing.T) {
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {Value: 100, CovenantType: consensus.COV_TYPE_P2PK},
	}
	makeTx := func(n int) *consensus.Tx {
		ins := make([]consensus.TxInput, n)
		for i := range ins {
			ins[i] = consensus.TxInput{PrevTxid: [32]byte{0x11}, PrevVout: 0, Sequence: 0}
		}
		return &consensus.Tx{TxKind: 0x00, TxNonce: 1, Inputs: ins, Witness: []consensus.WitnessItem{{}}}
	}
	if _, ok := feePrecheckP2PKInputValue(makeTx(0), utxos, 1, nil, nil); ok {
		t.Fatalf("len(Inputs)==0 must defer (ok=false)")
	}
	if _, ok := feePrecheckP2PKInputValue(makeTx(2), utxos, 1, nil, nil); ok {
		t.Fatalf("len(Inputs)==2 must defer (ok=false)")
	}
}

// TestMempoolCheapFeeFloorPrecheckDefersWhenOutputExceedsInput pins the
// `outputValue > inputValue` defer in cheapFeeFloorPrecheck
// (mempool_precheck_floor.go (`cheapFeeFloorPrecheck`)). Mirrors Rust
// rub166_precheck_defers_when_output_exceeds_input. Closes wave-20 P1
// finding #2 (Go parity gap).
func TestMempoolCheapFeeFloorPrecheckDefersWhenOutputExceedsInput(t *testing.T) {
	pubkey := make([]byte, consensus.ML_DSA_87_PUBKEY_BYTES)
	pubkeyHash := sha3.Sum256(pubkey)
	covData := append([]byte{consensus.SUITE_ID_ML_DSA_87}, pubkeyHash[:]...)
	signature := make([]byte, consensus.ML_DSA_87_SIG_BYTES+1)
	signature[len(signature)-1] = consensus.SIGHASH_ALL
	utxos := map[consensus.Outpoint]consensus.UtxoEntry{
		{Txid: [32]byte{0x11}, Vout: 0}: {
			Value: 50, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: covData,
		},
	}
	// Output value 100 > input value 50 → cheap precheck defers.
	tx := &consensus.Tx{
		TxKind:  0x00,
		TxNonce: 1,
		Inputs:  []consensus.TxInput{{PrevTxid: [32]byte{0x11}, PrevVout: 0, Sequence: 0}},
		Outputs: []consensus.TxOutput{{Value: 100, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: covData}},
		Witness: []consensus.WitnessItem{{SuiteID: consensus.SUITE_ID_ML_DSA_87, Pubkey: pubkey, Signature: signature}},
	}
	// Sanity: helper-level pre-conditions must be reachable.
	inputValue, ok := feePrecheckP2PKInputValue(tx, utxos, 1, nil, nil)
	if !ok {
		t.Fatalf("input precheck should accept; got ok=false")
	}
	outputValue, ok := feePrecheckP2PKOutputValue(tx.Outputs, 1, nil)
	if !ok {
		t.Fatalf("output precheck should accept; got ok=false")
	}
	if outputValue <= inputValue {
		t.Fatalf("test fixture sanity: outputValue (%d) must exceed inputValue (%d)", outputValue, inputValue)
	}
	// Wave-21 fix: actually call cheapFeeFloorPrecheck to exercise the
	// `outputValue > inputValue` defer at mempool_precheck_floor.go.
	// nil error = "deferred to slow path" (the precheck contract).
	snap := &chainStateAdmissionSnapshot{utxos: utxos}
	if err := cheapFeeFloorPrecheck(tx, snap, 1, 1, nil, nil); err != nil {
		t.Fatalf("output > input must defer (no error returned); got %v", err)
	}
}

// TestMempoolCheapFeeFloorPrecheckWeightZeroIsUnreachableViaPublicSurface
// documents that the `weight == 0` defer at mempool_precheck_floor.go
// (`err != nil || weight == 0`) is structurally unreachable through any
// public Go API: `consensus.TxWeightAndStats` returns weight > 0 for
// every structurally-valid tx, and weight==0 inputs would already fail
// the upstream input-count and witness guards. Rust counterpart
// `rub166_precheck_defers_when_weight_is_zero` exercises the branch
// directly because Rust's `cheap_fee_floor_precheck` accepts weight as
// a function parameter (RUB-167 single-walk); Go computes weight
// internally so the branch is defense-in-depth only. Renamed from
// the wave-20 `DefersWhenWeightIsZero` (which fictionally claimed
// reachable coverage) per pre-push-reviewer wave-20 finding #2 fix
// option (b).
func TestMempoolCheapFeeFloorPrecheckWeightZeroIsUnreachableViaPublicSurface(t *testing.T) {
	// emptyTx triggers the upstream `len(tx.Inputs) != 1` defer at
	// mempool_precheck_input.go BEFORE the weight check — confirming the
	// upstream guard is the actual gate.
	emptyTx := &consensus.Tx{TxKind: 0x00, TxNonce: 1, Inputs: []consensus.TxInput{}, Witness: []consensus.WitnessItem{}}
	snap := &chainStateAdmissionSnapshot{utxos: map[consensus.Outpoint]consensus.UtxoEntry{}}
	if err := cheapFeeFloorPrecheck(emptyTx, snap, 0, 1, nil, nil); err != nil {
		t.Fatalf("emptyTx should defer to slow path (input-count guard fires before weight check); got %v", err)
	}
	// Defense-in-depth assertion: even if a future change removes the
	// input-count guard, a structurally-malformed tx whose
	// TxWeightAndStats returns err would also defer at the
	// `err != nil` clause. Without injecting a fault into TxWeightAndStats
	// from the test (which would require a stub or test-only signature),
	// this branch cannot be exercised end-to-end. Documenting the
	// limitation here is the correct closure per parity-checker wave-20
	// finding option (b) ("rename to acknowledge unreachability").
}

// TestMempoolCheapPrecheckCachedDefaultRegistryIdentityAndCanonicalManifest
// pins Wave-22 cache invariants (Copilot wave-21 P2 #1+#2 follow-up):
// (a) the package-level cache pointer is stable across reads (no
// rebuild), (b) cached value satisfies IsCanonicalDefaultLiveManifest,
// (c) ML-DSA-87 lookup returns canonical params. Future change that
// swaps the cache to a non-canonical builder fails this test.
func TestMempoolCheapPrecheckCachedDefaultRegistryIdentityAndCanonicalManifest(t *testing.T) {
	if !cachedDefaultPrecheckSuiteRegistry.IsCanonicalDefaultLiveManifest() {
		t.Fatalf("cachedDefaultPrecheckSuiteRegistry must satisfy IsCanonicalDefaultLiveManifest")
	}
	params, ok := cachedDefaultPrecheckSuiteRegistry.Lookup(consensus.SUITE_ID_ML_DSA_87)
	if !ok {
		t.Fatalf("ML-DSA-87 must be present in cached default registry")
	}
	if params.SuiteID != consensus.SUITE_ID_ML_DSA_87 {
		t.Fatalf("expected SuiteID=%d, got %d", consensus.SUITE_ID_ML_DSA_87, params.SuiteID)
	}
	if params.AlgName != "ML-DSA-87" {
		t.Fatalf("expected AlgName=ML-DSA-87, got %s", params.AlgName)
	}
}

// TestMempoolCheapPrecheckCachedDefaultNativeSetsContainCanonicalSuite
// pins the Wave-22 native_spend / native_create cache invariants.
// Same rationale as the registry test: ensure the cached
// NativeSuiteSet is the canonical singleton (contains ML-DSA-87).
func TestMempoolCheapPrecheckCachedDefaultNativeSetsContainCanonicalSuite(t *testing.T) {
	if !cachedDefaultPrecheckNativeSpendSet.Contains(consensus.SUITE_ID_ML_DSA_87) {
		t.Fatalf("cachedDefaultPrecheckNativeSpendSet must contain SUITE_ID_ML_DSA_87")
	}
	if !cachedDefaultPrecheckNativeCreateSet.Contains(consensus.SUITE_ID_ML_DSA_87) {
		t.Fatalf("cachedDefaultPrecheckNativeCreateSet must contain SUITE_ID_ML_DSA_87")
	}
}

func TestMempoolValidateFeeFloorLockedWithFloorUsesMaxOfSnapAndLive(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	st, _ := testSpendableChainState(fromAddress, []uint64{1_000_000})
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 10, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	entry := &mempoolEntry{
		txid:   [32]byte{0x77},
		fee:    consensus.Uint128FromU64(5),
		weight: 1, // fee_rate = 5
		size:   1,
	}
	mp.mu.Lock()
	defer mp.mu.Unlock()

	mp.currentMinFeeRate = 1
	if err := mp.validateFeeFloorLockedWithFloor(entry /* snappedFloor */, 10); err == nil ||
		!strings.Contains(err.Error(), "mempool fee below rolling minimum") {
		t.Fatalf("snap=10 + live=1 must reject (max-floor=10 > fee_rate=5); got err=%v", err)
	}

	// Case 2 — raise race direction (snap-LOW, live-HIGH):
	// max(snap=1, live=10) = 10 → reject. NEVER admits below current
	// floor — closes the wave-7 Codex+Copilot raise-race finding.
	mp.currentMinFeeRate = 10
	if err := mp.validateFeeFloorLockedWithFloor(entry /* snappedFloor */, 1); err == nil ||
		!strings.Contains(err.Error(), "mempool fee below rolling minimum") {
		t.Fatalf("snap=1 + live=10 must reject (max-floor=10 > fee_rate=5); got err=%v", err)
	}

	// Case 3 — both low (steady state, no decay/raise during admission):
	// max(snap=1, live=1) = 1 → accept fee_rate=5.
	mp.currentMinFeeRate = 1
	if err := mp.validateFeeFloorLockedWithFloor(entry /* snappedFloor */, 1); err != nil {
		t.Fatalf("snap=1 + live=1 must accept fee_rate=5 (max-floor=1 <= 5); got err=%v", err)
	}

	// Case 4 — snap=0 falls through to live (post-clamp Default=1).
	// max(snap=0, live=1) = 1 → accept fee_rate=5.
	mp.currentMinFeeRate = 1
	if err := mp.validateFeeFloorLockedWithFloor(entry /* snappedFloor */, 0); err != nil {
		t.Fatalf("snap=0 + live=1 must accept fee_rate=5 (max-floor post-clamp=1 <= 5); got err=%v", err)
	}
}

func TestMempoolPolicySnapshot_DoesNotMutateForDaPolicy(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	// Stage C admission requires the input value to cover both the
	// relay-fee floor (weight * current_mempool_min_fee_rate) and the DA
	// floor (da_payload_len * (min_da_fee_rate + surcharge_per_byte)),
	// so this snapshot-mutation regression test has to provide enough
	// value for the admission path to actually exercise the DA helper.
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		PolicyDaSurchargePerByte: 1,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	txBytes := mustBuildSignedDaCommitTx(t, st.Utxos, outpoints[0], 50_000, 950_000, 1, fromKey, toAddress, []byte("0123456789"))
	nextHeight, _, err := nextBlockContext(st)
	if err != nil {
		t.Fatalf("nextBlockContext: %v", err)
	}
	blockMTP, err := mp.nextBlockMTP(nextHeight)
	if err != nil {
		t.Fatalf("nextBlockMTP: %v", err)
	}
	checked, err := consensus.CheckTransactionWithOwnedUtxoSetAndSuiteContext(
		txBytes,
		copyUtxoSet(st.Utxos),
		nextHeight,
		blockMTP,
		devnetGenesisChainID,
		mp.policy.RotationProvider,
		mp.policy.SuiteRegistry,
	)
	if err != nil {
		t.Fatalf("CheckTransactionWithOwnedUtxoSetAndSuiteContext: %v", err)
	}
	policyUtxos, err := policyInputSnapshot(checked.Tx, st.Utxos)
	if err != nil {
		t.Fatalf("policyInputSnapshot: %v", err)
	}
	before, err := policyInputSnapshot(checked.Tx, policyUtxos)
	if err != nil {
		t.Fatalf("policyInputSnapshot(before): %v", err)
	}

	if err := mp.applyPolicyAgainstState(checked, nextHeight, policyUtxos, mp.policySnapshot()); err != nil {
		t.Fatalf("applyPolicyAgainstState: %v", err)
	}
	if !reflect.DeepEqual(policyUtxos, before) {
		t.Fatalf("policy path mutated DA snapshot")
	}
}

func TestMempoolPolicyRejectsCoreSimplicityPreActivationBeforeConsensus(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{100})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		PolicyRejectSimplicityPreActivation: true,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	createTx := txWithOneInputOneOutput(outpoints[0].Txid, outpoints[0].Vout, 1, consensus.COV_TYPE_CORE_SIMPLICITY, simplicityCovenantDataForNodeTest([32]byte{0x53}, nil), nil)
	if err := mp.AddTx(createTx); err == nil || !strings.Contains(err.Error(), "CORE_SIMPLICITY output pre-ACTIVE") {
		t.Fatalf("expected CORE_SIMPLICITY output policy rejection, got %v", err)
	}
	if _, err := mp.RelayMetadata(createTx); err == nil || !strings.Contains(err.Error(), "CORE_SIMPLICITY output pre-ACTIVE") {
		t.Fatalf("expected relay CORE_SIMPLICITY output policy rejection, got %v", err)
	}

	var prev [32]byte
	prev[0] = 0x54
	st.Utxos[consensus.Outpoint{Txid: prev, Vout: 0}] = consensus.UtxoEntry{
		Value:        100,
		CovenantType: consensus.COV_TYPE_CORE_SIMPLICITY,
		CovenantData: simplicityCovenantDataForNodeTest([32]byte{0x55}, nil),
	}
	spendTx := txWithOneInputOneOutput(prev, 0, 99, consensus.COV_TYPE_P2PK, append([]byte(nil), fromAddress...), nil)
	if err := mp.AddTx(spendTx); err == nil || !strings.Contains(err.Error(), "CORE_SIMPLICITY spend pre-ACTIVE") {
		t.Fatalf("expected CORE_SIMPLICITY spend policy rejection, got %v", err)
	}
	if _, err := mp.RelayMetadata(spendTx); err == nil || !strings.Contains(err.Error(), "CORE_SIMPLICITY spend pre-ACTIVE") {
		t.Fatalf("expected relay CORE_SIMPLICITY spend policy rejection, got %v", err)
	}
}

func TestMempoolPolicyAllowsCoreSimplicityCreateWhenActive(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
	entry := st.Utxos[outpoints[0]]

	tx := &consensus.Tx{
		Version: 1,
		TxKind:  0x00,
		TxNonce: 1,
		Inputs:  []consensus.TxInput{{PrevTxid: outpoints[0].Txid, PrevVout: outpoints[0].Vout}},
		Outputs: []consensus.TxOutput{
			{Value: 100_000, CovenantType: consensus.COV_TYPE_CORE_SIMPLICITY, CovenantData: simplicityCovenantDataForNodeTest([32]byte{0x58}, nil)},
			{Value: entry.Value - 200_000, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: append([]byte(nil), fromAddress...)},
		},
	}
	if err := consensus.SignTransaction(tx, st.Utxos, devnetGenesisChainID, fromKey); err != nil {
		t.Fatalf("SignTransaction: %v", err)
	}
	txBytes, err := consensus.MarshalTx(tx)
	if err != nil {
		t.Fatalf("MarshalTx: %v", err)
	}

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		PolicyRejectSimplicityPreActivation: true,
		RotationProvider:                    testSimplicityRotation{activeAt: 0, chainID: devnetGenesisChainID},
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("expected active CORE_SIMPLICITY policy allow, got %v", err)
	}
}

func TestMempoolPolicyDoesNotMaskMalformedNonSimplicityOutput(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{100})
	txBytes := mustMarshalTxForNodeTest(t, &consensus.Tx{
		Version: 1,
		TxKind:  0x00,
		TxNonce: 1,
		Inputs:  []consensus.TxInput{{PrevTxid: outpoints[0].Txid, PrevVout: outpoints[0].Vout}},
		Outputs: []consensus.TxOutput{
			{Value: 1, CovenantType: consensus.COV_TYPE_CORE_SIMPLICITY, CovenantData: simplicityCovenantDataForNodeTest([32]byte{0x59}, nil)},
			{Value: 1, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: nil},
		},
	})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		PolicyRejectSimplicityPreActivation: true,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	for _, admit := range []struct {
		name string
		run  func() error
	}{
		{"AddTx", func() error { return mp.AddTx(txBytes) }},
		{"RelayMetadata", func() error { _, err := mp.RelayMetadata(txBytes); return err }},
	} {
		err := admit.run()
		if err == nil || !strings.Contains(err.Error(), "invalid CORE_P2PK covenant_data length") || strings.Contains(err.Error(), "CORE_SIMPLICITY output pre-ACTIVE") {
			t.Fatalf("%s err=%v, want malformed P2PK without CORE_SIMPLICITY policy masking", admit.name, err)
		}
	}
}

func TestMempoolPolicyRejectsNilCheckedTransaction(t *testing.T) {
	mp := &Mempool{}
	if err := mp.applyPolicyAgainstState(nil, 0, nil, MempoolConfig{}); err == nil || !strings.Contains(err.Error(), "nil checked transaction") {
		t.Fatalf("expected nil checked transaction rejection, got %v", err)
	}
	if err := mp.applyPolicyAgainstState(&consensus.CheckedTransaction{}, 0, nil, MempoolConfig{}); err == nil || !strings.Contains(err.Error(), "nil checked transaction") {
		t.Fatalf("expected nil checked tx rejection, got %v", err)
	}
	if kind := covenantPolicyKind(nil, nil, consensus.COV_TYPE_CORE_EXT); kind != "" {
		t.Fatalf("nil tx policy kind=%q", kind)
	}
	tx := &consensus.Tx{Inputs: []consensus.TxInput{{}}}
	if _, err := policyInputSnapshot(tx, map[consensus.Outpoint]consensus.UtxoEntry{}); err == nil || !strings.Contains(err.Error(), "utxo not found") {
		t.Fatalf("expected missing UTXO snapshot error, got %v", err)
	}
}

func TestMempoolPolicyPropagatesDaFeeComputationErrors(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{100})
	txBytes := mustBuildSignedDaCommitTx(t, st.Utxos, outpoints[0], 80, 10, 1, fromKey, toAddress, []byte("0123456789"))
	tx, _, _, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx(da): %v", err)
	}

	mp := &Mempool{
		chainState: &ChainState{},
		policy: MempoolConfig{
			PolicyDaSurchargePerByte: 1,
		},
	}
	_, daBytes, _, err := consensus.TxWeightAndStats(tx)
	if err != nil {
		t.Fatalf("TxWeightAndStats(da): %v", err)
	}
	if daBytes == 0 {
		t.Fatalf("test setup: DA fixture reported daBytes=0")
	}

	if err := mp.applyPolicyAgainstState(&consensus.CheckedTransaction{Tx: tx, DaBytes: daBytes}, 101, nil, mp.policySnapshot()); err == nil || !strings.Contains(err.Error(), "nil utxo set") {
		t.Fatalf("expected DA fee computation error, got %v", err)
	}
}

func TestMempoolPolicySkipsDaHelperForNonDaCheckedTx(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 900_000, 1, fromKey, fromAddress, toAddress)
	tx, _, _, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx(non-DA): %v", err)
	}
	_, daBytes, _, err := consensus.TxWeightAndStats(tx)
	if err != nil {
		t.Fatalf("TxWeightAndStats(non-DA): %v", err)
	}
	if daBytes != 0 {
		t.Fatalf("test setup: non-DA fixture reported daBytes=%d", daBytes)
	}

	mp := &Mempool{
		chainState: &ChainState{},
		policy: MempoolConfig{
			MinDaFeeRate:             1,
			PolicyDaSurchargePerByte: 1,
		},
	}
	policyUtxos, err := policyInputSnapshot(tx, st.Utxos)
	if err != nil {
		t.Fatalf("policyInputSnapshot: %v", err)
	}
	if err := mp.applyPolicyAgainstState(&consensus.CheckedTransaction{Tx: tx, DaBytes: daBytes}, 101, policyUtxos, mp.policySnapshot()); err != nil {
		t.Fatalf("non-DA checked tx should skip DA helper, got %v", err)
	}
}

func TestPolicyNeedsInputSnapshotForTxMatrix(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	parseFor := func(t *testing.T, txBytes []byte) *consensus.Tx {
		t.Helper()
		tx, _, _, _, err := consensus.ParseTx(txBytes)
		if err != nil {
			t.Fatalf("ParseTx: %v", err)
		}
		return tx
	}

	daBytesPayload := mustBuildSignedDaCommitTx(t, st.Utxos, outpoints[0], 50_000, 950_000, 1, fromKey, toAddress, []byte("0123456789"))
	nonDaBytesPayload := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 900_000, 1, fromKey, fromAddress, toAddress)
	daTx := parseFor(t, daBytesPayload)
	nonDaTx := parseFor(t, nonDaBytesPayload)
	if _, err := policyNeedsInputSnapshotForTx(nil, MempoolConfig{}); err == nil {
		t.Fatal("expected nil transaction snapshot decision error")
	}

	cases := []struct {
		name string
		cfg  MempoolConfig
		tx   *consensus.Tx
		want bool
	}{
		{name: "all_zero_non_da", cfg: MempoolConfig{}, tx: nonDaTx, want: true},
		{name: "all_zero_da", cfg: MempoolConfig{}, tx: daTx, want: true},
		{name: "min_da_fee_rate_only_non_da", cfg: MempoolConfig{MinDaFeeRate: 1}, tx: nonDaTx, want: true},
		{name: "min_da_fee_rate_only_da", cfg: MempoolConfig{MinDaFeeRate: 1}, tx: daTx, want: true},
		{name: "surcharge_only_non_da", cfg: MempoolConfig{PolicyDaSurchargePerByte: 1}, tx: nonDaTx, want: true},
		{name: "surcharge_only_da", cfg: MempoolConfig{PolicyDaSurchargePerByte: 1}, tx: daTx, want: true},
		{name: "simplicity_policy_non_da", cfg: MempoolConfig{PolicyRejectSimplicityPreActivation: true}, tx: nonDaTx, want: true},
		{name: "simplicity_policy_da", cfg: MempoolConfig{PolicyRejectSimplicityPreActivation: true}, tx: daTx, want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := policyNeedsInputSnapshotForTx(tc.tx, tc.cfg)
			if err != nil {
				t.Fatalf("policyNeedsInputSnapshotForTx error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got=%v want=%v", got, tc.want)
			}
		})
	}
}

func TestPolicyInputSnapshotCopiesOnlySpentInputs(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 2_000_000})

	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	tx, _, _, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx: %v", err)
	}

	snapshot, err := policyInputSnapshot(tx, st.Utxos)
	if err != nil {
		t.Fatalf("policyInputSnapshot: %v", err)
	}
	if len(snapshot) != 1 {
		t.Fatalf("snapshot len=%d, want 1", len(snapshot))
	}
	if _, ok := snapshot[outpoints[0]]; !ok {
		t.Fatalf("snapshot missing spent input")
	}
	if _, ok := snapshot[outpoints[1]]; ok {
		t.Fatalf("snapshot unexpectedly copied unrelated utxo")
	}

	entry := snapshot[outpoints[0]]
	entry.CovenantData[0] ^= 0xff
	snapshot[outpoints[0]] = entry
	if reflect.DeepEqual(snapshot[outpoints[0]].CovenantData, st.Utxos[outpoints[0]].CovenantData) {
		t.Fatal("mutating snapshot covenant data leaked into original utxo set")
	}
}

func TestChainStateAdmissionSnapshotForInputsCopiesOnlyRequestedEntries(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{100, 200})
	var missingTxid [32]byte
	missingTxid[0] = 0xee

	snapshot := st.admissionSnapshotForInputs([]consensus.Outpoint{
		outpoints[0],
		outpoints[0],
		{Txid: missingTxid, Vout: 9},
	})
	if snapshot == nil {
		t.Fatal("admissionSnapshotForInputs returned nil")
	}
	if len(snapshot.utxos) != 1 {
		t.Fatalf("snapshot len=%d, want 1", len(snapshot.utxos))
	}
	if _, ok := snapshot.utxos[outpoints[0]]; !ok {
		t.Fatal("snapshot missing requested input")
	}
	if _, ok := snapshot.utxos[outpoints[1]]; ok {
		t.Fatal("snapshot unexpectedly copied unrelated utxo")
	}

	entry := snapshot.utxos[outpoints[0]]
	entry.CovenantData[0] ^= 0xff
	snapshot.utxos[outpoints[0]] = entry
	if reflect.DeepEqual(snapshot.utxos[outpoints[0]].CovenantData, st.Utxos[outpoints[0]].CovenantData) {
		t.Fatal("mutating input snapshot leaked into original utxo set")
	}
}

func TestPolicyInputSnapshotRejectsMissingInput(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	tx, _, _, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx: %v", err)
	}

	delete(st.Utxos, outpoints[0])

	_, err = policyInputSnapshot(tx, st.Utxos)
	if err == nil || !strings.Contains(err.Error(), string(consensus.TX_ERR_MISSING_UTXO)) {
		t.Fatalf("expected missing utxo rejection, got %v", err)
	}
}

func TestMempoolDoubleSpend(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(tx1); err != nil {
		t.Fatalf("AddTx(tx1): %v", err)
	}
	tx1ID := txID(t, tx1)
	if got, ok := mp.pendingOutpoints.txidForOutpoint(outpoints[0]); !ok || got != tx1ID {
		t.Fatalf("pending-outpoint claim got %x ok=%v, want tx1 %x", got, ok, tx1ID)
	}
	seqAfterTx1 := mp.lastAdmissionSeq
	if err := mp.AddTx(tx2); err == nil {
		t.Fatalf("expected double-spend rejection")
	}
	if got := mp.Len(); got != 1 {
		t.Fatalf("mempool len=%d, want 1", got)
	}
	if mp.Contains(txID(t, tx2)) {
		t.Fatalf("conflicting tx entered mempool")
	}
	if got, ok := mp.pendingOutpoints.txidForOutpoint(outpoints[0]); !ok || got != tx1ID {
		t.Fatalf("pending-outpoint claim after conflict got %x ok=%v, want tx1 %x", got, ok, tx1ID)
	}
	if mp.lastAdmissionSeq != seqAfterTx1 {
		t.Fatalf("lastAdmissionSeq after conflict=%d, want %d", mp.lastAdmissionSeq, seqAfterTx1)
	}
}

func TestMempoolFullEvictsWorstByFeeWeight(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000, 1_000_000})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 2, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	txLow := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	txHigh := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
	txBest := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[2]}, 100_000, 300_000, 3, fromKey, fromAddress, toAddress)

	if err := mp.AddTx(txLow); err != nil {
		t.Fatalf("AddTx(low): %v", err)
	}
	if err := mp.AddTx(txHigh); err != nil {
		t.Fatalf("AddTx(high): %v", err)
	}
	if err := mp.AddTx(txBest); err != nil {
		t.Fatalf("AddTx(best): %v", err)
	}
	if got := mp.Len(); got != 2 {
		t.Fatalf("mempool len=%d, want 2", got)
	}
	if mp.Contains(txID(t, txLow)) {
		t.Fatal("lowest fee/weight tx remained after count-pressure eviction")
	}
	if !mp.Contains(txID(t, txHigh)) || !mp.Contains(txID(t, txBest)) {
		t.Fatal("capacity eviction removed a survivor with better fee/weight")
	}
	if got := mp.lastAdmissionSeq; got != 3 {
		t.Fatalf("lastAdmissionSeq=%d, want 3", got)
	}
	if got := mp.txs[txID(t, txBest)].admissionSeq; got != 3 {
		t.Fatalf("best admission_seq=%d, want 3", got)
	}
	if mp.currentMinFeeRate <= DefaultMempoolMinFeeRate {
		t.Fatalf("currentMinFeeRate=%d, want above base floor after actual eviction", mp.currentMinFeeRate)
	}

	selected := mp.SelectTransactions(3, 1<<20)
	if len(selected) != 2 {
		t.Fatalf("selected=%d, want 2", len(selected))
	}
	got := []string{txIDHex(t, selected[0]), txIDHex(t, selected[1])}
	wantBest := txIDHex(t, txBest)
	wantHigh := txIDHex(t, txHigh)
	if got[0] != wantBest || got[1] != wantHigh {
		t.Fatalf("selected=%v, want [%s %s]", got, wantBest, wantHigh)
	}
}

func TestMempoolCandidateWorstRejectsWithoutMutation(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
	tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 100_000, 2, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MaxTransactions: 10,
		MaxBytes:        len(tx1) + len(tx2) - 1,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	if err := mp.AddTx(tx1); err != nil {
		t.Fatalf("AddTx(tx1): %v", err)
	}
	before, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot before candidate-worst: %v", err)
	}
	usedBytes := mp.usedBytes
	if err := mp.AddTx(tx2); err == nil || !strings.Contains(err.Error(), "mempool capacity candidate rejected by eviction ordering") {
		t.Fatalf("expected candidate-worst rejection, got %v", err)
	}
	after, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot after candidate-worst: %v", err)
	}
	assertPostSlotRejectionSnapshot(t, before, after)
	if got := mp.Len(); got != 1 {
		t.Fatalf("mempool len=%d, want 1", got)
	}
	if mp.usedBytes != usedBytes {
		t.Fatalf("usedBytes=%d, want %d", mp.usedBytes, usedBytes)
	}
	if mp.Contains(txID(t, tx2)) {
		t.Fatalf("rejected byte-cap tx entered mempool")
	}
}

func TestMempoolCapacityRejectsBelowRollingFloorWithoutMutation(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
	txBelowFloor := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 1, 2, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 1, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if err := mp.AddTx(tx1); err != nil {
		t.Fatalf("AddTx(tx1): %v", err)
	}
	before, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot before below-floor: %v", err)
	}
	err = mp.AddTx(txBelowFloor)
	if err == nil || !strings.Contains(err.Error(), "mempool fee below rolling minimum") {
		t.Fatalf("expected below-floor rejection, got %v", err)
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitUnavailable {
		t.Fatalf("below-floor err=%v, want TxAdmitUnavailable", err)
	}
	after, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot after below-floor: %v", err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("below-floor capacity reject mutated mempool: before=%+v after=%+v", before, after)
	}
	if mp.Contains(txID(t, txBelowFloor)) {
		t.Fatal("below-floor capacity candidate entered mempool")
	}
}

func TestMempoolRollingFloorRejectsBelowCapacityWithoutMutation(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	txBelowFloor := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 1, 1, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 10, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	mp.currentMinFeeRate = 8
	before, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot before below-capacity below-floor: %v", err)
	}
	err = mp.AddTx(txBelowFloor)
	if err == nil || !strings.Contains(err.Error(), "mempool fee below rolling minimum") {
		t.Fatalf("expected below-capacity below-floor rejection, got %v", err)
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitUnavailable {
		t.Fatalf("below-capacity floor err=%v, want TxAdmitUnavailable", err)
	}
	after, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot after below-capacity below-floor: %v", err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("below-capacity floor reject mutated mempool: before=%+v after=%+v", before, after)
	}
	if mp.usedBytes != 0 || mp.lastAdmissionSeq != 0 || mp.currentMinFeeRate != 8 {
		t.Fatalf("below-capacity floor reject state usedBytes=%d seq=%d floor=%d", mp.usedBytes, mp.lastAdmissionSeq, mp.currentMinFeeRate)
	}
}

func TestMempoolAddEntryLockedRejectsBelowFloor(t *testing.T) {
	mp := &Mempool{maxTxs: 10, maxBytes: 100, currentMinFeeRate: 8}
	err := mp.addEntryLocked(&mempoolEntry{
		txid:   [32]byte{0x51},
		fee:    consensus.Uint128FromU64(7),
		weight: 1,
		size:   1,
	})
	if err == nil || !strings.Contains(err.Error(), "mempool fee below rolling minimum") {
		t.Fatalf("expected addEntryLocked below-floor rejection, got %v", err)
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) || txErr.Kind != TxAdmitUnavailable {
		t.Fatalf("addEntryLocked floor err=%v, want TxAdmitUnavailable", err)
	}
	if len(mp.txs) != 0 || mp.usedBytes != 0 || mp.lastAdmissionSeq != 0 || mp.currentMinFeeRate != 8 {
		t.Fatalf("addEntryLocked floor reject mutated mempool: len=%d used=%d seq=%d floor=%d", len(mp.txs), mp.usedBytes, mp.lastAdmissionSeq, mp.currentMinFeeRate)
	}
}

func TestMempoolRollingFloorAcceptsExactFloorBelowCapacity(t *testing.T) {
	const floor = uint64(8)
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	probe := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 1, 1, fromKey, fromAddress, toAddress)
	parsed, _, _, _, err := consensus.ParseTx(probe)
	if err != nil {
		t.Fatalf("ParseTx(probe): %v", err)
	}
	weight, _, _, err := consensus.TxWeightAndStats(parsed)
	if err != nil {
		t.Fatalf("TxWeightAndStats(probe): %v", err)
	}
	exactFee := weight * floor
	txExactFloor := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, exactFee, 1, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 10, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	mp.currentMinFeeRate = floor
	if err := mp.AddTx(txExactFloor); err != nil {
		t.Fatalf("AddTx(exact floor): %v", err)
	}
	entry := mp.txs[txID(t, txExactFloor)]
	if entry == nil {
		t.Fatal("exact-floor tx missing from mempool")
	}
	if feeRateBelowFloor(entry.fee, entry.weight, floor) {
		t.Fatalf("accepted exact-floor entry still below floor: fee=%d weight=%d floor=%d", entry.fee, entry.weight, floor)
	}
}

func TestMempoolByteCapEvictsToLowWater(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000, 1_000_000, 1_000_000})

	txLow := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	txMid := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
	txHigh := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[2]}, 100_000, 300_000, 3, fromKey, fromAddress, toAddress)
	txBest := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[3]}, 100_000, 400_000, 4, fromKey, fromAddress, toAddress)
	maxBytes := len(txLow) + len(txMid) + len(txHigh)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MaxTransactions: 10,
		MaxBytes:        maxBytes,
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	for _, item := range []struct {
		name string
		raw  []byte
	}{
		{name: "low", raw: txLow},
		{name: "mid", raw: txMid},
		{name: "high", raw: txHigh},
	} {
		if err := mp.AddTx(item.raw); err != nil {
			t.Fatalf("AddTx(%s): %v", item.name, err)
		}
	}
	if err := mp.AddTx(txBest); err != nil {
		t.Fatalf("AddTx(best): %v", err)
	}
	if got, wantMax := mp.usedBytes, mp.effectiveLowWaterBytesLocked(); got > wantMax {
		t.Fatalf("usedBytes=%d, want <= lowWater %d after byte-pressure eviction", got, wantMax)
	}
	if mp.Contains(txID(t, txLow)) || mp.Contains(txID(t, txMid)) {
		t.Fatal("byte-pressure low-water trim kept lower-priority evicted entries")
	}
	if !mp.Contains(txID(t, txHigh)) || !mp.Contains(txID(t, txBest)) {
		t.Fatal("byte-pressure low-water trim removed expected survivors")
	}
}

func TestMempoolSmallByteCapKeepsFittingCandidateAfterLowWaterTrim(t *testing.T) {
	for _, tc := range []struct {
		name     string
		maxBytes int
	}{
		{name: "one", maxBytes: 1},
		{name: "two", maxBytes: 2},
		{name: "five", maxBytes: 5},
		{name: "nine", maxBytes: 9},
	} {
		t.Run(tc.name, func(t *testing.T) {
			residentID := [32]byte{byte(0x60 + tc.maxBytes)}
			candidateID := [32]byte{byte(0x70 + tc.maxBytes)}
			mp := &Mempool{maxTxs: 10, maxBytes: tc.maxBytes}
			resident := &mempoolEntry{txid: residentID, fee: consensus.Uint128FromU64(1), weight: 1, size: tc.maxBytes}
			if err := mp.addEntryLocked(resident); err != nil {
				t.Fatalf("addEntryLocked(resident): %v", err)
			}

			candidate := &mempoolEntry{txid: candidateID, fee: consensus.Uint128FromU64(10), weight: 1, size: 1}
			if err := mp.addEntryLocked(candidate); err != nil {
				t.Fatalf("addEntryLocked(candidate): %v", err)
			}
			if got := mp.Len(); got != 1 {
				t.Fatalf("mempool len=%d, want 1 after small-cap low-water trim", got)
			}
			if got := mp.usedBytes; got != 1 {
				t.Fatalf("usedBytes=%d, want 1 after small-cap low-water trim", got)
			}
			if !mp.Contains(candidate.txid) {
				t.Fatal("candidate fitting the hard byte cap was not admitted")
			}
			if mp.Contains(resident.txid) {
				t.Fatal("small-cap low-water trim kept lower-priority resident")
			}
		})
	}
}

func TestMempoolBytePressureAdmitsCandidateLargerThanLowWaterWhenFitsHardCap(t *testing.T) {
	residentID := [32]byte{0x80}
	candidateID := [32]byte{0x81}
	mp := &Mempool{maxTxs: 10, maxBytes: 100}
	resident := &mempoolEntry{txid: residentID, fee: consensus.Uint128FromU64(1), weight: 1, size: 95}
	if err := mp.addEntryLocked(resident); err != nil {
		t.Fatalf("addEntryLocked(resident): %v", err)
	}
	if got, want := mp.effectiveLowWaterBytesLocked(), 90; got != want {
		t.Fatalf("lowWater=%d, want %d", got, want)
	}

	candidate := &mempoolEntry{txid: candidateID, fee: consensus.Uint128FromU64(100), weight: 1, size: 95}
	if err := mp.addEntryLocked(candidate); err != nil {
		t.Fatalf("addEntryLocked(candidate): %v", err)
	}
	if got := mp.Len(); got != 1 {
		t.Fatalf("mempool len=%d, want 1 after byte-pressure replacement", got)
	}
	if got := mp.usedBytes; got != 95 {
		t.Fatalf("usedBytes=%d, want candidate hard-cap size 95", got)
	}
	if !mp.Contains(candidateID) {
		t.Fatal("candidate fitting maxBytes was not admitted")
	}
	if mp.Contains(residentID) {
		t.Fatal("byte-pressure replacement kept lower-priority resident")
	}
}

func TestMempoolDuplicateRejectsBeforeEviction(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	tx := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 1, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if err := mp.AddTx(tx); err != nil {
		t.Fatalf("AddTx(tx): %v", err)
	}
	before, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot before duplicate: %v", err)
	}
	if err := mp.AddTx(tx); err == nil || !strings.Contains(err.Error(), "tx already in mempool") {
		t.Fatalf("expected duplicate rejection before eviction, got %v", err)
	}
	after, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot after duplicate: %v", err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("duplicate path mutated mempool: before=%+v after=%+v", before, after)
	}
}

func TestMempoolConflictRejectsBeforeEvictionUnderPressure(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	tx := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	conflictingHigherFee := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 1, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if err := mp.AddTx(tx); err != nil {
		t.Fatalf("AddTx(tx): %v", err)
	}
	before, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot before conflict: %v", err)
	}
	if err := mp.AddTx(conflictingHigherFee); err == nil || !strings.Contains(err.Error(), "mempool double-spend conflict") {
		t.Fatalf("expected conflict rejection before eviction, got %v", err)
	}
	after, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot after conflict: %v", err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("conflict path mutated mempool: before=%+v after=%+v", before, after)
	}
	if !mp.Contains(txID(t, tx)) || mp.Contains(txID(t, conflictingHigherFee)) {
		t.Fatal("conflict path replaced resident transaction")
	}
}

func TestMempoolRollingMinFeeDecaysOnlyOnConnectedBlockLowWater(t *testing.T) {
	st := NewChainState()
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxBytes: 1000})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	mp.currentMinFeeRate = 8
	mp.usedBytes = mp.effectiveLowWaterBytesLocked() - 1
	if err := mp.RemoveConflictingParsed(&consensus.ParsedBlock{}); err != nil {
		t.Fatalf("RemoveConflictingParsed: %v", err)
	}
	if got := mp.currentMinFeeRate; got != 8 {
		t.Fatalf("RemoveConflictingParsed decayed floor to %d, want 8", got)
	}
	if err := mp.EvictConfirmedParsed(&consensus.ParsedBlock{}); err != nil {
		t.Fatalf("EvictConfirmedParsed: %v", err)
	}
	if got := mp.currentMinFeeRate; got != 8 {
		t.Fatalf("EvictConfirmedParsed decayed floor to %d, want 8", got)
	}
	if got := canonicalMempoolFeeFloor(mp.currentMinFeeRate, mp.usedBytes, mp.effectiveLowWaterBytesLocked(), 1); got != 4 {
		t.Fatalf("connected-block low-water floor=%d, want 4", got)
	}
	mp.currentMinFeeRate = 8
	mp.usedBytes = mp.effectiveLowWaterBytesLocked()
	if got := canonicalMempoolFeeFloor(mp.currentMinFeeRate, mp.usedBytes, mp.effectiveLowWaterBytesLocked(), 1); got != 8 {
		t.Fatalf("boundary usedBytes floor=%d, want 8", got)
	}
	mp.currentMinFeeRate = 0
	mp.usedBytes = 0
	if preClamp, postDecay := canonicalMempoolFeeFloor(mp.currentMinFeeRate, mp.usedBytes, mp.effectiveLowWaterBytesLocked(), 0), canonicalMempoolFeeFloor(DefaultMempoolMinFeeRate, mp.usedBytes, mp.effectiveLowWaterBytesLocked(), 1); preClamp != DefaultMempoolMinFeeRate || postDecay != DefaultMempoolMinFeeRate {
		t.Fatalf("minimum floors pre=%d post=%d, want %d", preClamp, postDecay, DefaultMempoolMinFeeRate)
	}
}

func TestMempoolConnectedBlockDecaySeesConfirmedAndConflictingRemovals(t *testing.T) {
	spentByBlock := consensus.Outpoint{Txid: [32]byte{0xc1}, Vout: 2}
	confirmedID := [32]byte{0xa1}
	conflictingID := [32]byte{0xb1}
	mp := &Mempool{maxTxs: 10, maxBytes: 100, currentMinFeeRate: 8}
	confirmed := &mempoolEntry{
		txid:         confirmedID,
		wtxid:        confirmedID,
		fee:          consensus.Uint128FromU64(8),
		weight:       1,
		size:         5,
		admissionSeq: 1,
		source:       mempoolTxSourceLocal,
	}
	conflicting := &mempoolEntry{
		txid:         conflictingID,
		wtxid:        conflictingID,
		inputs:       []consensus.Outpoint{spentByBlock},
		fee:          consensus.Uint128FromU64(8),
		weight:       1,
		size:         90,
		admissionSeq: 2,
		source:       mempoolTxSourceLocal,
	}
	if err := mp.addEntryLocked(confirmed); err != nil {
		t.Fatalf("add confirmed entry: %v", err)
	}
	if err := mp.addEntryLocked(conflicting); err != nil {
		t.Fatalf("add conflicting entry: %v", err)
	}
	if got := mp.usedBytes; got != 95 {
		t.Fatalf("usedBytes before connected block=%d, want 95", got)
	}

	block := &consensus.ParsedBlock{
		Txids: [][32]byte{{0x01}, confirmedID},
		Txs: []*consensus.Tx{
			{},
			{Inputs: []consensus.TxInput{{PrevTxid: spentByBlock.Txid, PrevVout: spentByBlock.Vout}}},
		},
	}
	if err := mp.EvictConfirmedParsed(block); err != nil {
		t.Fatalf("EvictConfirmedParsed: %v", err)
	}
	if err := mp.RemoveConflictingParsed(block); err != nil {
		t.Fatalf("RemoveConflictingParsed: %v", err)
	}
	if mp.Contains(confirmedID) {
		t.Fatal("connected block left confirmed tx in mempool")
	}
	if mp.Contains(conflictingID) {
		t.Fatal("connected block left conflicting tx in mempool")
	}
	if got := mp.usedBytes; got != 0 {
		t.Fatalf("usedBytes after connected block=%d, want 0", got)
	}
	if got := canonicalMempoolFeeFloor(mp.currentMinFeeRate, mp.usedBytes, mp.effectiveLowWaterBytesLocked(), 1); got != 4 {
		t.Fatalf("currentMinFeeRate after confirmed+conflict removals=%d, want 4", got)
	}
}

func TestMempoolAddReorgTxUsesRollingFloor(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 10, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	mp.currentMinFeeRate = 8
	txBelowFloor := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 1, 1, fromKey, fromAddress, toAddress)

	err = mp.AddReorgTx(txBelowFloor)
	var admitErr *TxAdmitError
	if !errors.As(err, &admitErr) || admitErr.Kind != TxAdmitUnavailable {
		t.Fatalf("AddReorgTx below rolling floor error=%T %v, want TxAdmitUnavailable", err, err)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len after reorg floor reject=%d, want 0", got)
	}
	if mp.lastAdmissionSeq != 0 {
		t.Fatalf("lastAdmissionSeq after reorg floor reject=%d, want 0", mp.lastAdmissionSeq)
	}
	if got := mp.currentMinFeeRate; got != 8 {
		t.Fatalf("currentMinFeeRate after reorg floor reject=%d, want 8", got)
	}
}

func TestMempoolAddReorgTxUsesNormalCapacityAdmission(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	resident := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 300_000, 1, fromKey, fromAddress, toAddress)
	lowerReorg := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 100_000, 2, fromKey, fromAddress, toAddress)
	betterReorg := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 400_000, 3, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 1, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if err := mp.AddTx(resident); err != nil {
		t.Fatalf("AddTx(resident): %v", err)
	}
	before, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot before lower reorg: %v", err)
	}

	err = mp.AddReorgTx(lowerReorg)
	var admitErr *TxAdmitError
	if !errors.As(err, &admitErr) || admitErr.Kind != TxAdmitUnavailable {
		t.Fatalf("lower AddReorgTx err=%T %v, want TxAdmitUnavailable", err, err)
	}
	if !strings.Contains(err.Error(), "mempool capacity candidate rejected by eviction ordering") {
		t.Fatalf("lower AddReorgTx error=%v, want capacity candidate rejection", err)
	}
	afterReject, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot after lower reorg: %v", err)
	}
	assertPostSlotRejectionSnapshot(t, before, afterReject)
	if mp.Contains(txID(t, lowerReorg)) {
		t.Fatalf("lower reorg candidate entered mempool")
	}

	if err := mp.AddReorgTx(betterReorg); err != nil {
		t.Fatalf("AddReorgTx(better): %v", err)
	}
	if mp.Contains(txID(t, resident)) {
		t.Fatalf("normal capacity admission kept lower-priority resident")
	}
	betterEntry := mp.txs[txID(t, betterReorg)]
	if betterEntry == nil {
		t.Fatalf("better reorg candidate missing after normal admission")
	}
	if betterEntry.source != mempoolTxSourceReorg {
		t.Fatalf("better reorg source=%q, want %q", betterEntry.source, mempoolTxSourceReorg)
	}
}

func TestMempoolAddReorgTxRejectsConflictBeforeEviction(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	resident := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
	conflictingReorg := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 400_000, 2, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 1, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if err := mp.AddTx(resident); err != nil {
		t.Fatalf("AddTx(resident): %v", err)
	}
	before, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot before conflicting reorg: %v", err)
	}

	err = mp.AddReorgTx(conflictingReorg)
	var admitErr *TxAdmitError
	if !errors.As(err, &admitErr) || admitErr.Kind != TxAdmitConflict {
		t.Fatalf("conflicting AddReorgTx err=%T %v, want TxAdmitConflict", err, err)
	}
	after, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshot after conflicting reorg: %v", err)
	}
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("conflicting reorg tx mutated mempool: before=%+v after=%+v", before, after)
	}
	if mp.Contains(txID(t, conflictingReorg)) {
		t.Fatalf("conflicting reorg tx entered mempool")
	}
}

func TestMempoolByteCapAllowsExactBoundary(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
	tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MaxTransactions: 10,
		MaxBytes:        len(tx1) + len(tx2),
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	if err := mp.AddTx(tx1); err != nil {
		t.Fatalf("AddTx(tx1): %v", err)
	}
	if err := mp.AddTx(tx2); err != nil {
		t.Fatalf("AddTx(tx2) at exact byte cap: %v", err)
	}
	if got := mp.Len(); got != 2 {
		t.Fatalf("mempool len=%d, want 2", got)
	}
	if mp.usedBytes != len(tx1)+len(tx2) {
		t.Fatalf("usedBytes=%d, want %d", mp.usedBytes, len(tx1)+len(tx2))
	}
}

func TestMempoolAdmissionRejectsDoNotMutateByteAccounting(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 10})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
	txDoubleSpend := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 300_000, 2, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(tx1); err != nil {
		t.Fatalf("AddTx(tx1): %v", err)
	}
	wantBytes := mp.usedBytes
	wantLen := mp.Len()

	for _, tc := range []struct {
		name string
		raw  []byte
	}{
		{name: "duplicate", raw: tx1},
		{name: "double_spend", raw: txDoubleSpend},
		{name: "malformed", raw: []byte{0xde, 0xad}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := mp.AddTx(tc.raw); err == nil {
				t.Fatalf("expected rejection")
			}
			if got := mp.Len(); got != wantLen {
				t.Fatalf("mempool len=%d, want %d", got, wantLen)
			}
			if mp.usedBytes != wantBytes {
				t.Fatalf("usedBytes=%d, want %d", mp.usedBytes, wantBytes)
			}
		})
	}
}

func TestRestoreMempoolSnapshotRecomputesByteAccounting(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
	tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MaxTransactions: 10,
		MaxBytes:        len(tx1) + len(tx2),
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if err := mp.AddTx(tx1); err != nil {
		t.Fatalf("AddTx(tx1): %v", err)
	}
	snapshot, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshotMempool: %v", err)
	}
	if err := mp.AddTx(tx2); err != nil {
		t.Fatalf("AddTx(tx2): %v", err)
	}
	if err := installMempoolImageForTest(mp, snapshot); err != nil {
		t.Fatalf("installMempoolImageForTest: %v", err)
	}
	if got := mp.Len(); got != 1 {
		t.Fatalf("mempool len=%d, want 1", got)
	}
	tx1ID := txID(t, tx1)
	_, _, tx1WTxID, _, err := consensus.ParseTx(tx1)
	if err != nil {
		t.Fatalf("ParseTx(tx1): %v", err)
	}
	restored := mp.txs[tx1ID]
	if restored == nil {
		t.Fatalf("restored entry for tx1 missing")
	}
	if restored.wtxid != tx1WTxID {
		t.Fatalf("restored wtxid=%x, want %x", restored.wtxid, tx1WTxID)
	}
	if restored.admissionSeq != 1 {
		t.Fatalf("restored admission_seq=%d, want 1", restored.admissionSeq)
	}
	if restored.source != mempoolTxSourceLocal {
		t.Fatalf("restored source=%q, want %q", restored.source, mempoolTxSourceLocal)
	}
	if mp.lastAdmissionSeq != restored.admissionSeq {
		t.Fatalf("lastAdmissionSeq after restore=%d, want %d", mp.lastAdmissionSeq, restored.admissionSeq)
	}
	if mp.usedBytes != len(tx1) {
		t.Fatalf("usedBytes=%d, want %d", mp.usedBytes, len(tx1))
	}
	if mp.Contains(txID(t, tx2)) {
		t.Fatalf("restored mempool still contains tx2")
	}
	if err := mp.AddTx(tx2); err != nil {
		t.Fatalf("AddTx(tx2) after restore: %v", err)
	}
	if mp.usedBytes != len(tx1)+len(tx2) {
		t.Fatalf("usedBytes=%d, want %d after post-restore AddTx", mp.usedBytes, len(tx1)+len(tx2))
	}
}

func TestRestoreMempoolSnapshotPreservesAdmissionSeqHighWatermark(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 100_000, 2, fromKey, fromAddress, toAddress)
	tx3 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[2]}, 100_000, 100_000, 3, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(tx1); err != nil {
		t.Fatalf("AddTx(tx1): %v", err)
	}
	if err := mp.AddTx(tx2); err != nil {
		t.Fatalf("AddTx(tx2): %v", err)
	}
	mp.currentMinFeeRate = 7
	tx2ID := txID(t, tx2)
	mp.mu.Lock()
	if err := mp.removeTxLocked(tx2ID); err != nil {
		t.Fatalf("removeTxLocked: %v", err)
	}
	if mp.lastAdmissionSeq != 2 {
		t.Fatalf("lastAdmissionSeq after removing tx2=%d, want 2", mp.lastAdmissionSeq)
	}
	mp.mu.Unlock()

	snapshot, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshotMempool: %v", err)
	}
	if snapshot.lastAdmissionSeq != 2 {
		t.Fatalf("snapshot lastAdmissionSeq=%d, want 2", snapshot.lastAdmissionSeq)
	}
	if snapshot.currentMinFeeRate != 7 {
		t.Fatalf("snapshot currentMinFeeRate=%d, want 7", snapshot.currentMinFeeRate)
	}
	mp.currentMinFeeRate = 3
	if err := installMempoolImageForTest(mp, snapshot); err != nil {
		t.Fatalf("installMempoolImageForTest: %v", err)
	}
	if mp.lastAdmissionSeq != 2 {
		t.Fatalf("lastAdmissionSeq after restore=%d, want 2", mp.lastAdmissionSeq)
	}
	if mp.currentMinFeeRate != 7 {
		t.Fatalf("currentMinFeeRate after restore=%d, want 7", mp.currentMinFeeRate)
	}
	if err := mp.AddTx(tx3); err != nil {
		t.Fatalf("AddTx(tx3): %v", err)
	}
	tx3ID := txID(t, tx3)
	if got := mp.txs[tx3ID].admissionSeq; got != 3 {
		t.Fatalf("tx3 admissionSeq=%d, want 3", got)
	}
}

func TestSnapshotMempoolNormalizesRollingFloor(t *testing.T) {
	mp := &Mempool{pendingOutpoints: newPendingOutpointOwner(PendingOutpointTip{})}
	snapshot, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshotMempool: %v", err)
	}
	if snapshot.currentMinFeeRate != DefaultMempoolMinFeeRate {
		t.Fatalf("snapshot currentMinFeeRate=%d, want %d", snapshot.currentMinFeeRate, DefaultMempoolMinFeeRate)
	}
}

func TestRestoreMempoolSnapshotRejectsInvalidEntriesWithoutMutation(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
	txSecond := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
	txSecondID := txID(t, txSecond)
	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MaxTransactions: 10,
		MaxBytes:        len(txBytes) + len(txSecond),
	})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}
	snapshot, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshotMempool: %v", err)
	}
	wantTxID := txID(t, txBytes)
	wantBytes := mp.usedBytes
	txDoubleSpend := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 300_000, 2, fromKey, fromAddress, toAddress)
	doubleSpendID := txID(t, txDoubleSpend)
	snapshotEntry := func(txRaw []byte, id [32]byte, inputs []consensus.Outpoint) mempoolEntry {
		parsed, _, wtxid, _, err := consensus.ParseTx(txRaw)
		if err != nil {
			t.Fatalf("ParseTx(snapshotEntry): %v", err)
		}
		weight, _, _, err := consensus.TxWeightAndStats(parsed)
		if err != nil {
			t.Fatalf("TxWeightAndStats(snapshotEntry): %v", err)
		}
		return mempoolEntry{
			raw:          append([]byte(nil), txRaw...),
			txid:         id,
			wtxid:        wtxid,
			inputs:       append([]consensus.Outpoint(nil), inputs...),
			size:         len(txRaw),
			weight:       weight,
			admissionSeq: 99,
			source:       mempoolTxSourceLocal,
		}
	}
	cloneSnapshotForTest := func(base mempoolSnapshot) mempoolSnapshot {
		entries := make([]mempoolEntry, 0, len(base.entries))
		for i := range base.entries {
			entries = append(entries, cloneMempoolEntry(&base.entries[i]))
		}
		// The owner image travels with the entries: each case below corrupts
		// exactly one thing, so the record/claim binding must stay valid
		// everywhere else.
		return mempoolSnapshot{entries: entries, pending: base.pending, lastAdmissionSeq: base.lastAdmissionSeq, currentMinFeeRate: base.currentMinFeeRate}
	}
	withEditedFirst := func(edit func(*mempoolEntry)) func(mempoolSnapshot) mempoolSnapshot {
		return func(base mempoolSnapshot) mempoolSnapshot {
			bad := cloneSnapshotForTest(base)
			edit(&bad.entries[0])
			return bad
		}
	}

	for _, tc := range []struct {
		name      string
		configure func(*Mempool)
		mutate    func(mempoolSnapshot) mempoolSnapshot
		want      string
	}{
		{
			name:   "zero_size",
			mutate: withEditedFirst(func(entry *mempoolEntry) { entry.size = 0 }),
			want:   "invalid mempool snapshot entry size",
		},
		{
			name:   "zero_weight",
			mutate: withEditedFirst(func(entry *mempoolEntry) { entry.weight = 0 }),
			want:   "invalid mempool snapshot entry weight",
		},
		{
			name:   "weight_mismatch",
			mutate: withEditedFirst(func(entry *mempoolEntry) { entry.weight++ }),
			want:   "mempool snapshot entry weight mismatch",
		},
		{
			name:   "size_mismatch",
			mutate: withEditedFirst(func(entry *mempoolEntry) { entry.size = len(entry.raw) + 1 }),
			want:   "mempool snapshot entry size mismatch",
		},
		{
			name:   "malformed_raw",
			mutate: withEditedFirst(func(entry *mempoolEntry) { entry.raw, entry.size = []byte{0xde, 0xad}, 2 }),
			want:   "invalid mempool snapshot entry raw",
		},
		{
			name: "trailing_bytes",
			mutate: withEditedFirst(func(entry *mempoolEntry) {
				entry.raw = append(entry.raw, 0)
				entry.size = len(entry.raw)
			}),
			want: "mempool snapshot entry has trailing bytes",
		},
		{
			name: "txid_mismatch",
			mutate: withEditedFirst(func(entry *mempoolEntry) {
				entry.txid[0] ^= 0x01
			}),
			want: "mempool snapshot entry txid mismatch",
		},
		{
			name: "wtxid_mismatch",
			mutate: withEditedFirst(func(entry *mempoolEntry) {
				entry.wtxid[0] ^= 0x01
			}),
			want: "mempool snapshot entry wtxid mismatch",
		},
		{
			name:   "zero_admission_seq",
			mutate: withEditedFirst(func(entry *mempoolEntry) { entry.admissionSeq = 0 }),
			want:   "invalid mempool snapshot entry admission_seq",
		},
		{
			name:   "invalid_source",
			mutate: withEditedFirst(func(entry *mempoolEntry) { entry.source = "sidecar" }),
			want:   "invalid mempool snapshot entry source",
		},
		{
			name:   "input_count_mismatch",
			mutate: withEditedFirst(func(entry *mempoolEntry) { entry.inputs = nil }),
			want:   "mempool snapshot entry input count mismatch",
		},
		{
			name: "input_mismatch",
			mutate: withEditedFirst(func(entry *mempoolEntry) {
				entry.inputs[0].Vout++
			}),
			want: "mempool snapshot entry input mismatch",
		},
		{
			name: "duplicate_txid",
			mutate: func(base mempoolSnapshot) mempoolSnapshot {
				bad := cloneSnapshotForTest(base)
				bad.entries = append(bad.entries, bad.entries[0])
				return bad
			},
			want: "duplicate mempool snapshot txid",
		},
		{
			name: "duplicate_admission_seq",
			mutate: func(base mempoolSnapshot) mempoolSnapshot {
				bad := cloneSnapshotForTest(base)
				duplicate := snapshotEntry(txSecond, txSecondID, []consensus.Outpoint{outpoints[1]})
				duplicate.admissionSeq = bad.entries[0].admissionSeq
				bad.entries = append(bad.entries, duplicate)
				return bad
			},
			want: "duplicate mempool snapshot admission_seq",
		},
		{
			name: "duplicate_wtxid",
			mutate: func(base mempoolSnapshot) mempoolSnapshot {
				bad := cloneSnapshotForTest(base)
				duplicate := snapshotEntry(txSecond, txSecondID, []consensus.Outpoint{outpoints[1]})
				duplicate.wtxid = bad.entries[0].wtxid
				bad.entries = append(bad.entries, duplicate)
				return bad
			},
			want: "duplicate mempool snapshot wtxid",
		},
		{
			name: "admission_high_watermark_below_entry_max",
			mutate: func(base mempoolSnapshot) mempoolSnapshot {
				bad := cloneSnapshotForTest(base)
				bad.lastAdmissionSeq = bad.entries[0].admissionSeq - 1
				return bad
			},
			want: "mempool snapshot admission high-watermark below restored max",
		},
		{
			// The former standalone spenders map is gone, so a snapshot
			// double-spend is now caught by the record/claim bijection: the
			// second record over an already-claimed outpoint has no finalized
			// standard claim of its own, and the owner is the sole authority.
			name: "double_spending_record_without_claim",
			mutate: func(base mempoolSnapshot) mempoolSnapshot {
				bad := cloneSnapshotForTest(base)
				bad.entries = append(bad.entries, snapshotEntry(txDoubleSpend, doubleSpendID, []consensus.Outpoint{outpoints[0]}))
				bad.lastAdmissionSeq = bad.entries[len(bad.entries)-1].admissionSeq
				return bad
			},
			want: "zero or foreign pending-outpoint token",
		},
		{
			name: "aggregate_count_over_cap",
			configure: func(m *Mempool) {
				m.maxTxs = 1
			},
			mutate: func(base mempoolSnapshot) mempoolSnapshot {
				bad := cloneSnapshotForTest(base)
				bad.entries = append(bad.entries, snapshotEntry(txSecond, txSecondID, []consensus.Outpoint{outpoints[1]}))
				return bad
			},
			want: "mempool snapshot exceeds transaction cap",
		},
		{
			name: "aggregate_bytes_over_cap",
			configure: func(m *Mempool) {
				m.maxBytes = len(txBytes) + len(txSecond) - 1
			},
			mutate: func(base mempoolSnapshot) mempoolSnapshot {
				bad := cloneSnapshotForTest(base)
				bad.entries = append(bad.entries, snapshotEntry(txSecond, txSecondID, []consensus.Outpoint{outpoints[1]}))
				return bad
			},
			want: "mempool snapshot exceeds byte cap",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mp.maxTxs = 10
			mp.maxBytes = len(txBytes) + len(txSecond)
			if tc.configure != nil {
				tc.configure(mp)
			}
			bad := tc.mutate(snapshot)
			if err := installMempoolImageForTest(mp, bad); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected %q rejection, got %v", tc.want, err)
			}
			if got := mp.Len(); got != 1 {
				t.Fatalf("mempool len=%d, want 1 after rejected restore", got)
			}
			if !mp.Contains(wantTxID) {
				t.Fatalf("rejected restore removed existing tx %x", wantTxID)
			}
			if mp.usedBytes != wantBytes {
				t.Fatalf("usedBytes=%d, want %d after rejected restore", mp.usedBytes, wantBytes)
			}
		})
	}
}

func TestRestoreMempoolSnapshotAllowsExactCapacityBoundary(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
	tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
	source, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
		MaxTransactions: 2,
		MaxBytes:        len(tx1) + len(tx2),
	})
	if err != nil {
		t.Fatalf("new source mempool: %v", err)
	}
	if err := source.AddTx(tx1); err != nil {
		t.Fatalf("source AddTx(tx1): %v", err)
	}
	if err := source.AddTx(tx2); err != nil {
		t.Fatalf("source AddTx(tx2): %v", err)
	}
	snapshot, err := snapshotMempool(source)
	if err != nil {
		t.Fatalf("snapshotMempool: %v", err)
	}

	// Restore is SAME-owner restore: the snapshot's tokens belong to this
	// mempool's owner, so it is restored into the mempool it came from after
	// its records were cleared. A second Mempool has a second owner and every
	// token in this snapshot would be foreign to it.
	source.mu.Lock()
	for _, txBytes := range [][]byte{tx1, tx2} {
		if err := source.removeTxLocked(txID(t, txBytes)); err != nil {
			source.mu.Unlock()
			t.Fatalf("removeTxLocked: %v", err)
		}
	}
	source.mu.Unlock()
	if got := source.Len(); got != 0 {
		t.Fatalf("mempool len=%d after clearing, want 0", got)
	}

	if err := installMempoolImageForTest(source, snapshot); err != nil {
		t.Fatalf("installMempoolImageForTest exact boundary: %v", err)
	}
	if got := source.Len(); got != 2 {
		t.Fatalf("mempool len=%d, want 2", got)
	}
	if source.usedBytes != len(tx1)+len(tx2) {
		t.Fatalf("usedBytes=%d, want %d", source.usedBytes, len(tx1)+len(tx2))
	}
}

func TestMempoolAddTxHeightOverflow(t *testing.T) {
	st := &ChainState{HasTip: true, Height: ^uint64(0)} // MaxUint64
	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	err = mp.AddTx([]byte{0x01})
	if err == nil {
		t.Fatal("expected error for height overflow")
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) {
		t.Fatalf("expected TxAdmitError, got %T: %v", err, err)
	}
	if txErr.Kind != TxAdmitUnavailable {
		t.Fatalf("expected TxAdmitUnavailable, got %v", txErr.Kind)
	}
}

// residentClaim returns the resident entry for txid together with the live
// owner claim behind its exact token, so a test asserts the record/claim
// binding rather than the record alone.
func residentClaim(t *testing.T, mp *Mempool, txid [32]byte) (*mempoolEntry, *pendingOutpointClaim) {
	t.Helper()
	mp.mu.Lock()
	defer mp.mu.Unlock()
	entry, ok := mp.txs[txid]
	if !ok {
		t.Fatalf("no resident entry for %x", txid)
	}
	owner := mp.pendingOutpoints
	owner.mu.Lock()
	defer owner.mu.Unlock()
	return entry, owner.byToken[entry.token]
}

func ownerCounts(mp *Mempool) (outpoints int, claims int, highWater uint64) {
	owner := mp.pendingOutpoints
	owner.mu.Lock()
	defer owner.mu.Unlock()
	return len(owner.byOutpoint), len(owner.byToken), owner.tokenHighWater
}

// TestMempoolPendingOutpointAdmissionFinalizesExactlyOneToken proves the
// accepted admission row on the PUBLIC path: every input-bearing candidate
// reserves and finalizes exactly one standard token from the single owner
// bound at construction, the claim carries the entry's exact ordered inputs,
// and independent outpoints admit without interfering. It also pins the
// restart row: a fresh Mempool has a fresh owner for which every prior token
// is foreign.
func TestMempoolPendingOutpointAdmissionFinalizesExactlyOneToken(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	owner := mp.PendingOutpointOwner()
	if owner == nil || owner != mp.pendingOutpoints {
		t.Fatalf("PendingOutpointOwner()=%p, want the single bound owner %p", owner, mp.pendingOutpoints)
	}

	txs := [][]byte{
		mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress),
		mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 100_000, 2, fromKey, fromAddress, toAddress),
	}
	for i, txBytes := range txs {
		if err := mp.AddTx(txBytes); err != nil {
			t.Fatalf("AddTx(%d): %v", i, err)
		}
		entry, claim := residentClaim(t, mp, txID(t, txBytes))
		if entry.token.owner != owner || entry.token.seq != uint64(i+1) {
			t.Fatalf("entry %d token=(owner=%p,seq=%d), want seq %d from the bound owner", i, entry.token.owner, entry.token.seq, i+1)
		}
		if claim == nil || !claim.finalized || claim.domain != PendingOutpointStandardMempool || claim.txid != entry.txid {
			t.Fatalf("entry %d claim=%+v, want a finalized standard claim for %x", i, claim, entry.txid)
		}
		if !reflect.DeepEqual(claim.inputs, entry.inputs) {
			t.Fatalf("entry %d claim inputs=%v, want the entry inputs %v", i, claim.inputs, entry.inputs)
		}
		if got, ok := owner.txidForOutpoint(outpoints[i]); !ok || got != entry.txid {
			t.Fatalf("entry %d index row=(%x,%v), want %x", i, got, ok, entry.txid)
		}
	}
	gotOutpoints, gotClaims, gotHighWater := ownerCounts(mp)
	if gotOutpoints != 2 || gotClaims != 2 || gotHighWater != 2 {
		t.Fatalf("owner state=(outpoints=%d claims=%d high_water=%d), want 2/2/2", gotOutpoints, gotClaims, gotHighWater)
	}

	// Restart is not same-owner restore: a fresh Mempool gets a fresh owner and
	// every token the previous owner issued is foreign to it.
	restarted, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("restart mempool: %v", err)
	}
	if restarted.pendingOutpoints == owner {
		t.Fatal("restart reused the previous owner")
	}
	entry, _ := residentClaim(t, mp, txID(t, txs[0]))
	if got := testOwnerKind(t, restarted.pendingOutpoints.Release(entry.token)); got != PendingOutpointInternal {
		t.Fatalf("prior-owner token on the restarted owner kind=%d, want internal", got)
	}
	if gotOutpoints, gotClaims, gotHighWater := ownerCounts(restarted); gotOutpoints != 0 || gotClaims != 0 || gotHighWater != 0 {
		t.Fatalf("restarted owner state=(%d,%d,%d), want an empty fresh owner", gotOutpoints, gotClaims, gotHighWater)
	}
}

// TestMempoolPendingOutpointInputlessEntryHoldsZeroTokenAndNoClaim pins the
// input-less row: such a candidate performs the same no-op at the conflict slot
// the pre-owner loop did, consumes no sequence, carries the zero token and
// creates no claim — while an input-bearing sibling in the same mempool still
// holds its exact finalized claim. Terminal removal accepts both shapes.
func TestMempoolPendingOutpointInputlessEntryHoldsZeroTokenAndNoClaim(t *testing.T) {
	op := consensus.Outpoint{Txid: [32]byte{0x01}, Vout: 2}
	inputless := &mempoolEntry{txid: [32]byte{0x0a}, wtxid: [32]byte{0x0b}, fee: consensus.Uint128FromU64(1), weight: 1, size: 1}
	spending := &mempoolEntry{txid: [32]byte{0x0c}, wtxid: [32]byte{0x0d}, inputs: []consensus.Outpoint{op}, fee: consensus.Uint128FromU64(1), weight: 1, size: 1}

	mp := &Mempool{maxTxs: 10, maxBytes: 100}
	mp.mu.Lock()
	defer mp.mu.Unlock()
	for _, entry := range []*mempoolEntry{inputless, spending} {
		if err := mp.addEntryLocked(entry); err != nil {
			t.Fatalf("addEntryLocked(%x): %v", entry.txid, err)
		}
	}

	var zero PendingOutpointToken
	if inputless.token != zero {
		t.Fatalf("input-less entry token=%+v, want the zero token", inputless.token)
	}
	owner := mp.pendingOutpoints
	if spending.token == zero || spending.token.seq != 1 || owner.tokenHighWater != 1 {
		t.Fatalf("input-less admission consumed a sequence: spending seq=%d high_water=%d", spending.token.seq, owner.tokenHighWater)
	}
	if len(owner.byToken) != 1 || len(owner.byOutpoint) != 1 {
		t.Fatalf("owner holds claims=%d outpoints=%d, want exactly the spending entry's", len(owner.byToken), len(owner.byOutpoint))
	}
	if claim := owner.byToken[spending.token]; claim == nil || !claim.finalized {
		t.Fatalf("spending claim=%+v, want finalized", claim)
	}

	// The typed delta accepts the input-less shape on the terminal path too.
	if err := mp.removeTxLocked(inputless.txid); err != nil {
		t.Fatalf("removeTxLocked(input-less): %v", err)
	}
	if err := mp.removeTxLocked(spending.txid); err != nil {
		t.Fatalf("removeTxLocked(spending): %v", err)
	}
	if len(mp.txs) != 0 || len(owner.byToken) != 0 || len(owner.byOutpoint) != 0 {
		t.Fatalf("after removal records=%d claims=%d outpoints=%d, want all empty", len(mp.txs), len(owner.byToken), len(owner.byOutpoint))
	}
	if owner.tokenHighWater != 1 {
		t.Fatalf("token high-water=%d after removal, want the consumed sequence retained", owner.tokenHighWater)
	}
}

// TestMempoolPendingOutpointCapacityReplacementReleasesVictimTokens proves the
// capacity row: the evicting admission releases every exact victim token and
// installs plus finalizes the candidate in ONE typed delta, so no victim's
// outpoint survives its record and no sequence is reused.
func TestMempoolPendingOutpointCapacityReplacementReleasesVictimTokens(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000, 1_000_000})

	mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 2, MaxBytes: 1 << 20})
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txLow := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	txHigh := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
	txBest := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[2]}, 100_000, 300_000, 3, fromKey, fromAddress, toAddress)
	for _, txBytes := range [][]byte{txLow, txHigh} {
		if err := mp.AddTx(txBytes); err != nil {
			t.Fatalf("AddTx(setup %x): %v", txID(t, txBytes), err)
		}
	}
	victimEntry, _ := residentClaim(t, mp, txID(t, txLow))
	victimToken := victimEntry.token

	if err := mp.AddTx(txBest); err != nil {
		t.Fatalf("AddTx(best): %v", err)
	}
	if mp.Contains(txID(t, txLow)) {
		t.Fatal("victim survived capacity replacement")
	}
	owner := mp.PendingOutpointOwner()
	if _, ok := owner.txidForOutpoint(outpoints[0]); ok {
		t.Fatal("victim outpoint is still claimed after its record was evicted")
	}
	owner.mu.Lock()
	victimClaim := owner.byToken[victimToken]
	owner.mu.Unlock()
	if victimClaim != nil {
		t.Fatalf("victim claim %+v survived its record", victimClaim)
	}
	// An exact retry of the victim release is harmless and leaves no tombstone.
	if err := owner.Release(victimToken); err != nil {
		t.Fatalf("exact victim Release retry: %v", err)
	}
	for _, txBytes := range [][]byte{txHigh, txBest} {
		entry, claim := residentClaim(t, mp, txID(t, txBytes))
		if claim == nil || !claim.finalized || claim.txid != entry.txid {
			t.Fatalf("survivor %x claim=%+v, want its own finalized claim", entry.txid, claim)
		}
	}
	gotOutpoints, gotClaims, gotHighWater := ownerCounts(mp)
	if gotOutpoints != 2 || gotClaims != 2 || gotHighWater != 3 {
		t.Fatalf("owner state=(outpoints=%d claims=%d high_water=%d), want 2/2/3", gotOutpoints, gotClaims, gotHighWater)
	}
}

// TestMempoolPendingOutpointBlockCleanupReleasesExactTokens pins the terminal
// block rows: a connected block releases the exact tokens of both the entries
// it includes and the entries it conflicts with, and the public eviction entry
// point does the same for inclusion alone.
func TestMempoolPendingOutpointBlockCleanupReleasesExactTokens(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	included := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	resident := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 100_000, 2, fromKey, fromAddress, toAddress)
	// conflicting spends the same outpoint as resident but never enters the pool.
	conflicting := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 90_000, 100_000, 3, fromKey, fromAddress, toAddress)
	for _, txBytes := range [][]byte{included, resident} {
		if err := mp.AddTx(txBytes); err != nil {
			t.Fatalf("AddTx(%x): %v", txID(t, txBytes), err)
		}
	}

	parsed, err := consensus.ParseBlockBytes(buildMultiTxBlock(t, [32]byte{}, consensus.POW_LIMIT, 1, included, conflicting))
	if err != nil {
		t.Fatalf("ParseBlockBytes: %v", err)
	}
	if err := mp.EvictConfirmedParsed(parsed); err != nil {
		t.Fatalf("EvictConfirmedParsed: %v", err)
	}
	if err := mp.RemoveConflictingParsed(parsed); err != nil {
		t.Fatalf("RemoveConflictingParsed: %v", err)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0 after inclusion plus conflict cleanup", got)
	}
	gotOutpoints, gotClaims, gotHighWater := ownerCounts(mp)
	if gotOutpoints != 0 || gotClaims != 0 {
		t.Fatalf("owner still holds outpoints=%d claims=%d after cleanup", gotOutpoints, gotClaims)
	}
	if gotHighWater != 2 {
		t.Fatalf("token high-water=%d after cleanup, want the two consumed sequences retained", gotHighWater)
	}

	// Public EvictConfirmedParsed releases the exact token of an included entry.
	if err := mp.AddTx(resident); err != nil {
		t.Fatalf("AddTx(re-admit): %v", err)
	}
	entry, _ := residentClaim(t, mp, txID(t, resident))
	if entry.token.seq != 3 {
		t.Fatalf("re-admitted seq=%d, want 3 (no sequence reuse)", entry.token.seq)
	}
	evictBlock, err := consensus.ParseBlockBytes(buildSingleTxBlock(t, [32]byte{}, consensus.POW_LIMIT, 1, resident))
	if err != nil {
		t.Fatalf("ParseBlockBytes(evict): %v", err)
	}
	if err := mp.EvictConfirmedParsed(evictBlock); err != nil {
		t.Fatalf("EvictConfirmedParsed: %v", err)
	}
	if gotOutpoints, gotClaims, _ := ownerCounts(mp); gotOutpoints != 0 || gotClaims != 0 {
		t.Fatalf("owner still holds outpoints=%d claims=%d after eviction", gotOutpoints, gotClaims)
	}
}

// TestMempoolSnapshotPendingOutpointRoundTripRestoresExactTokens proves the
// same-owner rollback row: a restore rebuilds every record with its exact token
// and finalized claim, discards state admitted after the snapshot, and leaves
// both high-waters at their advanced values so no sequence is ever reused.
func TestMempoolSnapshotPendingOutpointRoundTripRestoresExactTokens(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	kept := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(kept); err != nil {
		t.Fatalf("AddTx(kept): %v", err)
	}
	keptEntry, _ := residentClaim(t, mp, txID(t, kept))
	keptToken := keptEntry.token

	snapshot, err := snapshotMempool(mp)
	if err != nil {
		t.Fatalf("snapshotMempool: %v", err)
	}
	discarded := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 100_000, 2, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(discarded); err != nil {
		t.Fatalf("AddTx(discarded): %v", err)
	}
	if err := installMempoolImageForTest(mp, snapshot); err != nil {
		t.Fatalf("installMempoolImageForTest: %v", err)
	}

	if mp.Contains(txID(t, discarded)) {
		t.Fatal("restore kept a record admitted after the snapshot")
	}
	restored, claim := residentClaim(t, mp, txID(t, kept))
	if restored.token != keptToken {
		t.Fatalf("restored token seq=%d, want the exact pre-snapshot seq %d", restored.token.seq, keptToken.seq)
	}
	if claim == nil || !claim.finalized || claim.txid != restored.txid {
		t.Fatalf("restored claim=%+v, want the exact finalized claim", claim)
	}
	if _, ok := mp.PendingOutpointOwner().txidForOutpoint(outpoints[1]); ok {
		t.Fatal("restore left the discarded candidate's outpoint claimed")
	}
	gotOutpoints, gotClaims, gotHighWater := ownerCounts(mp)
	if gotOutpoints != 1 || gotClaims != 1 {
		t.Fatalf("restored owner state=(outpoints=%d claims=%d), want 1/1", gotOutpoints, gotClaims)
	}
	if gotHighWater != 2 {
		t.Fatalf("token high-water=%d after restore, want the advanced value 2 retained", gotHighWater)
	}
	// The retained high-water is what stops a reused sequence after the abort.
	next := mustReserve(t, mp.PendingOutpointOwner(), [32]byte{0xfe}, testOutpoint(77))
	if next.seq != 3 {
		t.Fatalf("post-restore reservation seq=%d, want 3", next.seq)
	}
}

// TestMempoolSnapshotPendingOutpointRejectsBrokenClaimBinding pins all four
// snapshot rejection rows: a record whose token has no finalized standard claim,
// a standard claim with no record, an input-less record carrying a token, and a
// claim whose inputs disagree with its record. A refused restore publishes
// NEITHER half — records and owner claims both stay exactly pre-restore.
func TestMempoolSnapshotPendingOutpointRejectsBrokenClaimBinding(t *testing.T) {
	base := func(t *testing.T) (*Mempool, mempoolSnapshot, [32]byte) {
		t.Helper()
		fromKey := mustNodeMLDSA87Keypair(t)
		toKey := mustNodeMLDSA87Keypair(t)
		fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
		toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
		st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
		mp, err := NewMempool(st, nil, devnetGenesisChainID)
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
		if err := mp.AddTx(txBytes); err != nil {
			t.Fatalf("AddTx: %v", err)
		}
		snapshot, err := snapshotMempool(mp)
		if err != nil {
			t.Fatalf("snapshotMempool: %v", err)
		}
		return mp, snapshot, txID(t, txBytes)
	}
	cases := map[string]func(*mempoolSnapshot){
		"record without claim": func(s *mempoolSnapshot) { s.pending.claims = nil },
		"claim without record": func(s *mempoolSnapshot) { s.entries = nil },
		"token on input-less record": func(s *mempoolSnapshot) {
			s.entries[0].inputs = nil
			s.pending.claims = nil
		},
		"claim input mismatch": func(s *mempoolSnapshot) {
			s.pending.claims[0].inputs = []consensus.Outpoint{testOutpoint(123)}
		},
	}
	for name, corrupt := range cases {
		t.Run(name, func(t *testing.T) {
			mp, snapshot, txid := base(t)
			beforeLen := mp.Len()
			mp.mu.RLock()
			claimedOutpoint := mp.txs[txid].inputs[0]
			mp.mu.RUnlock()
			beforeOutpoints, beforeClaims, beforeHighWater := ownerCounts(mp)
			corrupt(&snapshot)
			if err := installMempoolImageForTest(mp, snapshot); err == nil {
				t.Fatalf("restore accepted a %s snapshot", name)
			}
			if mp.Len() != beforeLen || !mp.Contains(txid) {
				t.Fatalf("refused restore published records: len=%d contains=%v", mp.Len(), mp.Contains(txid))
			}
			// The OWNER half too: publishing the snapshot image while the records
			// stay pre-restore is exactly the torn state this ordering forbids.
			gotOutpoints, gotClaims, gotHighWater := ownerCounts(mp)
			if gotOutpoints != beforeOutpoints || gotClaims != beforeClaims || gotHighWater != beforeHighWater {
				t.Fatalf("refused restore published owner state=(%d,%d,%d), want (%d,%d,%d)",
					gotOutpoints, gotClaims, gotHighWater, beforeOutpoints, beforeClaims, beforeHighWater)
			}
			if got, ok := mp.PendingOutpointOwner().txidForOutpoint(claimedOutpoint); !ok || got != txid {
				t.Fatalf("refused restore rewrote the owner index: got=(%x,%v), want %x", got, ok, txid)
			}
		})
	}
}

func TestMempoolAddTxBlockMTPError(t *testing.T) {
	// Empty blockStore + non-zero height → prevTimestampsFromStore fails.
	dir := t.TempDir()
	store := mustCreateBlockStore(t, BlockStorePath(dir))
	st := &ChainState{HasTip: true, Height: 50}
	mp, err := NewMempool(st, store, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	err = mp.AddTx([]byte{0x01})
	if err == nil {
		t.Fatal("expected error for missing block timestamps")
	}
	var txErr *TxAdmitError
	if !errors.As(err, &txErr) {
		t.Fatalf("expected TxAdmitError, got %T: %v", err, err)
	}
	if txErr.Kind != TxAdmitUnavailable {
		t.Fatalf("expected TxAdmitUnavailable, got %v", txErr.Kind)
	}
}

func TestMempoolEviction(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}

	block := buildSingleTxBlock(t, [32]byte{}, consensus.POW_LIMIT, 1, txBytes)
	if err := mp.EvictConfirmed(block); err != nil {
		t.Fatalf("EvictConfirmed: %v", err)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
	if got := mp.usedBytes; got != 0 {
		t.Fatalf("usedBytes=%d, want 0", got)
	}
}

func TestMempoolSelectByFee(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txLow := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	txHigh := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 300_000, 2, fromKey, fromAddress, toAddress)
	txMid := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[2]}, 100_000, 200_000, 3, fromKey, fromAddress, toAddress)
	for _, txBytes := range [][]byte{txLow, txHigh, txMid} {
		if err := mp.AddTx(txBytes); err != nil {
			t.Fatalf("AddTx: %v", err)
		}
	}

	selected := mp.SelectTransactions(2, 1<<20)
	if len(selected) != 2 {
		t.Fatalf("selected=%d, want 2", len(selected))
	}
	if got, want := txIDHex(t, selected[0]), txIDHex(t, txHigh); got != want {
		t.Fatalf("selected[0]=%s, want %s", got, want)
	}
	if got, want := txIDHex(t, selected[1]), txIDHex(t, txMid); got != want {
		t.Fatalf("selected[1]=%s, want %s", got, want)
	}
}

func TestMinerMineOneSelectsFromMempool(t *testing.T) {
	dir := t.TempDir()
	store := mustCreateBlockStore(t, BlockStorePath(dir))

	var tipHash [32]byte
	for height := uint64(0); height <= 100; height++ {
		hash, _ := mustPutBlock(t, store, height, byte(height), height+1, []byte{byte(height)})
		if height == 100 {
			tipHash = hash
		}
	}

	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
	st.HasTip = true
	st.Height = 100
	st.TipHash = tipHash

	syncEngine, err := NewSyncEngine(st, store, DefaultSyncConfig(nil, devnetGenesisChainID, ChainStatePath(dir)))
	if err != nil {
		t.Fatalf("new sync engine: %v", err)
	}
	mp, err := NewMempool(st, store, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	syncEngine.SetMempool(mp)

	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}

	cfg := DefaultMinerConfig()
	cfg.TimestampSource = func() uint64 { return 124 }
	miner, err := NewMiner(st, store, syncEngine, cfg)
	if err != nil {
		t.Fatalf("new miner: %v", err)
	}
	mined, err := miner.MineOne(context.Background(), nil)
	if err != nil {
		t.Fatalf("MineOne: %v", err)
	}
	if mined.TxCount != 2 {
		t.Fatalf("tx_count=%d, want 2", mined.TxCount)
	}
	if got := mp.Len(); got != 0 {
		t.Fatalf("mempool len=%d, want 0", got)
	}
}

// TestTxAdmitErrorCauseCompatibility pins the cause bridge: the hash branch attaches the
// sentinel, Unwrap exposes only it, same-text errors never match, errors.As still reaches
// the outer pointer, and Error(), Kind and the count buckets are unchanged.
func TestTxAdmitErrorCauseCompatibility(t *testing.T) {
	f := newDANonReplayFixture(t, 1)
	bad := f.signed(daNonReplayTxSpec{kind: 0x02, daID: [32]byte{0xca}, payload: []byte("cause"), chunkHash: [32]byte{0xff}, literalChunkHash: true})
	_, hashErr := f.relay.AdmitDA(bad.raw, publicPeer(t, "cause"))
	require(t, errors.Is(hashErr, ErrDARelayChunkHashMismatch) && hashErr.Error() == "DA chunk payload hash mismatch", "the hash branch did not attach the sentinel: %v", hashErr)
	_, _, _, _, _, siblingErr := parseDAAdmission(bad.raw)
	require(t, siblingErr != nil && siblingErr.Error() == hashErr.Error() && !errors.Is(siblingErr, ErrDARelayChunkHashMismatch), "parseDAAdmission's same-text sibling must stay cause-free: %v", siblingErr)
	require(t, (*TxAdmitError)(nil).Unwrap() == nil, "nil receiver unwraps to %v, want nil", (*TxAdmitError)(nil).Unwrap())
	bare := txAdmitRejected("DA chunk payload hash mismatch")
	wrapped := txAdmitRejected("DA chunk payload hash mismatch")
	wrapped.cause = ErrDARelayChunkHashMismatch
	require(t, bare.Unwrap() == nil && !errors.Is(bare, ErrDARelayChunkHashMismatch), "cause-free same-text error matched the sentinel: unwrap=%v", bare.Unwrap())
	require(t, errors.Is(wrapped, ErrDARelayChunkHashMismatch) && !errors.Is(wrapped, errors.New(ErrDARelayChunkHashMismatch.Error())), "errors.Is must select by sentinel identity, never by text")
	outer := fmt.Errorf("relay: %w", wrapped)
	var admit *TxAdmitError
	require(t, errors.As(outer, &admit) && admit == wrapped && errors.Is(outer, ErrDARelayChunkHashMismatch), "errors.As reached %p, want the outer pointer %p", admit, wrapped)
	require(t, wrapped.Error() == bare.Error() && wrapped.Kind == TxAdmitRejected && wrapped.Message == "DA chunk payload hash mismatch", "cause changed the public rendering: %q %s", wrapped.Error(), wrapped.Kind)
	m := &Mempool{}
	for _, err := range []error{wrapped, bare, txAdmitConflict("c"), txAdmitUnavailable("u"), nil} {
		m.noteAdmissionResult(err)
	}
	require(t, m.AdmissionCounts() == MempoolAdmissionCounts{Accepted: 1, Conflict: 1, Rejected: 2, Unavailable: 1}, "admission counts with a caused error=%+v", m.AdmissionCounts())
}

func mustNodeMLDSA87Keypair(t *testing.T) *consensus.MLDSA87Keypair {
	t.Helper()
	kp, err := consensus.NewMLDSA87Keypair()
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "unsupported") {
			t.Skipf("ML-DSA backend unavailable: %v", err)
		}
		t.Fatalf("NewMLDSA87Keypair: %v", err)
	}
	t.Cleanup(func() { kp.Close() })
	return kp
}

func testSpendableChainState(fromAddress []byte, values []uint64) (*ChainState, []consensus.Outpoint) {
	st := NewChainState()
	st.HasTip = true
	st.Height = 100
	st.TipHash[0] = 0x11
	outpoints := make([]consensus.Outpoint, 0, len(values))
	for i, value := range values {
		var txid [32]byte
		txid[0] = byte(i + 1)
		txid[31] = byte(i + 9)
		op := consensus.Outpoint{Txid: txid, Vout: uint32(i)}
		st.Utxos[op] = consensus.UtxoEntry{
			Value:             value,
			CovenantType:      consensus.COV_TYPE_P2PK,
			CovenantData:      append([]byte(nil), fromAddress...),
			CreationHeight:    1,
			CreatedByCoinbase: true,
		}
		outpoints = append(outpoints, op)
	}
	return st, outpoints
}

func mustBuildSignedTransferTx(
	t *testing.T,
	utxos map[consensus.Outpoint]consensus.UtxoEntry,
	inputs []consensus.Outpoint,
	amount uint64,
	fee uint64,
	nonce uint64,
	signer *consensus.MLDSA87Keypair,
	changeAddress []byte,
	toAddress []byte,
) []byte {
	t.Helper()
	txInputs := make([]consensus.TxInput, 0, len(inputs))
	var totalIn uint64
	for _, op := range inputs {
		entry, ok := utxos[op]
		if !ok {
			t.Fatalf("missing utxo for %x:%d", op.Txid, op.Vout)
		}
		totalIn += entry.Value
		txInputs = append(txInputs, consensus.TxInput{
			PrevTxid: op.Txid,
			PrevVout: op.Vout,
			Sequence: 0,
		})
	}
	change := totalIn - amount - fee
	outputs := []consensus.TxOutput{{
		Value:        amount,
		CovenantType: consensus.COV_TYPE_P2PK,
		CovenantData: append([]byte(nil), toAddress...),
	}}
	if change > 0 {
		outputs = append(outputs, consensus.TxOutput{
			Value:        change,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: append([]byte(nil), changeAddress...),
		})
	}

	tx := &consensus.Tx{
		Version:  1,
		TxKind:   0x00,
		TxNonce:  nonce,
		Inputs:   txInputs,
		Outputs:  outputs,
		Locktime: 0,
	}
	if err := consensus.SignTransaction(tx, utxos, devnetGenesisChainID, signer); err != nil {
		t.Fatalf("SignTransaction: %v", err)
	}
	txBytes, err := consensus.MarshalTx(tx)
	if err != nil {
		t.Fatalf("MarshalTx: %v", err)
	}
	return txBytes
}

func corruptFirstWitnessSignature(t *testing.T, txBytes []byte) []byte {
	t.Helper()
	tx, _, _, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx before corrupt: %v", err)
	}
	if len(tx.Witness) == 0 || len(tx.Witness[0].Signature) == 0 {
		t.Fatal("expected first witness signature")
	}
	tx.Witness[0].Signature[0] ^= 0xFF
	out, err := consensus.MarshalTx(tx)
	if err != nil {
		t.Fatalf("MarshalTx after corrupt: %v", err)
	}
	return out
}

func mustBuildSignedAnchorOutputTx(
	t *testing.T,
	utxos map[consensus.Outpoint]consensus.UtxoEntry,
	input consensus.Outpoint,
	anchorValue uint64,
	fee uint64,
	nonce uint64,
	signer *consensus.MLDSA87Keypair,
	changeAddress []byte,
) []byte {
	t.Helper()
	entry, ok := utxos[input]
	if !ok {
		t.Fatalf("missing utxo for %x:%d", input.Txid, input.Vout)
	}
	var anchorData [32]byte
	anchorData[0] = 0x42
	tx := &consensus.Tx{
		Version: 1,
		TxKind:  0x00,
		TxNonce: nonce,
		Inputs: []consensus.TxInput{{
			PrevTxid: input.Txid,
			PrevVout: input.Vout,
			Sequence: 0,
		}},
		Outputs: []consensus.TxOutput{
			{Value: anchorValue, CovenantType: consensus.COV_TYPE_ANCHOR, CovenantData: anchorData[:]},
			{Value: entry.Value - anchorValue - fee, CovenantType: consensus.COV_TYPE_P2PK, CovenantData: append([]byte(nil), changeAddress...)},
		},
		Locktime: 0,
	}
	if err := consensus.SignTransaction(tx, utxos, devnetGenesisChainID, signer); err != nil {
		t.Fatalf("SignTransaction(anchor): %v", err)
	}
	txBytes, err := consensus.MarshalTx(tx)
	if err != nil {
		t.Fatalf("MarshalTx(anchor): %v", err)
	}
	return txBytes
}

func mustBuildSignedDaCommitTx(
	t *testing.T,
	utxos map[consensus.Outpoint]consensus.UtxoEntry,
	input consensus.Outpoint,
	amount uint64,
	fee uint64,
	nonce uint64,
	signer *consensus.MLDSA87Keypair,
	toAddress []byte,
	manifest []byte,
) []byte {
	t.Helper()
	return mustBuildSignedDaCommitTxWithChunkCount(t, utxos, input, amount, fee, nonce, signer, toAddress, 1, manifest)
}

func mustBuildSignedDaCommitTxWithChunkCount(
	t *testing.T,
	utxos map[consensus.Outpoint]consensus.UtxoEntry,
	input consensus.Outpoint,
	amount uint64,
	fee uint64,
	nonce uint64,
	signer *consensus.MLDSA87Keypair,
	toAddress []byte,
	chunkCount uint16,
	manifest []byte,
) []byte {
	t.Helper()
	tx := &consensus.Tx{
		Version: 1,
		TxKind:  0x01,
		TxNonce: nonce,
		Inputs: []consensus.TxInput{{
			PrevTxid: input.Txid,
			PrevVout: input.Vout,
			Sequence: 0,
		}},
		Outputs: []consensus.TxOutput{{
			Value:        amount,
			CovenantType: consensus.COV_TYPE_P2PK,
			CovenantData: append([]byte(nil), toAddress...),
		}},
		Locktime:  0,
		DaPayload: append([]byte(nil), manifest...),
		DaCommitCore: &consensus.DaCommitCore{
			ChunkCount:  chunkCount,
			BatchNumber: 1,
		},
	}
	if err := consensus.SignTransaction(tx, utxos, devnetGenesisChainID, signer); err != nil {
		t.Fatalf("SignTransaction(da): %v", err)
	}
	txBytes, err := consensus.MarshalTx(tx)
	if err != nil {
		t.Fatalf("MarshalTx(da): %v", err)
	}
	return txBytes
}

func txIDHex(t *testing.T, txBytes []byte) string {
	t.Helper()
	txid := txID(t, txBytes)
	return fmt.Sprintf("%x", txid[:])
}

func txID(t *testing.T, txBytes []byte) [32]byte {
	t.Helper()
	_, txid, _, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx: %v", err)
	}
	return txid
}

func TestTxAdmitErrorKinds(t *testing.T) {
	assertKind := func(t *testing.T, err error, wantKind TxAdmitErrorKind) {
		t.Helper()
		var txErr *TxAdmitError
		if !errors.As(err, &txErr) {
			t.Fatalf("expected *TxAdmitError, got %T: %v", err, err)
		}
		if txErr.Kind != wantKind {
			t.Fatalf("kind=%q, want %q (msg=%q)", txErr.Kind, wantKind, txErr.Message)
		}
	}

	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())

	t.Run("nil mempool", func(t *testing.T) {
		var mp *Mempool
		err := mp.AddTx([]byte{0x00})
		assertKind(t, err, TxAdmitUnavailable)
	})

	t.Run("duplicate tx conflict", func(t *testing.T) {
		st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
		mp, err := NewMempool(st, nil, devnetGenesisChainID)
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		tx := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
		if err := mp.AddTx(tx); err != nil {
			t.Fatalf("first AddTx: %v", err)
		}
		err = mp.AddTx(tx)
		assertKind(t, err, TxAdmitConflict)
	})

	t.Run("double spend conflict", func(t *testing.T) {
		st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
		mp, err := NewMempool(st, nil, devnetGenesisChainID)
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
		tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
		if err := mp.AddTx(tx1); err != nil {
			t.Fatalf("first AddTx: %v", err)
		}
		err = mp.AddTx(tx2)
		assertKind(t, err, TxAdmitConflict)
	})

	t.Run("mempool full unavailable", func(t *testing.T) {
		st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})
		mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 1, MaxBytes: 1 << 20})
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
		tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[1]}, 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
		if err := mp.AddTx(tx1); err != nil {
			t.Fatalf("first AddTx: %v", err)
		}
		err = mp.AddTx(tx2)
		assertKind(t, err, TxAdmitUnavailable)
	})

	t.Run("rolling floor unavailable", func(t *testing.T) {
		st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
		mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{MaxTransactions: 10, MaxBytes: 1 << 20})
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		mp.currentMinFeeRate = 8
		tx := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 1, 1, fromKey, fromAddress, toAddress)
		err = mp.AddTx(tx)
		assertKind(t, err, TxAdmitUnavailable)
	})

	t.Run("invalid tx rejected", func(t *testing.T) {
		st, _ := testSpendableChainState(fromAddress, []uint64{1_000_000})
		mp, err := NewMempool(st, nil, devnetGenesisChainID)
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		// Garbage bytes that fail consensus.CheckTransaction → rejected.
		err = mp.AddTx([]byte{0xDE, 0xAD})
		assertKind(t, err, TxAdmitRejected)
	})
}

func TestTxAdmitErrorMessage(t *testing.T) {
	err := &TxAdmitError{Kind: TxAdmitConflict, Message: "tx already in mempool"}
	if err.Error() != "tx already in mempool" {
		t.Fatalf("Error()=%q, want %q", err.Error(), "tx already in mempool")
	}
}

func TestMempoolAllTxIDsReturnsEveryEntry(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}

	want := make(map[[32]byte]struct{})
	for i := 0; i < 3; i++ {
		txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[i]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
		if err := mp.AddTx(txBytes); err != nil {
			t.Fatalf("AddTx[%d]: %v", i, err)
		}
		_, txid, _, _, err := consensus.ParseTx(txBytes)
		if err != nil {
			t.Fatalf("ParseTx[%d]: %v", i, err)
		}
		want[txid] = struct{}{}
	}

	got := mp.AllTxIDs()
	if len(got) != 3 {
		t.Fatalf("AllTxIDs len=%d, want 3", len(got))
	}
	for _, id := range got {
		if _, ok := want[id]; !ok {
			t.Fatalf("AllTxIDs returned unexpected txid %x", id)
		}
	}
}

func TestMempoolAllTxIDsSortedDeterministic(t *testing.T) {
	// Verify that sorting AllTxIDs produces deterministic lexicographic order;
	// handlers sort the IDs before presenting them.
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000, 1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	var ids [][32]byte
	for i := 0; i < 3; i++ {
		txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[i]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
		if err := mp.AddTx(txBytes); err != nil {
			t.Fatalf("AddTx[%d]: %v", i, err)
		}
		_, txid, _, _, err := consensus.ParseTx(txBytes)
		if err != nil {
			t.Fatalf("ParseTx[%d]: %v", i, err)
		}
		ids = append(ids, txid)
	}
	got := mp.AllTxIDs()
	if len(got) != 3 {
		t.Fatalf("AllTxIDs len=%d, want 3", len(got))
	}
	// Replicate handler sort: lexicographic on hex-encoded txid.
	sort.Slice(got, func(i, j int) bool {
		return hex.EncodeToString(got[i][:]) < hex.EncodeToString(got[j][:])
	})
	sort.Slice(ids, func(i, j int) bool {
		return hex.EncodeToString(ids[i][:]) < hex.EncodeToString(ids[j][:])
	})
	for i := range ids {
		if got[i] != ids[i] {
			t.Fatalf("sorted[%d]: got %x, want %x", i, got[i], ids[i])
		}
	}
}

func TestMempoolAllTxIDsEmpty(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	st, _ := testSpendableChainState(fromAddress, []uint64{100})
	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if got := mp.AllTxIDs(); len(got) != 0 {
		t.Fatalf("AllTxIDs on empty mempool returned %d entries, want 0", len(got))
	}
}

func TestMempoolAllTxIDsNilReceiver(t *testing.T) {
	var mp *Mempool
	if got := mp.AllTxIDs(); got != nil {
		t.Fatalf("AllTxIDs on nil receiver=%v, want nil", got)
	}
}

func TestMempoolTxByIDReturnsRawAndDefensiveCopy(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}
	_, txid, _, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx: %v", err)
	}

	got, ok := mp.TxByID(txid)
	if !ok {
		t.Fatalf("TxByID ok=false, want true")
	}
	if !bytes.Equal(got, txBytes) {
		t.Fatalf("TxByID raw mismatch")
	}

	// Defensive-copy invariant: mutate the returned slice and verify the
	// mempool entry remains intact via a second TxByID call.
	got[0] ^= 0xff
	got2, ok2 := mp.TxByID(txid)
	if !ok2 {
		t.Fatalf("TxByID second call ok=false")
	}
	if !bytes.Equal(got2, txBytes) {
		t.Fatalf("mempool entry mutated by caller — defensive copy broken")
	}
}

func TestMempoolTxByIDMissing(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	st, _ := testSpendableChainState(fromAddress, []uint64{100})
	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	var unknown [32]byte
	raw, ok := mp.TxByID(unknown)
	if ok || raw != nil {
		t.Fatalf("TxByID on unknown txid returned raw=%v ok=%v, want nil,false", raw, ok)
	}
}

func TestMempoolTxByIDNilReceiver(t *testing.T) {
	var mp *Mempool
	var id [32]byte
	raw, ok := mp.TxByID(id)
	if ok || raw != nil {
		t.Fatalf("TxByID on nil receiver returned raw=%v ok=%v, want nil,false", raw, ok)
	}
}

// TestMempoolRetainedTxByID drives the RUB-1166 hostile matrix R0-R11 over the
// atomic retained snapshot: nil and absent rows, every admission source, the
// test-only corruption rows, the defensive copy, every removal and restore path,
// and a concurrent reader racing complete remove/restore cycles.
func TestMempoolRetainedTxByID(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	defaultCfg := MempoolConfig{MaxTransactions: 10, MaxBytes: 1 << 20}
	mustOK := func(t *testing.T, what string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	newPool := func(t *testing.T, cfg MempoolConfig, values ...uint64) (*Mempool, *ChainState, []consensus.Outpoint) {
		t.Helper()
		st, outpoints := testSpendableChainState(fromAddress, values)
		mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, cfg)
		mustOK(t, "new mempool", err)
		return mp, st, outpoints
	}
	buildTx := func(t *testing.T, st *ChainState, op consensus.Outpoint, fee, nonce uint64) []byte {
		t.Helper()
		return mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{op}, 100_000, fee, nonce, fromKey, fromAddress, toAddress)
	}
	// wantHit pins the complete triple against the admitted bytes themselves.
	wantHit := func(t *testing.T, mp *Mempool, raw []byte) RetainedTxSnapshot {
		t.Helper()
		_, txid, wtxid, _, err := consensus.ParseTx(raw)
		mustOK(t, "ParseTx", err)
		got, ok := mp.RetainedTxByID(txid)
		if !ok || got.IndexedTxID != txid || got.AdmissionWTxID != wtxid || !bytes.Equal(got.Raw, raw) {
			t.Fatalf("snapshot=(%x,%x,%x,%v), want (%x,%x,%x,true)", got.IndexedTxID, got.AdmissionWTxID, got.Raw, ok, txid, wtxid, raw)
		}
		return got
	}
	wantMiss := func(t *testing.T, label string, mp *Mempool, key [32]byte) {
		t.Helper()
		got, ok := mp.RetainedTxByID(key)
		if ok || got.IndexedTxID != ([32]byte{}) || got.AdmissionWTxID != ([32]byte{}) || got.Raw != nil {
			t.Fatalf("%s=(%x,%x,%x,%v), want the zero snapshot and false", label, got.IndexedTxID, got.AdmissionWTxID, got.Raw, ok)
		}
	}

	t.Run("R0/R1 nil receiver and absent txid", func(t *testing.T) {
		var nilPool *Mempool
		wantMiss(t, "nil receiver", nilPool, [32]byte{})
		mp, _, _ := newPool(t, defaultCfg, 1_000_000)
		wantMiss(t, "absent txid", mp, [32]byte{0xAB})
	})

	t.Run("R2/R8 every admission source yields the exact triple as a defensive copy", func(t *testing.T) {
		mp, st, outpoints := newPool(t, defaultCfg, 1_000_000, 1_000_000, 1_000_000)
		for i, admit := range []func([]byte) error{mp.AddTx, mp.AddRemoteTx, mp.AddReorgTx} {
			raw := buildTx(t, st, outpoints[i], 300_000, uint64(i+1))
			mustOK(t, "admit", admit(raw))
			mutated := wantHit(t, mp, raw)
			mutated.Raw[0] ^= 0xFF
			wantHit(t, mp, raw)
		}
	})

	t.Run("R3 present nil row is corruption, not absence", func(t *testing.T) {
		key := [32]byte{0x5A}
		mp := &Mempool{txs: map[[32]byte]*mempoolEntry{key: nil}}
		got, ok := mp.RetainedTxByID(key)
		if !ok || got.IndexedTxID != key || got.AdmissionWTxID != ([32]byte{}) || got.Raw != nil {
			t.Fatalf("present nil row=(%x,%x,%x,%v), want (%x, zero, nil, true)", got.IndexedTxID, got.AdmissionWTxID, got.Raw, ok, key)
		}
		wantMiss(t, "absent key in the same pool", mp, [32]byte{0x5B})
	})

	// One row corrupted in every dimension at once: the snapshot must report the
	// primary row's own stored values, unclassified and unrepaired.
	t.Run("R4-R7 corrupted row is reported from the primary index", func(t *testing.T) {
		mp, st, outpoints := newPool(t, defaultCfg, 1_000_000)
		raw := buildTx(t, st, outpoints[0], 100_000, 1)
		mustOK(t, "AddTx", mp.AddTx(raw))
		id := txID(t, raw)
		wantWTxID := [32]byte{0xC0, 0xDE}
		wantRaw := append(append([]byte(nil), raw...), 0xFF)
		mp.mu.Lock()
		entry := mp.txs[id]
		delete(mp.wtxids, entry.wtxid) // R6: the secondary binding goes missing.
		entry.wtxid = wantWTxID        // R4: the stored wtxid no longer matches Raw.
		entry.txid = [32]byte{0x99}    // R5: the entry field disagrees with the index.
		entry.raw = wantRaw            // R7: retained bytes are no longer canonical.
		mp.mu.Unlock()
		got, ok := mp.RetainedTxByID(id)
		if !ok || got.IndexedTxID != id || got.AdmissionWTxID != wantWTxID || !bytes.Equal(got.Raw, wantRaw) {
			t.Fatalf("corrupted row=(%x,%x,%x,%v), want (%x,%x,%x,true)", got.IndexedTxID, got.AdmissionWTxID, got.Raw, ok, id, wantWTxID, wantRaw)
		}
		mp.mu.RLock()
		defer mp.mu.RUnlock()
		if len(mp.wtxids) != 0 {
			t.Fatalf("the read repaired the secondary wtxid index: %d bindings", len(mp.wtxids))
		}
	})

	t.Run("R9 removal makes the row absent", func(t *testing.T) {
		removalCase := func(t *testing.T, label string, cfg MempoolConfig, remove func(*testing.T, *Mempool, *ChainState, []consensus.Outpoint, []byte, mempoolSnapshot)) {
			t.Helper()
			mp, st, outpoints := newPool(t, cfg, 1_000_000, 1_000_000)
			empty, err := snapshotMempool(mp)
			mustOK(t, "snapshotMempool", err)
			raw := buildTx(t, st, outpoints[0], 100_000, 1)
			mustOK(t, "AddTx", mp.AddTx(raw))
			wantHit(t, mp, raw)
			remove(t, mp, st, outpoints, raw, empty)
			wantMiss(t, label, mp, txID(t, raw))
		}
		removalCase(t, "confirmed", defaultCfg, func(t *testing.T, mp *Mempool, _ *ChainState, _ []consensus.Outpoint, raw []byte, _ mempoolSnapshot) {
			mustOK(t, "EvictConfirmed", mp.EvictConfirmed(buildSingleTxBlock(t, [32]byte{}, consensus.POW_LIMIT, 1, raw)))
		})
		removalCase(t, "conflict", defaultCfg, func(t *testing.T, mp *Mempool, st *ChainState, ops []consensus.Outpoint, _ []byte, _ mempoolSnapshot) {
			conflicting := buildTx(t, st, ops[0], 200_000, 2)
			coinbase := coinbaseWithWitnessCommitmentAndP2PKValueAtHeight(t, 1, consensus.BlockSubsidy(1, 0))
			mustOK(t, "RemoveConflicting", mp.RemoveConflicting(buildMultiTxBlock(t, [32]byte{}, consensus.POW_LIMIT, 1, coinbase, conflicting)))
		})
		removalCase(t, "capacity", MempoolConfig{MaxTransactions: 1, MaxBytes: 1 << 20}, func(t *testing.T, mp *Mempool, st *ChainState, ops []consensus.Outpoint, _ []byte, _ mempoolSnapshot) {
			mustOK(t, "AddTx(better fee)", mp.AddTx(buildTx(t, st, ops[1], 400_000, 2)))
		})
		removalCase(t, "empty restore", defaultCfg, func(t *testing.T, mp *Mempool, _ *ChainState, _ []consensus.Outpoint, _ []byte, empty mempoolSnapshot) {
			mustOK(t, "installMempoolImageForTest", installMempoolImageForTest(mp, empty))
		})
	})

	t.Run("R10 restore observations are complete", func(t *testing.T) {
		mp, st, outpoints := newPool(t, defaultCfg, 1_000_000, 1_000_000)
		tx1 := buildTx(t, st, outpoints[0], 200_000, 1)
		mustOK(t, "AddTx(tx1)", mp.AddTx(tx1))
		withTx1, err := snapshotMempool(mp)
		mustOK(t, "snapshotMempool", err)
		tx2 := buildTx(t, st, outpoints[1], 200_000, 2)
		mustOK(t, "AddTx(tx2)", mp.AddTx(tx2))
		mustOK(t, "installMempoolImageForTest", installMempoolImageForTest(mp, withTx1))
		wantHit(t, mp, tx1)
		wantMiss(t, "row dropped by the restore", mp, txID(t, tx2))
		// A rejected restore publishes nothing, so the pre-restore tuple stands.
		broken := withTx1
		broken.lastAdmissionSeq = 0
		if err := installMempoolImageForTest(mp, broken); err == nil {
			t.Fatal("restore accepted an admission high-water below the restored max")
		}
		wantHit(t, mp, tx1)
	})

	t.Run("R11 concurrent remove and restore never yields a mixed tuple", func(t *testing.T) {
		mp, st, outpoints := newPool(t, defaultCfg, 1_000_000)
		raw := buildTx(t, st, outpoints[0], 100_000, 1)
		mustOK(t, "AddTx", mp.AddTx(raw))
		_, id, wtxid, _, err := consensus.ParseTx(raw)
		mustOK(t, "ParseTx", err)
		mp.mu.RLock()
		entryA := mp.txs[id]
		mp.mu.RUnlock()
		wtxidB := wtxid
		wtxidB[0] ^= 0xFF
		rawB := append(append([]byte(nil), raw...), 0xEE)
		entryB := &mempoolEntry{txid: id, wtxid: wtxidB, raw: rawB}
		withRow, err := snapshotMempool(mp)
		mustOK(t, "snapshotMempool", err)
		block := buildSingleTxBlock(t, [32]byte{}, consensus.POW_LIMIT, 1, raw)
		done, stopped := make(chan struct{}), make(chan struct{})
		defer func() { close(done); <-stopped }()
		go func() {
			defer close(stopped)
			for {
				select {
				case <-done:
					return
				default:
				}
				// Absence is a legal observation; a hit must be one whole row.
				if got, ok := mp.RetainedTxByID(id); ok && (got.IndexedTxID != id || (got.AdmissionWTxID != wtxid || !bytes.Equal(got.Raw, raw)) && (got.AdmissionWTxID != wtxidB || !bytes.Equal(got.Raw, rawB))) {
					t.Errorf("mixed tuple: (%x,%x,%x)", got.IndexedTxID, got.AdmissionWTxID, got.Raw)
					return
				}
			}
		}()
		// Public-path cycles kill stale/fabricated hits and prove race-cleanliness.
		for i := 0; i < 200; i++ {
			mustOK(t, "EvictConfirmed", mp.EvictConfirmed(block))
			mustOK(t, "installMempoolImageForTest", installMempoolImageForTest(mp, withRow))
		}
		// Divergent-incarnation swap kills fields assembled across two incarnations.
		incarnations := [2]*mempoolEntry{entryA, entryB}
		for i := 0; i < 200; i++ {
			mp.mu.Lock()
			mp.txs[id] = incarnations[i%2]
			mp.mu.Unlock()
		}
	})
}

func TestMempoolContainsReflectsAdmission(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	_, txid, _, _, err := consensus.ParseTx(txBytes)
	if err != nil {
		t.Fatalf("ParseTx: %v", err)
	}

	if mp.Contains(txid) {
		t.Fatalf("Contains before admit=true, want false")
	}
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}
	if !mp.Contains(txid) {
		t.Fatalf("Contains after admit=false, want true")
	}
	var other [32]byte
	if mp.Contains(other) {
		t.Fatalf("Contains for unrelated txid=true, want false")
	}
}

func TestMempoolContainsNilReceiver(t *testing.T) {
	var mp *Mempool
	var id [32]byte
	if mp.Contains(id) {
		t.Fatalf("Contains on nil receiver=true, want false")
	}
}

// TestMempoolBytesUsedTracksUsedBytes pins the BytesUsed gauge: empty
// mempool reports 0; after a successful AddTx BytesUsed reflects the
// raw transaction byte size accounted in the existing usedBytes field.
// This is the metric scrape source for rubin_node_mempool_bytes.
func TestMempoolBytesUsedTracksUsedBytes(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if got := mp.BytesUsed(); got != 0 {
		t.Fatalf("BytesUsed empty=%d, want 0", got)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}
	if got := mp.BytesUsed(); got != len(txBytes) {
		t.Fatalf("BytesUsed=%d, want %d (raw tx size)", got, len(txBytes))
	}
}

// TestMempoolBytesUsedNilReceiver pins the nil-safety contract used by
// the /metrics rendering path: a nil mempool reports 0 bytes without
// panicking, so the scrape rendering can call BytesUsed unconditionally.
func TestMempoolBytesUsedNilReceiver(t *testing.T) {
	var mp *Mempool
	if got := mp.BytesUsed(); got != 0 {
		t.Fatalf("BytesUsed nil receiver=%d, want 0", got)
	}
}

// TestMempoolAdmissionCountsAcceptedBumpsExactlyOnce pins that a happy
// AddTx call increments only the Accepted bucket of the admission
// counters and leaves the other three buckets at zero.
func TestMempoolAdmissionCountsAcceptedBumpsExactlyOnce(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if got := mp.AdmissionCounts(); got != (MempoolAdmissionCounts{}) {
		t.Fatalf("AdmissionCounts pre-AddTx=%+v, want zero", got)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}
	got := mp.AdmissionCounts()
	if got.Accepted != 1 || got.Conflict != 0 || got.Rejected != 0 || got.Unavailable != 0 {
		t.Fatalf("AdmissionCounts after accepted AddTx=%+v, want only Accepted=1", got)
	}
}

// TestMempoolAdmissionCountsConflictBumpsExactlyOnce pins that a
// duplicate-txid AddTx call routes to the Conflict bucket. The first
// AddTx accepts; the second AddTx with the same bytes hits the
// validateAdmissionLocked duplicate-spender path which returns
// txAdmitConflict.
func TestMempoolAdmissionCountsConflictBumpsExactlyOnce(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("first AddTx: %v", err)
	}
	dupErr := mp.AddTx(txBytes)
	if dupErr == nil {
		t.Fatalf("duplicate AddTx unexpectedly accepted")
	}
	var admitErr *TxAdmitError
	if !errors.As(dupErr, &admitErr) || admitErr.Kind != TxAdmitConflict {
		t.Fatalf("duplicate AddTx err=%v (kind=%v), want TxAdmitConflict", dupErr, func() any {
			if admitErr != nil {
				return admitErr.Kind
			}
			return "<nil>"
		}())
	}
	got := mp.AdmissionCounts()
	if got.Accepted != 1 {
		t.Fatalf("AdmissionCounts.Accepted=%d, want 1 (first AddTx)", got.Accepted)
	}
	if got.Conflict != 1 || got.Rejected != 0 || got.Unavailable != 0 {
		t.Fatalf("AdmissionCounts after duplicate=%+v, want Conflict=1", got)
	}
}

// TestMempoolAdmissionCountsRejectedBumpsExactlyOnce pins that an
// AddTx call rejected by the parse-time path (here: trailing bytes
// after canonical tx) routes to the Rejected bucket via the
// txAdmitRejected helper inside checkTransactionWithSnapshot.
func TestMempoolAdmissionCountsRejectedBumpsExactlyOnce(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	// Append a trailing byte to force the "trailing bytes after canonical
	// tx" reject path inside checkTransactionWithSnapshot.
	bad := append([]byte{}, txBytes...)
	bad = append(bad, 0x00)
	addErr := mp.AddTx(bad)
	if addErr == nil {
		t.Fatalf("malformed AddTx unexpectedly accepted")
	}
	var admitErr *TxAdmitError
	if !errors.As(addErr, &admitErr) || admitErr.Kind != TxAdmitRejected {
		t.Fatalf("malformed AddTx err=%v, want TxAdmitRejected", addErr)
	}
	got := mp.AdmissionCounts()
	if got.Rejected != 1 || got.Accepted != 0 || got.Conflict != 0 || got.Unavailable != 0 {
		t.Fatalf("AdmissionCounts after malformed=%+v, want Rejected=1", got)
	}
}

// TestMempoolAdmissionCountsUnavailableBumpsExactlyOnce pins that an
// AddTx call hitting the nil-chainstate guard routes to the
// Unavailable bucket. nil-chainstate is the explicit unavailable
// branch documented in AddTx.
func TestMempoolAdmissionCountsUnavailableBumpsExactlyOnce(t *testing.T) {
	mp := &Mempool{} // chainState nil — exercises txAdmitUnavailable("nil chainstate")
	addErr := mp.AddTx([]byte{0x00})
	if addErr == nil {
		t.Fatalf("AddTx on nil-chainstate mempool unexpectedly accepted")
	}
	var admitErr *TxAdmitError
	if !errors.As(addErr, &admitErr) || admitErr.Kind != TxAdmitUnavailable {
		t.Fatalf("AddTx err=%v, want TxAdmitUnavailable", addErr)
	}
	got := mp.AdmissionCounts()
	if got.Unavailable != 1 || got.Accepted != 0 || got.Conflict != 0 || got.Rejected != 0 {
		t.Fatalf("AdmissionCounts after unavailable=%+v, want Unavailable=1", got)
	}
}

// TestMempoolAdmissionCountsNilReceiver pins the nil-safety contract
// used by /metrics rendering: a nil mempool returns the zero-value
// MempoolAdmissionCounts struct without panicking.
func TestMempoolAdmissionCountsNilReceiver(t *testing.T) {
	var mp *Mempool
	if got := mp.AdmissionCounts(); got != (MempoolAdmissionCounts{}) {
		t.Fatalf("AdmissionCounts nil receiver=%+v, want zero struct", got)
	}
}

// TestMempoolStatsNilReceiver pins the nil-safety contract used by
// /metrics rendering: a nil mempool returns counters/sizes 0 and
// MinFeeRate=DefaultMempoolMinFeeRate, mirroring the existing
// CurrentMinFeeRateSnapshot nil-safe convention so /metrics on an
// uninitialized state advertises the baseline floor instead of 0.
// Without panicking either way.
func TestMempoolStatsNilReceiver(t *testing.T) {
	var mp *Mempool
	want := MempoolStats{MinFeeRate: DefaultMempoolMinFeeRate}
	if got := mp.Stats(); got != want {
		t.Fatalf("Stats nil receiver=%+v, want %+v", got, want)
	}
}

// TestMempoolStatsReadsLiveStateNotConfigDefaults asserts that
// MaxBytes / LowWaterBytes / MinFeeRate are read from the mempool's
// current struct fields, not from MempoolConfig defaults. This is the
// Linear "max_bytes, low_water_bytes, and min_fee_rate are reported
// from current mempool state, not hardcoded duplicates" invariant.
func TestMempoolStatsReadsLiveStateNotConfigDefaults(t *testing.T) {
	mp := &Mempool{
		maxTxs:            5,
		maxBytes:          12345,
		lowWaterBytes:     6789,
		currentMinFeeRate: 42,
	}
	got := mp.Stats()
	if got.MaxBytes != 12345 {
		t.Fatalf("MaxBytes=%d, want 12345", got.MaxBytes)
	}
	if got.LowWaterBytes != 6789 {
		t.Fatalf("LowWaterBytes=%d, want 6789", got.LowWaterBytes)
	}
	if got.MinFeeRate != 42 {
		t.Fatalf("MinFeeRate=%d, want 42", got.MinFeeRate)
	}
	if got.TxCount != 0 {
		t.Fatalf("TxCount=%d, want 0", got.TxCount)
	}
	if got.BytesUsed != 0 {
		t.Fatalf("BytesUsed=%d, want 0", got.BytesUsed)
	}
	if got.EvictedResidentTotal != 0 {
		t.Fatalf("EvictedResidentTotal=%d, want 0", got.EvictedResidentTotal)
	}
}

// TestMempoolStatsScrapePurity asserts that two consecutive Stats()
// calls observe the same EvictedResidentTotal: reading the snapshot
// must not mutate any underlying counter or gauge. This protects the
// /metrics endpoint contract that scraping is a pure observation.
func TestMempoolStatsScrapePurity(t *testing.T) {
	mp := &Mempool{maxTxs: 5, maxBytes: 12345}
	mp.evictedResidentTotal.Store(7)
	first := mp.Stats()
	second := mp.Stats()
	if first != second {
		t.Fatalf("Stats() not pure: first=%+v second=%+v", first, second)
	}
	if first.EvictedResidentTotal != 7 {
		t.Fatalf("EvictedResidentTotal=%d, want 7", first.EvictedResidentTotal)
	}
}

// TestMempoolStatsResidentEvictionIncrementsExactlyOnce force-runs
// the eviction code path: a 1-slot mempool with one resident, then
// addEntryLocked with a higher-fee candidate that displaces the
// resident. EvictedResidentTotal must increase by exactly one.
// Mirrors the Linear invariant "A resident-entry capacity eviction
// increments the eviction counter exactly once."
func TestMempoolStatsResidentEvictionIncrementsExactlyOnce(t *testing.T) {
	mp := &Mempool{maxTxs: 1, maxBytes: 100}
	resident := &mempoolEntry{
		txid:   [32]byte{0x01},
		fee:    consensus.Uint128FromU64(10),
		weight: 1,
		size:   1,
	}
	if err := mp.addEntryLocked(resident); err != nil {
		t.Fatalf("addEntryLocked(resident): %v", err)
	}
	if got := mp.Stats().EvictedResidentTotal; got != 0 {
		t.Fatalf("EvictedResidentTotal after first admit=%d, want 0", got)
	}
	candidate := &mempoolEntry{
		txid:   [32]byte{0x02},
		fee:    consensus.Uint128FromU64(100),
		weight: 1,
		size:   1,
	}
	if err := mp.addEntryLocked(candidate); err != nil {
		t.Fatalf("addEntryLocked(displacing candidate): %v", err)
	}
	if got := mp.Stats().EvictedResidentTotal; got != 1 {
		t.Fatalf("EvictedResidentTotal after eviction=%d, want 1", got)
	}
}

// TestMempoolStatsResidentEvictionIncrementsByNOnMultiTrim asserts
// the "+1 per evicted resident" contract under byte-pressure where
// one admission removes more than one resident. Capacity-trimming
// to low_water_bytes can evict multiple residents in a single
// addEntryLocked call; the counter must track the actual number of
// removed residents, not just "did at least one eviction happen".
// Reviewer P2 finding on PR #1405: the prior single-eviction test
// left this path uncovered.
func TestMempoolStatsResidentEvictionIncrementsByNOnMultiTrim(t *testing.T) {
	// Force byte-pressure trim with multiple evictions.
	// maxBytes=20 → defaultMempoolLowWaterBytes(20) = (20/10)*9 = 18.
	// maxTxs=10 keeps the count limit non-binding so every eviction
	// here is byte-pressure-driven, not count-pressure-driven.
	// 4 residents of size=5 each fill the pool to 20 bytes.
	// A candidate of size=10 (fee=1000, evicts the worst residents
	// per the eviction-ordering comparator) admits with target =
	// mempoolBytePressureTarget(18, 10) = 18; capacity-trimming
	// must drop residents until usedBytes+candidateSize <= 18,
	// which after evicting 3 residents (15 bytes residents + 10
	// candidate = 25 still over... actually after 3 evictions one
	// resident remains: 5 bytes + 10 candidate = 15 <= 18 ✓).
	mp := &Mempool{maxTxs: 10, maxBytes: 20}
	residents := []*mempoolEntry{
		{txid: [32]byte{0x01}, fee: consensus.Uint128FromU64(1), weight: 1, size: 5},
		{txid: [32]byte{0x02}, fee: consensus.Uint128FromU64(2), weight: 1, size: 5},
		{txid: [32]byte{0x03}, fee: consensus.Uint128FromU64(3), weight: 1, size: 5},
		{txid: [32]byte{0x04}, fee: consensus.Uint128FromU64(4), weight: 1, size: 5},
	}
	for _, r := range residents {
		if err := mp.addEntryLocked(r); err != nil {
			t.Fatalf("addEntryLocked(resident=%x): %v", r.txid[0], err)
		}
	}
	if got := mp.Stats().EvictedResidentTotal; got != 0 {
		t.Fatalf("EvictedResidentTotal after seeding=%d, want 0", got)
	}
	candidate := &mempoolEntry{
		txid:   [32]byte{0x10},
		fee:    consensus.Uint128FromU64(1000),
		weight: 1,
		size:   10,
	}
	if err := mp.addEntryLocked(candidate); err != nil {
		t.Fatalf("addEntryLocked(displacing candidate): %v", err)
	}
	// Counter must reflect every evicted resident, not just one.
	// After admission: TxCount = (4 - N_evicted) + 1; the test
	// asserts the symmetric counter increment instead of hardcoding
	// N because the exact eviction count depends on the eviction
	// comparator's tie-breaking, but it must equal "len(residents) -
	// (TxCount - 1)" by construction.
	stats := mp.Stats()
	wantEvicted := uint64(len(residents)) - uint64(stats.TxCount-1)
	if stats.EvictedResidentTotal != wantEvicted {
		t.Fatalf("EvictedResidentTotal=%d, want %d (one bump per evicted resident; TxCount=%d, started with %d residents)",
			stats.EvictedResidentTotal, wantEvicted, stats.TxCount, len(residents))
	}
	// Sanity: at least 2 residents must have been evicted to make
	// this a real multi-trim case, distinct from the single-evict
	// test above.
	if stats.EvictedResidentTotal < 2 {
		t.Fatalf("EvictedResidentTotal=%d, want >=2 to exercise multi-resident trim",
			stats.EvictedResidentTotal)
	}
}

// TestMempoolStatsCandidateWorstRejectionDoesNotCount asserts that
// rejecting an incoming candidate at capacity (because it is the
// worst entry per eviction ordering) MUST NOT increment the
// resident-eviction counter. No resident is removed in that path.
// Mirrors Linear "Candidate rejection at capacity ... is not counted
// as a resident eviction."
func TestMempoolStatsCandidateWorstRejectionDoesNotCount(t *testing.T) {
	mp := &Mempool{maxTxs: 1, maxBytes: 100}
	resident := &mempoolEntry{
		txid:   [32]byte{0x10},
		fee:    consensus.Uint128FromU64(100),
		weight: 1,
		size:   1,
	}
	if err := mp.addEntryLocked(resident); err != nil {
		t.Fatalf("addEntryLocked(resident): %v", err)
	}
	worstCandidate := &mempoolEntry{
		txid:   [32]byte{0x11},
		fee:    consensus.Uint128FromU64(1),
		weight: 1,
		size:   1,
	}
	err := mp.addEntryLocked(worstCandidate)
	if err == nil || !strings.Contains(err.Error(), "mempool capacity candidate rejected by eviction ordering") {
		t.Fatalf("expected candidate-worst rejection, got %v", err)
	}
	if got := mp.Stats().EvictedResidentTotal; got != 0 {
		t.Fatalf("EvictedResidentTotal after candidate-worst rejection=%d, want 0", got)
	}
}

// TestMempoolStatsFeeFloorRejectionDoesNotCount asserts that
// rejecting an incoming candidate below the rolling minimum fee rate
// MUST NOT increment the resident-eviction counter. Fee-floor
// rejection happens in validateFeeFloorLocked before the eviction
// plan is consulted, so no resident is touched. Mirrors Linear
// "fee-floor rejection is not counted as a resident eviction."
func TestMempoolStatsFeeFloorRejectionDoesNotCount(t *testing.T) {
	mp := &Mempool{
		maxTxs:            10,
		maxBytes:          1000,
		currentMinFeeRate: 1000,
	}
	tooCheap := &mempoolEntry{
		txid:   [32]byte{0x21},
		fee:    consensus.Uint128FromU64(1),
		weight: 1,
		size:   1,
	}
	err := mp.addEntryLocked(tooCheap)
	if err == nil || !strings.Contains(err.Error(), "mempool fee below rolling minimum") {
		t.Fatalf("expected fee-floor rejection, got %v", err)
	}
	if got := mp.Stats().EvictedResidentTotal; got != 0 {
		t.Fatalf("EvictedResidentTotal after fee-floor rejection=%d, want 0", got)
	}
}

func TestExtractTxInputsReturnsCorrectOutpoints(t *testing.T) {
	checked := &consensus.CheckedTransaction{
		Tx: &consensus.Tx{
			Inputs: []consensus.TxInput{
				{PrevTxid: [32]byte{0x01}, PrevVout: 7},
				{PrevTxid: [32]byte{0x02}, PrevVout: 3},
				{PrevTxid: [32]byte{0xff}, PrevVout: 0},
			},
		},
	}
	inputs := extractTxInputs(checked)
	if len(inputs) != 3 {
		t.Fatalf("got %d inputs, want 3", len(inputs))
	}
	for i, want := range []struct {
		txid [32]byte
		vout uint32
	}{
		{[32]byte{0x01}, 7},
		{[32]byte{0x02}, 3},
		{[32]byte{0xff}, 0},
	} {
		if inputs[i].Txid != want.txid || inputs[i].Vout != want.vout {
			t.Fatalf("input[%d] = {%x, %d}, want {%x, %d}", i, inputs[i].Txid, inputs[i].Vout, want.txid, want.vout)
		}
	}
}

func TestValidateChainSnapshotRejectsNil(t *testing.T) {
	_, err := validateChainSnapshot(nil)
	if err == nil {
		t.Fatal("expected error for nil snapshot")
	}
	var admit *TxAdmitError
	if !errors.As(err, &admit) || admit.Kind != TxAdmitUnavailable {
		t.Fatalf("expected TxAdmitUnavailable, got %T %v", err, err)
	}
}

// TestMempoolSigCacheOwnerReusesPositiveResultsAcrossValidations pins the one
// mempool-owned positive signature cache on the live validation seam that
// currently routes through it (RelayMetadata -> validateTransactionWithConsensus):
// empty at construction, populated by a successful validation, reused by a later
// validation of the exact same tuple, per-mempool (never shared), and never an
// admission-result cache — a hit publishes no entry and claims no outpoint.
func TestMempoolSigCacheOwnerReusesPositiveResultsAcrossValidations(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	if mp.sigCache == nil {
		t.Fatal("mempool must own a signature cache")
	}
	if mp.sigCache.Len() != 0 || mp.sigCache.Hits() != 0 || mp.sigCache.Misses() != 0 {
		t.Fatalf("cache must be empty at construction: len=%d hits=%d misses=%d",
			mp.sigCache.Len(), mp.sigCache.Hits(), mp.sigCache.Misses())
	}

	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if _, err := mp.RelayMetadata(txBytes); err != nil {
		t.Fatalf("RelayMetadata: %v", err)
	}
	if mp.sigCache.Len() != 1 || mp.sigCache.Misses() != 1 || mp.sigCache.Hits() != 0 {
		t.Fatalf("after first validation: len=%d misses=%d hits=%d, want 1/1/0",
			mp.sigCache.Len(), mp.sigCache.Misses(), mp.sigCache.Hits())
	}

	// A second mempool owns a separate, empty cache.
	other, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool (other): %v", err)
	}
	if other.sigCache == mp.sigCache || other.sigCache.Len() != 0 {
		t.Fatalf("each mempool must own exactly one fresh cache: len=%d", other.sigCache.Len())
	}

	if _, err := mp.RelayMetadata(txBytes); err != nil {
		t.Fatalf("RelayMetadata (repeat): %v", err)
	}
	if mp.sigCache.Hits() != 1 {
		t.Fatalf("repeated validation must hit the cache: hits=%d", mp.sigCache.Hits())
	}
	if mp.sigCache.Len() != 1 {
		t.Fatalf("a hit must not insert: len=%d", mp.sigCache.Len())
	}
	if mp.Len() != 0 || mp.AdmissionCounts() != (MempoolAdmissionCounts{}) {
		t.Fatalf("a cache hit is not an admission: len=%d counts=%+v", mp.Len(), mp.AdmissionCounts())
	}
}

// TestMempoolSigCacheAlternateWitnessWithSameTxidCannotHit covers the hostile
// row: a re-signed transaction has the same txid but different signature bytes,
// so it must MISS and verify on its own.
func TestMempoolSigCacheAlternateWitnessWithSameTxidCannotHit(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	first := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	second := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
	if bytes.Equal(first, second) {
		t.Skip("signature backend is deterministic: no alternate witness representation available")
	}
	_, firstTxid, _, _, err := consensus.ParseTx(first)
	if err != nil {
		t.Fatalf("ParseTx(first): %v", err)
	}
	_, secondTxid, _, _, err := consensus.ParseTx(second)
	if err != nil {
		t.Fatalf("ParseTx(second): %v", err)
	}
	if firstTxid != secondTxid {
		t.Fatalf("txid mismatch: %x vs %x", firstTxid, secondTxid)
	}

	if _, err := mp.RelayMetadata(first); err != nil {
		t.Fatalf("RelayMetadata(first): %v", err)
	}
	if _, err := mp.RelayMetadata(second); err != nil {
		t.Fatalf("RelayMetadata(second): %v", err)
	}
	if mp.sigCache.Hits() != 0 {
		t.Fatalf("alternate witness bytes must not hit: hits=%d", mp.sigCache.Hits())
	}
	if mp.sigCache.Misses() != 2 || mp.sigCache.Len() != 2 {
		t.Fatalf("alternate witness must verify on its own: misses=%d len=%d, want 2/2",
			mp.sigCache.Misses(), mp.sigCache.Len())
	}
}

// TestMempoolSigCacheAddTxPathReusesPositiveResults pins the LIVE admission
// path (AddTx -> addTxWithSource -> checkTransactionWithSnapshot) on the one
// mempool-owned cache. Backend executions are counted by Misses(): the live
// seam calls the backend exactly once per miss and never on a hit, which
// clients/go/consensus/verify_sig_registry_test.go pins directly against the
// backend call counter. So a second admission of the same signed transaction
// that adds a hit and no miss added zero backend calls.
//
// The repeat is rejected as a duplicate, which is the point: a positive cache
// hit is not an admission-result hit — every non-backend check still runs.
func TestMempoolSigCacheAddTxPathReusesPositiveResults(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})

	mp, err := NewMempool(st, nil, devnetGenesisChainID)
	if err != nil {
		t.Fatalf("new mempool: %v", err)
	}
	txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)

	if err := mp.AddTx(txBytes); err != nil {
		t.Fatalf("AddTx: %v", err)
	}
	if mp.sigCache.Misses() != 1 || mp.sigCache.Hits() != 0 || mp.sigCache.Len() != 1 {
		t.Fatalf("first AddTx must miss and insert once: misses=%d hits=%d len=%d, want 1/0/1",
			mp.sigCache.Misses(), mp.sigCache.Hits(), mp.sigCache.Len())
	}

	if err := mp.AddTx(txBytes); err == nil {
		t.Fatal("duplicate AddTx unexpectedly accepted")
	}
	if mp.sigCache.Hits() != 1 {
		t.Fatalf("repeat admission must hit the owner cache: hits=%d, want 1", mp.sigCache.Hits())
	}
	if mp.sigCache.Misses() != 1 {
		t.Fatalf("repeat admission must add zero backend calls: misses=%d, want 1", mp.sigCache.Misses())
	}
	if mp.sigCache.Len() != 1 {
		t.Fatalf("a hit must not insert: len=%d, want 1", mp.sigCache.Len())
	}
	if mp.Len() != 1 {
		t.Fatalf("duplicate must not be admitted twice: len=%d, want 1", mp.Len())
	}
}

// TestMempoolSigCacheNilCacheAddTxMatchesBaseline proves the nil-cache AddTx
// path is exact uncached behavior: same accept, same duplicate rejection text,
// same resident set as the cache-owning mempool over identical state.
func TestMempoolSigCacheNilCacheAddTxMatchesBaseline(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())

	admit := func(t *testing.T, nilCache bool) (string, int, [32]byte) {
		t.Helper()
		st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
		mp, err := NewMempool(st, nil, devnetGenesisChainID)
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		if nilCache {
			mp.sigCache = nil
		}
		txBytes := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{outpoints[0]}, 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
		if err := mp.AddTx(txBytes); err != nil {
			t.Fatalf("AddTx (nilCache=%v): %v", nilCache, err)
		}
		dupErr := mp.AddTx(txBytes)
		if dupErr == nil {
			t.Fatalf("duplicate AddTx unexpectedly accepted (nilCache=%v)", nilCache)
		}
		return dupErr.Error(), mp.Len(), txID(t, txBytes)
	}

	cachedErr, cachedLen, cachedTxid := admit(t, false)
	uncachedErr, uncachedLen, uncachedTxid := admit(t, true)
	if cachedErr != uncachedErr || cachedLen != uncachedLen || cachedTxid != uncachedTxid {
		t.Fatalf("nil cache diverged from baseline: err %q vs %q, len %d vs %d, txid %x vs %x",
			uncachedErr, cachedErr, uncachedLen, cachedLen, uncachedTxid, cachedTxid)
	}
}

// TestMempoolSigCacheWarmCacheNeverBuysAdmission pins the hostile admission
// rows on the LIVE AddTx path: a positive cache hit is never an admission-result
// hit. The conflict, capacity, and fee-floor rows warm the owner cache through
// the validation-only relay seam (which claims no outpoint); the saturation row
// fills the cache directly via the legacy Insert API. All four rows prove the
// conflict, saturation/FIFO, capacity, and fee-floor authorities fire exactly
// as they would with a cold cache.
func TestMempoolSigCacheWarmCacheNeverBuysAdmission(t *testing.T) {
	fromKey := mustNodeMLDSA87Keypair(t)
	toKey := mustNodeMLDSA87Keypair(t)
	fromAddress := consensus.P2PKCovenantDataForPubkey(fromKey.PubkeyBytes())
	toAddress := consensus.P2PKCovenantDataForPubkey(toKey.PubkeyBytes())
	newPool := func(t *testing.T) (*Mempool, *ChainState, []consensus.Outpoint) {
		t.Helper()
		st, outpoints := testSpendableChainState(fromAddress, []uint64{1_000_000})
		mp, err := NewMempool(st, nil, devnetGenesisChainID)
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		return mp, st, outpoints
	}

	// The cache is warmed with the CONFLICTING transaction's own signature, so
	// its verification is a genuine hit — admission is still refused by the
	// pending-outpoint authority with its exact existing message.
	t.Run("conflict", func(t *testing.T) {
		mp, st, ops := newPool(t)
		txA := mustBuildSignedTransferTx(t, st.Utxos, ops[:1], 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
		txB := mustBuildSignedTransferTx(t, st.Utxos, ops[:1], 100_000, 200_000, 2, fromKey, fromAddress, toAddress)
		txidA, txidB := txID(t, txA), txID(t, txB)
		if txidA == txidB {
			t.Fatalf("conflict row needs distinct txids, got %x twice", txidA)
		}
		if _, err := mp.RelayMetadata(txB); err != nil {
			t.Fatalf("RelayMetadata(txB): %v", err)
		}
		if err := mp.AddTx(txA); err != nil {
			t.Fatalf("AddTx(txA): %v", err)
		}
		if mp.sigCache.Len() != 2 || mp.sigCache.Hits() != 0 || mp.sigCache.Misses() != 2 {
			t.Fatalf("warm state: len=%d hits=%d misses=%d, want 2/0/2",
				mp.sigCache.Len(), mp.sigCache.Hits(), mp.sigCache.Misses())
		}
		var txErr *TxAdmitError
		if err := mp.AddTx(txB); !errors.As(err, &txErr) || txErr.Kind != TxAdmitConflict {
			t.Fatalf("AddTx(txB) err=%T %v, want TxAdmitConflict", err, err)
		}
		if want := fmt.Sprintf("mempool double-spend conflict with %x", txidA); txErr.Message != want {
			t.Fatalf("conflict message %q, want %q", txErr.Message, want)
		}
		if mp.Len() != 1 || mp.txs[txidA] == nil || mp.txs[txidB] != nil {
			t.Fatalf("conflicting tx changed the resident set: len=%d", mp.Len())
		}
		if got := mp.AdmissionCounts(); got != (MempoolAdmissionCounts{Accepted: 1, Conflict: 1}) {
			t.Fatalf("admission counts=%+v, want accepted=1 conflict=1", got)
		}
		// B's signature WAS answered from cache (hit, no backend call) and the
		// rejected admission inserted nothing.
		if mp.sigCache.Hits() != 1 || mp.sigCache.Misses() != 2 || mp.sigCache.Len() != 2 {
			t.Fatalf("after conflict: hits=%d misses=%d len=%d, want 1/2/2",
				mp.sigCache.Hits(), mp.sigCache.Misses(), mp.sigCache.Len())
		}
	})

	// Saturation at the production capacity constant is a performance state,
	// never a verdict. The fill uses the exported legacy-domain API: those keys
	// only occupy capacity, they live in a different key domain than the live
	// seam's binding-inclusive keys and can never be mistaken for a result.
	t.Run("saturation", func(t *testing.T) {
		mp, st, ops := newPool(t)
		var digest [32]byte
		for i := 0; i < mempoolSigCacheCapacity; i++ {
			digest[0], digest[1], digest[2] = byte(i), byte(i>>8), byte(i>>16)
			mp.sigCache.Insert(0x01, []byte("saturation-pubkey"), []byte("saturation-sig"), digest)
		}
		if mp.sigCache.Len() != mempoolSigCacheCapacity {
			t.Fatalf("cache not saturated: len=%d, want %d", mp.sigCache.Len(), mempoolSigCacheCapacity)
		}
		txBytes := mustBuildSignedTransferTx(t, st.Utxos, ops[:1], 100_000, 100_000, 1, fromKey, fromAddress, toAddress)
		if err := mp.AddTx(txBytes); err != nil {
			t.Fatalf("saturated cache must never reject admission: %v", err)
		}
		if mp.Len() != 1 || mp.txs[txID(t, txBytes)] == nil {
			t.Fatalf("tx not resident after admission: len=%d", mp.Len())
		}
		// FIFO evicted exactly one older entry, and the newest entry survived:
		// revalidating the same transaction hits with no new backend call.
		if _, err := mp.RelayMetadata(txBytes); err != nil {
			t.Fatalf("RelayMetadata(repeat): %v", err)
		}
		if mp.sigCache.Len() != mempoolSigCacheCapacity || mp.sigCache.Hits() != 1 || mp.sigCache.Misses() != 1 {
			t.Fatalf("under saturation: len=%d hits=%d misses=%d, want %d/1/1",
				mp.sigCache.Len(), mp.sigCache.Hits(), mp.sigCache.Misses(), mempoolSigCacheCapacity)
		}
	})

	// The candidate's own signature is warmed through the validation-only
	// relay seam (a genuine hit target, mirroring "conflict" above), then the
	// mempool is filled so the candidate is the worst-ranked entry by
	// fee/weight. The capacity authority must reject it with its exact
	// existing message and leave the resident set untouched, exactly as it
	// would for a cold candidate.
	t.Run("capacity", func(t *testing.T) {
		st, ops := testSpendableChainState(fromAddress, []uint64{1_000_000, 1_000_000})
		tx1 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{ops[0]}, 100_000, 200_000, 1, fromKey, fromAddress, toAddress)
		tx2 := mustBuildSignedTransferTx(t, st.Utxos, []consensus.Outpoint{ops[1]}, 100_000, 100_000, 2, fromKey, fromAddress, toAddress)
		mp, err := NewMempoolWithConfig(st, nil, devnetGenesisChainID, MempoolConfig{
			MaxTransactions: 10,
			MaxBytes:        len(tx1) + len(tx2) - 1,
		})
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		if err := mp.AddTx(tx1); err != nil {
			t.Fatalf("AddTx(tx1): %v", err)
		}
		if _, err := mp.RelayMetadata(tx2); err != nil {
			t.Fatalf("RelayMetadata(tx2): %v", err)
		}
		if mp.sigCache.Len() != 2 || mp.sigCache.Hits() != 0 || mp.sigCache.Misses() != 2 {
			t.Fatalf("warm state: len=%d hits=%d misses=%d, want 2/0/2",
				mp.sigCache.Len(), mp.sigCache.Hits(), mp.sigCache.Misses())
		}
		before, err := snapshotMempool(mp)
		if err != nil {
			t.Fatalf("snapshot before warm-capacity: %v", err)
		}
		usedBytes := mp.usedBytes
		addErr := mp.AddTx(tx2)
		if addErr == nil {
			t.Fatalf("expected candidate-worst rejection, got nil")
		}
		var txErr *TxAdmitError
		if !errors.As(addErr, &txErr) || txErr.Kind != TxAdmitUnavailable {
			t.Fatalf("warm-capacity err=%v, want TxAdmitUnavailable", addErr)
		}
		if want := "mempool capacity candidate rejected by eviction ordering"; txErr.Message != want {
			t.Fatalf("warm-capacity message %q, want %q", txErr.Message, want)
		}
		after, err := snapshotMempool(mp)
		if err != nil {
			t.Fatalf("snapshot after warm-capacity: %v", err)
		}
		// The capacity reject fires after the conflict slot already reserved
		// (and released) a token for the candidate, exactly like the cold
		// candidate-worst row in TestMempoolCandidateWorstRejectsWithoutMutation.
		assertPostSlotRejectionSnapshot(t, before, after)
		if mp.Len() != 1 || mp.txs[txID(t, tx1)] == nil || mp.txs[txID(t, tx2)] != nil {
			t.Fatalf("warm-capacity candidate changed resident set: len=%d", mp.Len())
		}
		if mp.usedBytes != usedBytes {
			t.Fatalf("warm-capacity usedBytes=%d, want %d", mp.usedBytes, usedBytes)
		}
		// The warm hit answered tx2's signature from cache: one additional
		// hit, zero additional backend executions (misses unchanged), and
		// the hit itself inserted nothing new.
		if mp.sigCache.Hits() != 1 || mp.sigCache.Misses() != 2 || mp.sigCache.Len() != 2 {
			t.Fatalf("after warm-capacity: hits=%d misses=%d len=%d, want 1/2/2",
				mp.sigCache.Hits(), mp.sigCache.Misses(), mp.sigCache.Len())
		}
	})

	// The floor authority has two enforcement points: a cheap pre-signature
	// fast-reject (single-input plain P2PK only, see cheapFeeFloorPrecheck)
	// and the locked post-validation check (validateFeeFloorLockedWithFloor)
	// that runs after every witness is verified. A single-input candidate
	// never reaches the cache on this row: the fast path intercepts it
	// before verification (feePrecheckP2PKInputValue requires
	// len(tx.Inputs)==1 && len(tx.Witness)==1). To exercise a GENUINE warm
	// hit on the floor authority, the candidate spends two outpoints (two
	// witness items), which is outside the fast path's single-witness
	// eligibility and always defers to the full validation + locked floor
	// check below.
	t.Run("floor", func(t *testing.T) {
		st, ops := testSpendableChainState(fromAddress, []uint64{600_000, 600_000})
		txBelowFloor := mustBuildSignedTransferTx(t, st.Utxos, ops, 100_000, 1, 1, fromKey, fromAddress, toAddress)
		mp, err := NewMempool(st, nil, devnetGenesisChainID)
		if err != nil {
			t.Fatalf("new mempool: %v", err)
		}
		// The below-floor rejection message is deterministic here: the tx is
		// built above with a fixed fee, ML-DSA-87 witnesses are fixed-length
		// so the weight is stable, and a fresh mempool's rolling floor is
		// DefaultMempoolMinFeeRate. Compute it once via the same parse/weight
		// helpers the production floor checks use, and pin the exact string.
		parsedFloorTx, _, _, _, err := consensus.ParseTx(txBelowFloor)
		if err != nil {
			t.Fatalf("ParseTx(txBelowFloor): %v", err)
		}
		floorWeight, _, _, err := consensus.TxWeightAndStats(parsedFloorTx)
		if err != nil {
			t.Fatalf("TxWeightAndStats(txBelowFloor): %v", err)
		}
		wantFloorMsg := fmt.Sprintf("mempool fee below rolling minimum: fee=%s weight=%d min_fee_rate=%d",
			consensus.Uint128FromU64(1).String(), floorWeight, DefaultMempoolMinFeeRate)
		// RelayMetadata runs full validation (warming the cache for both
		// witness items) BEFORE its own read-only floor check, so it also
		// rejects below-floor — proving the read-only relay seam never
		// buys admission either, exactly like the AddTx path below.
		if _, relayErr := mp.RelayMetadata(txBelowFloor); relayErr == nil || relayErr.Error() != wantFloorMsg {
			t.Fatalf("RelayMetadata(txBelowFloor) = %v, want %q", relayErr, wantFloorMsg)
		}
		if mp.sigCache.Len() != 2 || mp.sigCache.Hits() != 0 || mp.sigCache.Misses() != 2 {
			t.Fatalf("warm state: len=%d hits=%d misses=%d, want 2/0/2",
				mp.sigCache.Len(), mp.sigCache.Hits(), mp.sigCache.Misses())
		}
		before, err := snapshotMempool(mp)
		if err != nil {
			t.Fatalf("snapshot before warm-floor: %v", err)
		}
		addErr := mp.AddTx(txBelowFloor)
		if addErr == nil {
			t.Fatalf("expected below-floor rejection, got nil")
		}
		var txErr *TxAdmitError
		if !errors.As(addErr, &txErr) || txErr.Kind != TxAdmitUnavailable {
			t.Fatalf("warm-floor err=%v, want TxAdmitUnavailable", addErr)
		}
		if txErr.Message != wantFloorMsg {
			t.Fatalf("warm-floor message = %q, want %q", txErr.Message, wantFloorMsg)
		}
		after, err := snapshotMempool(mp)
		if err != nil {
			t.Fatalf("snapshot after warm-floor: %v", err)
		}
		// The locked floor check runs after reserveEntryInputsLocked already
		// reserved (and released) a token for the two-input candidate.
		assertPostSlotRejectionSnapshot(t, before, after)
		if mp.Len() != 0 || mp.txs[txID(t, txBelowFloor)] != nil {
			t.Fatalf("warm-floor candidate entered resident set: len=%d", mp.Len())
		}
		// Both of the candidate's witness items were answered from cache
		// (two additional hits, the tx's two witness tuples), zero
		// additional backend executions, and no new insertion.
		if mp.sigCache.Hits() != 2 || mp.sigCache.Misses() != 2 || mp.sigCache.Len() != 2 {
			t.Fatalf("after warm-floor: hits=%d misses=%d len=%d, want 2/2/2",
				mp.sigCache.Hits(), mp.sigCache.Misses(), mp.sigCache.Len())
		}
	})
}

// daGuardRejectMessage is the exact standard-domain rejection text, written
// here as a literal instead of read from production so that changing the
// production message fails these rows rather than following them.
const daGuardRejectMessage = "standard mempool accepts only tx_kind=0x00"

// daGuardCandidate is one signed DA candidate: its row label and its raw bytes.
type daGuardCandidate struct {
	name string
	raw  []byte
}

// daGuardCandidates returns a fresh standard mempool plus the signed
// {DA_COMMIT 0x01, DA_CHUNK 0x02} pair daAdmissionTestMempool builds over its
// alternating outpoints, so a row cannot silently cover only one DA kind.
func daGuardCandidates(t *testing.T) (*Mempool, []daGuardCandidate) {
	t.Helper()
	mp, raw := daAdmissionTestMempool(t, 2)
	return mp, []daGuardCandidate{{"commit_0x01", raw[0]}, {"chunk_0x02", raw[1]}}
}

// daGuardWrapper adapts one public standard producer to a common
// error-returning shape. The set below is exactly the public referencing set
// of addTxWithSource; its remaining references are in-package tests.
type daGuardWrapper struct {
	name   string
	source mempoolTxSource
	admit  func(*Mempool, []byte) error
}

func daGuardWrappers() []daGuardWrapper {
	return []daGuardWrapper{
		{"AddTx", mempoolTxSourceLocal, func(mp *Mempool, raw []byte) error { return mp.AddTx(raw) }},
		{"AddRemoteTx", mempoolTxSourceRemote, func(mp *Mempool, raw []byte) error { return mp.AddRemoteTx(raw) }},
		{"AddReorgTx", mempoolTxSourceReorg, func(mp *Mempool, raw []byte) error { return mp.AddReorgTx(raw) }},
		{"AddRemoteTxForRelay", mempoolTxSourceRemote, func(mp *Mempool, raw []byte) error { return mp.AddRemoteTxForRelay(raw, nil).Err }},
	}
}

// requireDAKindReject pins the exact public rejection by equality on both the
// kind and the whole message, never by substring and never by errors.Is.
func requireDAKindReject(t *testing.T, err error) {
	t.Helper()
	var admitErr *TxAdmitError
	if !errors.As(err, &admitErr) || admitErr.Kind != TxAdmitRejected || admitErr.Message != daGuardRejectMessage || err.Error() != daGuardRejectMessage {
		t.Fatalf("err=%v (%T), want TxAdmitRejected %q", err, err, daGuardRejectMessage)
	}
}

// daGuardContext is the owner's current exact admission context.
func daGuardContext(t *testing.T, mp *Mempool) *PendingOutpointAdmissionContext {
	t.Helper()
	admission, ok := mp.PendingOutpointOwner().AdmissionContext()
	if !ok {
		t.Fatal("owner admission context unavailable")
	}
	return &admission
}

// daGuardImage is the same-instance admission image a refusal must leave
// untouched: the canonical M/O fingerprint, plus the four things that
// fingerprint does not carry — index nilness, lowWaterBytes, the
// resident-eviction counter, and the caller's own raw candidate bytes. Owner
// nilness is restated below although the fingerprint already carries it. It is
// not a whole-instance no-change promise: m.sigCache is excluded because the
// earlier signature validation inserts a positive entry before the guard
// runs, which the state contract permits.
func daGuardImage(t *testing.T, mp *Mempool, raw []byte) string {
	t.Helper()
	fingerprint := canonicalMOImageFingerprint(t, mp, 0)
	mp.mu.RLock()
	defer mp.mu.RUnlock()
	return fmt.Sprintf("%s txsNil=%v wtxidsNil=%v ownerNil=%v low=%d evicted=%d caller=%x",
		fingerprint, mp.txs == nil, mp.wtxids == nil, mp.pendingOutpoints == nil,
		mp.lowWaterBytes, mp.evictedResidentTotal.Load(), raw)
}

// requireDAGuardPreservedImage runs one refusal and pins both halves of the
// state contract: the image is identical across the call, and the four
// admission buckets read exactly wantCounts afterwards.
func requireDAGuardPreservedImage(t *testing.T, mp *Mempool, raw []byte, admit func() error, wantCounts MempoolAdmissionCounts) {
	t.Helper()
	before := daGuardImage(t, mp, raw)
	requireDAKindReject(t, admit())
	if after := daGuardImage(t, mp, raw); after != before {
		t.Fatalf("admission image changed across the refusal:\n before %s\n after  %s", before, after)
	}
	if got := mp.AdmissionCounts(); got != wantCounts {
		t.Fatalf("admission counts=%+v, want %+v", got, wantCounts)
	}
}

// TestMempoolRejectsDAKindAcrossAllEntryPoints is the A1/A2 row: each of the
// four public standard producers refuses both signed DA kinds with the exact
// standard-domain rejection and no state effect, refuses the same bytes again
// on the same pool without accumulating anything, and still admits an ordinary
// tx_kind=0x00 with its existing source, residency, owner token and counter.
func TestMempoolRejectsDAKindAcrossAllEntryPoints(t *testing.T) {
	for _, wrapper := range daGuardWrappers() {
		for _, index := range []int{0, 1} {
			mp, candidates := daGuardCandidates(t)
			candidate := candidates[index]
			t.Run("reject/"+wrapper.name+"/"+candidate.name, func(t *testing.T) {
				requireDAGuardPreservedImage(t, mp, candidate.raw, func() error { return wrapper.admit(mp, candidate.raw) }, MempoolAdmissionCounts{Rejected: 1})
				requireDAGuardPreservedImage(t, mp, candidate.raw, func() error { return wrapper.admit(mp, candidate.raw) }, MempoolAdmissionCounts{Rejected: 2})
			})
		}
	}
	for _, wrapper := range daGuardWrappers() {
		t.Run("ordinary/"+wrapper.name, func(t *testing.T) {
			h := newRelayHarness(t, nil, 1_000_000)
			raw := h.tx(0, 100_000, 100_000, 1)
			if err := wrapper.admit(h.mp, raw); err != nil {
				t.Fatalf("ordinary tx_kind=0x00 admission: %v", err)
			}
			h.mp.mu.RLock()
			entry, seq, used := h.mp.txs[txID(t, raw)], h.mp.lastAdmissionSeq, h.mp.usedBytes
			h.mp.mu.RUnlock()
			if entry == nil || entry.source != wrapper.source || entry.token == (PendingOutpointToken{}) || seq != 1 || used != len(raw) {
				t.Fatalf("entry=%+v seq=%d used=%d, want one retained entry under source %q holding an owner token", entry, seq, used, wrapper.source)
			}
			if got := h.mp.AdmissionCounts(); got != (MempoolAdmissionCounts{Accepted: 1}) {
				t.Fatalf("ordinary counts=%+v, want exactly one Accepted", got)
			}
		})
	}
	t.Run("ordinary/AddRemoteTxForRelay/typed", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		raw := h.tx(0, 100_000, 100_000, 1)
		got := h.mp.AddRemoteTxForRelay(raw, nil)
		if got.Err != nil || got.Disposition != RelayAdmissionRetained || got.TxID != txID(t, raw) || got.WTxID == ([32]byte{}) || got.HasAdmissionContext || got.AdmissionContext != (PendingOutpointAdmissionContext{}) {
			t.Fatalf("typed ordinary result=%+v, want RETAINED with parsed identities and no cache-authorizing context", got)
		}
	})
}

// TestMempoolDAKindGuardPreservesIdentityPrecedence is the A3 row: the resident
// txid slot and then the resident wtxid slot decide before the kind slot, and
// txid wins when both are resident. The index rows below are TEST-ARRANGED
// structural precedence witnesses; they claim no naturally reachable DA
// residency and no same-wtxid/different-txid candidate.
func TestMempoolDAKindGuardPreservesIdentityPrecedence(t *testing.T) {
	resident := [32]byte{0xAB}
	rows := []struct {
		name    string
		arrange func(mp *Mempool, txid, wtxid [32]byte)
		wantMsg string
	}{
		{"arranged_txid_present", func(mp *Mempool, txid, _ [32]byte) { mp.txs[txid] = &mempoolEntry{txid: txid} }, "tx already in mempool"},
		{"arranged_wtxid_only", func(mp *Mempool, _, wtxid [32]byte) { mp.wtxids[wtxid] = resident }, fmt.Sprintf("mempool wtxid conflict with %x", resident)},
		{"arranged_both_txid_wins", func(mp *Mempool, txid, wtxid [32]byte) {
			mp.txs[txid] = &mempoolEntry{txid: txid}
			mp.wtxids[wtxid] = resident
		}, "tx already in mempool"},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			mp, candidates := daGuardCandidates(t)
			raw := candidates[0].raw
			tx, txid, wtxid, _, err := consensus.ParseTx(raw)
			if err != nil || tx.TxKind != 0x01 {
				t.Fatalf("ParseTx(commit) err=%v, want a signed kind 0x01 candidate", err)
			}
			mp.mu.Lock()
			row.arrange(mp, txid, wtxid)
			mp.mu.Unlock()
			before := daGuardImage(t, mp, raw)
			got := mp.AddRemoteTxForRelay(raw, daGuardContext(t, mp))
			var admitErr *TxAdmitError
			if !errors.As(got.Err, &admitErr) || admitErr.Kind != TxAdmitConflict || admitErr.Message != row.wantMsg {
				t.Fatalf("err=%v, want TxAdmitConflict %q rather than the kind rejection", got.Err, row.wantMsg)
			}
			if got.Disposition != RelayAdmissionDuplicate || got.HasAdmissionContext {
				t.Fatalf("disposition=%v hasContext=%v, want DUPLICATE with no published context", got.Disposition, got.HasAdmissionContext)
			}
			if after := daGuardImage(t, mp, raw); after != before {
				t.Fatalf("identity refusal changed the image:\n before %s\n after  %s", before, after)
			}
			if counts := mp.AdmissionCounts(); counts != (MempoolAdmissionCounts{Conflict: 1}) {
				t.Fatalf("counts=%+v, want exactly one Conflict", counts)
			}
		})
	}
}

// TestMempoolDAKindGuardDoesNotReservePendingOutpoints is the A4 owner half and
// the A7 reuse row: a refused DA candidate claims no pending outpoint, so a
// real standard input conflict never decides its outcome, a never-initialized
// owner and index pair is not lazily created, and the same confirmed input is
// still spendable by an ordinary transaction on the SAME pool with no owner,
// index or chainstate reset in between.
func TestMempoolDAKindGuardDoesNotReservePendingOutpoints(t *testing.T) {
	t.Run("kind_wins_over_a_real_standard_input_conflict", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		if err := h.mp.AddTx(h.tx(0, 100_000, 100_000, 1)); err != nil {
			t.Fatalf("seed the standard claim on outpoint 0: %v", err)
		}
		da := mustBuildSignedDaCommitTx(t, h.st.Utxos, h.outpoints[0], 100_000, 900_000, 2, h.fromKey, h.toAddr, []byte("0123456789"))
		before := cloneDAAdmissionOwner(h.mp.PendingOutpointOwner())
		requireDAGuardPreservedImage(t, h.mp, da, func() error { return h.mp.AddTx(da) }, MempoolAdmissionCounts{Accepted: 1, Rejected: 1})
		if after := cloneDAAdmissionOwner(h.mp.PendingOutpointOwner()); !reflect.DeepEqual(after, before) {
			t.Fatalf("owner image changed across the refusal:\n before %+v\n after  %+v", before, after)
		}
		if got := h.mp.AddRemoteTxForRelay(da, nil); got.Disposition != RelayAdmissionStableTerminalReject {
			t.Fatalf("disposition=%v, want STABLE_TERMINAL_REJECT rather than the later CONFLICT", got.Disposition)
		}
	})
	t.Run("never_initialized_owner_and_indexes_stay_nil", func(t *testing.T) {
		mp, candidates := daGuardCandidates(t)
		mp.mu.Lock()
		mp.pendingOutpoints, mp.txs, mp.wtxids = nil, nil, nil
		mp.mu.Unlock()
		raw := candidates[1].raw
		requireDAGuardPreservedImage(t, mp, raw, func() error { return mp.AddTx(raw) }, MempoolAdmissionCounts{Rejected: 1})
		mp.mu.RLock()
		defer mp.mu.RUnlock()
		if mp.pendingOutpoints != nil || mp.txs != nil || mp.wtxids != nil {
			t.Fatal("the refusal lazily created an owner or an index")
		}
	})
	t.Run("reject_then_ordinary_reuse_of_the_same_confirmed_input", func(t *testing.T) {
		h := newRelayHarness(t, nil, 1_000_000)
		da := mustBuildSignedDaCommitTx(t, h.st.Utxos, h.outpoints[0], 100_000, 900_000, 1, h.fromKey, h.toAddr, []byte("0123456789"))
		requireDAGuardPreservedImage(t, h.mp, da, func() error { return h.mp.AddTx(da) }, MempoolAdmissionCounts{Rejected: 1})
		if err := h.mp.AddTx(h.tx(0, 100_000, 100_000, 2)); err != nil {
			t.Fatalf("ordinary spend of the still-unclaimed confirmed input: %v", err)
		}
		if counts := h.mp.AdmissionCounts(); counts != (MempoolAdmissionCounts{Accepted: 1, Rejected: 1}) {
			t.Fatalf("counts=%+v, want one rejection followed by one acceptance", counts)
		}
	})
}

// TestMempoolDAKindGuardLeavesCapacitySequenceAndOwnerUnchanged is the rest of
// A4: each later standard pressure is arranged INDEPENDENTLY and the kind
// rejection still wins, leaving the complete image, the eviction counter and
// the four admission buckets exact.
func TestMempoolDAKindGuardLeavesCapacitySequenceAndOwnerUnchanged(t *testing.T) {
	rows := []struct {
		name    string
		cfg     *MempoolConfig
		values  []uint64
		arrange func(t *testing.T, h *relayHarness) MempoolAdmissionCounts
	}{
		{"raised_rolling_floor", nil, []uint64{1_000_000}, func(t *testing.T, h *relayHarness) MempoolAdmissionCounts {
			h.mp.SetCurrentMinFeeRateForTest(1 << 40)
			return MempoolAdmissionCounts{}
		}},
		{"full_count_capacity", &MempoolConfig{MaxTransactions: 1, MaxBytes: 1 << 20}, []uint64{1_000_000, 1_000_000}, func(t *testing.T, h *relayHarness) MempoolAdmissionCounts {
			if err := h.mp.AddTx(h.tx(1, 100_000, 100_000, 9)); err != nil {
				t.Fatalf("seed the single resident entry: %v", err)
			}
			return MempoolAdmissionCounts{Accepted: 1}
		}},
		{"insufficient_byte_capacity", &MempoolConfig{MaxTransactions: 10, MaxBytes: 1024}, []uint64{1_000_000}, func(*testing.T, *relayHarness) MempoolAdmissionCounts {
			return MempoolAdmissionCounts{}
		}},
		{"exhausted_admission_sequence", nil, []uint64{1_000_000}, func(_ *testing.T, h *relayHarness) MempoolAdmissionCounts {
			h.mp.mu.Lock()
			h.mp.lastAdmissionSeq = ^uint64(0)
			h.mp.mu.Unlock()
			return MempoolAdmissionCounts{}
		}},
		{"exhausted_owner_token_sequence", nil, []uint64{1_000_000}, func(_ *testing.T, h *relayHarness) MempoolAdmissionCounts {
			owner := h.mp.PendingOutpointOwner()
			owner.mu.Lock()
			owner.tokenHighWater = ^uint64(0)
			owner.mu.Unlock()
			return MempoolAdmissionCounts{}
		}},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			h := newRelayHarness(t, row.cfg, row.values...)
			want := row.arrange(t, h)
			want.Rejected++
			da := mustBuildSignedDaCommitTx(t, h.st.Utxos, h.outpoints[0], 100_000, 900_000, 1, h.fromKey, h.toAddr, []byte("0123456789"))
			requireDAGuardPreservedImage(t, h.mp, da, func() error { return h.mp.AddTx(da) }, want)
		})
	}
}

// TestMempoolDAKindGuardRelayResult is the A5 row: every published field of the
// typed producer result for a refused DA candidate, over both signed DA kinds.
// Only the exact complete context the read-only probe proved for this call
// authorizes evidence; a nil, generation-stale or tip-mismatched one publishes
// none, and the later reserve/context refusal never preempts the kind
// rejection.
func TestMempoolDAKindGuardRelayResult(t *testing.T) {
	rows := []struct {
		name        string
		expected    func(t *testing.T, mp *Mempool) *PendingOutpointAdmissionContext
		wantContext bool
	}{
		{"nil_expected_context", func(*testing.T, *Mempool) *PendingOutpointAdmissionContext { return nil }, false},
		{"exact_current_complete_context", daGuardContext, true},
		{"same_tip_stale_generation", func(t *testing.T, mp *Mempool) *PendingOutpointAdmissionContext {
			stale := daGuardContext(t, mp)
			owner := mp.PendingOutpointOwner()
			if _, err := owner.beginTransition(); err != nil {
				t.Fatalf("beginTransition: %v", err)
			}
			if err := owner.commitStableTip(pendingOutpointTipOf(mp.chainState)); err != nil {
				t.Fatalf("commitStableTip: %v", err)
			}
			return stale
		}, false},
		{"mismatched_tip_same_generation", func(t *testing.T, mp *Mempool) *PendingOutpointAdmissionContext {
			mismatched := *daGuardContext(t, mp)
			mismatched.StableTip.Height++
			mismatched.StableTip.Hash[0] ^= 0xFF
			return &mismatched
		}, false},
	}
	for _, row := range rows {
		for _, index := range []int{0, 1} {
			mp, candidates := daGuardCandidates(t)
			candidate := candidates[index]
			t.Run(row.name+"/"+candidate.name, func(t *testing.T) {
				expected := row.expected(t, mp)
				got := mp.AddRemoteTxForRelay(candidate.raw, expected)
				requireDAKindReject(t, got.Err)
				if got.Disposition != RelayAdmissionStableTerminalReject {
					t.Fatalf("disposition=%v, want STABLE_TERMINAL_REJECT", got.Disposition)
				}
				if got.TxID != txID(t, candidate.raw) || got.WTxID == ([32]byte{}) {
					t.Fatalf("identities=%x/%x, want the producer-parsed candidate pair", got.TxID, got.WTxID)
				}
				want := PendingOutpointAdmissionContext{}
				if row.wantContext {
					want = *expected
				}
				if got.HasAdmissionContext != row.wantContext || got.AdmissionContext != want {
					t.Fatalf("hasContext=%v context=%+v, want %v and %+v", got.HasAdmissionContext, got.AdmissionContext, row.wantContext, want)
				}
			})
		}
	}
}

// TestMempoolDAKindGuardPreservesEarlierErrors is R1-R4: every terminal,
// canonical, consensus and DA-policy refusal that already owned a DA candidate
// still owns it, with its exact baseline kind and disposition and a message
// pinned by containment, because the DA-policy texts carry trailing detail.
// Each row varies exactly one dimension of an otherwise valid signed DA
// candidate.
func TestMempoolDAKindGuardPreservesEarlierErrors(t *testing.T) {
	rows := []struct {
		name       string
		build      func(t *testing.T) (*Mempool, []byte)
		wantKind   TxAdmitErrorKind
		wantDisp   RelayAdmissionDisposition
		wantMsg    string
		wantIDs    bool
		wantCounts MempoolAdmissionCounts
	}{
		{"R1_nil_receiver", func(t *testing.T) (*Mempool, []byte) {
			_, candidates := daGuardCandidates(t)
			return nil, candidates[0].raw
		}, TxAdmitUnavailable, RelayAdmissionUnavailable, "nil mempool", false, MempoolAdmissionCounts{}},
		{"R1_nil_chainstate", func(t *testing.T) (*Mempool, []byte) {
			_, candidates := daGuardCandidates(t)
			return &Mempool{}, candidates[0].raw
		}, TxAdmitUnavailable, RelayAdmissionUnavailable, "nil chainstate", false, MempoolAdmissionCounts{Unavailable: 1}},
		{"R1_terminal_admission_guard", func(t *testing.T) (*Mempool, []byte) {
			mp, candidates := daGuardCandidates(t)
			mp.chainState.admissionMu.notifyTerminal()
			return mp, candidates[0].raw
		}, TxAdmitUnavailable, RelayAdmissionUnavailable, "pending-outpoint owner admission context unavailable", false, MempoolAdmissionCounts{Unavailable: 1}},
		{"R2_truncated_bytes", func(t *testing.T) (*Mempool, []byte) {
			mp, candidates := daGuardCandidates(t)
			return mp, candidates[0].raw[:len(candidates[0].raw)-1]
		}, TxAdmitRejected, RelayAdmissionStableTerminalReject, "TX_ERR_PARSE: unexpected EOF (bytes)", false, MempoolAdmissionCounts{Rejected: 1}},
		{"R2_trailing_bytes", func(t *testing.T) (*Mempool, []byte) {
			mp, candidates := daGuardCandidates(t)
			return mp, append(append([]byte(nil), candidates[0].raw...), 0x00)
		}, TxAdmitRejected, RelayAdmissionStableTerminalReject, "trailing bytes after canonical tx", false, MempoolAdmissionCounts{Rejected: 1}},
		{"R2_unsupported_kind_byte", func(t *testing.T) (*Mempool, []byte) {
			mp, candidates := daGuardCandidates(t)
			unsupported := append([]byte(nil), candidates[0].raw...)
			unsupported[4] = 0x03
			return mp, unsupported
		}, TxAdmitRejected, RelayAdmissionStableTerminalReject, "TX_ERR_PARSE: unsupported tx_kind", false, MempoolAdmissionCounts{Rejected: 1}},
		{"R2_invalid_kind_payload_shape", func(t *testing.T) (*Mempool, []byte) {
			mp, candidates := daGuardCandidates(t)
			reshaped := append([]byte(nil), candidates[0].raw...)
			reshaped[4] = 0x02
			return mp, reshaped
		}, TxAdmitRejected, RelayAdmissionStableTerminalReject, "TX_ERR_PARSE: da_payload_len out of range for tx_kind=0x02", false, MempoolAdmissionCounts{Rejected: 1}},
		{"R3_invalid_signature", func(t *testing.T) (*Mempool, []byte) {
			mp, candidates := daGuardCandidates(t)
			return mp, corruptFirstWitnessSignature(t, candidates[0].raw)
		}, TxAdmitRejected, RelayAdmissionStableTerminalReject, "TX_ERR_SIG_INVALID: CORE_P2PK signature invalid", true, MempoolAdmissionCounts{Rejected: 1}},
		{"R3_missing_confirmed_utxo", func(t *testing.T) (*Mempool, []byte) {
			mp, candidates := daGuardCandidates(t)
			tx, _, _, _, err := consensus.ParseTx(candidates[0].raw)
			if err != nil {
				t.Fatalf("ParseTx(commit): %v", err)
			}
			delete(mp.chainState.Utxos, consensus.Outpoint{Txid: tx.Inputs[0].PrevTxid, Vout: tx.Inputs[0].PrevVout})
			return mp, candidates[0].raw
		}, TxAdmitRejected, RelayAdmissionMissingDependency, "TX_ERR_MISSING_UTXO: utxo not found", true, MempoolAdmissionCounts{Rejected: 1}},
		{"R3_zero_tx_nonce", func(t *testing.T) (*Mempool, []byte) {
			h := newRelayHarness(t, nil, 1_000_000)
			return h.mp, mustBuildSignedDaCommitTx(t, h.st.Utxos, h.outpoints[0], 100_000, 900_000, 0, h.fromKey, h.toAddr, []byte("0123456789"))
		}, TxAdmitRejected, RelayAdmissionStableTerminalReject, "TX_ERR_TX_NONCE_INVALID: tx_nonce must be >= 1 for non-coinbase", true, MempoolAdmissionCounts{Rejected: 1}},
		{"R3_retired_core_ext_covenant", func(t *testing.T) (*Mempool, []byte) {
			h := newRelayHarness(t, nil, 1_000_000)
			tx := &consensus.Tx{
				Version: 1, TxKind: 0x01, TxNonce: 7,
				Inputs:       []consensus.TxInput{{PrevTxid: h.outpoints[0].Txid, PrevVout: h.outpoints[0].Vout}},
				Outputs:      []consensus.TxOutput{{Value: 100_000, CovenantType: 0x0102, CovenantData: []byte{0x01}}},
				DaPayload:    []byte("0123456789"),
				DaCommitCore: &consensus.DaCommitCore{ChunkCount: 1, BatchNumber: 1},
			}
			if err := consensus.SignTransaction(tx, h.st.Utxos, devnetGenesisChainID, h.fromKey); err != nil {
				t.Fatalf("SignTransaction(retired covenant): %v", err)
			}
			return h.mp, mustMarshalTxForNodeTest(t, tx)
		}, TxAdmitRejected, RelayAdmissionStableTerminalReject, "TX_ERR_COVENANT_TYPE_INVALID: unknown covenant_type", true, MempoolAdmissionCounts{Rejected: 1}},
		{"R4_da_fee_below_stage_c_floor", func(t *testing.T) (*Mempool, []byte) {
			h := newRelayHarness(t, &MempoolConfig{PolicyDaSurchargePerByte: 1}, 100)
			return h.mp, mustBuildSignedDaCommitTx(t, h.st.Utxos, h.outpoints[0], 99, 1, 1, h.fromKey, h.toAddr, []byte("0123456789"))
		}, TxAdmitRejected, RelayAdmissionStableTerminalReject, "DA fee below Stage C floor", true, MempoolAdmissionCounts{Rejected: 1}},
		{"R4_declared_chunk_budget_exceeded", func(t *testing.T) (*Mempool, []byte) {
			h := newRelayHarness(t, &MempoolConfig{PolicyMaxDaBytesPerBlock: consensus.CHUNK_BYTES - 1}, 1_000_000)
			return h.mp, mustBuildSignedDaCommitTxWithChunkCount(t, h.st.Utxos, h.outpoints[0], 50_000, 950_000, 1, h.fromKey, h.toAddr, 1, []byte("0123456789"))
		}, TxAdmitRejected, RelayAdmissionStableTerminalReject, "DA declared chunk budget exceeded", true, MempoolAdmissionCounts{Rejected: 1}},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			mp, raw := row.build(t)
			got := mp.AddRemoteTxForRelay(raw, nil)
			var admitErr *TxAdmitError
			if !errors.As(got.Err, &admitErr) {
				t.Fatalf("err=%v (%T), want a *TxAdmitError", got.Err, got.Err)
			}
			// Runs before the generic check below so a preemption reports
			// itself instead of being masked by the message mismatch.
			if admitErr.Message == daGuardRejectMessage {
				t.Fatalf("the kind guard preempted the earlier %s refusal", row.name)
			}
			if admitErr.Kind != row.wantKind || !strings.Contains(admitErr.Message, row.wantMsg) {
				t.Fatalf("err=%v, want %s carrying %q", got.Err, row.wantKind, row.wantMsg)
			}
			if got.Disposition != row.wantDisp || got.HasAdmissionContext || got.AdmissionContext != (PendingOutpointAdmissionContext{}) {
				t.Fatalf("disposition=%v hasContext=%v, want %v with no published context", got.Disposition, got.HasAdmissionContext, row.wantDisp)
			}
			if hasIDs := got.TxID != ([32]byte{}); hasIDs != row.wantIDs {
				t.Fatalf("published identity=%v (%x), want %v", hasIDs, got.TxID, row.wantIDs)
			}
			if counts := mp.AdmissionCounts(); counts != row.wantCounts {
				t.Fatalf("counts=%+v, want %+v", counts, row.wantCounts)
			}
			if mp.Len() != 0 {
				t.Fatalf("mempool len=%d after an earlier refusal, want 0", mp.Len())
			}
		})
	}
}
