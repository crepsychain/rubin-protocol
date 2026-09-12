use std::cmp::Ordering;
#[cfg(test)]
use std::collections::HashSet;
use std::collections::{BinaryHeap, HashMap};
use std::sync::OnceLock;

use rubin_consensus::uint128_json::{compare_fee_rate as compare_fee_rate_exact, fee_below_rate};
use rubin_consensus::{
    apply_non_coinbase_tx_basic_update_with_mtp_and_suite_context,
    constants::{COV_TYPE_CORE_SIMPLICITY, MAX_RELAY_MSG_BYTES},
    parse_block_header_bytes, parse_tx, tx_weight_and_stats_public, validate_tx_covenants_genesis,
    DefaultRotationProvider, NativeSuiteSet, Outpoint, RotationProvider, SuiteRegistry,
};

use crate::pending_outpoint_owner::{
    PendingOutpointAdmissionContext, PendingOutpointError, PendingOutpointErrorKind,
    PendingOutpointOwnerHandle, PendingOutpointTip, PendingOutpointToken,
};
use crate::sync::{SuiteContext, SyncEngine};
use crate::{BlockStore, ChainState};

const MAX_TX_POOL_TRANSACTIONS: usize = 300;
const DEFAULT_TX_POOL_MAX_BYTES: usize = MAX_RELAY_MSG_BYTES as usize;
const TX_POOL_LOW_WATER_NUMERATOR: usize = 9;
const TX_POOL_LOW_WATER_DENOMINATOR: usize = 10;

const _: () = assert!(DEFAULT_TX_POOL_MAX_BYTES as u64 == MAX_RELAY_MSG_BYTES);

/// Hot-path cache for the canonical default-fallback `SuiteRegistry`
/// used by `fee_precheck_p2pk_input_value` when the caller passes
/// `registry=None`. Fast-reject path is the spam-flood path; without
/// the cache, every plain P2PK candidate allocated a fresh
/// `BTreeMap<u8, SuiteParams>`. Mirror of Go
/// `cachedDefaultPrecheckSuiteRegistry` in
/// `clients/go/node/mempool_precheck_input.go`.
///
/// Defense-in-depth: the cached registry is verified to satisfy
/// `is_canonical_default_live_manifest()` on first init so a future
/// constants drift fail-closes here instead of silently shipping wrong
/// SuiteParams to the precheck. Mirrors the
/// `default_runtime_suite_registry` pattern in `verify_sig_openssl.rs`.
fn cached_default_registry() -> &'static SuiteRegistry {
    static REG: OnceLock<SuiteRegistry> = OnceLock::new();
    REG.get_or_init(|| {
        let r = SuiteRegistry::default_registry();
        // Drift fail-closed: panic at process init (not per-tx) if the
        // canonical manifest invariant is violated. Uses `assert!`
        // (NOT `debug_assert!`): the check MUST run in release builds
        // too, otherwise the cached precheck registry could silently
        // ship drifted SuiteParams in production while the slow
        // verification path (`verify_sig_openssl::default_runtime_suite_registry`)
        // returns a typed error — creating a release-only
        // classification split. Mirrors Go's package-init `panic()`
        // in `cachedDefaultPrecheckSuiteRegistry`. Closes Copilot
        // wave-22 P1 + Codex wave-22 P2.
        assert!(
            r.is_canonical_default_live_manifest(),
            "cached_default_registry: SuiteRegistry::default_registry() drifted from canonical live manifest"
        );
        r
    })
}

/// Hot-path cache for the canonical default-fallback
/// `native_spend_suites` set used by `fee_precheck_p2pk_input_value`
/// when the caller passes `rotation=None`.
/// `DefaultRotationProvider::native_spend_suites` returns the SAME set
/// for every height (`{SUITE_ID_ML_DSA_87}`); without the cache, every
/// plain P2PK candidate allocated a fresh `BTreeSet<u8>`. Mirror of Go
/// package-level cached set in `clients/go/node/mempool_precheck_input.go`.
fn cached_default_native_spend_set() -> &'static NativeSuiteSet {
    static SET: OnceLock<NativeSuiteSet> = OnceLock::new();
    SET.get_or_init(|| DefaultRotationProvider.native_spend_suites(0))
}

/// Hot-path cache for the canonical default-fallback
/// `native_create_suites` set used by `fee_precheck_p2pk_output_value`
/// when the caller passes `rotation=None`. Same rationale as
/// `cached_default_native_spend_set`.
fn cached_default_native_create_set() -> &'static NativeSuiteSet {
    static SET: OnceLock<NativeSuiteSet> = OnceLock::new();
    SET.get_or_init(|| DefaultRotationProvider.native_create_suites(0))
}

/// Default rolling mempool minimum fee rate used by callers without a live
/// rolling-floor source. Mirrors Go's `DefaultMempoolMinFeeRate` so the Rust
/// DA Stage C predicate enforces the same baseline relay floor when no live
/// mempool state is available; this is the documented Go pattern, not a
/// parallel rolling-floor invention. Live rolling-floor wiring is tracked
/// separately as the Rust standard mempool policy task.
pub const DEFAULT_MEMPOOL_MIN_FEE_RATE: u64 = 1;

/// Default spec-side DA per-byte floor used when a caller does not override
/// `policy_min_da_fee_rate`. Mirrors Go's `DefaultMinDaFeeRate`
/// (`POLICY_MEMPOOL_ADMISSION_GENESIS.md` Stage C `min_da_fee_rate`). Kept
/// as a separate constant from `DEFAULT_MEMPOOL_MIN_FEE_RATE` so a future
/// change to the relay floor cannot silently change the DA floor.
pub const DEFAULT_MIN_DA_FEE_RATE: u64 = 1;

#[derive(Debug, Clone)]
pub struct TxPoolConfig {
    pub policy_da_surcharge_per_byte: u64,
    pub policy_reject_non_coinbase_anchor_outputs: bool,
    /// Mirror of Go `MempoolConfig.PolicyRejectSimplicityPreActivation`:
    /// non-consensus pre-activation guardrail for CORE_SIMPLICITY
    /// (0x0106). When true, transactions that create or spend a
    /// CORE_SIMPLICITY output are rejected by admission/relay policy
    /// until the rotation provider reports the Simplicity deployment
    /// active at the next block height. Policy-only; consensus validity
    /// is unaffected.
    pub policy_reject_simplicity_pre_activation: bool,
    pub suite_context: Option<SuiteContext>,
    /// Rolling local mempool floor used by the Stage C relay-fee term.
    /// Defaults to `DEFAULT_MEMPOOL_MIN_FEE_RATE`; a live rolling floor
    /// source is wired in when the Rust standard mempool policy ships.
    pub policy_current_mempool_min_fee_rate: u64,
    /// Spec-side DA per-byte floor (`POLICY_MEMPOOL_ADMISSION_GENESIS.md`
    /// Stage C `min_da_fee_rate`). Defaults to `DEFAULT_MIN_DA_FEE_RATE`,
    /// kept separate from `DEFAULT_MEMPOOL_MIN_FEE_RATE` so a future
    /// change to the relay floor cannot silently change the DA floor.
    /// Zero disables only the `da_fee_floor` term; the surcharge term is
    /// governed independently by `policy_da_surcharge_per_byte`.
    pub policy_min_da_fee_rate: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxPoolEntry {
    pub raw: Vec<u8>,
    pub inputs: Vec<Outpoint>,
    /// Authoritative admitted fee: the exact u128 scalar consensus derived.
    /// `weight` stays u64 per the policy contract.
    pub fee: u128,
    pub weight: u64,
    pub size: usize,
    /// Caller-declared admission origin. Mirrors Go `mempoolEntry.source`
    /// (clients/go/node/mempool.go). Recorded for observability /
    /// downstream filtering; NOT consulted by `compare_entries_for_mining`
    /// or `compare_admit_priority` — admission ordering is source-blind.
    pub source: TxSource,
}

/// Defensive rollback snapshot for Rust `TxPool`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct TxPoolSnapshot {
    current_mempool_min_fee_rate: u64,
    owner_context: PendingOutpointAdmissionContext,
    entries: Vec<TxPoolSnapshotEntry>,
    next_heap_id: u64,
    used_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
struct TxPoolSnapshotEntry {
    txid: [u8; 32],
    wtxid: [u8; 32],
    entry: TxPoolEntry,
    heap_id: u64,
}

#[derive(Debug)]
pub struct TxPool {
    cfg: TxPoolConfig,
    txs: HashMap<[u8; 32], TxPoolEntry>,
    owner: Option<PendingOutpointOwnerHandle>,
    owner_tokens: HashMap<[u8; 32], PendingOutpointToken>,
    worst_heap: BinaryHeap<WorstEntryKey>,
    // Stable admission sequence per resident txid. It also tags worst_heap
    // entries for lazy stale-entry filtering.
    heap_seqs: HashMap<[u8; 32], u64>,
    next_heap_id: u64,
    max_transactions: usize,
    max_bytes: usize,
    low_water_bytes: usize,
    used_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorstEntryKey {
    txid: [u8; 32],
    fee: u128,
    weight: u64,
    heap_id: u64,
}

struct AdmitPriority<'a> {
    fee: u128,
    weight: u64,
    tie: &'a [u8],
}

struct CapacityPlanEntry<'a> {
    txid: [u8; 32],
    entry: &'a TxPoolEntry,
    candidate: bool,
    admission_seq: u64,
}

#[derive(Clone, Copy)]
enum CapacityOrdering {
    LegacyCountPressure,
    BytePressure,
}

impl Ord for WorstEntryKey {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_priority_values(
            AdmitPriority {
                fee: other.fee,
                weight: other.weight,
                tie: &other.txid,
            },
            AdmitPriority {
                fee: self.fee,
                weight: self.weight,
                tie: &self.txid,
            },
        )
    }
}

impl PartialOrd for WorstEntryKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Caller-declared origin of a mempool admission. Mirrors Go
/// `mempoolTxSource` (clients/go/node/mempool.go) with three variants:
/// `Local` (RPC submit), `Remote` (p2p relay), `Reorg` (sync-reorg
/// requeue). Source is recorded on the admitted entry but does NOT
/// grant admission priority or bypass any validation step. Closed enum
/// gives compile-time exhaustiveness; Go's `validMempoolTxSource`
/// runtime check has no Rust analog because invalid variants cannot be
/// constructed (parity-improved).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TxSource {
    Local,
    Remote,
    Reorg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxPoolAdmitErrorKind {
    Conflict,
    Rejected,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxPoolAdmitError {
    pub kind: TxPoolAdmitErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelayTxMetadata {
    /// Authoritative admitted fee, carried unnarrowed to relay ordering.
    pub fee: u128,
    pub size: usize,
}

impl std::fmt::Display for TxPoolAdmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TxPoolAdmitError {}

impl TxPool {
    pub fn new() -> Self {
        Self::new_with_config(TxPoolConfig::default())
    }

    pub fn new_with_config(cfg: TxPoolConfig) -> Self {
        Self::new_with_owner(cfg, None)
    }

    pub(crate) fn new_bound_with_config(
        cfg: TxPoolConfig,
        owner: PendingOutpointOwnerHandle,
    ) -> Self {
        Self::new_with_owner(cfg, Some(owner))
    }

    fn new_with_owner(cfg: TxPoolConfig, owner: Option<PendingOutpointOwnerHandle>) -> Self {
        let max_bytes = DEFAULT_TX_POOL_MAX_BYTES;
        Self {
            cfg,
            txs: HashMap::new(),
            owner,
            owner_tokens: HashMap::new(),
            worst_heap: BinaryHeap::new(),
            heap_seqs: HashMap::new(),
            next_heap_id: 0,
            max_transactions: MAX_TX_POOL_TRANSACTIONS,
            max_bytes,
            low_water_bytes: default_tx_pool_low_water_bytes(max_bytes),
            used_bytes: 0,
        }
    }

    fn admission_context(
        &mut self,
        chain_state: &ChainState,
    ) -> Result<(PendingOutpointOwnerHandle, PendingOutpointAdmissionContext), TxPoolAdmitError>
    {
        let expected_tip = PendingOutpointTip::from_chain_state(chain_state);
        let owner = self
            .owner
            .get_or_insert_with(|| PendingOutpointOwnerHandle::new(expected_tip))
            .clone();
        let context = owner.admission_context().map_err(owner_error)?;
        if context.tip != expected_tip {
            return Err(unavailable(
                "pending-outpoint owner context does not match chain tip",
            ));
        }
        Ok((owner, context))
    }

    pub(crate) fn reconcile_pending_outpoint_owner(
        &mut self,
        engine: &mut SyncEngine,
    ) -> Result<(), String> {
        match (engine.pending_outpoint_owner(), self.owner.clone()) {
            (None, None) => {
                let tip = PendingOutpointTip::from_chain_state(&engine.chain_state);
                let owner = PendingOutpointOwnerHandle::new(tip);
                engine.bind_pending_outpoint_owner(owner.clone())?;
                self.owner = Some(owner);
                Ok(())
            }
            (Some(sync_owner), Some(pool_owner)) if sync_owner.same_owner(&pool_owner) => Ok(()),
            _ => Err("pending-outpoint owner mismatch".into()),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_owner_handle(&self) -> Option<PendingOutpointOwnerHandle> {
        self.owner.clone()
    }

    #[cfg(test)]
    fn test_owner(&self) -> PendingOutpointOwnerHandle {
        self.owner.clone().expect("test owner")
    }

    #[cfg(test)]
    fn owner_txid_for_outpoint(&self, outpoint: &Outpoint) -> Option<[u8; 32]> {
        self.test_owner()
            .txid_for_outpoint(outpoint)
            .expect("owner lookup")
    }

    #[cfg(test)]
    fn owner_row_count(&self) -> usize {
        self.test_owner().test_row_count().expect("owner rows")
    }

    #[cfg(test)]
    fn owner_claim_is_finalized(&self, txid: &[u8; 32]) -> bool {
        let Some(entry) = self.txs.get(txid) else {
            return false;
        };
        if entry.inputs.is_empty() {
            return !self.owner_tokens.contains_key(txid);
        }
        let Some(owner) = self.owner.clone() else {
            return false;
        };
        let Some(token) = self.owner_tokens.get(txid) else {
            return false;
        };
        owner.lock().is_ok_and(|guard| {
            guard
                .validate_standard_entry(&owner, *txid, &entry.inputs, Some(token), true)
                .is_ok()
        })
    }

    pub fn len(&self) -> usize {
        self.txs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.txs.is_empty()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn snapshot(&self) -> Result<TxPoolSnapshot, TxPoolAdmitError> {
        let owner_context = match self.owner.clone() {
            Some(owner) => owner.admission_context().map_err(owner_error)?,
            None => PendingOutpointAdmissionContext {
                tip: PendingOutpointTip::default(),
                generation: 0,
            },
        };
        let mut entries = Vec::with_capacity(self.txs.len());
        let mut used_bytes = 0usize;
        let mut max_heap_id = 0u64;
        for (txid, entry) in &self.txs {
            let heap_id = self
                .heap_seqs
                .get(txid)
                .copied()
                .ok_or_else(|| rejected("txpool snapshot missing heap sequence"))?;
            if heap_id == 0 {
                return Err(rejected("invalid txpool snapshot heap id"));
            }
            used_bytes = used_bytes
                .checked_add(entry.size)
                .ok_or_else(|| unavailable("txpool snapshot byte accounting overflow"))?;
            max_heap_id = max_heap_id.max(heap_id);
            let wtxid = validate_txpool_snapshot_entry(*txid, None, entry)?;
            if !entry.inputs.is_empty() {
                let owner = self
                    .owner
                    .clone()
                    .ok_or_else(|| rejected("txpool snapshot missing pending-outpoint owner"))?;
                let guard = owner.lock().map_err(owner_error)?;
                guard
                    .validate_standard_entry(
                        &owner,
                        *txid,
                        &entry.inputs,
                        self.owner_tokens.get(txid),
                        true,
                    )
                    .map_err(owner_error)?;
            }
            entries.push(TxPoolSnapshotEntry {
                txid: *txid,
                wtxid,
                entry: entry.clone(),
                heap_id,
            });
        }
        entries.sort_by_key(|item| item.txid);
        if self.used_bytes != used_bytes {
            return Err(rejected("txpool snapshot used_bytes mismatch"));
        }
        if self.next_heap_id < max_heap_id {
            return Err(rejected("txpool snapshot heap high-watermark below max"));
        }
        if u64::MAX - self.next_heap_id <= (self.max_transactions as u64).saturating_add(1) {
            return Err(rejected("txpool snapshot heap near saturation"));
        }
        let floor = self.cfg.policy_current_mempool_min_fee_rate;
        Ok(TxPoolSnapshot {
            current_mempool_min_fee_rate: floor.max(DEFAULT_MEMPOOL_MIN_FEE_RATE),
            owner_context,
            entries,
            next_heap_id: self.next_heap_id,
            used_bytes,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn restore_snapshot(
        &mut self,
        snapshot: &TxPoolSnapshot,
    ) -> Result<(), TxPoolAdmitError> {
        if self.max_transactions == 0 || self.max_bytes == 0 {
            return Err(unavailable("invalid txpool snapshot restore limits"));
        }
        if snapshot.entries.len() > self.max_transactions {
            return Err(unavailable(format!(
                "txpool snapshot exceeds transaction cap: count={} max={}",
                snapshot.entries.len(),
                self.max_transactions
            )));
        }

        let mut txs = HashMap::with_capacity(snapshot.entries.len());
        let mut owner_tokens = HashMap::with_capacity(snapshot.entries.len());
        let mut wtxids = HashMap::with_capacity(snapshot.entries.len());
        let mut snapshot_outpoints = HashMap::new();
        let mut heap_seqs = HashMap::with_capacity(snapshot.entries.len());
        let mut admission_seqs = HashMap::with_capacity(snapshot.entries.len());
        let mut worst_heap = BinaryHeap::with_capacity(snapshot.entries.len());
        let mut used_bytes = 0usize;
        let mut max_heap_id = 0u64;
        let owner = PendingOutpointOwnerHandle::new_with_context(snapshot.owner_context);

        for item in &snapshot.entries {
            if txs.contains_key(&item.txid) {
                return Err(rejected(format!(
                    "duplicate txpool snapshot txid {}",
                    hex::encode(item.txid)
                )));
            }
            let heap_id = item.heap_id;
            if heap_id == 0 {
                return Err(rejected(format!(
                    "invalid txpool snapshot heap id for txid {}: heap_id=0",
                    hex::encode(item.txid)
                )));
            }
            if let Some(existing) = wtxids.insert(item.wtxid, item.txid) {
                return Err(rejected(format!(
                    "duplicate txpool snapshot wtxid {} existing={} new={}",
                    hex::encode(item.wtxid),
                    hex::encode(existing),
                    hex::encode(item.txid)
                )));
            }
            validate_txpool_snapshot_entry(item.txid, Some(item.wtxid), &item.entry)?;
            if let Some(existing) = admission_seqs.insert(heap_id, item.txid) {
                return Err(rejected(format!(
                    "duplicate txpool snapshot heap id {heap_id} existing={} new={}",
                    hex::encode(existing),
                    hex::encode(item.txid)
                )));
            }
            let next_used = used_bytes
                .checked_add(item.entry.size)
                .ok_or_else(|| unavailable("txpool snapshot byte accounting overflow"))?;
            if item.entry.size > self.max_bytes || next_used > self.max_bytes {
                return Err(unavailable(format!(
                    "txpool snapshot exceeds byte cap: used={} entry={} max={}",
                    used_bytes, item.entry.size, self.max_bytes
                )));
            }
            let entry = item.entry.clone();
            if !entry.inputs.is_empty() {
                for input in &entry.inputs {
                    if let Some(existing) = snapshot_outpoints.insert(input.clone(), item.txid) {
                        return Err(rejected(format!(
                            "duplicate txpool snapshot spender txid={} vout={} existing={} new={}",
                            hex::encode(input.txid),
                            input.vout,
                            hex::encode(existing),
                            hex::encode(item.txid)
                        )));
                    }
                }
                let token = owner
                    .reserve(snapshot.owner_context, item.txid, entry.inputs.clone())
                    .map_err(owner_error)?;
                owner.finalize(&token).map_err(owner_error)?;
                owner_tokens.insert(item.txid, token);
            }
            used_bytes = next_used;
            max_heap_id = max_heap_id.max(heap_id);
            heap_seqs.insert(item.txid, heap_id);
            worst_heap.push(WorstEntryKey {
                txid: item.txid,
                fee: item.entry.fee,
                weight: item.entry.weight,
                heap_id,
            });
            txs.insert(item.txid, entry);
        }

        if snapshot.used_bytes != used_bytes {
            return Err(rejected(format!(
                "txpool snapshot used_bytes mismatch: snapshot={} computed={}",
                snapshot.used_bytes, used_bytes
            )));
        }
        if snapshot.next_heap_id < max_heap_id {
            return Err(rejected(format!(
                "txpool snapshot heap high-watermark below restored max: next={} max={}",
                snapshot.next_heap_id, max_heap_id
            )));
        }
        if u64::MAX - snapshot.next_heap_id <= (self.max_transactions as u64).saturating_add(1) {
            return Err(rejected("txpool snapshot heap near saturation"));
        }

        let floor = snapshot.current_mempool_min_fee_rate;
        self.cfg.policy_current_mempool_min_fee_rate = floor.max(DEFAULT_MEMPOOL_MIN_FEE_RATE);
        self.txs = txs;
        self.owner = Some(owner);
        self.owner_tokens = owner_tokens;
        self.heap_seqs = heap_seqs;
        self.worst_heap = worst_heap;
        self.next_heap_id = snapshot.next_heap_id;
        self.used_bytes = used_bytes;
        Ok(())
    }

    /// Returns the txids of every transaction currently in the pool.
    /// The ordering of the returned vector is not guaranteed to be stable
    /// between calls; callers that require a deterministic order must sort.
    pub fn all_txids(&self) -> Vec<[u8; 32]> {
        self.txs.keys().copied().collect()
    }

    /// Returns a defensive clone of the raw transaction bytes for a pool entry
    /// with the given txid. Returns `None` if no matching entry is present.
    pub fn tx_by_id(&self, txid: &[u8; 32]) -> Option<Vec<u8>> {
        self.txs.get(txid).map(|entry| entry.raw.clone())
    }

    /// Returns the caller-declared `TxSource` recorded on the pool entry
    /// with the given txid. Returns `None` if no matching entry is
    /// present. Source is observability metadata only and does not
    /// affect admission ordering or eviction priority. Intended for
    /// downstream observability / source-counter telemetry / parity
    /// tests in producer-wiring slices (RUB-169..173).
    pub fn entry_source(&self, txid: &[u8; 32]) -> Option<TxSource> {
        self.txs.get(txid).map(|entry| entry.source)
    }

    /// Inject a raw entry for testing without full transaction validation.
    #[cfg(test)]
    pub fn inject_test_entry(&mut self, txid: [u8; 32], raw: Vec<u8>) {
        let size = raw.len();
        self.insert_entry(
            txid,
            TxPoolEntry {
                raw,
                inputs: Vec::new(),
                fee: 0,
                weight: size as u64,
                size,
                source: TxSource::Local,
            },
        );
    }

    /// Reports whether a transaction with the given txid is currently present
    /// in the pool.
    pub fn contains(&self, txid: &[u8; 32]) -> bool {
        self.txs.contains_key(txid)
    }

    pub fn select_transactions(&self, max_count: usize, max_bytes: usize) -> Vec<Vec<u8>> {
        self.select_transactions_with_filter(max_count, max_bytes, |_| false)
    }

    /// Sorted mining selection that drops any raw for which `filter` returns true
    /// before the count/byte caps (mirror of Go `pickMinerCandidateEntries`, where the
    /// `isMiningDATxRaw` skip precedes both caps). `select_transactions` is the
    /// unfiltered case; the miner passes `is_mining_da_tx_raw` to exclude individual
    /// DA txs from flat candidate selection.
    pub fn select_transactions_with_filter(
        &self,
        max_count: usize,
        max_bytes: usize,
        filter: impl Fn(&[u8]) -> bool,
    ) -> Vec<Vec<u8>> {
        if max_count == 0 || max_bytes == 0 {
            return Vec::new();
        }
        let mut entries: Vec<(&[u8; 32], &TxPoolEntry)> = self.txs.iter().collect();
        entries.sort_by(compare_entries_for_mining);
        let mut selected = Vec::with_capacity(entries.len().min(max_count));
        let mut used_bytes = 0usize;
        for entry in entries {
            if filter(&entry.1.raw) {
                continue;
            }
            if selected.len() >= max_count {
                break;
            }
            if entry.1.size > max_bytes.saturating_sub(used_bytes) {
                continue;
            }
            selected.push(entry.1.raw.clone());
            used_bytes += entry.1.size;
        }
        selected
    }

    /// Backward-compatible admission entry that defaults the caller-
    /// declared source to `TxSource::Local`. New producer wiring should
    /// call `add_tx_with_source` directly with the appropriate variant
    /// so source provenance is recorded on the entry. Returns just the
    /// txid (drops `RelayTxMetadata`); use `admit_with_metadata` or
    /// `add_tx_with_source` directly to receive the `RelayTxMetadata`
    /// alongside the txid.
    pub fn admit(
        &mut self,
        tx_bytes: &[u8],
        chain_state: &ChainState,
        block_store: Option<&BlockStore>,
        chain_id: [u8; 32],
    ) -> Result<[u8; 32], TxPoolAdmitError> {
        self.admit_with_metadata(tx_bytes, chain_state, block_store, chain_id)
            .map(|(txid, _)| txid)
    }

    /// Backward-compatible metadata-returning admission entry that
    /// defaults the caller-declared source to `TxSource::Local`. Mirrors
    /// the legacy admission API surface; new code should call
    /// `add_tx_with_source` instead with the appropriate variant.
    pub fn admit_with_metadata(
        &mut self,
        tx_bytes: &[u8],
        chain_state: &ChainState,
        block_store: Option<&BlockStore>,
        chain_id: [u8; 32],
    ) -> Result<([u8; 32], RelayTxMetadata), TxPoolAdmitError> {
        self.add_tx_with_source(
            tx_bytes,
            chain_state,
            block_store,
            chain_id,
            TxSource::Local,
        )
    }

    /// Canonical admission entry with caller-declared source provenance.
    /// Mirrors Go `Mempool.addTxWithSource` (clients/go/node/mempool.go)
    /// which is the shared body behind Go's `AddTx`/`AddRemoteTx`/
    /// `AddReorgTx` public entries. The `source` argument is recorded
    /// on the resulting `TxPoolEntry.source` for observability and
    /// downstream filtering, but does NOT affect any validation,
    /// admission ordering, or capacity-eviction priority — admission
    /// is source-blind.
    ///
    /// Rust's closed `TxSource` enum makes Go's runtime
    /// `validMempoolTxSource` check unnecessary; invalid variants
    /// cannot be constructed (parity-improved).
    pub fn add_tx_with_source(
        &mut self,
        tx_bytes: &[u8],
        chain_state: &ChainState,
        block_store: Option<&BlockStore>,
        chain_id: [u8; 32],
        source: TxSource,
    ) -> Result<([u8; 32], RelayTxMetadata), TxPoolAdmitError> {
        let (owner, owner_context) = self.admission_context(chain_state)?;
        let (tx, txid, _wtxid, consumed) =
            parse_tx(tx_bytes).map_err(|err| rejected(format!("transaction rejected: {err}")))?;
        if consumed != tx_bytes.len() {
            return Err(rejected("transaction rejected: non-canonical tx bytes"));
        }
        let inputs: Vec<Outpoint> = tx
            .inputs
            .iter()
            .map(|input| Outpoint {
                txid: input.prev_txid,
                vout: input.prev_vout,
            })
            .collect();

        // RUB-167 single-walk invariant: extract weight + da_bytes once
        // here and reuse for both post-consensus policy
        // (`apply_policy` -> `reject_da_anchor_tx_policy`) and the
        // final `validate_fee_floor` call. The same `(weight, da_bytes)`
        // tuple anchors both the DA-side classification and the
        // rolling-floor classification so they cannot diverge.
        let (weight, da_bytes, _) = tx_weight_and_stats_public(&tx)
            .map_err(|err| rejected(format!("transaction rejected: {err}")))?;

        let next_height = next_block_height(chain_state)?;
        let block_mtp = next_block_mtp(block_store, next_height)?;
        let (rotation, registry): (Option<&dyn RotationProvider>, Option<&SuiteRegistry>) =
            match self.cfg.suite_context.as_ref() {
                Some(ctx) => (Some(ctx.rotation.as_ref()), Some(ctx.registry.as_ref())),
                None => (None, None),
            };

        // RUB-166 fast-reject: cheap fee-floor precheck for plain P2PK
        // spam BEFORE expensive ML-DSA signature verification in
        // `apply_non_coinbase_tx_basic_update_*` below, but AFTER all
        // chain-context resolution (`next_block_height`,
        // `next_block_mtp`, `suite_context` lookup). Chain-context
        // failures (height overflow, missing tip MTP) keep their
        // existing error precedence so `admit_with_metadata` does NOT
        // diverge from `relay_metadata` on the (chain-context-error vs
        // fee-floor-fail) precedence.
        // Mirrors Go's `cheapFeeFloorPrecheck` call site inside
        // `checkTransactionWithSnapshot` at
        // `clients/go/node/mempool_precheck.go` `checkTransactionWithSnapshot` (RUB-165 PR #1415):
        // Go runs the precheck after `nextBlockContext`/`nextBlockMTP`
        // and immediately before the expensive consensus apply call.
        // The helper bails (`Ok(())`) for any tx shape it cannot
        // soundly classify (DA / multi-input / non-P2PK covenant /
        // overspend) and the existing expensive path keeps its error
        // precedence. Reuses the RUB-167 pre-computed `weight`. Same
        // `Unavailable` error class + same message format as
        // `validate_fee_floor`, so the rolling-floor classification
        // stays uniform across the fast-reject path and the
        // post-consensus path.
        cheap_fee_floor_precheck(
            &tx,
            &chain_state.utxos,
            weight,
            self.cfg.policy_current_mempool_min_fee_rate,
            next_height,
            rotation,
            registry,
        )?;
        // Mirror of Go `checkTransactionWithSnapshot`
        // (`clients/go/node/mempool_precheck.go`): the CORE_SIMPLICITY
        // pre-activation policy gate runs BEFORE consensus validation so
        // the policy reason — not the consensus reject — is the
        // observable admission outcome (Go pins this with
        // `TestMempoolPolicyRejectsCoreSimplicityPreActivationBeforeConsensus`).
        // Scoped to CORE_SIMPLICITY-involved txs (the exact set the gate
        // acts on): for those, a missing input rejects with the consensus
        // TX_ERR_MISSING_UTXO shape FIRST (Go `policyInputSnapshot`
        // precedence) so the policy reason never masks it. Non-Simplicity
        // txs (coinbase, plain spends) keep their pre-existing consensus
        // reject path untouched — full policyInputSnapshot parity for
        // every tx is a separate general-policy surface, not this gate.
        if self.cfg.policy_reject_simplicity_pre_activation
            && covenant_policy_kind(&tx, &chain_state.utxos, COV_TYPE_CORE_SIMPLICITY).is_some()
        {
            reject_missing_policy_inputs(&tx, &chain_state.utxos)?;
            if let Some(reason) = reject_core_simplicity_pre_activation(
                &tx,
                &chain_state.utxos,
                next_height,
                rotation,
            )
            .map_err(rejected)?
            {
                return Err(rejected(reason));
            }
        }
        let (_, summary) = apply_non_coinbase_tx_basic_update_with_mtp_and_suite_context(
            &tx,
            txid,
            &chain_state.utxos,
            next_height,
            block_mtp,
            block_mtp,
            chain_id,
            rotation,
            registry,
        )
        .map_err(|err| rejected(format!("transaction rejected: {err}")))?;
        // RUB-18/RUB-162 ordering: run post-consensus policy before
        // mempool duplicate/conflict checks, then run the final rolling
        // floor after duplicate/conflict checks. This mirrors Go:
        // `checkTransactionWithSnapshot` performs cheap precheck,
        // consensus validation, and `applyPolicyAgainstState` before
        // `validateNonCapacityAdmissionLocked`; `validateCapacityAdmissionLocked`
        // then applies the final rolling floor after duplicate/conflict
        // rejection. Relay has no duplicate/conflict boundary, so it
        // still uses `apply_post_consensus_policy_with_floor` below.
        // `#[rustfmt::skip]` keeps the call on one line so the
        // tarpaulin / Codacy diff-coverage tool attributes hits to a
        // single statement instead of per-arg lines (multi-line calls
        // leave several args marked "Not covered" even when the call
        // executes). Locals shorten the argument list.
        let utxos = &chain_state.utxos;
        let cfg = &self.cfg;
        #[rustfmt::skip]
        let policy_result = apply_post_consensus_policy_without_floor(&tx, utxos, weight, da_bytes, next_height, cfg);
        policy_result?;

        if self.txs.contains_key(&txid) {
            return Err(conflict("tx already in mempool"));
        }
        let raw = tx_bytes.to_vec();
        let mut reservation = if inputs.is_empty() {
            None
        } else {
            Some(
                owner
                    .reserve(owner_context, txid, inputs.clone())
                    .map_err(owner_error)?,
            )
        };
        if let Err(error) = validate_fee_floor(
            summary.fee,
            weight,
            self.cfg.policy_current_mempool_min_fee_rate,
        ) {
            release_reservation(&owner, reservation.as_ref());
            return Err(error);
        }

        let entry = TxPoolEntry {
            raw,
            inputs,
            fee: summary.fee,
            weight,
            size: tx_bytes.len(),
            source,
        };

        // Go-parity capacity admission runs after structural, chain,
        // policy, and rolling-floor checks. The low-water byte cap is an
        // eviction target under pressure, not a hard upper bound on a
        // fitting candidate.
        let evicted = match self.capacity_eviction_plan(txid, &entry) {
            Ok(evicted) => evicted,
            Err(error) => {
                release_reservation(&owner, reservation.as_ref());
                return Err(error);
            }
        };
        if let Err(error) = self.preflight_standard_delta(&evicted, reservation.is_some()) {
            release_reservation(&owner, reservation.as_ref());
            return Err(error);
        }
        let mut candidate = Some(entry);
        if let Err(error) = self.commit_standard_delta(
            &owner,
            owner_context,
            txid,
            &mut candidate,
            &mut reservation,
            &evicted,
        ) {
            release_reservation(&owner, reservation.as_ref());
            return Err(error);
        }
        Ok((
            txid,
            RelayTxMetadata {
                fee: summary.fee,
                size: tx_bytes.len(),
            },
        ))
    }

    pub fn relay_metadata_for_bytes(
        &self,
        tx_bytes: &[u8],
        chain_state: &ChainState,
        block_store: Option<&BlockStore>,
        chain_id: [u8; 32],
    ) -> Result<RelayTxMetadata, TxPoolAdmitError> {
        relay_metadata(tx_bytes, chain_state, block_store, chain_id, &self.cfg)
    }

    /// Remove transactions by txid (e.g. after block confirmation).
    pub fn evict_txids(&mut self, txids: &[[u8; 32]]) {
        self.remove_entries_exact(txids);
    }

    pub fn remove_conflicting_inputs(&mut self, txs: &[rubin_consensus::Tx]) {
        let mut outpoints = Vec::new();
        for tx in txs {
            outpoints.extend(tx.inputs.iter().map(|input| Outpoint {
                txid: input.prev_txid,
                vout: input.prev_vout,
            }));
        }
        self.remove_conflicting_outpoints(&outpoints);
    }

    pub fn remove_conflicting_outpoints(&mut self, outpoints: &[Outpoint]) {
        let Some(owner) = self.owner.clone() else {
            return;
        };
        let mut conflicting = Vec::new();
        for outpoint in outpoints {
            match owner.txid_for_outpoint(outpoint) {
                Ok(Some(txid)) if !conflicting.contains(&txid) => conflicting.push(txid),
                Ok(_) => {}
                Err(_) => return,
            }
        }
        self.evict_txids(&conflicting);
    }

    fn preflight_standard_delta(
        &mut self,
        evicted: &[[u8; 32]],
        needs_token: bool,
    ) -> Result<(), TxPoolAdmitError> {
        if self.next_heap_id == u64::MAX {
            return Err(unavailable("txpool heap sequence exhausted"));
        }
        if evicted.iter().any(|txid| !self.txs.contains_key(txid)) {
            return Err(unavailable("txpool eviction plan changed before commit"));
        }
        self.txs
            .try_reserve(1)
            .map_err(|_| unavailable("txpool entry capacity unavailable"))?;
        self.heap_seqs
            .try_reserve(1)
            .map_err(|_| unavailable("txpool heap index capacity unavailable"))?;
        if needs_token {
            self.owner_tokens
                .try_reserve(1)
                .map_err(|_| unavailable("txpool token capacity unavailable"))?;
        }
        self.worst_heap
            .try_reserve(1)
            .map_err(|_| unavailable("txpool heap capacity unavailable"))?;
        Ok(())
    }

    fn commit_standard_delta(
        &mut self,
        owner: &PendingOutpointOwnerHandle,
        context: PendingOutpointAdmissionContext,
        txid: [u8; 32],
        candidate: &mut Option<TxPoolEntry>,
        candidate_token: &mut Option<PendingOutpointToken>,
        evicted: &[[u8; 32]],
    ) -> Result<(), TxPoolAdmitError> {
        let entry = candidate
            .as_ref()
            .ok_or_else(|| rejected("txpool candidate missing before commit"))?;
        let mut owner_guard = owner.lock().map_err(owner_error)?;
        owner_guard.check_available(context).map_err(owner_error)?;
        if evicted.len() > MAX_TX_POOL_TRANSACTIONS {
            return Err(unavailable("txpool delta victim bound exceeded"));
        }
        for (index, victim_txid) in evicted.iter().enumerate() {
            if *victim_txid == txid || evicted[..index].contains(victim_txid) {
                return Err(unavailable("txpool delta victim shape invalid"));
            }
        }
        if let Some(token) = candidate_token.as_ref() {
            owner_guard
                .check_candidate(owner, token)
                .map_err(owner_error)?;
        }
        owner_guard
            .validate_standard_entry(owner, txid, &entry.inputs, candidate_token.as_ref(), false)
            .map_err(owner_error)?;
        for evicted_txid in evicted {
            let Some(victim) = self.txs.get(evicted_txid) else {
                return Err(unavailable("txpool eviction plan changed before commit"));
            };
            owner_guard
                .validate_standard_entry(
                    owner,
                    *evicted_txid,
                    &victim.inputs,
                    self.owner_tokens.get(evicted_txid),
                    true,
                )
                .map_err(owner_error)?;
        }
        let entry = candidate
            .take()
            .ok_or_else(|| rejected("txpool candidate missing during commit"))?;
        for evicted_txid in evicted {
            if self.remove_entry_raw(evicted_txid).is_some() {
                if let Some(token) = self.owner_tokens.remove(evicted_txid) {
                    owner_guard.drop_after_validation(&token);
                }
            }
        }
        self.insert_entry_raw(txid, entry);
        if let Some(token) = candidate_token.take() {
            self.owner_tokens.insert(txid, token.clone());
            owner_guard.finalize_after_validation(&token);
        }
        drop(owner_guard);
        self.compact_worst_heap_if_needed();
        Ok(())
    }

    fn remove_entries_exact(&mut self, txids: &[[u8; 32]]) {
        let mut live = Vec::new();
        for txid in txids {
            if self.txs.contains_key(txid) && !live.contains(txid) {
                live.push(*txid);
            }
        }
        if live.is_empty() {
            return;
        }
        let Some(owner) = self.owner.clone() else {
            if live.iter().all(|txid| {
                self.txs
                    .get(txid)
                    .is_some_and(|entry| entry.inputs.is_empty())
            }) {
                for txid in &live {
                    self.remove_entry_raw(txid);
                }
                self.compact_worst_heap_if_needed();
            }
            return;
        };
        let Ok(mut owner_guard) = owner.lock() else {
            return;
        };
        for txid in &live {
            let Some(entry) = self.txs.get(txid) else {
                return;
            };
            if owner_guard
                .validate_standard_entry(
                    &owner,
                    *txid,
                    &entry.inputs,
                    self.owner_tokens.get(txid),
                    true,
                )
                .is_err()
            {
                return;
            }
        }
        for txid in &live {
            if self.remove_entry_raw(txid).is_some() {
                if let Some(token) = self.owner_tokens.remove(txid) {
                    owner_guard.drop_after_validation(&token);
                }
            }
        }
        drop(owner_guard);
        self.compact_worst_heap_if_needed();
    }

    fn insert_entry_raw(&mut self, txid: [u8; 32], entry: TxPoolEntry) {
        self.next_heap_id = self.next_heap_id.saturating_add(1);
        let heap_id = self.next_heap_id;
        self.used_bytes = self.used_bytes.saturating_add(entry.size);
        self.heap_seqs.insert(txid, heap_id);
        self.worst_heap.push(WorstEntryKey {
            txid,
            fee: entry.fee,
            weight: entry.weight,
            heap_id,
        });
        self.txs.insert(txid, entry);
    }

    fn remove_entry_raw(&mut self, txid: &[u8; 32]) -> Option<TxPoolEntry> {
        if let Some(entry) = self.txs.remove(txid) {
            self.heap_seqs.remove(txid);
            self.used_bytes = self.used_bytes.saturating_sub(entry.size);
            return Some(entry);
        }
        None
    }

    #[cfg(test)]
    fn insert_entry(&mut self, txid: [u8; 32], entry: TxPoolEntry) {
        let token = if entry.inputs.is_empty() {
            None
        } else {
            let owner = self
                .owner
                .get_or_insert_with(|| {
                    PendingOutpointOwnerHandle::new(PendingOutpointTip::default())
                })
                .clone();
            let context = owner.admission_context().expect("test owner context");
            let token = owner
                .reserve(context, txid, entry.inputs.clone())
                .expect("test owner reservation");
            owner.finalize(&token).expect("test owner finalization");
            Some(token)
        };
        self.insert_entry_raw(txid, entry);
        if let Some(token) = token {
            self.owner_tokens.insert(txid, token);
        }
    }

    #[cfg(test)]
    fn remove_entry(&mut self, txid: &[u8; 32]) {
        self.remove_entries_exact(&[*txid]);
    }

    fn capacity_eviction_plan(
        &self,
        candidate_txid: [u8; 32],
        candidate: &TxPoolEntry,
    ) -> Result<Vec<[u8; 32]>, TxPoolAdmitError> {
        if self.max_transactions == 0 || self.max_bytes == 0 {
            return Err(unavailable(format!(
                "invalid mempool capacity limits: max_txs={} max_bytes={}",
                self.max_transactions, self.max_bytes
            )));
        }
        if candidate_txid == [0u8; 32] || candidate.size == 0 || candidate.weight == 0 {
            return Err(rejected(
                "tx pool capacity invariant violated: invalid candidate metadata",
            ));
        }
        if candidate.size > self.max_bytes {
            return Err(unavailable(format!(
                "mempool byte limit exceeded: current={} tx={} max={}",
                self.used_bytes, candidate.size, self.max_bytes
            )));
        }

        let count_pressure = self.txs.len() >= self.max_transactions;
        let byte_pressure = self.used_bytes > self.max_bytes.saturating_sub(candidate.size);
        if !count_pressure && !byte_pressure {
            return Ok(Vec::new());
        }

        let target_bytes = if byte_pressure {
            tx_pool_byte_pressure_target(self.effective_low_water_bytes(), candidate.size)
        } else {
            self.max_bytes
        };
        let ordering = if byte_pressure {
            CapacityOrdering::BytePressure
        } else {
            CapacityOrdering::LegacyCountPressure
        };
        let plan_cap = self
            .txs
            .len()
            .checked_add(1)
            .ok_or_else(|| unavailable("txpool capacity plan overflow"))?;
        let mut total_count = self.txs.len().saturating_add(1);
        let mut total_bytes = self.used_bytes.saturating_add(candidate.size);
        let mut plan_pool = Vec::new();
        plan_pool
            .try_reserve(plan_cap)
            .map_err(|_| unavailable("txpool capacity plan unavailable"))?;
        let mut admission_seqs = HashMap::new();
        admission_seqs
            .try_reserve(self.txs.len())
            .map_err(|_| unavailable("txpool capacity sequence unavailable"))?;
        let mut evicted = Vec::new();
        evicted
            .try_reserve(self.txs.len())
            .map_err(|_| unavailable("txpool eviction plan unavailable"))?;
        for (txid, entry) in &self.txs {
            if *txid == [0u8; 32] {
                return Err(rejected(
                    "tx pool capacity invariant violated: invalid resident metadata",
                ));
            }
            let Some(admission_seq) = self.heap_seqs.get(txid).copied() else {
                return Err(rejected(
                    "tx pool capacity invariant violated: missing heap sequence",
                ));
            };
            if let Some(existing) = admission_seqs.insert(admission_seq, *txid) {
                return Err(rejected(format!(
                    "tx pool capacity invariant violated: duplicate heap sequence {admission_seq} existing={} new={}",
                    hex::encode(existing),
                    hex::encode(txid)
                )));
            }
            if entry.size == 0 || entry.weight == 0 {
                return Err(rejected(
                    "tx pool capacity invariant violated: invalid resident metadata",
                ));
            }
            plan_pool.push(CapacityPlanEntry {
                txid: *txid,
                entry,
                candidate: false,
                admission_seq,
            });
        }
        plan_pool.push(CapacityPlanEntry {
            txid: candidate_txid,
            entry: candidate,
            candidate: true,
            admission_seq: 0,
        });

        while (total_count > self.max_transactions || total_bytes > target_bytes)
            && !plan_pool.is_empty()
        {
            let worst_index = worst_capacity_plan_index(&plan_pool, ordering);
            let worst = plan_pool.remove(worst_index);
            if worst.candidate {
                return Err(unavailable(
                    "mempool capacity candidate rejected by eviction ordering",
                ));
            }
            if total_bytes < worst.entry.size {
                return Err(unavailable("mempool eviction byte accounting underflow"));
            }
            total_count = total_count.saturating_sub(1);
            total_bytes -= worst.entry.size;
            evicted.push(worst.txid);
        }

        if total_count > self.max_transactions || total_bytes > self.max_bytes {
            return Err(unavailable(format!(
                "mempool capacity remains exceeded after dry-run eviction: count={}/{} bytes={}/{}",
                total_count, self.max_transactions, total_bytes, self.max_bytes
            )));
        }
        Ok(evicted)
    }

    fn effective_low_water_bytes(&self) -> usize {
        if self.low_water_bytes > 0 || self.max_bytes == 0 {
            return self.low_water_bytes;
        }
        default_tx_pool_low_water_bytes(self.max_bytes)
    }

    #[cfg(test)]
    fn set_capacity_for_test(&mut self, max_transactions: usize, max_bytes: usize) {
        self.max_transactions = max_transactions;
        self.max_bytes = max_bytes;
        self.low_water_bytes = default_tx_pool_low_water_bytes(max_bytes);
    }

    #[cfg(test)]
    fn insert_capacity_checked_entry_for_test(
        &mut self,
        txid: [u8; 32],
        entry: TxPoolEntry,
    ) -> Result<(), TxPoolAdmitError> {
        for evicted_txid in self.capacity_eviction_plan(txid, &entry)? {
            self.remove_entry(&evicted_txid);
        }
        self.insert_entry(txid, entry);
        Ok(())
    }

    // PR-1410 wave-3 — the historical TxPool impl-method that performed
    // the rolling-floor check (the `_locked` suffix referred to its
    // TxPool-state lock convenience) was removed. After the wave-3
    // drift-prevention helper extraction, the free `validate_fee_floor`
    // predicate is the single source-of-truth call. `relay_metadata` uses
    // `apply_post_consensus_policy_with_floor`; admission uses the
    // policy-only subhelper, performs duplicate/conflict checks at the Go
    // boundary, then calls the same `validate_fee_floor` predicate. The
    // historical `_locked` suffix no longer carries meaning — the
    // predicate is stateless on the cfg field. Tests call the free
    // `validate_fee_floor` directly.

    #[cfg(test)]
    fn seed_worst_heap(&mut self) {
        let max_stale_tail = self.txs.len().saturating_add(1);
        if self.heap_seqs.len() == self.txs.len()
            && (self.txs.is_empty() || !self.worst_heap.is_empty())
            && self.worst_heap.len() <= max_stale_tail
        {
            return;
        }
        self.rebuild_worst_heap();
    }

    #[cfg(test)]
    fn current_worst_txid(&mut self) -> Option<[u8; 32]> {
        self.seed_worst_heap();
        loop {
            let head = self.worst_heap.peek()?;
            match self.heap_seqs.get(&head.txid) {
                Some(heap_id) if *heap_id == head.heap_id => return Some(head.txid),
                _ => {
                    self.worst_heap.pop();
                }
            }
        }
    }

    fn compact_worst_heap_if_needed(&mut self) {
        let live = self.txs.len();
        let threshold = live.saturating_mul(2);
        if self.worst_heap.len() > threshold {
            self.worst_heap.retain(|entry| {
                self.heap_seqs
                    .get(&entry.txid)
                    .is_some_and(|heap_id| *heap_id == entry.heap_id)
            });
        }
    }

    #[cfg(test)]
    fn rebuild_worst_heap(&mut self) {
        let mut rebuilt = BinaryHeap::with_capacity(self.txs.len());
        let live_txids: HashSet<[u8; 32]> = self.txs.keys().copied().collect();
        self.heap_seqs.retain(|txid, _| live_txids.contains(txid));
        for (txid, entry) in &self.txs {
            let heap_id = match self.heap_seqs.get(txid).copied() {
                Some(heap_id) => heap_id,
                None => {
                    self.next_heap_id = self.next_heap_id.saturating_add(1);
                    let heap_id = self.next_heap_id;
                    self.heap_seqs.insert(*txid, heap_id);
                    heap_id
                }
            };
            rebuilt.push(WorstEntryKey {
                txid: *txid,
                fee: entry.fee,
                weight: entry.weight,
                heap_id,
            });
        }
        self.worst_heap = rebuilt;
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn validate_txpool_snapshot_entry(
    txid: [u8; 32],
    wtxid: Option<[u8; 32]>,
    entry: &TxPoolEntry,
) -> Result<[u8; 32], TxPoolAdmitError> {
    if txid == [0u8; 32] {
        return Err(rejected("invalid txpool snapshot entry txid"));
    }
    if entry.size == 0 {
        return Err(rejected("invalid txpool snapshot entry size"));
    }
    if entry.weight == 0 {
        return Err(rejected("invalid txpool snapshot entry weight"));
    }
    if entry.size != entry.raw.len() {
        return Err(rejected("txpool snapshot entry size mismatch"));
    }
    let (tx, raw_txid, raw_wtxid, consumed) = parse_tx(&entry.raw).map_err(|err| {
        rejected(format!(
            "invalid txpool snapshot entry raw for txid {}: {err}",
            hex::encode(txid)
        ))
    })?;
    if consumed != entry.raw.len() {
        return Err(rejected("txpool snapshot entry has trailing bytes"));
    }
    if raw_txid != txid {
        return Err(rejected(format!(
            "txpool snapshot entry txid mismatch: entry={} raw={}",
            hex::encode(txid),
            hex::encode(raw_txid)
        )));
    }
    if let Some(wtxid) = wtxid {
        if wtxid != raw_wtxid {
            return Err(rejected(format!(
                "txpool snapshot entry wtxid mismatch: entry={} raw={} txid={}",
                hex::encode(wtxid),
                hex::encode(raw_wtxid),
                hex::encode(txid)
            )));
        }
    }
    let (weight, _, _) = tx_weight_and_stats_public(&tx)
        .map_err(|err| rejected(format!("invalid txpool snapshot entry weight: {err}")))?;
    if entry.weight != weight {
        return Err(rejected(format!(
            "txpool snapshot entry weight mismatch: entry={} computed={} txid={}",
            entry.weight,
            weight,
            hex::encode(txid)
        )));
    }
    let inputs: Vec<Outpoint> = tx
        .inputs
        .iter()
        .map(|input| Outpoint {
            txid: input.prev_txid,
            vout: input.prev_vout,
        })
        .collect();
    if entry.inputs != inputs {
        return Err(rejected("txpool snapshot entry input list mismatch"));
    }
    // Go validates string source; Rust's closed `TxSource` cannot represent invalid values.
    Ok(raw_wtxid)
}

/// Returns the metadata a relay peer needs to forward the transaction
/// (fee + serialized size). Runs full structural + chainstate validation
/// AND enforces the rolling-relay-fee floor inline via
/// `apply_post_consensus_policy_with_floor` -> `validate_fee_floor`.
///
/// Cross-client parity: Go `RelayMetadata` (`clients/go/node/mempool.go`)
/// also enforces the rolling floor read-only after structural + chainstate
/// validation. Both relay metadata paths return `Unavailable` for otherwise
/// valid below-floor txs. Neither path inserts into the mempool or performs
/// duplicate, conflict, or capacity admission checks.
pub(crate) fn relay_metadata(
    tx_bytes: &[u8],
    chain_state: &ChainState,
    block_store: Option<&BlockStore>,
    chain_id: [u8; 32],
    cfg: &TxPoolConfig,
) -> Result<RelayTxMetadata, TxPoolAdmitError> {
    let (tx, txid, _wtxid, consumed) =
        parse_tx(tx_bytes).map_err(|err| rejected(format!("transaction rejected: {err}")))?;
    if consumed != tx_bytes.len() {
        return Err(rejected("transaction rejected: non-canonical tx bytes"));
    }

    let next_height = next_block_height(chain_state)?;
    let block_mtp = next_block_mtp(block_store, next_height)?;
    let (rotation, registry): (Option<&dyn RotationProvider>, Option<&SuiteRegistry>) =
        match cfg.suite_context.as_ref() {
            Some(ctx) => (Some(ctx.rotation.as_ref()), Some(ctx.registry.as_ref())),
            None => (None, None),
        };
    // Mirror of Go `checkParsedTransactionWithSnapshot`
    // (`clients/go/node/mempool.go`): the CORE_SIMPLICITY pre-activation
    // policy gate runs BEFORE consensus validation on the relay path too,
    // so relay and admission report the same policy reason. Scoped to
    // CORE_SIMPLICITY-involved txs, as on the admission path: a missing
    // input rejects with the consensus TX_ERR_MISSING_UTXO shape FIRST
    // (Go `policyInputSnapshot` precedence) so the policy reason never
    // masks it; non-Simplicity txs keep their pre-existing reject path.
    if cfg.policy_reject_simplicity_pre_activation
        && covenant_policy_kind(&tx, &chain_state.utxos, COV_TYPE_CORE_SIMPLICITY).is_some()
    {
        reject_missing_policy_inputs(&tx, &chain_state.utxos)?;
        if let Some(reason) =
            reject_core_simplicity_pre_activation(&tx, &chain_state.utxos, next_height, rotation)
                .map_err(rejected)?
        {
            return Err(rejected(reason));
        }
    }
    let (_, summary) = apply_non_coinbase_tx_basic_update_with_mtp_and_suite_context(
        &tx,
        txid,
        &chain_state.utxos,
        next_height,
        block_mtp,
        block_mtp,
        chain_id,
        rotation,
        registry,
    )
    .map_err(|err| rejected(format!("transaction rejected: {err}")))?;
    // RUB-162/RUB-197 relay drift-prevention: relay must run the
    // post-consensus policy sequence (cfg-clone with cfg-zero override
    // -> apply_policy -> rolling-floor enforcement) read-only. Admission
    // uses the same policy-only helper and `validate_fee_floor` predicate,
    // but keeps the Go-compatible duplicate/conflict boundary between
    // those two calls.
    // RUB-167 single-walk invariant: one `tx_weight_and_stats_public`
    // call here feeds both the DA-side and the rolling-floor
    // classifications inside the wrapper.
    let (weight, da_bytes, _) = tx_weight_and_stats_public(&tx)
        .map_err(|err| rejected(format!("transaction rejected: {err}")))?;
    // `#[rustfmt::skip]` keeps the call on one line for tarpaulin /
    // Codacy diff-coverage attribution (mirror of admit_with_metadata).
    let utxos = &chain_state.utxos;
    #[rustfmt::skip]
    let policy_result = apply_post_consensus_policy_with_floor(&tx, utxos, summary.fee, weight, da_bytes, next_height, cfg);
    policy_result?;

    Ok(RelayTxMetadata {
        fee: summary.fee,
        size: tx_bytes.len(),
    })
}

/// Relay-side post-consensus policy sequence. Mirrors Go's
/// `RelayMetadata`: apply policy first, then enforce the rolling floor
/// read-only. Admission cannot call this whole helper because Go places
/// duplicate/conflict checks between `applyPolicyAgainstState` and the
/// final rolling-floor check; admission therefore calls the policy-only
/// subhelper, performs duplicate/conflict checks, and then calls the same
/// `validate_fee_floor` predicate.
///
/// Order is mandatory and must NOT be changed without updating the
/// matching Go reference: apply_policy(cfg-zero) runs first to
/// classify DA-side rejections as Rejected (terminal), then
/// validate_fee_floor with the original cfg's
/// `policy_current_mempool_min_fee_rate` runs to classify rolling-
/// relay-floor rejections as Unavailable (transient/retryable). Relay and
/// admission must use the same predicate so peer relay
/// (`tx_relay::handle_received_tx`, which gates on `relay_metadata`)
/// never propagates a tx that local `admit` would reject.
///
/// Caller responsibilities (kept out of this helper to preserve
/// existing call-site semantics):
///   - parse_tx + canonical bytes check + apply_non_coinbase_tx_basic_update
///     must complete BEFORE this helper (signature verification +
///     consensus state validation are upstream).
///   - `weight` and `da_bytes` are extracted by the caller from a
///     single `tx_weight_and_stats_public(tx)` call (admit reads them
///     at the start of the function; relay extracts them between
///     apply_non_coinbase and this helper). `fee` is computed
///     separately by the upstream `apply_non_coinbase_tx_basic_update_*`
///     step (admit/relay receive it as `summary.fee`). The same
///     `(weight, da_bytes)` pair is passed straight through to
///     `apply_policy` and the same `weight` is reused by
///     `validate_fee_floor`, so the DA-side and rolling-floor
///     classifications operate on identical `weight` values (RUB-167
///     single-walk invariant). Admission preserves the same single-walk
///     invariant even though its policy-only and rolling-floor calls are
///     separated by the Go-compatible duplicate/conflict boundary.
///   - The miner caller (`reject_candidate`) deliberately does NOT
///     use this helper; miner has its own policy_cfg construction
///     because it has no rolling-floor equivalent (Go
///     `applyPolicyAgainstState` in clients/go/node/mempool.go documents this
///     same exception). Miner reuses the same single-walk pattern by
///     calling `tx_weight_and_stats_public` once before invoking
///     `apply_policy`.
fn apply_post_consensus_policy_with_floor(
    tx: &rubin_consensus::Tx,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
    fee: u128,
    weight: u64,
    da_bytes: u64,
    next_height: u64,
    cfg: &TxPoolConfig,
) -> Result<(), TxPoolAdmitError> {
    apply_post_consensus_policy_without_floor(tx, utxos, weight, da_bytes, next_height, cfg)?;
    validate_fee_floor(fee, weight, cfg.policy_current_mempool_min_fee_rate)
}

fn apply_post_consensus_policy_without_floor(
    tx: &rubin_consensus::Tx,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
    weight: u64,
    da_bytes: u64,
    next_height: u64,
    cfg: &TxPoolConfig,
) -> Result<(), TxPoolAdmitError> {
    let mut policy_cfg = cfg.clone();
    policy_cfg.policy_current_mempool_min_fee_rate = 0;
    apply_policy(tx, weight, da_bytes, utxos, next_height, &policy_cfg).map_err(rejected)
}

impl Default for TxPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TxPoolConfig {
    fn default() -> Self {
        Self {
            policy_da_surcharge_per_byte: 0,
            policy_reject_non_coinbase_anchor_outputs: true,
            // Mirror of Go `DefaultMinerConfig` -> `DefaultMempoolConfig`:
            // the CORE_SIMPLICITY pre-activation guardrail defaults ON.
            policy_reject_simplicity_pre_activation: true,
            suite_context: None,
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: DEFAULT_MIN_DA_FEE_RATE,
        }
    }
}

fn next_block_height(chain_state: &ChainState) -> Result<u64, TxPoolAdmitError> {
    if !chain_state.has_tip {
        return Ok(0);
    }
    if chain_state.height == u64::MAX {
        return Err(unavailable("height overflow"));
    }
    Ok(chain_state.height + 1)
}

fn next_block_mtp(
    block_store: Option<&BlockStore>,
    next_height: u64,
) -> Result<u64, TxPoolAdmitError> {
    let Some(block_store) = block_store else {
        return Ok(0);
    };
    if next_height == 0 {
        return Ok(0);
    }
    let mut window_len = 11u64;
    if next_height < window_len {
        window_len = next_height;
    }
    let mut out = Vec::with_capacity(window_len as usize);
    for idx in 0..window_len {
        let height = next_height - 1 - idx;
        let Some(hash) = block_store
            .canonical_hash(height)
            .map_err(|err| unavailable(err.to_string()))?
        else {
            return Err(unavailable(
                "missing canonical header for timestamp context",
            ));
        };
        let header_bytes = block_store
            .get_header_by_hash(hash)
            .map_err(|err| unavailable(err.to_string()))?;
        let header =
            parse_block_header_bytes(&header_bytes).map_err(|err| unavailable(err.to_string()))?;
        out.push(header.timestamp);
    }
    Ok(mtp_median(next_height, &out))
}

fn mtp_median(next_height: u64, prev_timestamps: &[u64]) -> u64 {
    let mut window_len = 11usize;
    if next_height < window_len as u64 {
        window_len = next_height as usize;
    }
    if prev_timestamps.len() < window_len {
        if prev_timestamps.is_empty() {
            return 0;
        }
        window_len = prev_timestamps.len();
    }
    let mut window = prev_timestamps[..window_len].to_vec();
    window.sort_unstable();
    window[(window.len() - 1) / 2]
}

fn conflict(message: impl Into<String>) -> TxPoolAdmitError {
    TxPoolAdmitError {
        kind: TxPoolAdmitErrorKind::Conflict,
        message: message.into(),
    }
}

fn owner_error(error: PendingOutpointError) -> TxPoolAdmitError {
    match error.kind {
        PendingOutpointErrorKind::Conflict => conflict(error.message),
        PendingOutpointErrorKind::Unavailable => unavailable(error.message),
        PendingOutpointErrorKind::Internal => unavailable(error.message),
    }
}

fn release_reservation(owner: &PendingOutpointOwnerHandle, token: Option<&PendingOutpointToken>) {
    if let Some(token) = token {
        let _ = owner.release(token);
    }
}

fn rejected(message: impl Into<String>) -> TxPoolAdmitError {
    TxPoolAdmitError {
        kind: TxPoolAdmitErrorKind::Rejected,
        message: message.into(),
    }
}

fn unavailable(message: impl Into<String>) -> TxPoolAdmitError {
    TxPoolAdmitError {
        kind: TxPoolAdmitErrorKind::Unavailable,
        message: message.into(),
    }
}

/// Returns true if `fee` is below the rolling fee floor for `weight`.
/// Mirrors Go `feeRateBelowFloor` in clients/go/node/mempool.go
/// using full-precision u128 cross-multiplication (`fee < weight * floor`).
/// u128 holds any `u64 * u64` product losslessly, so the comparison is
/// well-defined at every input including `weight == u64::MAX` and
/// `floor == u64::MAX`.
///
/// Zero weight returns true: `validateFeeFloorLocked` callers treat
/// `weight == 0` as an uncomputable rate ("treat as below floor"),
/// matching Go's documented branch.
///
/// Floor argument is clamped up to `DEFAULT_MEMPOOL_MIN_FEE_RATE` if
/// smaller, mirroring Go `feeRateBelowFloor`'s in-helper clamp at
/// clients/go/node/mempool.go (in-helper clamp inside `feeRateBelowFloor`; grep `DefaultMempoolMinFeeRate` in fn body). Callers therefore always
/// receive at-least-DEFAULT enforcement even if cfg-static or
/// rolling-floor sources zero the field.
fn fee_rate_below_floor(fee: u128, weight: u64, floor: u64) -> bool {
    if weight == 0 {
        return true;
    }
    let floor = floor.max(DEFAULT_MEMPOOL_MIN_FEE_RATE);
    fee_below_rate(fee, weight, floor)
}

/// Free-function predicate enforcing the rolling-relay-floor invariant.
/// `relay_metadata` calls it through `apply_post_consensus_policy_with_floor`;
/// admission calls it after its Go-compatible duplicate/conflict boundary.
/// Both paths use this one source-of-truth check. Returns `Unavailable`
/// (transient / retryable) on rolling-floor failure, mirroring Go
/// `validateFeeFloorLocked` at
/// clients/go/node/mempool.go (`validateFeeFloorLocked` wrapper + `validateFeeFloorLockedWithFloor` body). The `DEFAULT_MEMPOOL_MIN_FEE_RATE`
/// clamp lives inside `fee_rate_below_floor` (Go-parity at
/// clients/go/node/mempool.go in-helper clamp inside `feeRateBelowFloor` — grep `DefaultMempoolMinFeeRate` in fn body. The error message surfaces
/// the post-clamp value for operator clarity.
fn validate_fee_floor(fee: u128, weight: u64, cfg_floor: u64) -> Result<(), TxPoolAdmitError> {
    if fee_rate_below_floor(fee, weight, cfg_floor) {
        let surfaced_floor = cfg_floor.max(DEFAULT_MEMPOOL_MIN_FEE_RATE);
        return Err(unavailable(format!(
            "mempool fee below rolling minimum: fee={fee} weight={weight} min_fee_rate={surfaced_floor}"
        )));
    }
    Ok(())
}

/// RUB-166 cheap fee-floor precheck for plain P2PK spam fast-reject.
/// Mirrors Go's `cheapFeeFloorPrecheck` at
/// `clients/go/node/mempool_precheck_floor.go` `cheapFeeFloorPrecheck` (RUB-165, PR #1415, merge SHA
/// `ed3be97`).
///
/// Conservatism (verbatim Go logic): only fast-rejects when ALL of:
///   - `tx.tx_kind == 0x00` (plain transfer; no DA / CORE_ANCHOR lanes)
///   - `tx.da_payload` is empty
///   - exactly one input, whose outpoint resolves in `utxos` to a
///     COV_TYPE_P2PK entry
///   - every output is COV_TYPE_P2PK
///   - sum of output values does not overflow and does not exceed the
///     input value (overspend defers to expensive path)
///   - `weight > 0`
///
/// Anything else returns `Ok(())` so the existing expensive admission
/// path handles classification (signature verify / state validation /
/// policy ordering preserved exactly).
///
/// `weight` is passed in by the caller per the RUB-167 single-walk
/// invariant; the same `weight` value is reused downstream by
/// `validate_fee_floor` so DA-side and rolling-floor classifications
/// operate on identical values (this precheck plus the downstream check
/// use the same `weight`, so a fee-rate decision here matches the final
/// decision).
///
/// On below-floor reject the error message uses the verbatim
/// `validate_fee_floor` format ("mempool fee below rolling minimum:
/// fee=X weight=Y min_fee_rate=Z" with `min_fee_rate` set to
/// `cfg_floor.max(DEFAULT_MEMPOOL_MIN_FEE_RATE)` post-clamp). For all
/// production callers (which pass `policy_current_mempool_min_fee_rate`
/// at or above `DEFAULT_MEMPOOL_MIN_FEE_RATE`) this is byte-identical
/// to Go's `cheapFeeFloorPrecheck` output. Test configurations
/// passing `current_min_fee_rate` below `DEFAULT_MEMPOOL_MIN_FEE_RATE`
/// print the post-clamp value (matches Rust's existing
/// `validate_fee_floor` surface). Error class is
/// `TxPoolAdmitErrorKind::Unavailable` so callers may retry once the
/// rolling floor drops.
fn cheap_fee_floor_precheck(
    tx: &rubin_consensus::Tx,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
    weight: u64,
    current_min_fee_rate: u64,
    next_height: u64,
    rotation: Option<&dyn RotationProvider>,
    registry: Option<&SuiteRegistry>,
) -> Result<(), TxPoolAdmitError> {
    // Wave-4 class-closure conservatism: defer when the slow path
    // (`apply_non_coinbase_tx_basic_update_*` + `validate_tx_covenants_genesis`)
    // would return `Rejected` (terminal). Without these defers a
    // below-floor tx that ALSO has a structural defect would be
    // misclassified as transient `Unavailable` instead of `Rejected`
    // (terminal), masking the structural error and allowing callers to
    // retry forever.
    //
    // tx_nonce == 0 for non-coinbase: the slow path returns
    // `Rejected("tx_nonce must be >= 1 for non-coinbase")` at
    // `clients/rust/crates/rubin-consensus/src/utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`.
    // Same defer added in Go's `cheapFeeFloorPrecheck` so admit/relay
    // endpoints behave identically across clients.
    if tx.tx_kind != 0x00 || !tx.da_payload.is_empty() || tx.tx_nonce == 0 {
        return Ok(());
    }
    let Some(input_value) =
        fee_precheck_p2pk_input_value(tx, utxos, next_height, rotation, registry)
    else {
        return Ok(());
    };
    let Some(output_value) = fee_precheck_p2pk_output_value(&tx.outputs, next_height, rotation)
    else {
        return Ok(());
    };
    if output_value > input_value {
        return Ok(());
    }
    if weight == 0 {
        return Ok(());
    }
    // Single-input P2PK shape only, so this difference cannot exceed u64 and
    // the local u64 arithmetic is exact here. It is a fast-reject hint, never
    // the authoritative admitted fee: that stays the consensus-derived u128
    // value carried by the apply summary.
    let fee = u128::from(input_value - output_value);
    validate_fee_floor(fee, weight, current_min_fee_rate)
}

/// Returns the P2PK input value when `tx` has exactly one input AND
/// that input is BOTH structurally valid (witness count == 1, no
/// coinbase-prevout marker, empty script_sig, sequence in standard
/// range) AND resolves in `utxos` to a `COV_TYPE_P2PK` entry. Returns
/// `None` for any other shape so the caller defers to the expensive
/// admission path.
///
/// Wave-4 class-closure conservatism: each input-side guard mirrors a
/// terminal-reject branch in the slow path
/// `apply_non_coinbase_tx_basic_update_*` at
/// `clients/rust/crates/rubin-consensus/src/utxo_basic.rs`. Without
/// these defers a below-floor tx with a structurally-defective input
/// would be misclassified as transient `Unavailable` instead of
/// terminal `Rejected`, masking the structural error and allowing
/// callers to retry forever. Mirrors Go's `feePrecheckP2PKInputValue`
/// in `clients/go/node/mempool_precheck_input.go` (Go got the same
/// wave-4 guard set in this PR; helpers were extracted from
/// `mempool.go` in the wave-9..11 file split).
///
/// Scope-cap: P2PK signature-verification failure is NOT classified
/// here. Verifying ML-DSA signatures is the expensive operation this
/// fast-reject is designed to avoid. Below-floor txs with invalid
/// signatures may surface as rolling-floor `Unavailable` until the
/// fee floor no longer applies; this is intentional.
fn fee_precheck_p2pk_input_value(
    tx: &rubin_consensus::Tx,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
    next_height: u64,
    rotation: Option<&dyn RotationProvider>,
    registry: Option<&SuiteRegistry>,
) -> Option<u64> {
    use rubin_consensus::constants::{COINBASE_MATURITY, MAX_P2PK_COVENANT_DATA};
    use rubin_consensus::is_valid_sighash_type;
    use sha3::{Digest, Sha3_256};
    if tx.inputs.len() != 1 {
        return None;
    }
    if tx.witness.len() != 1 {
        return None;
    }
    let input = &tx.inputs[0];
    // Coinbase-prevout marker on a non-coinbase input: slow path
    // returns Rejected (terminal) via the parse-time check at
    // `clients/rust/crates/rubin-consensus/src/utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`.
    if input.prev_txid == [0u8; 32] && input.prev_vout == 0xffff_ffff {
        return None;
    }
    // Non-empty script_sig on a P2PK input: slow path returns Rejected
    // (terminal) via the parse-time check at `utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`.
    if !input.script_sig.is_empty() {
        return None;
    }
    // Out-of-range sequence on a P2PK input: slow path returns Rejected
    // (terminal) via the parse-time check at `utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`.
    if input.sequence > 0x7fff_ffff {
        return None;
    }
    let outpoint = Outpoint {
        txid: input.prev_txid,
        vout: input.prev_vout,
    };
    let entry = utxos.get(&outpoint)?;
    if entry.covenant_type != rubin_consensus::constants::COV_TYPE_P2PK {
        return None;
    }
    if entry.created_by_coinbase
        && (next_height < entry.creation_height
            || next_height - entry.creation_height < COINBASE_MATURITY)
    {
        return None;
    }
    // Wave-14 witness-item structural validation. Each defer mirrors a
    // terminal-reject branch in the slow-path `validate_p2pk_spend_q`
    // at `clients/rust/crates/rubin-consensus/src/spend_verify.rs` `validate_p2pk_spend_q`.
    // ML-DSA signature verification itself stays out of the precheck by
    // design (it is the expensive operation this fast-reject is built to
    // skip); cheap structural checks below exercise everything except
    // the cryptographic verify step + key-binding sha3.
    let w = &tx.witness[0];
    // Wave-22 hot-path cache (Copilot wave-21 P2 #1+#2): when the
    // caller passes `rotation=None`, reuse the cached default
    // native_spend set instead of allocating a fresh `BTreeSet<u8>`
    // per tx via `DefaultRotationProvider::native_spend_suites`. The
    // default set is height-independent so a single static instance
    // is correct for all heights.
    let in_native_spend = match rotation {
        Some(rp) => rp.native_spend_suites(next_height).contains(w.suite_id),
        None => cached_default_native_spend_set().contains(w.suite_id),
    };
    if !in_native_spend {
        return None; // mirrors spend_verify.rs `validate_p2pk_spend_q` SigAlgInvalid
    }
    // Wave-22 fix per Copilot wave-21 P2 #1: cached static fallback
    // (no per-tx BTreeMap construction in the spam-reject hot path).
    // Closure (not bare fn-ptr) is required: lifetime variance of
    // `&'static SuiteRegistry` vs `&'1 SuiteRegistry` (the inferred
    // lifetime of `registry: Option<&SuiteRegistry>`) means
    // `unwrap_or_else(cached_default_registry)` fails type-check
    // (E0521). Allowing `clippy::redundant_closure` for this site.
    #[allow(clippy::redundant_closure)]
    let reg: &SuiteRegistry = registry.unwrap_or_else(|| cached_default_registry());
    let params = reg.lookup(w.suite_id)?; // mirrors `validate_p2pk_spend_q` SigAlgInvalid
    if w.pubkey.len() as u64 != params.pubkey_len || w.signature.len() as u64 != params.sig_len + 1
    {
        return None; // mirrors `validate_p2pk_spend_q` SigNoncanonical
    }
    // Wave-15 panic-safety + suite consistency. The covenant_data length
    // check MUST precede the [0] / [1..33] indexing because
    // `chain_state_from_disk` accepts arbitrary persisted covenant_data
    // bytes without per-covenant structure validation, so a corrupted
    // on-disk UTXO entry could otherwise panic the admission loop on the
    // next spend. Mirror of slow-path spend_verify.rs (`validate_p2pk_spend_q`).
    if entry.covenant_data.len() as u64 != MAX_P2PK_COVENANT_DATA {
        return None; // panic-safety + mirrors `validate_p2pk_spend_q` CovenantTypeInvalid
    }
    if entry.covenant_data[0] != w.suite_id {
        return None; // mirrors `validate_p2pk_spend_q` CovenantTypeInvalid
    }
    // Wave-16 sighash trailer: defer only on INVALID sighash type. The
    // slow path's `is_valid_sighash_type` (sighash.rs (`is_valid_sighash_type`)) accepts six
    // canonical trailers (SIGHASH_ALL/NONE/SINGLE × ANYONECANPAY); only
    // bytes outside that set are terminal-rejected. Wave-15's literal
    // `== SIGHASH_ALL` check over-deferred 5/6 valid types and let
    // attackers flip the trailer byte to bypass the cheap reject —
    // hostile-reviewer P1. Free check (single byte compare).
    let &trailer = w.signature.last()?;
    if !is_valid_sighash_type(trailer) {
        return None;
    }
    // Wave-15 key-binding: SHA3(pubkey) must match covenant_data[1..33].
    // Cost: one SHA3 hash on a ~2.6KB pubkey ≪ ML-DSA verify (the
    // documented scope-cap). Slow path's `verify_mldsa_key_and_sig_q`
    // at spend_verify.rs (`validate_p2pk_spend_q`) returns
    // `SigInvalid("CORE_P2PK key binding mismatch")`.
    // Wave-17: use sha3 crate directly instead of re-exporting
    // sha3_256 from rubin-consensus (revert wave-15 lib.rs export to
    // keep consensus public surface unchanged — Copilot wave-15+16 P1).
    let pubkey_hash: [u8; 32] = {
        let mut h = Sha3_256::new();
        h.update(&w.pubkey);
        h.finalize().into()
    };
    if pubkey_hash[..] != entry.covenant_data[1..33] {
        return None;
    }
    Some(entry.value)
}

/// Returns the sum of P2PK output values when every output is a
/// CONSENSUS-VALID `COV_TYPE_P2PK` (non-zero value, exactly
/// `MAX_P2PK_COVENANT_DATA == 33` byte covenant_data, and a suite_id
/// in the active `native_create_suites(next_height)` set) and the
/// running sum does not overflow `u64`. Returns `None` for any other
/// shape so the caller defers to the expensive admission path. The
/// extra structural guards mirror the slow-path checks at
/// `clients/rust/crates/rubin-consensus/src/covenant_genesis.rs` `validate_tx_covenants_genesis`
/// — without them a below-floor tx with consensus-invalid P2PK
/// outputs would be misclassified as transient `Unavailable` instead
/// of permanent `Rejected`. Mirrors Go's `feePrecheckP2PKOutputValue`
/// in `clients/go/node/mempool_precheck_output.go` (full function
/// body including the `bits.Add64` overflow defer; Go got the same
/// wave-4 guard set in this PR; helper was extracted from
/// `mempool.go` in the wave-9..11 file split).
fn fee_precheck_p2pk_output_value(
    outputs: &[rubin_consensus::TxOutput],
    next_height: u64,
    rotation: Option<&dyn RotationProvider>,
) -> Option<u64> {
    use rubin_consensus::constants::MAX_P2PK_COVENANT_DATA;
    // Wave-22 hot-path cache: same rationale as input-side helper.
    let owned_create_set: Option<NativeSuiteSet> =
        rotation.map(|rp| rp.native_create_suites(next_height));
    let native_suites: &NativeSuiteSet = owned_create_set
        .as_ref()
        .unwrap_or_else(|| cached_default_native_create_set());
    let mut total: u64 = 0;
    for output in outputs {
        if output.covenant_type != rubin_consensus::constants::COV_TYPE_P2PK {
            return None;
        }
        // Wave-4 class-closure conservatism: each guard mirrors a
        // permanent-reject branch in `validate_tx_covenants_genesis`
        // (covenant_genesis.rs `validate_tx_covenants_genesis`).
        if output.value == 0 {
            return None;
        }
        if output.covenant_data.len() as u64 != MAX_P2PK_COVENANT_DATA {
            return None;
        }
        let suite_id = output.covenant_data[0];
        if !native_suites.contains(suite_id) {
            return None;
        }
        total = total.checked_add(output.value)?;
    }
    Some(total)
}

/// `weight` and `da_bytes` MUST be the result of one
/// `tx_weight_and_stats_public(tx)` call performed by the caller before
/// invoking this function (see `reject_da_anchor_tx_policy` docstring
/// for the full RUB-167 single-walk invariant).
pub(crate) fn apply_policy(
    tx: &rubin_consensus::Tx,
    weight: u64,
    da_bytes: u64,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
    next_height: u64,
    cfg: &TxPoolConfig,
) -> Result<(), String> {
    if cfg.policy_reject_non_coinbase_anchor_outputs {
        reject_non_coinbase_anchor_outputs(tx)?;
    }
    reject_da_anchor_tx_policy(
        tx,
        weight,
        da_bytes,
        utxos,
        cfg.policy_current_mempool_min_fee_rate,
        cfg.policy_min_da_fee_rate,
        cfg.policy_da_surcharge_per_byte,
    )?;
    // Mirror of Go `applyPolicyAgainstStateSimplicity`
    // (`clients/go/node/mempolicy_helpers.go`). On the admission/relay
    // paths this arm is unreachable in practice — the pre-consensus gate
    // in `add_tx_with_source`/`relay_metadata` fires first — but it is
    // the arm that covers the miner candidate path (`apply_policy` is
    // the Rust miner's policy vehicle, like Go's
    // `rejectCandidateSimplicityPolicy`).
    if cfg.policy_reject_simplicity_pre_activation {
        let rotation = cfg
            .suite_context
            .as_ref()
            .map(|ctx| ctx.rotation.as_ref() as &dyn RotationProvider);
        if let Some(reason) =
            reject_core_simplicity_pre_activation(tx, utxos, next_height, rotation)?
        {
            return Err(reason);
        }
    }
    Ok(())
}

/// Missing-input precedence for the CORE_SIMPLICITY gate, mirroring Go
/// `policyInputSnapshot` (`clients/go/node/mempool_policy.go:150`): Go
/// resolves input outpoints into the policy snapshot BEFORE
/// `rejectCoreSimplicityPreActivation` and rejects a missing input with
/// `TX_ERR_MISSING_UTXO: utxo not found` first, so the policy reason
/// never masks the consensus structural error. Called ONLY for
/// CORE_SIMPLICITY-involved txs (the call sites gate on
/// `covenant_policy_kind(.., COV_TYPE_CORE_SIMPLICITY)`): Rust must
/// reject actively rather than fall through, because its consensus
/// pipeline checks output covenants (incl. the CORE_SIMPLICITY
/// deployment gate) before input resolution, so a fall-through would
/// surface `TX_ERR_COVENANT_TYPE_INVALID` instead of Go's
/// missing-UTXO-first precedence. The message is the BARE consensus
/// `TxError` text (`utxo_basic.rs` "utxo not found"), byte-identical to
/// Go's `txAdmitRejected(err.Error())`. Non-Simplicity txs (coinbase,
/// plain spends) are deliberately NOT routed here — they keep their
/// pre-existing consensus reject path; extending Go's unconditional
/// policyInputSnapshot to every tx is a separate general-policy surface.
/// The miner path also has NO such guard: Go's miner helper calls the
/// gate directly on the chain UTXO view (no snapshot).
fn reject_missing_policy_inputs(
    tx: &rubin_consensus::Tx,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
) -> Result<(), TxPoolAdmitError> {
    for input in &tx.inputs {
        let outpoint = Outpoint {
            txid: input.prev_txid,
            vout: input.prev_vout,
        };
        if !utxos.contains_key(&outpoint) {
            // BARE `TxError` text (no "transaction rejected: " wrapper): Go's
            // policyInputSnapshot rejects via `txAdmitRejected(err.Error())`
            // (`clients/go/node/mempool_precheck.go`), whose harness `err` is
            // the plain `TX_ERR_MISSING_UTXO: utxo not found`. Verified equal
            // on both CLIs for plain-P2PK AND CORE_SIMPLICITY missing-input
            // txs. The surrounding parse/consensus rejects keep their
            // pre-existing `transaction rejected:` wrapper — that Rust-wide
            // convention vs Go's bare text is a separate, pre-existing
            // divergence outside this contract.
            let err = rubin_consensus::TxError::new(
                rubin_consensus::ErrorCode::TxErrMissingUtxo,
                "utxo not found",
            );
            return Err(rejected(format!("{err}")));
        }
    }
    Ok(())
}

/// Mirror of Go `rejectCoreSimplicityPreActivation`
/// (`clients/go/node/policy_simplicity.go`): pre-activation fail-closed
/// CORE_SIMPLICITY (0x0106) mempool/relay/miner policy.
///
/// Returns `Ok(None)` when the transaction neither creates nor spends a
/// CORE_SIMPLICITY output, or when the rotation provider reports the
/// Simplicity deployment active at `height`. Returns `Ok(Some(reason))`
/// with the policy reject reason (`CORE_SIMPLICITY {output|spend}
/// pre-ACTIVE` — byte-identical to Go, no client name) for a
/// structurally valid pre-activation create/spend. Returns
/// `Err(message)` when the forced-active genesis revalidation surfaces a
/// consensus structural error: a malformed non-Simplicity covenant must
/// keep its consensus error instead of being masked by the policy
/// reason (Go `TestMempoolPolicyDoesNotMaskMalformedNonSimplicityOutput`);
/// the message is the BARE `TxError` text (no `transaction rejected: `
/// wrapper), byte-identical to Go's `txAdmitRejected(err.Error())`, which
/// the caller forwards through `rejected(...)` unchanged.
///
/// Go's `SimplicityDeploymentProvider.SimplicityActiveAtHeight` can fail
/// ("CORE_SIMPLICITY deployment lookup failure" reject branch); the Rust
/// `RotationProvider::simplicity_active_at_height` is infallible, so
/// that branch is statically unreachable here.
fn reject_core_simplicity_pre_activation(
    tx: &rubin_consensus::Tx,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
    height: u64,
    rotation: Option<&dyn RotationProvider>,
) -> Result<Option<String>, String> {
    let Some(kind) = covenant_policy_kind(tx, utxos, COV_TYPE_CORE_SIMPLICITY) else {
        return Ok(None);
    };
    let active = rotation
        .map(|provider| provider.simplicity_active_at_height(height))
        .unwrap_or(false);
    if active {
        return Ok(None);
    }
    let forced = ActiveSimplicityGenesisRotation { inner: rotation };
    if let Err(err) = validate_tx_covenants_genesis(tx, height, Some(&forced)) {
        // BARE text to match Go: `rejectCoreSimplicityPreActivation` returns
        // the raw `ValidateTxCovenantsGenesis` error and the caller emits
        // `txAdmitRejected(err.Error())` (no wrapper), so a masked-consensus
        // reject reads identically Go vs Rust.
        return Err(format!("{err}"));
    }
    Ok(Some(format!("CORE_SIMPLICITY {kind} pre-ACTIVE")))
}

/// Mirror of Go `activeSimplicityGenesisRotation`: delegates suite
/// rotation to the wrapped provider (or `DefaultRotationProvider` when
/// none is configured, mirroring the Go nil fallback) while forcing the
/// Simplicity deployment active, so the pre-activation policy can re-run
/// genesis covenant checks without the deployment gate masking
/// structural errors.
struct ActiveSimplicityGenesisRotation<'a> {
    inner: Option<&'a dyn RotationProvider>,
}

impl RotationProvider for ActiveSimplicityGenesisRotation<'_> {
    fn native_create_suites(&self, height: u64) -> NativeSuiteSet {
        match self.inner {
            Some(inner) => inner.native_create_suites(height),
            None => DefaultRotationProvider.native_create_suites(height),
        }
    }

    fn native_spend_suites(&self, height: u64) -> NativeSuiteSet {
        match self.inner {
            Some(inner) => inner.native_spend_suites(height),
            None => DefaultRotationProvider.native_spend_suites(height),
        }
    }

    fn simplicity_active_at_height(&self, _height: u64) -> bool {
        true
    }
}

fn covenant_policy_kind(
    tx: &rubin_consensus::Tx,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
    covenant_type: u16,
) -> Option<&'static str> {
    if tx
        .outputs
        .iter()
        .any(|output| output.covenant_type == covenant_type)
    {
        return Some("output");
    }
    for input in &tx.inputs {
        let outpoint = Outpoint {
            txid: input.prev_txid,
            vout: input.prev_vout,
        };
        if utxos
            .get(&outpoint)
            .is_some_and(|entry| entry.covenant_type == covenant_type)
        {
            return Some("spend");
        }
    }
    None
}

pub(crate) fn reject_non_coinbase_anchor_outputs(tx: &rubin_consensus::Tx) -> Result<(), String> {
    if tx
        .outputs
        .iter()
        .any(|output| output.covenant_type == rubin_consensus::constants::COV_TYPE_ANCHOR)
    {
        return Err("non-coinbase CORE_ANCHOR is non-standard (policy)".to_string());
    }
    Ok(())
}

/// Stage C DA fee policy aligned with Go's `RejectDaAnchorTxPolicy`
/// (`POLICY_MEMPOOL_ADMISSION_GENESIS.md` Stage C):
///
/// ```text
/// fee(tx)             = sum_inputs - sum_outputs
/// relay_fee_floor(tx) = weight(tx) * current_mempool_min_fee_rate
/// da_fee_floor(tx)    = da_payload_len(tx) * min_da_fee_rate
/// da_surcharge(tx)    = da_payload_len(tx) * da_surcharge_per_byte
/// da_required_fee(tx) = da_fee_floor(tx) + da_surcharge(tx)
/// required_fee(tx)    = max(relay_fee_floor(tx), da_required_fee(tx))
/// reject if fee(tx) < required_fee(tx)
/// ```
///
/// Arithmetic is checked widening; any overflow rejects fail-closed as a
/// policy error. The helper does not change consensus validity. For non-DA
/// transactions (`da_payload_len == 0`) the helper short-circuits with
/// `Ok(())` and applies no DA-specific term; relay-fee-floor enforcement
/// for non-DA transactions remains the standard mempool admission path's
/// responsibility.
///
/// Inputs:
/// - `current_mempool_min_fee_rate`: rolling local mempool floor. Callers
///   without a live rolling-floor source pass
///   `DEFAULT_MEMPOOL_MIN_FEE_RATE` (mirrors Go's documented
///   `DefaultMempoolMinFeeRate` pattern, not a parallel floor invention).
/// - `min_da_fee_rate`: spec-side DA per-byte floor (Stage C
///   `min_da_fee_rate`, default `1`).
/// - `da_surcharge_per_byte`: operator-tunable DA per-byte surcharge
///   added on top of the spec-side floor; `0` disables only the
///   surcharge term, not `da_fee_floor`.
///
/// `weight` and `da_bytes` are passed in by the caller, computed once
/// per admission/relay decision via `tx_weight_and_stats_public(tx)`.
/// Callers MUST pass values from a single `tx_weight_and_stats_public`
/// call performed AFTER `parse_tx` succeeds and BEFORE invoking
/// `apply_policy`; the same `weight` MUST be reused by any sibling
/// rolling-fee-floor check (`validate_fee_floor`) so the DA-side
/// classification and rolling-floor classification operate on identical
/// values.
///
/// This is the deliberate INVERSE of Go's `RejectDaAnchorTxPolicy`
/// (`clients/go/node/policy_da_anchor.go`), which recomputes
/// `weight, daBytes, _, err := consensus.TxWeightAndStats(tx)`
/// internally and explicitly distrusts caller-supplied values. The
/// Rust helper trusts the caller because (a) it stays `pub(crate)` so
/// every callsite is audited in-crate (`admit_with_metadata`,
/// `relay_metadata`, `apply_post_consensus_policy_with_floor`,
/// `apply_post_consensus_policy_without_floor`, `miner::reject_candidate`,
/// and the `run_da_policy` test helper all walk weight at the call site),
/// and (b) reusing one `weight` value across both
/// `reject_da_anchor_tx_policy` (which also consumes `da_bytes`) and
/// `validate_fee_floor` removes the drift class where the DA and
/// rolling-floor halves could otherwise see different `weight` values.
pub(crate) fn reject_da_anchor_tx_policy(
    tx: &rubin_consensus::Tx,
    weight: u64,
    da_bytes: u64,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
    current_mempool_min_fee_rate: u64,
    min_da_fee_rate: u64,
    da_surcharge_per_byte: u64,
) -> Result<(), String> {
    if da_bytes == 0 {
        // Non-DA transaction: the helper only enforces the DA half of the
        // Stage C admission contract. Non-DA relay-fee floor enforcement
        // is performed by the free `validate_fee_floor` predicate: relay
        // calls it via `apply_post_consensus_policy_with_floor`, while
        // admission calls it after the Go-compatible duplicate/conflict
        // boundary. Both paths zero-out the rolling floor before invoking
        // this helper so the DA-side classification is preserved without
        // double-charging the relay floor inside the DA helper. The helper
        // deliberately short-circuits here for non-DA transactions and does
        // not compute fee or apply any DA-specific term.
        return Ok(());
    }
    // The relay-floor term is exact and never rejects: weight and the rolling
    // rate are both u64, so their full product is at most (2^64-1)^2 and fits
    // u128 by construction. Mirrors Go `mulU64Exact` in policy_da_anchor.go
    // and the already-exact `fee_below_rate` predicate. The DA-side terms
    // below keep their u64 overflow errors, whose reject reasons are pinned
    // by the shared CV-DA-FEE-FLOOR vector and both consensus-CLI models.
    let relay_floor = u128::from(weight) * u128::from(current_mempool_min_fee_rate);
    let da_floor = da_bytes.checked_mul(min_da_fee_rate).ok_or_else(|| {
        format!(
            "DA fee floor overflow (da_payload_len={da_bytes} min_da_fee_rate={min_da_fee_rate}): u64 overflow"
        )
    })?;
    let da_surcharge = da_bytes.checked_mul(da_surcharge_per_byte).ok_or_else(|| {
        format!(
            "DA surcharge overflow (da_payload_len={da_bytes} surcharge_per_byte={da_surcharge_per_byte}): u64 overflow"
        )
    })?;
    let da_required = da_floor.checked_add(da_surcharge).ok_or_else(|| {
        format!(
            "DA required fee overflow (da_fee_floor={da_floor} da_surcharge={da_surcharge}): u64 overflow"
        )
    })?;
    let required = relay_floor.max(u128::from(da_required));
    if required == 0 {
        // DA tx but every Stage C rate-derived fee term is zero: the
        // relay-floor term is zero and both DA-side terms are zero.
        // Nothing to enforce; admit without fee compute.
        return Ok(());
    }
    let fee = compute_fee_no_verify(tx, utxos)
        .map_err(|err| format!("cannot compute fee for DA tx (policy): {err}"))?;
    if fee < required {
        return Err(format!(
            "DA fee below Stage C floor (fee={fee} required_fee={required} relay_fee_floor={relay_floor} da_fee_floor={da_floor} da_surcharge={da_surcharge} weight={weight} da_payload_len={da_bytes})"
        ));
    }
    Ok(())
}

/// Derives the policy-side fee as an exact u128, matching the consensus
/// derivation's domain and Go `computeFeeNoVerify` in `policy_da_anchor.go`.
///
/// Per-input and per-output values stay u64; only their sums and the
/// difference widen, so a DA transaction whose fee exceeds u64 is measured
/// against the Stage C floor exactly instead of failing an overflow check.
/// Two u64-max inputs sum to 2^64 and derive a fee, where a u64 accumulator
/// returned `sum_in overflow` and diverged from Go.
pub(crate) fn compute_fee_no_verify(
    tx: &rubin_consensus::Tx,
    utxos: &HashMap<Outpoint, rubin_consensus::UtxoEntry>,
) -> Result<u128, String> {
    if tx.inputs.is_empty() {
        return Err("missing inputs".to_string());
    }
    let mut sum_in = 0u128;
    for input in &tx.inputs {
        let outpoint = Outpoint {
            txid: input.prev_txid,
            vout: input.prev_vout,
        };
        let entry = utxos
            .get(&outpoint)
            .ok_or_else(|| "missing utxo".to_string())?;
        sum_in = sum_in
            .checked_add(u128::from(entry.value))
            .ok_or_else(|| "sum_in overflow".to_string())?;
    }
    let mut sum_out = 0u128;
    for output in &tx.outputs {
        sum_out = sum_out
            .checked_add(u128::from(output.value))
            .ok_or_else(|| "sum_out overflow".to_string())?;
    }
    if sum_out > sum_in {
        return Err("overspend".to_string());
    }
    Ok(sum_in - sum_out)
}

fn compare_entries_for_mining(
    a: &(&[u8; 32], &TxPoolEntry),
    b: &(&[u8; 32], &TxPoolEntry),
) -> Ordering {
    let (a_txid, a_entry) = *a;
    let (b_txid, b_entry) = *b;
    match compare_fee_rate(a_entry, b_entry) {
        Ordering::Greater => Ordering::Less,
        Ordering::Less => Ordering::Greater,
        Ordering::Equal => match b_entry.fee.cmp(&a_entry.fee) {
            Ordering::Equal => match a_entry.weight.cmp(&b_entry.weight) {
                // Final tie-break matches Go parity
                // (`clients/go/node/mempool.go`, `sortMempoolEntries`:
                // `bytes.Compare(entries[i].txid[:], entries[j].txid[:])`)
                // and the Rust-internal `compare_priority_values` admit
                // comparator, both of which tie by TXID (lexicographic).
                // Using raw serialized bytes here (prior behavior) caused
                // deterministic ordering to drift between clients on
                // equal-fee/weight/size transaction sets.
                Ordering::Equal => a_txid.cmp(b_txid),
                other => other,
            },
            other => other,
        },
    }
}

#[cfg(test)]
fn compare_admit_priority(
    txid_a: [u8; 32],
    a: &TxPoolEntry,
    txid_b: [u8; 32],
    b: &TxPoolEntry,
) -> Ordering {
    compare_priority_values(
        AdmitPriority {
            fee: a.fee,
            weight: a.weight,
            tie: &txid_a,
        },
        AdmitPriority {
            fee: b.fee,
            weight: b.weight,
            tie: &txid_b,
        },
    )
}

fn compare_priority_values(a: AdmitPriority<'_>, b: AdmitPriority<'_>) -> Ordering {
    match compare_fee_rate_values(a.fee, a.weight, b.fee, b.weight) {
        Ordering::Equal => match a.fee.cmp(&b.fee) {
            Ordering::Equal => match b.weight.cmp(&a.weight) {
                Ordering::Equal => b.tie.cmp(a.tie),
                other => other,
            },
            other => other,
        },
        other => other,
    }
}

fn compare_fee_rate(a: &TxPoolEntry, b: &TxPoolEntry) -> Ordering {
    compare_fee_rate_values(a.fee, a.weight, b.fee, b.weight)
}

fn default_tx_pool_low_water_bytes(max_bytes: usize) -> usize {
    if max_bytes == 0 {
        return 0;
    }
    let low_water = (max_bytes / TX_POOL_LOW_WATER_DENOMINATOR) * TX_POOL_LOW_WATER_NUMERATOR
        + ((max_bytes % TX_POOL_LOW_WATER_DENOMINATOR) * TX_POOL_LOW_WATER_NUMERATOR
            / TX_POOL_LOW_WATER_DENOMINATOR);
    if low_water == 0 {
        return 1;
    }
    low_water
}

fn tx_pool_byte_pressure_target(low_water_bytes: usize, candidate_size: usize) -> usize {
    low_water_bytes.max(candidate_size)
}

fn worst_capacity_plan_index(
    plan_pool: &[CapacityPlanEntry<'_>],
    ordering: CapacityOrdering,
) -> usize {
    let mut worst_index = 0;
    for i in 1..plan_pool.len() {
        if capacity_plan_entry_worse(&plan_pool[i], &plan_pool[worst_index], ordering) {
            worst_index = i;
        }
    }
    worst_index
}

fn capacity_plan_entry_worse(
    a: &CapacityPlanEntry<'_>,
    b: &CapacityPlanEntry<'_>,
    ordering: CapacityOrdering,
) -> bool {
    let priority = match ordering {
        CapacityOrdering::LegacyCountPressure => compare_count_pressure_priority(a, b),
        CapacityOrdering::BytePressure => compare_capacity_priority(a, b),
    };
    match priority {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => a.txid > b.txid,
    }
}

fn compare_count_pressure_priority(
    a: &CapacityPlanEntry<'_>,
    b: &CapacityPlanEntry<'_>,
) -> Ordering {
    compare_priority_values(
        AdmitPriority {
            fee: a.entry.fee,
            weight: a.entry.weight,
            tie: &a.txid,
        },
        AdmitPriority {
            fee: b.entry.fee,
            weight: b.entry.weight,
            tie: &b.txid,
        },
    )
}

fn compare_capacity_priority(a: &CapacityPlanEntry<'_>, b: &CapacityPlanEntry<'_>) -> Ordering {
    match compare_fee_rate_values(a.entry.fee, a.entry.weight, b.entry.fee, b.entry.weight) {
        Ordering::Equal => match a.entry.fee.cmp(&b.entry.fee) {
            Ordering::Equal => match a.admission_seq.cmp(&b.admission_seq) {
                Ordering::Equal => Ordering::Equal,
                other => other,
            },
            other => other,
        },
        other => other,
    }
}

/// Compares fee_a/weight_a against fee_b/weight_b by exact
/// cross-multiplication over all 192 product bits of a u128 fee by a u64
/// weight. Mirrors Go `consensus.CompareFeeRate`.
fn compare_fee_rate_values(fee_a: u128, weight_a: u64, fee_b: u128, weight_b: u64) -> Ordering {
    compare_fee_rate_exact(fee_a, weight_a, fee_b, weight_b)
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;

    use rubin_consensus::block::BLOCK_HEADER_BYTES;
    use rubin_consensus::constants::{
        COV_TYPE_ANCHOR, COV_TYPE_CORE_EXT, COV_TYPE_CORE_SIMPLICITY, COV_TYPE_P2PK, MAX_TX_INPUTS,
        SUITE_ID_SENTINEL, TX_WIRE_VERSION,
    };
    use rubin_consensus::{
        marshal_tx, p2pk_covenant_data_for_pubkey, parse_tx, sign_transaction,
        tx_weight_and_stats_public, DaChunkCore, Mldsa87Keypair, Outpoint, Tx, TxInput, TxOutput,
        UtxoEntry, WitnessItem,
    };

    use super::{
        cheap_fee_floor_precheck, compare_admit_priority, compare_entries_for_mining,
        compare_fee_rate, compute_fee_no_verify, conflict, default_tx_pool_low_water_bytes,
        fee_precheck_p2pk_input_value, fee_precheck_p2pk_output_value, mtp_median,
        next_block_height, next_block_mtp, reject_da_anchor_tx_policy, rejected, relay_metadata,
        tx_pool_byte_pressure_target, unavailable, TxPool, TxPoolAdmitErrorKind, TxPoolConfig,
        TxPoolEntry, TxPoolSnapshot, TxPoolSnapshotEntry, TxSource, DEFAULT_MEMPOOL_MIN_FEE_RATE,
        MAX_TX_POOL_TRANSACTIONS,
    };
    use crate::{
        block_store_path, default_sync_config, devnet_genesis_block_bytes, devnet_genesis_chain_id,
        test_helpers::signed_conflicting_p2pk_state_and_txs, BlockStore, ChainState, SyncEngine,
    };

    #[derive(serde::Deserialize)]
    struct FixtureFile<T> {
        vectors: Vec<T>,
    }

    #[derive(Clone, serde::Deserialize)]
    struct FixtureUtxo {
        txid: String,
        vout: u32,
        value: u64,
        covenant_type: u16,
        covenant_data: String,
        creation_height: u64,
        created_by_coinbase: bool,
    }

    #[derive(Clone, serde::Deserialize)]
    struct PositiveTxVector {
        id: String,
        tx_hex: String,
        #[serde(default)]
        chain_id: Option<String>,
        height: u64,
        expect_ok: bool,
        utxos: Vec<FixtureUtxo>,
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ))
    }

    fn create_block_store(prefix: &str) -> (BlockStore, PathBuf) {
        let dir = unique_temp_dir(prefix);
        fs::create_dir_all(&dir).expect("mkdir");
        let store = BlockStore::create(block_store_path(&dir)).expect("blockstore");
        (store, dir)
    }

    fn test_entry(fee: u128, weight: u64, size: usize, source: TxSource) -> TxPoolEntry {
        TxPoolEntry {
            raw: vec![0xA5; size],
            inputs: Vec::new(),
            fee,
            weight,
            size,
            source,
        }
    }

    #[test]
    #[rustfmt::skip]
    fn compact_defensive_branch_coverage() {
        let snapshot=|p:&TxPool|(p.txs.clone(),p.owner_tokens.clone(),p.heap_seqs.clone(),p.worst_heap.clone().into_sorted_vec(),p.next_heap_id,p.used_bytes);
        let input=Outpoint{txid:[3;32],vout:0}; let mut poisoned=TxPool::new(); poisoned.insert_entry([3;32],TxPoolEntry{inputs:vec![input.clone()],..test_entry(1,1,1,TxSource::Local)});
        let owner=poisoned.test_owner(); let clone=owner.clone(); let _=std::thread::spawn(move||{let _guard=clone.lock().unwrap(); panic!("poison");}).join(); let before=snapshot(&poisoned);
        poisoned.remove_conflicting_outpoints(std::slice::from_ref(&input)); assert_eq!(snapshot(&poisoned),before); poisoned.evict_txids(&[[3;32]]); assert_eq!(snapshot(&poisoned),before);
        let mut invalid=TxPool::new(); invalid.insert_entry([4;32],TxPoolEntry{inputs:vec![input.clone()],..test_entry(2,1,1,TxSource::Local)}); invalid.owner_tokens.remove(&[4;32]); let before=snapshot(&invalid); let owner_before=(invalid.test_owner().test_high_water(),invalid.owner_row_count(),invalid.owner_txid_for_outpoint(&input)); invalid.evict_txids(&[[4;32]]); assert_eq!(snapshot(&invalid),before); assert_eq!((invalid.test_owner().test_high_water(),invalid.owner_row_count(),invalid.owner_txid_for_outpoint(&input)),owner_before);
        assert_eq!(invalid.select_transactions(1,usize::MAX).len(),1); assert!(invalid.select_transactions(0,usize::MAX).is_empty()); assert!(invalid.select_transactions_with_filter(1,usize::MAX,|_|true).is_empty()); assert!(invalid.select_transactions(1,0).is_empty());
        let mut oversized=TxPool::new(); oversized.txs.insert([8;32],test_entry(1,1,2,TxSource::Local)); assert!(oversized.select_transactions(1,1).is_empty());
        let mut pool=TxPool::new(); pool.insert_entry([7;32],test_entry(3,2,2,TxSource::Reorg)); let saved_id=pool.next_heap_id; pool.next_heap_id=u64::MAX; let before=snapshot(&pool); let e=pool.preflight_standard_delta(&[],false).unwrap_err(); assert_eq!((e.kind,e.message.as_str()),(TxPoolAdmitErrorKind::Unavailable,"txpool heap sequence exhausted")); assert_eq!(snapshot(&pool),before);
        pool.next_heap_id=saved_id; let before=snapshot(&pool); let e=pool.preflight_standard_delta(&[[9;32]],false).unwrap_err(); assert_eq!((e.kind,e.message.as_str()),(TxPoolAdmitErrorKind::Unavailable,"txpool eviction plan changed before commit")); assert_eq!(snapshot(&pool),before);
        let before=snapshot(&pool); assert!(pool.remove_entry_raw(&[9;32]).is_none()); assert_eq!(snapshot(&pool),before);
    }

    fn txpool_snapshot_entry_from_raw(
        raw: Vec<u8>,
        fee: u128,
        source: TxSource,
        heap_id: u64,
    ) -> ([u8; 32], TxPoolSnapshotEntry) {
        let (tx, txid, wtxid, consumed) = parse_tx(&raw).expect("parse snapshot raw");
        assert_eq!(consumed, raw.len(), "snapshot raw must be canonical");
        let inputs = tx
            .inputs
            .iter()
            .map(|input| Outpoint {
                txid: input.prev_txid,
                vout: input.prev_vout,
            })
            .collect();
        let (weight, _, _) = tx_weight_and_stats_public(&tx).expect("snapshot weight");
        let size = raw.len();
        (
            txid,
            TxPoolSnapshotEntry {
                txid,
                wtxid,
                entry: TxPoolEntry {
                    raw,
                    inputs,
                    fee,
                    weight,
                    size,
                    source,
                },
                heap_id,
            },
        )
    }

    fn txpool_snapshot_test_pool() -> (TxPool, [u8; 32], [u8; 32]) {
        let (_state_a, raw_a) = signed_p2pk_state_and_tx(
            20_000,
            vec![TxOutput {
                value: 8,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x31; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );
        let (_state_b, raw_b) = core_ext_spend_state_and_tx(7);
        let (txid_a, item_a) = txpool_snapshot_entry_from_raw(raw_a, 19_992, TxSource::Remote, 1);
        let (txid_b, item_b) = txpool_snapshot_entry_from_raw(raw_b, 1, TxSource::Reorg, 2);
        let max_bytes = item_a.entry.size + item_b.entry.size + 100;
        let mut pool = TxPool::new_with_config(TxPoolConfig {
            policy_current_mempool_min_fee_rate: 7,
            ..TxPoolConfig::default()
        });
        pool.set_capacity_for_test(7, max_bytes);
        pool.insert_entry(txid_a, item_a.entry);
        pool.insert_entry(txid_b, item_b.entry);
        (pool, txid_a, txid_b)
    }

    fn genesis_coinbase_bytes() -> Vec<u8> {
        let block = devnet_genesis_block_bytes();
        assert_eq!(
            block[BLOCK_HEADER_BYTES], 0x01,
            "expected single-tx genesis fixture"
        );
        let tx_start = BLOCK_HEADER_BYTES + 1;
        let (_tx, _txid, _wtxid, consumed) = parse_tx(&block[tx_start..]).expect("parse tx");
        block[tx_start..tx_start + consumed].to_vec()
    }

    fn parse_hex32_test(name: &str, value: &str) -> [u8; 32] {
        let raw = hex::decode(value).unwrap_or_else(|err| panic!("{name} hex: {err}"));
        assert_eq!(raw.len(), 32, "{name} must be 32 bytes");
        let mut out = [0u8; 32];
        out.copy_from_slice(&raw);
        out
    }

    fn fixture_utxos_to_map(items: &[FixtureUtxo]) -> HashMap<Outpoint, UtxoEntry> {
        let mut out = HashMap::with_capacity(items.len());
        for item in items {
            out.insert(
                Outpoint {
                    txid: parse_hex32_test("fixture utxo txid", &item.txid),
                    vout: item.vout,
                },
                UtxoEntry {
                    value: item.value,
                    covenant_type: item.covenant_type,
                    covenant_data: hex::decode(&item.covenant_data)
                        .expect("fixture covenant_data hex"),
                    creation_height: item.creation_height,
                    created_by_coinbase: item.created_by_coinbase,
                },
            );
        }
        out
    }

    fn positive_fixture_vector() -> PositiveTxVector {
        const UTXO_BASIC_FIXTURE_JSON: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../conformance/fixtures/CV-UTXO-BASIC.json"
        ));
        let fixture: FixtureFile<PositiveTxVector> =
            serde_json::from_str(UTXO_BASIC_FIXTURE_JSON).expect("parse positive fixture");
        fixture
            .vectors
            .into_iter()
            .find(|vector| vector.id == "CV-U-06")
            .expect("positive fixture vector")
    }

    fn fixture_chain_id(chain_id: Option<&str>) -> [u8; 32] {
        chain_id
            .map(|value| parse_hex32_test("chain_id", value))
            .unwrap_or([0u8; 32])
    }

    fn chain_state_from_positive_fixture(vector: &PositiveTxVector) -> ChainState {
        let mut state = ChainState::new();
        state.has_tip = vector.height > 0;
        state.height = vector.height.saturating_sub(1);
        state.utxos = fixture_utxos_to_map(&vector.utxos);
        state
    }

    fn empty_core_ext_covenant_data(ext_id: u16) -> Vec<u8> {
        core_ext_covenant_data_with_payload(ext_id, &[])
    }

    fn core_ext_covenant_data_with_payload(ext_id: u16, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&ext_id.to_le_bytes());
        rubin_consensus::encode_compact_size(payload.len() as u64, &mut out);
        out.extend_from_slice(payload);
        out
    }

    fn signed_p2pk_state_and_tx(
        input_value: u64,
        outputs: Vec<TxOutput>,
        tx_kind: u8,
        da_chunk_core: Option<DaChunkCore>,
        da_payload: Vec<u8>,
    ) -> (ChainState, Vec<u8>) {
        let keypair = match Mldsa87Keypair::generate() {
            Ok(value) => value,
            Err(err) => panic!("OpenSSL signer unavailable for txpool policy test: {err}"),
        };
        let pubkey = keypair.pubkey_bytes();
        let outpoint = Outpoint {
            txid: [0x11; 32],
            vout: 0,
        };
        let mut state = ChainState::new();
        state.utxos.insert(
            outpoint.clone(),
            UtxoEntry {
                value: input_value,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&pubkey),
                creation_height: 0,
                created_by_coinbase: false,
            },
        );

        let mut tx = Tx {
            version: TX_WIRE_VERSION,
            tx_kind,
            tx_nonce: 7,
            inputs: vec![TxInput {
                prev_txid: outpoint.txid,
                prev_vout: outpoint.vout,
                script_sig: Vec::new(),
                sequence: 0,
            }],
            outputs,
            locktime: 0,
            da_commit_core: None,
            da_chunk_core,
            witness: Vec::new(),
            da_payload,
        };
        sign_transaction(&mut tx, &state.utxos, [0u8; 32], &keypair).expect("sign tx");
        let raw = marshal_tx(&tx).expect("marshal tx");
        (state, raw)
    }

    fn core_ext_spend_state_and_tx(ext_id: u16) -> (ChainState, Vec<u8>) {
        let input = Outpoint {
            txid: [0x33; 32],
            vout: 0,
        };
        let mut state = ChainState::new();
        state.utxos.insert(
            input.clone(),
            UtxoEntry {
                value: 10,
                covenant_type: COV_TYPE_CORE_EXT,
                covenant_data: empty_core_ext_covenant_data(ext_id),
                creation_height: 0,
                created_by_coinbase: false,
            },
        );
        let tx = Tx {
            version: TX_WIRE_VERSION,
            tx_kind: 0x00,
            tx_nonce: 9,
            inputs: vec![TxInput {
                prev_txid: input.txid,
                prev_vout: input.vout,
                script_sig: Vec::new(),
                sequence: 0,
            }],
            outputs: vec![TxOutput {
                value: 9,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x44; 2592]),
            }],
            locktime: 0,
            da_commit_core: None,
            da_chunk_core: None,
            witness: vec![WitnessItem {
                suite_id: SUITE_ID_SENTINEL,
                pubkey: Vec::new(),
                signature: Vec::new(),
            }],
            da_payload: Vec::new(),
        };
        let raw = marshal_tx(&tx).expect("marshal core_ext spend");
        (state, raw)
    }

    #[test]
    fn mtp_median_uses_sorted_middle_of_window() {
        let got = mtp_median(5, &[9, 3, 5, 1, 7]);
        assert_eq!(got, 5);
    }

    #[test]
    fn mtp_median_uses_available_history_when_window_is_short() {
        assert_eq!(mtp_median(3, &[9]), 9);
    }

    #[test]
    fn admit_rejects_parse_errors() {
        let mut pool = TxPool::new();
        let err = pool
            .admit(&[], &ChainState::new(), None, [0u8; 32])
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("transaction rejected"));
    }

    #[test]
    fn admit_rejects_sub_floor_conformance_tx_as_unavailable_with_atomicity() {
        // RUB-162 Phase A migration rationale (per controller Q2 record):
        //   - old assumption: TxPool::new() admits the canonical conformance
        //     tx; pre-RUB-162 admit_with_metadata did not enforce the rolling
        //     fee floor as a separate Unavailable classifier.
        //   - new invariant: admit_with_metadata classifies relay-floor
        //     failure as Unavailable via validate_fee_floor (mirrors
        //     Go validateFeeFloorLocked). The conformance fixture has fee=10
        //     weight=7653 (fee_rate ≈ 0.0013, far below DEFAULT=1) and the
        //     floor cannot be lowered via cfg because validate_fee_floor
        //     clamps to DEFAULT (Go parity).
        //   - why it reaches policy path: tx is well-formed; floor check is
        //     after apply_policy.
        //   - replacement coverage: this test now asserts Unavailable
        //     directly (the new correct behaviour); the floor-compliant
        //     index-population smoke equivalent lives in
        //     `rub162_admit_da_above_both_floors_admits_with_indexes`,
        //     DA fixture. The conformance fixture itself (RUB-54 scope)
        //     is unchanged; only the in-test pool config is adapted.
        let vector = positive_fixture_vector();
        assert!(vector.expect_ok, "{} should be positive fixture", vector.id);
        let raw = hex::decode(&vector.tx_hex).expect("tx hex");
        let (_tx, _txid, _wtxid, consumed) = parse_tx(&raw).expect("parse tx");
        assert_eq!(consumed, raw.len(), "{}", vector.id);

        let state = chain_state_from_positive_fixture(&vector);
        let mut pool = TxPool::new();
        let err = pool
            .admit(
                &raw,
                &state,
                None,
                fixture_chain_id(vector.chain_id.as_deref()),
            )
            .expect_err("conformance fixture has fee_rate well below DEFAULT_MEMPOOL_MIN_FEE_RATE");
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("mempool fee below rolling minimum"));

        assert_eq!(pool.len(), 0, "no entry inserted on Unavailable");
        assert!(
            pool.owner_row_count() == 0,
            "no owner claim rows inserted on Err"
        );
        assert_eq!(pool.heap_seqs.len(), 0, "no heap state on Err");
    }

    /// PR-1410 wave-2 migration: this test previously asserted that
    /// `relay_metadata` Ok'd the conformance fixture (fee=10 / weight≈
    /// 7653 ⇒ fee/weight ≈ 0.0013, far below DEFAULT_MEMPOOL_MIN_FEE_RATE).
    /// After the wave-2 fix `relay_metadata` enforces the same rolling
    /// floor as `admit_with_metadata`, so the conformance fixture is
    /// now sub-floor and rejects with Unavailable from the new floor
    /// check inside `relay_metadata`. The polarity-accurate name
    /// (`relay_metadata_rejects_sub_floor_conformance_tx_as_unavailable`)
    /// reflects the asserted behaviour.
    #[test]
    fn relay_metadata_rejects_sub_floor_conformance_tx_as_unavailable() {
        let vector = positive_fixture_vector();
        let raw = hex::decode(&vector.tx_hex).expect("tx hex");
        let state = chain_state_from_positive_fixture(&vector);

        let err = relay_metadata(
            &raw,
            &state,
            None,
            fixture_chain_id(vector.chain_id.as_deref()),
            &TxPoolConfig::default(),
        )
        .expect_err("conformance fixture has fee_rate well below DEFAULT_MEMPOOL_MIN_FEE_RATE");

        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(
            err.message.contains("mempool fee below rolling minimum"),
            "expected rolling-floor message, got: {}",
            err.message
        );
    }

    #[test]
    fn new_pool_starts_empty() {
        let pool = TxPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn default_impl_matches_empty_pool() {
        let pool = TxPool::default();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn txpool_snapshot_round_trip_preserves_rust_state_and_go_field_mapping() {
        let (mut pool, txid_a, txid_b) = txpool_snapshot_test_pool();
        let selected_before = pool.select_transactions(10, usize::MAX);
        let heap_before = pool.worst_heap.clone().into_sorted_vec();
        let snapshot = pool.snapshot().expect("snapshot");

        pool.evict_txids(&[txid_a, txid_b]);
        pool.cfg.policy_current_mempool_min_fee_rate = 99;
        assert!(pool.is_empty(), "live pool mutated before restore");

        pool.restore_snapshot(&snapshot)
            .expect("valid snapshot restores");
        assert_eq!(&pool.snapshot().expect("snapshot"), &snapshot);
        assert_eq!(pool.worst_heap.clone().into_sorted_vec(), heap_before);
        for item in &snapshot.entries {
            assert_eq!(pool.heap_seqs.get(&item.txid), Some(&item.heap_id));
            for input in &item.entry.inputs {
                assert_eq!(pool.owner_txid_for_outpoint(input), Some(item.txid));
            }
        }
        assert_eq!(pool.select_transactions(10, usize::MAX), selected_before);
        assert_eq!(pool.cfg.policy_current_mempool_min_fee_rate, 7);

        let lenient_snapshot = TxPool::new_with_config(TxPoolConfig {
            policy_current_mempool_min_fee_rate: 0,
            ..TxPoolConfig::default()
        })
        .snapshot()
        .expect("lenient snapshot");
        pool.restore_snapshot(&lenient_snapshot)
            .expect("empty snapshot restores");
        assert!(pool.cfg.policy_current_mempool_min_fee_rate == DEFAULT_MEMPOOL_MIN_FEE_RATE);
    }

    #[test]
    fn txpool_snapshot_restore_rejects_corruption_without_replacing_live_state() {
        let (guard_pool, _guard_a, _guard_b) = txpool_snapshot_test_pool();
        let guard_snapshot = guard_pool.snapshot().expect("guard snapshot");
        let (mut target, _target_a, _target_b) = txpool_snapshot_test_pool();
        let mut duplicate_spender = guard_snapshot.clone();
        let (_conflict_state, raw_a, raw_b) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let (_txid_a, item_a) = txpool_snapshot_entry_from_raw(raw_a, 7690, TxSource::Local, 1);
        let (_txid_b, item_b) = txpool_snapshot_entry_from_raw(raw_b, 7691, TxSource::Remote, 2);
        duplicate_spender.entries = vec![item_a, item_b];
        duplicate_spender.used_bytes = duplicate_spender
            .entries
            .iter()
            .map(|item| item.entry.size)
            .sum();
        duplicate_spender.next_heap_id = 2;
        target.set_capacity_for_test(7, duplicate_spender.used_bytes + 1);

        let mut reject = |poison: TxPoolSnapshot, needle: &str| {
            let before = target.snapshot().expect("target snapshot before poison");
            let err = target
                .restore_snapshot(&poison)
                .expect_err("poison snapshot must fail closed");
            assert!(
                err.message.contains(needle),
                "expected error containing {needle:?}, got: {}",
                err.message
            );
            assert_eq!(
                &target.snapshot().expect("target snapshot after poison"),
                &before
            );
        };

        let mut duplicate_txid = guard_snapshot.clone();
        let duplicate_item = duplicate_txid.entries[0].clone();
        duplicate_txid.entries.push(duplicate_item);
        reject(duplicate_txid, "duplicate txpool snapshot txid");
        reject(duplicate_spender, "duplicate txpool snapshot spender");
        let mut duplicate_heap = guard_snapshot.clone();
        duplicate_heap.entries[1].heap_id = duplicate_heap.entries[0].heap_id;
        reject(duplicate_heap, "duplicate txpool snapshot heap id");
        let mut duplicate_wtxid = guard_snapshot.clone();
        duplicate_wtxid.entries[1].wtxid = duplicate_wtxid.entries[0].wtxid;
        reject(duplicate_wtxid, "duplicate txpool snapshot wtxid");
        let mut missing_heap = guard_snapshot.clone();
        missing_heap.entries[0].heap_id = 0;
        reject(missing_heap, "invalid txpool snapshot heap id");
        let mut invalid_raw = guard_snapshot.clone();
        invalid_raw.entries[0].entry.raw = vec![0xff];
        invalid_raw.entries[0].entry.size = 1;
        reject(invalid_raw, "invalid txpool snapshot entry raw");
        let mut txid_mismatch = guard_snapshot.clone();
        txid_mismatch.entries[0].txid = [0x99; 32];
        reject(txid_mismatch, "txid mismatch");
        let mut zero_txid = guard_snapshot.clone();
        zero_txid.entries[0].txid = [0u8; 32];
        reject(zero_txid, "invalid txpool snapshot entry txid");
        let mut wtxid_mismatch = guard_snapshot.clone();
        wtxid_mismatch.entries[0].wtxid = [0x88; 32];
        reject(wtxid_mismatch, "wtxid mismatch");
        let mut weight_mismatch = guard_snapshot.clone();
        weight_mismatch.entries[0].entry.weight += 1;
        reject(weight_mismatch, "weight mismatch");
        let mut input_mismatch = guard_snapshot.clone();
        input_mismatch.entries[0].entry.inputs.clear();
        reject(input_mismatch, "input list mismatch");
        let mut zero_size = guard_snapshot.clone();
        zero_size.entries[0].entry.size = 0;
        reject(zero_size, "invalid txpool snapshot entry size");
        let reject_live_cap = |max_txs, max_bytes, needle| {
            let mut target = TxPool::new();
            target.set_capacity_for_test(max_txs, max_bytes);
            let before = target.snapshot().expect("empty target snapshot");
            let err = target
                .restore_snapshot(&guard_snapshot)
                .expect_err("live capacity must fail closed");
            assert!(err.message.contains(needle), "got: {}", err.message);
            assert_eq!(&target.snapshot().expect("after"), &before);
        };
        reject_live_cap(1, usize::MAX, "transaction cap");
        reject_live_cap(7, guard_snapshot.entries[0].entry.size, "byte cap");
        let mut used_mismatch = guard_snapshot.clone();
        used_mismatch.used_bytes += 1;
        reject(used_mismatch, "used_bytes mismatch");
        let mut high_water = guard_snapshot.clone();
        high_water.next_heap_id = 1;
        reject(high_water, "high-watermark");
        let mut saturated = guard_snapshot.clone();
        saturated.next_heap_id = u64::MAX - 8;
        reject(saturated, "heap near saturation");
        target.used_bytes += 1;
        assert!(target.snapshot().is_err());
        target.used_bytes -= 1;
        target.next_heap_id = 1;
        assert!(target.snapshot().is_err());
        target.next_heap_id = u64::MAX - 8;
        assert!(target.snapshot().is_err());
    }

    #[test]
    fn next_block_height_handles_tip_states() {
        assert_eq!(next_block_height(&ChainState::new()).expect("height"), 0);

        let mut state = ChainState::new();
        state.has_tip = true;
        state.height = 7;
        assert_eq!(next_block_height(&state).expect("height"), 8);

        state.height = u64::MAX;
        let err = next_block_height(&state).unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("height overflow"));
    }

    #[test]
    fn next_block_mtp_handles_missing_store_context() {
        assert_eq!(next_block_mtp(None, 0).expect("mtp"), 0);

        let (store, dir) = create_block_store("rubin-txpool-mtp");
        assert_eq!(next_block_mtp(Some(&store), 0).expect("mtp"), 0);
        let err = next_block_mtp(Some(&store), 1).unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("missing canonical header"));
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn next_block_mtp_reads_timestamp_context_from_store() {
        let (store, dir) = create_block_store("rubin-txpool-mtp-success");
        let mut engine = SyncEngine::new(
            ChainState::new(),
            Some(store.clone()),
            default_sync_config(None, devnet_genesis_chain_id(), None),
        )
        .expect("sync");
        engine
            .apply_block(&devnet_genesis_block_bytes(), None)
            .expect("apply genesis");
        let reopened = BlockStore::open(block_store_path(&dir)).expect("reopen");
        assert!(next_block_mtp(Some(&reopened), 1).expect("mtp") > 0);
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn admit_rejects_duplicate_txid() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let mut pool = TxPool::new();
        pool.admit(&raw, &state, None, devnet_genesis_chain_id())
            .expect("first admit");
        let err = pool
            .admit(&raw, &state, None, devnet_genesis_chain_id())
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Conflict);
        assert!(err.message.contains("already in mempool"));
    }

    #[test]
    fn owner_context_and_token_exhaustion_preserve_admission_order() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let owner = super::PendingOutpointOwnerHandle::new(
            super::PendingOutpointTip::from_chain_state(&state),
        );
        let mut pool = TxPool::new_bound_with_config(TxPoolConfig::default(), owner.clone());
        owner.begin_transition().expect("fence owner");
        let err = pool
            .admit(&[0xff], &state, None, devnet_genesis_chain_id())
            .expect_err("owner context precedes parse");
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        owner.reopen_old_tip().expect("reopen owner");
        let mut tip_b = state.clone();
        tip_b.has_tip = true;
        tip_b.height = 1;
        tip_b.tip_hash = [0xB; 32];
        let before = (owner.test_high_water(), pool.owner_row_count());
        let err = pool
            .admit(&[0xff], &tip_b, None, devnet_genesis_chain_id())
            .expect_err("tip mismatch precedes parse");
        assert_eq!(
            err.message,
            "pending-outpoint owner context does not match chain tip"
        );
        assert_eq!((owner.test_high_water(), pool.owner_row_count()), before);
        owner.test_set_high_water(u64::MAX);
        let err = pool
            .admit(&[0xff], &state, None, devnet_genesis_chain_id())
            .expect_err("parse precedes exhausted reservation");
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        let err = pool
            .admit(&raw, &state, None, devnet_genesis_chain_id())
            .expect_err("valid reservation is exhausted");
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);

        let duplicate_owner = super::PendingOutpointOwnerHandle::new(
            super::PendingOutpointTip::from_chain_state(&state),
        );
        let mut duplicate_pool =
            TxPool::new_bound_with_config(TxPoolConfig::default(), duplicate_owner.clone());
        duplicate_pool
            .admit(&raw, &state, None, devnet_genesis_chain_id())
            .expect("first admission");
        duplicate_owner.test_set_high_water(u64::MAX);
        let err = duplicate_pool
            .admit(&raw, &state, None, devnet_genesis_chain_id())
            .expect_err("duplicate precedes exhausted reservation");
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Conflict);
    }

    #[test]
    fn admit_rejects_non_canonical_trailing_bytes() {
        let mut raw = genesis_coinbase_bytes();
        raw.push(0x00);
        let err = TxPool::new()
            .admit(&raw, &ChainState::new(), None, devnet_genesis_chain_id())
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("non-canonical tx bytes"));
    }

    #[test]
    fn relay_metadata_rejects_core_ext_outputs_as_unsupported_runtime() {
        let (state, raw) = signed_p2pk_state_and_tx(
            10,
            vec![TxOutput {
                value: 9,
                covenant_type: COV_TYPE_CORE_EXT,
                covenant_data: empty_core_ext_covenant_data(7),
            }],
            0x00,
            None,
            Vec::new(),
        );

        let err =
            relay_metadata(&raw, &state, None, [0u8; 32], &TxPoolConfig::default()).unwrap_err();

        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("TX_ERR_COVENANT_TYPE_INVALID"));
    }

    /// Mirror of Go `simplicityCovenantDataForNodeTest`: 32-byte program
    /// CMR followed by compactSize(state_len) with an empty state.
    fn simplicity_covenant_data(cmr_byte: u8) -> Vec<u8> {
        let mut out = vec![cmr_byte; 32];
        rubin_consensus::encode_compact_size(0, &mut out);
        out
    }

    /// Mirror of Go `testSimplicityRotation`: default native suites plus
    /// a Simplicity deployment that is active at every height.
    struct SimplicityActiveRotation;

    impl rubin_consensus::RotationProvider for SimplicityActiveRotation {
        fn native_create_suites(&self, height: u64) -> rubin_consensus::NativeSuiteSet {
            rubin_consensus::DefaultRotationProvider.native_create_suites(height)
        }

        fn native_spend_suites(&self, height: u64) -> rubin_consensus::NativeSuiteSet {
            rubin_consensus::DefaultRotationProvider.native_spend_suites(height)
        }

        fn simplicity_active_at_height(&self, _height: u64) -> bool {
            true
        }
    }

    /// Mirror of the Go test `MempoolConfig` literal (zero floors, only
    /// the Simplicity pre-activation guardrail enabled).
    fn simplicity_policy_only_config() -> TxPoolConfig {
        TxPoolConfig {
            policy_da_surcharge_per_byte: 0,
            policy_reject_non_coinbase_anchor_outputs: false,
            policy_reject_simplicity_pre_activation: true,
            suite_context: None,
            policy_current_mempool_min_fee_rate: 0,
            policy_min_da_fee_rate: 0,
        }
    }

    /// Mirror of Go `txWithOneInputOneOutput`: a deliberately UNSIGNED
    /// single-input transaction (the pre-activation policy gate must
    /// fire before signature verification).
    fn unsigned_one_input_tx(prev: &Outpoint, outputs: Vec<TxOutput>) -> Vec<u8> {
        let tx = Tx {
            version: TX_WIRE_VERSION,
            tx_kind: 0x00,
            tx_nonce: 11,
            inputs: vec![TxInput {
                prev_txid: prev.txid,
                prev_vout: prev.vout,
                script_sig: Vec::new(),
                sequence: 0,
            }],
            outputs,
            locktime: 0,
            da_commit_core: None,
            da_chunk_core: None,
            witness: Vec::new(),
            da_payload: Vec::new(),
        };
        marshal_tx(&tx).expect("marshal unsigned policy tx")
    }

    #[test]
    fn mempool_policy_rejects_core_simplicity_pre_activation_before_consensus() {
        // Mirror of Go
        // `TestMempoolPolicyRejectsCoreSimplicityPreActivationBeforeConsensus`:
        // the policy reason — not a consensus reject — is the observable
        // outcome for BOTH admission and relay, on the create side and
        // the spend side, with unsigned transactions.
        let funding = Outpoint {
            txid: [0x11; 32],
            vout: 0,
        };
        let mut state = ChainState::new();
        state.utxos.insert(
            funding.clone(),
            UtxoEntry {
                value: 100,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&[0x44; 2592]),
                creation_height: 0,
                created_by_coinbase: false,
            },
        );
        let cfg = simplicity_policy_only_config();
        let chain_id = [0u8; 32];

        let create_raw = unsigned_one_input_tx(
            &funding,
            vec![TxOutput {
                value: 1,
                covenant_type: COV_TYPE_CORE_SIMPLICITY,
                covenant_data: simplicity_covenant_data(0x53),
            }],
        );
        let err = TxPool::new_with_config(cfg.clone())
            .admit(&create_raw, &state, None, chain_id)
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(
            err.message.contains("CORE_SIMPLICITY output pre-ACTIVE"),
            "admit create: {}",
            err.message
        );
        let err = relay_metadata(&create_raw, &state, None, chain_id, &cfg).unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(
            err.message.contains("CORE_SIMPLICITY output pre-ACTIVE"),
            "relay create: {}",
            err.message
        );

        let prev = Outpoint {
            txid: [0x54; 32],
            vout: 0,
        };
        state.utxos.insert(
            prev.clone(),
            UtxoEntry {
                value: 100,
                covenant_type: COV_TYPE_CORE_SIMPLICITY,
                covenant_data: simplicity_covenant_data(0x55),
                creation_height: 0,
                created_by_coinbase: false,
            },
        );
        let spend_raw = unsigned_one_input_tx(
            &prev,
            vec![TxOutput {
                value: 99,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&[0x44; 2592]),
            }],
        );
        let err = TxPool::new_with_config(cfg.clone())
            .admit(&spend_raw, &state, None, chain_id)
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(
            err.message.contains("CORE_SIMPLICITY spend pre-ACTIVE"),
            "admit spend: {}",
            err.message
        );
        let err = relay_metadata(&spend_raw, &state, None, chain_id, &cfg).unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(
            err.message.contains("CORE_SIMPLICITY spend pre-ACTIVE"),
            "relay spend: {}",
            err.message
        );
    }

    #[test]
    fn mempool_policy_missing_input_precedes_simplicity_policy_reason() {
        // Mirror of Go `policyInputSnapshot` precedence (semantic-review
        // finding, RUB-528): a missing input outpoint keeps the
        // consensus TX_ERR_MISSING_UTXO reject — the CORE_SIMPLICITY
        // policy reason must NOT mask it, on both admission and relay.
        let missing = Outpoint {
            txid: [0x66; 32],
            vout: 0,
        };
        let state = ChainState::new();
        let raw = unsigned_one_input_tx(
            &missing,
            vec![TxOutput {
                value: 1,
                covenant_type: COV_TYPE_CORE_SIMPLICITY,
                covenant_data: simplicity_covenant_data(0x66),
            }],
        );
        let cfg = simplicity_policy_only_config();
        for (name, err) in [
            (
                "admit",
                TxPool::new_with_config(cfg.clone())
                    .admit(&raw, &state, None, [0u8; 32])
                    .unwrap_err(),
            ),
            (
                "relay",
                relay_metadata(&raw, &state, None, [0u8; 32], &cfg).unwrap_err(),
            ),
        ] {
            assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected, "{name}");
            // Exact BARE string — byte-identical to Go's
            // `txAdmitRejected(err.Error())` (verified equal on both CLIs).
            // Pins the missing-UTXO precedence AND the no-wrapper parity
            // without a shared vector; the policy reason must not mask it.
            assert_eq!(
                err.message, "TX_ERR_MISSING_UTXO: utxo not found",
                "{name}: missing input must keep the bare consensus missing-UTXO reject"
            );
        }
    }

    #[test]
    fn mempool_policy_allows_core_simplicity_create_when_active() {
        // Mirror of Go `TestMempoolPolicyAllowsCoreSimplicityCreateWhenActive`:
        // with a rotation provider reporting the Simplicity deployment
        // active, the policy gate steps aside and a signed create admits.
        let (state, raw) = signed_p2pk_state_and_tx(
            1_000_000,
            vec![
                TxOutput {
                    value: 100_000,
                    covenant_type: COV_TYPE_CORE_SIMPLICITY,
                    covenant_data: simplicity_covenant_data(0x58),
                },
                TxOutput {
                    value: 800_000,
                    covenant_type: COV_TYPE_P2PK,
                    covenant_data: p2pk_covenant_data_for_pubkey(&[0x44; 2592]),
                },
            ],
            0x00,
            None,
            Vec::new(),
        );
        let mut cfg = simplicity_policy_only_config();
        cfg.suite_context = Some(crate::sync::SuiteContext {
            rotation: std::sync::Arc::new(SimplicityActiveRotation),
            registry: std::sync::Arc::new(
                rubin_consensus::SuiteRegistry::default_registry().clone(),
            ),
        });
        TxPool::new_with_config(cfg)
            .admit(&raw, &state, None, [0u8; 32])
            .expect("active CORE_SIMPLICITY create must pass policy and admit");
    }

    #[test]
    fn mempool_policy_does_not_mask_malformed_non_simplicity_output() {
        // Mirror of Go `TestMempoolPolicyDoesNotMaskMalformedNonSimplicityOutput`:
        // the forced-active genesis revalidation must surface the
        // malformed P2PK consensus error instead of the policy reason.
        let funding = Outpoint {
            txid: [0x11; 32],
            vout: 0,
        };
        let mut state = ChainState::new();
        state.utxos.insert(
            funding.clone(),
            UtxoEntry {
                value: 100,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&[0x44; 2592]),
                creation_height: 0,
                created_by_coinbase: false,
            },
        );
        let raw = unsigned_one_input_tx(
            &funding,
            vec![
                TxOutput {
                    value: 1,
                    covenant_type: COV_TYPE_CORE_SIMPLICITY,
                    covenant_data: simplicity_covenant_data(0x59),
                },
                TxOutput {
                    value: 1,
                    covenant_type: COV_TYPE_P2PK,
                    covenant_data: Vec::new(),
                },
            ],
        );
        let cfg = simplicity_policy_only_config();
        for (name, err) in [
            (
                "admit",
                TxPool::new_with_config(cfg.clone())
                    .admit(&raw, &state, None, [0u8; 32])
                    .unwrap_err(),
            ),
            (
                "relay",
                relay_metadata(&raw, &state, None, [0u8; 32], &cfg).unwrap_err(),
            ),
        ] {
            assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected, "{name}");
            assert!(
                err.message
                    .contains("invalid CORE_P2PK covenant_data length")
                    && !err.message.contains("CORE_SIMPLICITY output pre-ACTIVE"),
                "{name}: malformed P2PK must not be masked by the Simplicity policy: {}",
                err.message
            );
        }
    }

    #[test]
    fn admit_rejects_pool_full() {
        // RUB-162 Phase A migration rationale (per controller Path A
        // approval 2026-05-03 + RESP P1 #2 reorder fix 2026-05-03):
        //   - old assumption: input=10/output=10 → fee=0 candidate
        //     reaches the pool-full ordering check and admit returns
        //     Unavailable("tx pool full") because the pre-RUB-162
        //     admit_with_metadata had no rolling-floor classification.
        //   - new invariant: validate_fee_floor runs BEFORE the
        //     pool-full check (Go validateCapacityAdmissionLocked order
        //     at clients/go/node/mempool.go `addEntryLockedWithFloor` capacity-eviction block. Sub-floor
        //     candidates fail the floor check first.
        //   - reachability: tx is well-formed; the pool-full ordering
        //     branch is the test goal.
        //   - replacement coverage: input bumped to 7700 so fee=7690 ≥
        //     weight (~7653) ⇒ candidate passes floor and reaches the
        //     pool-full ordering check this test is asserting. A
        //     dedicated rub162_admit_full_pool_below_floor_returns_floor_
        //     error_not_pool_full test below pins the new ordering for
        //     sub-floor candidates.
        let (state, raw) = signed_p2pk_state_and_tx(
            7700,
            vec![TxOutput {
                value: 10,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x55; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );
        let (_tx, txid, _wtxid, consumed) = parse_tx(&raw).expect("parse tx");
        assert_eq!(consumed, raw.len());

        let mut pool = TxPool::new();
        for idx in 0..MAX_TX_POOL_TRANSACTIONS {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&(idx as u64 + 1).to_le_bytes());
            if key == txid {
                key[8] = 1;
            }
            // Worst-pool entries sized to outrank candidate fee_rate so
            // pool-full ordering rejects (worst fee_rate=10 > candidate
            // fee_rate≈1.005, candidate cannot beat worst).
            pool.insert_entry(
                key,
                TxPoolEntry {
                    raw: vec![0xff],
                    inputs: Vec::new(),
                    fee: 10,
                    weight: 1,
                    size: 1,
                    source: TxSource::Local,
                },
            );
        }
        let err = pool.admit(&raw, &state, None, [0u8; 32]).unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        // PR-1410 wave-2 — error message split per Go-parity at
        // clients/go/node/mempool.go (`validateCapacityAdmissionLocked` eviction-ordering reject). Eviction-ordering
        // rejection is now distinct from internal capacity invariant
        // failure.
        assert!(
            err.message
                .contains("mempool capacity candidate rejected by eviction ordering"),
            "expected eviction-ordering message, got: {}",
            err.message
        );
        assert_eq!(
            pool.owner_row_count(),
            0,
            "capacity failure releases candidate claim"
        );
        assert_eq!(
            pool.owner
                .clone()
                .expect("candidate owner")
                .test_high_water(),
            1,
            "capacity failure retains exactly one reserved token sequence"
        );
    }

    /// PR-1410 wave-2 — direct `insert_entry` populates the worst_heap
    /// without going through admit_with_metadata, so this test
    /// exercises the routine eviction-ordering rejection path with a
    /// fully populated heap (not the corruption invariants). Pinning
    /// the new error-message format prevents future regressions that
    /// would collapse this case back into a generic "tx pool full"
    /// message and lose Go-parity at clients/go/node/mempool.go (`validateCapacityAdmissionLocked` eviction-ordering reject).
    #[test]
    fn rub162_admit_pool_full_eviction_ordering_message_distinct_from_invariant() {
        let (state, raw) = signed_p2pk_state_and_tx(
            7700,
            vec![TxOutput {
                value: 10,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x77; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );
        let (_tx, txid, _wtxid, consumed) = parse_tx(&raw).expect("parse tx");
        assert_eq!(consumed, raw.len());

        let mut pool = TxPool::new();
        for idx in 0..MAX_TX_POOL_TRANSACTIONS {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&(idx as u64 + 1).to_le_bytes());
            if key == txid {
                key[8] = 1;
            }
            pool.insert_entry(
                key,
                TxPoolEntry {
                    raw: vec![0xff],
                    inputs: Vec::new(),
                    fee: 10,
                    weight: 1,
                    size: 1,
                    source: TxSource::Local,
                },
            );
        }
        let err = pool.admit(&raw, &state, None, [0u8; 32]).unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(
            err.message
                .contains("mempool capacity candidate rejected by eviction ordering"),
            "expected eviction-ordering message (Go-parity `validateCapacityAdmissionLocked` in clients/go/node/mempool.go), got: {}",
            err.message
        );
        // Pin: legitimate eviction-ordering rejection MUST NOT collapse
        // into the internal invariant-violation message.
        assert!(
            !err.message.contains("invariant violated"),
            "eviction-ordering rejection must not surface as invariant violation; got: {}",
            err.message
        );
    }

    #[test]
    fn admit_evicts_lowest_priority_when_pool_full() {
        // RUB-162 Phase A migration rationale (per controller Q2 record):
        //   - old assumption: input=10/output=8 → fee=2 with weight≈7653
        //     admits because pre-RUB-162 admit_with_metadata didn't enforce
        //     the rolling fee floor; the test exercised eviction-ordering.
        //   - new invariant: admit_with_metadata enforces the rolling fee
        //     floor (DEFAULT_MEMPOOL_MIN_FEE_RATE=1) via
        //     validate_fee_floor → Unavailable when fee/weight < 1.
        //   - why it reaches policy path: tx is well-formed; floor check
        //     is after apply_policy, before eviction.
        //   - replacement coverage: input bumped to 7661 so fee = 7661 - 8
        //     = 7653 ≥ weight (≈7653) ⇒ fee/weight ≥ 1 ⇒ passes the
        //     default floor; eviction-ordering invariant remains under test.
        //     Cfg-zero opt-out is impossible because the floor is clamped
        //     to DEFAULT inside validate_fee_floor (Go parity).
        let (state, raw) = signed_p2pk_state_and_tx(
            7661,
            vec![TxOutput {
                value: 8,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x66; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );
        let (_tx, txid, _wtxid, consumed) = parse_tx(&raw).expect("parse tx");
        assert_eq!(consumed, raw.len());

        let mut pool = TxPool::new();
        let worst = [0x11; 32];
        let worst_input = Outpoint {
            txid: [0xF1; 32],
            vout: 0,
        };
        pool.insert_entry(
            worst,
            TxPoolEntry {
                raw: vec![0x01],
                inputs: vec![worst_input.clone()],
                fee: 0,
                weight: 100,
                size: 1,
                source: TxSource::Local,
            },
        );
        for idx in 1..MAX_TX_POOL_TRANSACTIONS {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&(idx as u64 + 1).to_le_bytes());
            if key == txid || key == worst {
                key[8] = 1;
            }
            pool.insert_entry(
                key,
                TxPoolEntry {
                    raw: vec![0xff],
                    inputs: Vec::new(),
                    fee: 1,
                    weight: 1,
                    size: 1,
                    source: TxSource::Local,
                },
            );
        }

        let admitted = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect("admit should evict");
        assert_eq!(admitted, txid);
        assert_eq!(pool.txs.len(), MAX_TX_POOL_TRANSACTIONS);
        assert!(pool.txs.contains_key(&txid));
        assert!(!pool.txs.contains_key(&worst));
        assert!(pool.owner_claim_is_finalized(&txid));
        assert!(pool.owner_txid_for_outpoint(&worst_input).is_none());
    }

    #[test]
    fn admit_reject_on_full_preserves_future_eviction() {
        // RUB-162 Phase A migration rationale (per controller Q2 record):
        //   - old assumption: pre-RUB-162 admit_with_metadata did not enforce
        //     the rolling fee floor; the test exercised heap-state
        //     preservation across reject + admit cycle.
        //   - new invariant: admit_with_metadata enforces the rolling fee
        //     floor via validate_fee_floor → Unavailable.
        //   - why it reaches policy path: txs are well-formed; floor check
        //     is after apply_policy.
        //   - replacement coverage: input_value bumped to make fee_rate ≥ 1
        //     so both candidates pass the floor. The "worse" candidate now
        //     has fee=7652, the "better" has fee=8000. Pool worst entry
        //     bumped to fee_rate ≈ 0.9 so worse-candidate fee_rate (~1.0)
        //     beats it but only marginally — preserves the test's intended
        //     eviction-ordering interaction. Heap-state preservation
        //     invariant (P1 #2 atomicity-related) remains under test.
        // Better candidate: fee = 20000 - 8 = 19992; weight ≈ 7653 (ML-DSA
        // witness) → fee_rate ≈ 2.61. Above floor; should beat worst pool
        // entry (fee_rate = 2.0).
        let (state_better, raw_better) = signed_p2pk_state_and_tx(
            20_000,
            vec![TxOutput {
                value: 8,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x76; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );
        // Worse candidate: fee = 7662 - 9 = 7653; weight ≈ 7653 → fee_rate
        // = 1.0. Above floor (passes validate_fee_floor) but below
        // worst pool entry's fee_rate (2.0); pool-full eviction comparator
        // rejects with Unavailable.
        let (state_worse, raw_worse) = signed_p2pk_state_and_tx(
            7_662,
            vec![TxOutput {
                value: 9,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x77; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );

        let mut pool = TxPool::new();
        let worst = [0x11; 32];
        // Worst pool entry: fee_rate = 20000/10000 = 2.0. Above the default
        // rolling floor (1) so the eviction-ordering test exercises the
        // comparator branch that compares "worse candidate at floor" vs
        // "above-floor resident worst". Inserted through insert_entry to
        // keep capacity indexes coherent while bypassing full tx validation.
        pool.insert_entry(
            worst,
            TxPoolEntry {
                raw: vec![0x01; raw_worse.len()],
                inputs: Vec::new(),
                fee: 20_000,
                weight: 10_000,
                size: raw_worse.len(),
                source: TxSource::Local,
            },
        );
        for idx in 1..MAX_TX_POOL_TRANSACTIONS {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&(idx as u64 + 1).to_le_bytes());
            if key == worst {
                key[8] = 1;
            }
            pool.insert_entry(
                key,
                TxPoolEntry {
                    raw: vec![0xff],
                    inputs: Vec::new(),
                    fee: 3,
                    weight: 1,
                    size: 1,
                    source: TxSource::Local,
                },
            );
        }

        let err = pool
            .admit(&raw_worse, &state_worse, None, [0u8; 32])
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(pool.txs.contains_key(&worst));

        let admitted = pool
            .admit(&raw_better, &state_better, None, [0u8; 32])
            .expect("admit better should still evict");
        assert_eq!(pool.txs.len(), MAX_TX_POOL_TRANSACTIONS);
        assert!(pool.txs.contains_key(&admitted));
        assert!(!pool.txs.contains_key(&worst));
    }

    #[test]
    fn rub196_byte_cap_evicts_to_low_water_through_public_admit() {
        for (max_bytes, want) in [(0, 0), (1, 1), (9, 8), (10, 9), (11, 9)] {
            assert_eq!(default_tx_pool_low_water_bytes(max_bytes), want);
        }
        assert_eq!(tx_pool_byte_pressure_target(90, 95), 95);
        let (state, raw) = signed_p2pk_state_and_tx(
            40_000,
            vec![TxOutput {
                value: 10,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x90; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );
        let (_tx, best_txid, _wtxid, consumed) = parse_tx(&raw).expect("parse best");
        assert_eq!(consumed, raw.len());
        let size = raw.len();
        let mut pool = TxPool::new();
        pool.set_capacity_for_test(10, size * 3);
        for (txid, fee, source) in [
            ([0x31; 32], 1, TxSource::Local),
            ([0x32; 32], 2, TxSource::Remote),
            ([0x33; 32], 3, TxSource::Reorg),
        ] {
            pool.insert_entry(txid, test_entry(fee, 1, size, source));
        }
        assert_eq!(
            pool.admit(&raw, &state, None, [0u8; 32]).unwrap(),
            best_txid
        );
        assert!(pool.used_bytes <= pool.effective_low_water_bytes());
        assert!(!pool.txs.contains_key(&[0x31; 32]));
        assert!(!pool.txs.contains_key(&[0x32; 32]));
        assert!(pool.txs.contains_key(&[0x33; 32]));
        assert!(pool.txs.contains_key(&best_txid));
    }

    #[test]
    fn rub196_byte_capacity_direct_edges() {
        let mut larger = TxPool::new();
        larger.set_capacity_for_test(10, 100);
        larger.insert_entry([0x41; 32], test_entry(1, 1, 95, TxSource::Local));
        larger
            .insert_capacity_checked_entry_for_test(
                [0x42; 32],
                test_entry(100, 1, 95, TxSource::Remote),
            )
            .unwrap();
        assert_eq!(larger.used_bytes, 95);
        assert!(larger.txs.contains_key(&[0x42; 32]));
        assert!(!larger.txs.contains_key(&[0x41; 32]));

        let mut worst = TxPool::new();
        worst.set_capacity_for_test(10, 100);
        worst.insert_entry([0x51; 32], test_entry(100, 1, 95, TxSource::Local));
        let snapshot = (
            worst.len(),
            worst.used_bytes,
            worst.heap_seqs.clone(),
            worst.next_heap_id,
        );
        let err = worst
            .insert_capacity_checked_entry_for_test(
                [0x52; 32],
                test_entry(1, 1, 95, TxSource::Reorg),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err
            .message
            .contains("candidate rejected by eviction ordering"));
        assert_eq!(
            (
                worst.len(),
                worst.used_bytes,
                worst.heap_seqs,
                worst.next_heap_id
            ),
            snapshot
        );

        let mut tie = TxPool::new();
        tie.set_capacity_for_test(1, 100);
        tie.insert_entry([0x01; 32], test_entry(10, 1, 50, TxSource::Local));
        assert!(tie
            .insert_capacity_checked_entry_for_test(
                [0xff; 32],
                test_entry(10, 1, 50, TxSource::Remote)
            )
            .is_err());
        assert!(tie.txs.contains_key(&[0x01; 32]));

        let mut count_only_tie = TxPool::new();
        count_only_tie.set_capacity_for_test(1, 1_000);
        count_only_tie.insert_entry([0x80; 32], test_entry(10, 1, 50, TxSource::Local));
        count_only_tie
            .insert_capacity_checked_entry_for_test(
                [0x01; 32],
                test_entry(10, 1, 50, TxSource::Remote),
            )
            .expect("count-only pressure preserves legacy txid tie-break");
        assert!(count_only_tie.txs.contains_key(&[0x01; 32]));
        assert!(!count_only_tie.txs.contains_key(&[0x80; 32]));
    }

    #[test]
    fn rub196_capacity_rejects_bad_edges_without_mutation() {
        let mut invalid_limits = TxPool::new();
        invalid_limits.set_capacity_for_test(0, 100);
        let err = invalid_limits
            .insert_capacity_checked_entry_for_test(
                [0x53; 32],
                test_entry(100, 1, 1, TxSource::Local),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("invalid mempool capacity limits"));

        let mut invalid_candidate = TxPool::new();
        invalid_candidate.set_capacity_for_test(10, 100);
        let err = invalid_candidate
            .insert_capacity_checked_entry_for_test(
                [0u8; 32],
                test_entry(100, 1, 1, TxSource::Local),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("invalid candidate metadata"));

        let mut oversize = TxPool::new();
        oversize.set_capacity_for_test(10, 100);
        let err = oversize
            .insert_capacity_checked_entry_for_test(
                [0x54; 32],
                test_entry(100, 1, 101, TxSource::Local),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert_eq!((oversize.len(), oversize.used_bytes), (0, 0));

        let mut missing_seq = TxPool::new();
        missing_seq.set_capacity_for_test(1, 100);
        missing_seq
            .txs
            .insert([0x56; 32], test_entry(1, 1, 50, TxSource::Local));
        let err = missing_seq
            .insert_capacity_checked_entry_for_test(
                [0x57; 32],
                test_entry(100, 1, 50, TxSource::Remote),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("missing heap sequence"));

        let mut invalid_resident = TxPool::new();
        invalid_resident.set_capacity_for_test(1, 100);
        invalid_resident.insert_entry([0u8; 32], test_entry(1, 1, 50, TxSource::Local));
        let err = invalid_resident
            .insert_capacity_checked_entry_for_test(
                [0x58; 32],
                test_entry(100, 1, 50, TxSource::Remote),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("invalid resident metadata"));

        let mut duplicate_seq = TxPool::new();
        duplicate_seq.set_capacity_for_test(2, 100);
        duplicate_seq.insert_entry([0x59; 32], test_entry(1, 1, 50, TxSource::Local));
        duplicate_seq.insert_entry([0x5a; 32], test_entry(2, 1, 50, TxSource::Local));
        let first_seq = *duplicate_seq.heap_seqs.get(&[0x59; 32]).unwrap();
        duplicate_seq.heap_seqs.insert([0x5a; 32], first_seq);
        let err = duplicate_seq
            .insert_capacity_checked_entry_for_test(
                [0x5b; 32],
                test_entry(100, 1, 1, TxSource::Remote),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("duplicate heap sequence"));

        let mut zero_weight = TxPool::new();
        zero_weight.set_capacity_for_test(1, 100);
        zero_weight.insert_entry([0x5c; 32], test_entry(1, 0, 50, TxSource::Local));
        let err = zero_weight
            .insert_capacity_checked_entry_for_test(
                [0x5d; 32],
                test_entry(100, 1, 50, TxSource::Remote),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("invalid resident metadata"));

        let mut underflow = TxPool::new();
        underflow.set_capacity_for_test(1, 100);
        underflow.insert_entry([0x5b; 32], test_entry(1, 1, 95, TxSource::Local));
        underflow.used_bytes = 0;
        let err = underflow
            .insert_capacity_checked_entry_for_test(
                [0x5c; 32],
                test_entry(100, 1, 1, TxSource::Remote),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("eviction byte accounting underflow"));
        assert!(underflow.txs.contains_key(&[0x5b; 32]));

        let mut stale_low_water = TxPool::new();
        stale_low_water.max_bytes = 100;
        stale_low_water.low_water_bytes = 0;
        assert_eq!(stale_low_water.effective_low_water_bytes(), 90);

        let mut exceeded_after_plan = TxPool::new();
        exceeded_after_plan.set_capacity_for_test(10, 100);
        exceeded_after_plan.low_water_bytes = 200;
        exceeded_after_plan.insert_entry([0x5e; 32], test_entry(1, 1, 95, TxSource::Local));
        let err = exceeded_after_plan
            .insert_capacity_checked_entry_for_test(
                [0x5f; 32],
                test_entry(100, 1, 10, TxSource::Remote),
            )
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("capacity remains exceeded"));

        let mut rebuild = TxPool::new();
        rebuild
            .txs
            .insert([0x60; 32], test_entry(1, 1, 1, TxSource::Local));
        assert_eq!(rebuild.current_worst_txid(), Some([0x60; 32]));
        assert!(rebuild.heap_seqs.contains_key(&[0x60; 32]));
    }

    #[test]
    fn rub196_source_does_not_affect_byte_capacity_ordering() {
        let mut outcomes = Vec::new();
        for (resident_source, candidate_source) in [
            (TxSource::Local, TxSource::Remote),
            (TxSource::Reorg, TxSource::Local),
        ] {
            let mut pool = TxPool::new();
            pool.set_capacity_for_test(10, 100);
            pool.insert_entry([0x61; 32], test_entry(1, 1, 40, resident_source));
            pool.insert_entry([0x62; 32], test_entry(50, 1, 40, TxSource::Local));
            pool.insert_capacity_checked_entry_for_test(
                [0x63; 32],
                test_entry(100, 1, 40, candidate_source),
            )
            .unwrap();
            outcomes.push((
                pool.txs.contains_key(&[0x61; 32]),
                pool.txs.contains_key(&[0x62; 32]),
                pool.txs.contains_key(&[0x63; 32]),
                pool.used_bytes,
            ));
        }
        assert_eq!(
            outcomes,
            vec![(false, true, true, 80), (false, true, true, 80)]
        );
    }

    #[test]
    fn admit_rejects_mempool_input_conflict() {
        let (state, resident, conflicting) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let mut pool = TxPool::new();
        pool.admit(&resident, &state, None, devnet_genesis_chain_id())
            .expect("resident admit");
        let err = pool
            .admit(&conflicting, &state, None, devnet_genesis_chain_id())
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Conflict);
        assert!(err.message.contains("double-spend conflict"));
    }

    #[test]
    fn select_transactions_orders_by_fee_rate_then_fee() {
        let mut pool = TxPool::new();
        pool.txs.insert(
            [0x11; 32],
            TxPoolEntry {
                raw: vec![0x11],
                inputs: Vec::new(),
                fee: 20,
                weight: 100,
                size: 20,
                source: TxSource::Local,
            },
        );
        pool.txs.insert(
            [0x22; 32],
            TxPoolEntry {
                raw: vec![0x22],
                inputs: Vec::new(),
                fee: 15,
                weight: 90,
                size: 10,
                source: TxSource::Local,
            },
        );
        pool.txs.insert(
            [0x33; 32],
            TxPoolEntry {
                raw: vec![0x33],
                inputs: Vec::new(),
                fee: 15,
                weight: 80,
                size: 10,
                source: TxSource::Local,
            },
        );

        let selected = pool.select_transactions(3, 40);
        assert_eq!(selected, vec![vec![0x11], vec![0x33], vec![0x22]]);
    }

    #[test]
    fn txpool_fee_rate_uses_weight_not_size() {
        let size_favored = TxPoolEntry {
            raw: vec![0x41],
            inputs: Vec::new(),
            fee: 4,
            weight: 4,
            size: 1,
            source: TxSource::Local,
        };
        let weight_favored = TxPoolEntry {
            raw: vec![0x21],
            inputs: Vec::new(),
            fee: 2,
            weight: 1,
            size: 1,
            source: TxSource::Local,
        };

        assert_eq!(
            compare_fee_rate(&size_favored, &weight_favored),
            Ordering::Less,
            "fee/weight must rank 2/1 above 4/4 even when fee/size favors 4/1",
        );
        assert_eq!(
            compare_admit_priority([0x41; 32], &size_favored, [0x21; 32], &weight_favored),
            Ordering::Less,
        );

        let size_favored_txid: [u8; 32] = [0x41; 32];
        let weight_favored_txid: [u8; 32] = [0x21; 32];
        assert_eq!(
            compare_entries_for_mining(
                &(&weight_favored_txid, &weight_favored),
                &(&size_favored_txid, &size_favored),
            ),
            Ordering::Less,
        );
    }

    /// RUB-1127: the policy layer ranks the exact u128 fee end to end —
    /// admission floor, miner ordering, admit priority, live selection, and
    /// the capacity victim choice. Every row straddles u64: 2^64 truncates to
    /// 0 under a narrowing to the low 64 bits while 2^64-1 truncates to
    /// itself, so a narrowed comparison inverts each verdict rather than
    /// merely losing precision. Mirrors Go
    /// `TestMempoolPolicyAboveU64FeesAdmitOrderAndEvictExactly`.
    #[test]
    fn policy_above_u64_fees_admit_order_and_evict_exactly() {
        let above_u64 = 1u128 << 64;
        let below_u64 = u128::from(u64::MAX);
        assert!(above_u64 > below_u64, "row must straddle u64");

        // Admission floor: weight*floor is 1000, which the exact fee clears
        // and a fee truncated to its low 64 bits (0) does not.
        super::validate_fee_floor(above_u64, 1, 1000)
            .expect("fee above u64 must clear a weight*floor of 1000");

        let wide_txid: [u8; 32] = [0x92; 32];
        let narrow_txid: [u8; 32] = [0x93; 32];
        let wide = TxPoolEntry {
            raw: vec![0x92],
            inputs: Vec::new(),
            fee: above_u64,
            weight: 1,
            size: 1,
            source: TxSource::Local,
        };
        let narrow = TxPoolEntry {
            raw: vec![0x93],
            inputs: Vec::new(),
            fee: below_u64,
            weight: 1,
            size: 1,
            source: TxSource::Local,
        };

        // Miner ordering: lower `Ordering` sorts first.
        assert_eq!(
            compare_entries_for_mining(&(&wide_txid, &wide), &(&narrow_txid, &narrow)),
            Ordering::Less,
            "the above-u64 fee must sort ahead of 2^64-1",
        );
        // Admit priority: the below-u64 entry is the worse one.
        assert_eq!(
            compare_admit_priority(narrow_txid, &narrow, wide_txid, &wide),
            Ordering::Less,
        );

        let mut pool = TxPool::new();
        pool.set_capacity_for_test(2, 100);
        pool.insert_entry(wide_txid, wide.clone());
        pool.insert_entry(narrow_txid, narrow.clone());

        // Live miner selection over the straddling pair.
        assert_eq!(
            pool.select_transactions(2, 100),
            vec![vec![0x92], vec![0x93]],
        );

        // Capacity victim: count pressure must remove the below-u64 entry and
        // keep the above-u64 one.
        pool.insert_capacity_checked_entry_for_test(
            [0x94; 32],
            test_entry(1u128 << 65, 1, 1, TxSource::Remote),
        )
        .expect("candidate above both resident fees must be admitted");
        assert!(
            pool.txs.contains_key(&wide_txid),
            "above-u64 entry must survive capacity eviction"
        );
        assert!(
            !pool.txs.contains_key(&narrow_txid),
            "below-u64 entry must be the capacity victim"
        );
    }

    #[test]
    fn select_transactions_respects_count_and_size_caps() {
        let mut pool = TxPool::new();
        pool.txs.insert(
            [0x44; 32],
            TxPoolEntry {
                raw: vec![0x44, 0x44],
                inputs: Vec::new(),
                fee: 5,
                weight: 10,
                size: 2,
                source: TxSource::Local,
            },
        );
        pool.txs.insert(
            [0x55; 32],
            TxPoolEntry {
                raw: vec![0x55, 0x55, 0x55, 0x55],
                inputs: Vec::new(),
                fee: 100,
                weight: 10,
                size: 4,
                source: TxSource::Local,
            },
        );
        pool.txs.insert(
            [0x66; 32],
            TxPoolEntry {
                raw: vec![0x66],
                inputs: Vec::new(),
                fee: 1,
                weight: 10,
                size: 1,
                source: TxSource::Local,
            },
        );

        let selected = pool.select_transactions(1, 3);
        assert_eq!(selected, vec![vec![0x44, 0x44]]);
    }

    #[test]
    fn remove_conflicting_inputs_evicts_conflicting_mempool_entries() {
        // RUB-162 Phase A migration rationale (per controller Q2 record,
        // Path A approval 2026-05-03):
        //   - old assumption: input=20 / first_output=10 → fee=10 with
        //     weight ≈ 7653 admits because pre-RUB-162 admit_with_metadata
        //     did not enforce the rolling fee floor on non-DA txs.
        //   - new invariant: admit_with_metadata enforces the rolling fee
        //     floor (DEFAULT_MEMPOOL_MIN_FEE_RATE=1) for every admission
        //     via validate_fee_floor.
        //   - reachability: tx is well-formed (parses, basic checks pass);
        //     floor check is after apply_policy. Original test goal is to
        //     exercise remove_conflicting_inputs via a previously-admitted
        //     tx + a block tx that double-spends the same input.
        //   - replacement coverage: input bumped to 7700 so the admitted
        //     tx fee = 7700 - 10 = 7690 ≥ weight (≈7653); the block tx
        //     fee = 7700 - 9 = 7691 also above floor. Original conflict
        //     scenario preserved.
        let (state, admitted_raw, block_raw) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let mut pool = TxPool::new();
        let admitted_txid = pool
            .admit(&admitted_raw, &state, None, devnet_genesis_chain_id())
            .expect("admit");
        let (block_tx, _txid, _wtxid, consumed) = parse_tx(&block_raw).expect("parse tx");
        assert_eq!(consumed, block_raw.len());

        pool.remove_conflicting_inputs(&[block_tx]);

        assert!(!pool.txs.contains_key(&admitted_txid));
        assert!(pool.is_empty());
    }

    #[test]
    fn stale_worst_heap_is_compacted_after_removals() {
        let mut pool = TxPool::new();
        for idx in 0..64u8 {
            let txid = [idx; 32];
            pool.insert_entry(
                txid,
                TxPoolEntry {
                    raw: vec![idx],
                    inputs: Vec::new(),
                    fee: u128::from(idx) + 1,
                    weight: u64::from(idx) + 1,
                    size: 1,
                    source: TxSource::Local,
                },
            );
        }

        for idx in 0..63u8 {
            let txid = [idx; 32];
            pool.remove_entry(&txid);
        }

        assert_eq!(pool.txs.len(), 1);
        assert_eq!(pool.heap_seqs.len(), 1);
        assert_eq!(pool.worst_heap.len(), 1);
    }

    #[test]
    fn current_worst_txid_skips_stale_heap_tail_without_rebuild() {
        let mut pool = TxPool::new();
        for idx in 0..2u8 {
            let txid = [idx + 1; 32];
            pool.insert_entry(
                txid,
                TxPoolEntry {
                    raw: vec![idx],
                    inputs: Vec::new(),
                    fee: u128::from(idx) + 10,
                    weight: u64::from(idx) + 10,
                    size: 1,
                    source: TxSource::Local,
                },
            );
        }

        pool.remove_entry(&[1; 32]);
        pool.insert_entry(
            [3; 32],
            TxPoolEntry {
                raw: vec![3],
                inputs: Vec::new(),
                fee: 30,
                weight: 30,
                size: 1,
                source: TxSource::Local,
            },
        );

        assert_eq!(pool.txs.len(), 2);
        assert_eq!(pool.heap_seqs.len(), 2);
        assert_eq!(pool.worst_heap.len(), 3);

        let next_heap_id = pool.next_heap_id;
        assert!(pool.current_worst_txid().is_some());
        assert_eq!(pool.next_heap_id, next_heap_id);
    }

    #[test]
    fn mining_sort_helpers_cover_zero_weight_and_tiebreaks() {
        let zero = TxPoolEntry {
            raw: vec![0x00],
            inputs: Vec::new(),
            fee: 10,
            weight: 0,
            size: 10,
            source: TxSource::Local,
        };
        let normal = TxPoolEntry {
            raw: vec![0x01],
            inputs: Vec::new(),
            fee: 20,
            weight: 20,
            size: 10,
            source: TxSource::Local,
        };
        assert_eq!(compare_fee_rate(&zero, &normal), Ordering::Equal);

        let high_fee = TxPoolEntry {
            raw: vec![0x03],
            inputs: Vec::new(),
            fee: 30,
            weight: 10,
            size: 10,
            source: TxSource::Local,
        };
        let low_fee = TxPoolEntry {
            raw: vec![0x02],
            inputs: Vec::new(),
            fee: 20,
            weight: 10,
            size: 10,
            source: TxSource::Local,
        };
        assert_eq!(
            compare_admit_priority([0x03; 32], &high_fee, [0x02; 32], &low_fee),
            Ordering::Greater
        );
        let high_txid: [u8; 32] = [0x03; 32];
        let low_txid: [u8; 32] = [0x02; 32];
        assert_eq!(
            compare_entries_for_mining(&(&high_txid, &high_fee), &(&low_txid, &low_fee)),
            Ordering::Less
        );

        let lighter = TxPoolEntry {
            raw: vec![0x04],
            inputs: Vec::new(),
            fee: 20,
            weight: 5,
            size: 10,
            source: TxSource::Local,
        };
        let heavier = TxPoolEntry {
            raw: vec![0x05],
            inputs: Vec::new(),
            fee: 20,
            weight: 8,
            size: 10,
            source: TxSource::Local,
        };
        let lighter_txid: [u8; 32] = [0x04; 32];
        let heavier_txid: [u8; 32] = [0x05; 32];
        assert_eq!(
            compare_entries_for_mining(&(&lighter_txid, &lighter), &(&heavier_txid, &heavier),),
            Ordering::Less
        );

        // Final tie-break is TXID (lexicographic), matching Go parity
        // (`clients/go/node/mempool.go`, `sortMempoolEntries`) and the
        // Rust admit-priority comparator. Same fee/size/weight, different
        // TXIDs — lower TXID sorts first.
        let equal_a = TxPoolEntry {
            raw: vec![0xFF],
            inputs: Vec::new(),
            fee: 20,
            weight: 10,
            size: 10,
            source: TxSource::Local,
        };
        let equal_b = TxPoolEntry {
            raw: vec![0xAA],
            inputs: Vec::new(),
            fee: 20,
            weight: 10,
            size: 10,
            source: TxSource::Local,
        };
        let lo_txid: [u8; 32] = [0x01; 32];
        let hi_txid: [u8; 32] = [0x02; 32];
        assert_eq!(
            compare_entries_for_mining(&(&lo_txid, &equal_a), &(&hi_txid, &equal_b)),
            Ordering::Less,
            "lower txid must sort first regardless of raw bytes",
        );
    }

    #[test]
    fn admit_reports_unavailable_on_height_overflow() {
        let raw = genesis_coinbase_bytes();
        let mut state = ChainState::new();
        state.has_tip = true;
        state.height = u64::MAX;

        let err = TxPool::new()
            .admit(&raw, &state, None, devnet_genesis_chain_id())
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("height overflow"));
    }

    #[test]
    fn admit_reports_unavailable_when_timestamp_context_missing() {
        let raw = genesis_coinbase_bytes();
        let mut state = ChainState::new();
        state.has_tip = true;
        state.height = 0;

        let (store, dir) = create_block_store("rubin-txpool-admit-mtp");
        let err = TxPool::new()
            .admit(&raw, &state, Some(&store), devnet_genesis_chain_id())
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("missing canonical header"));
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn admit_reports_unavailable_when_header_lookup_fails() {
        // RUB-162 Phase A migration rationale (per controller Path A
        // approval 2026-05-03 + RESP P1 #2 reorder fix 2026-05-03):
        //   - old assumption: positive_fixture_vector tx (fee=10/weight=
        //     7653) reaches next_block_mtp -> get_header_by_hash which
        //     fails with "read header"; pre-RUB-162 admit_with_metadata
        //     did not enforce the rolling fee floor at all.
        //   - new invariant: validate_fee_floor runs AFTER
        //     apply_policy as the post-apply Go-parity placement
        //     (mirrors clients/go/node/mempool.go (`addEntryLockedWithFloor` capacity-eviction block)
        //     validateCapacityAdmissionLocked called from
        //     addToMempoolLocked AFTER applyPolicyAgainstState). The
        //     conformance fixture is sub-floor so admit returns
        //     Unavailable("mempool fee below rolling minimum") AFTER
        //     apply_policy succeeds but BEFORE reaching the post-apply
        //     pool-full / capacity branches.
        //   - reachability: test's goal is the header_lookup failure
        //     path (block_store.get_header_by_hash → Err propagated
        //     as Unavailable). header_lookup happens inside
        //     next_block_mtp which runs BEFORE both apply_policy and
        //     validate_fee_floor, so floor-compliance is NOT a
        //     reachability requirement for this branch — any well-formed
        //     tx with chain_state has_tip=true / height=0 and a
        //     canonical tip whose header is missing from the
        //     block_store reaches the header_lookup branch.
        //   - replacement coverage: build a signed P2PK tx (input=7700)
        //     inline with chain_state has_tip=true / height=0; the
        //     input value is left at 7700 only for consistency with the
        //     general RUB-162 floor-compliant fixture policy across the
        //     test module (harmless here because reachability does not
        //     depend on it). The header_lookup-failure invariant remains
        //     under test.
        let (mut state, raw) = signed_p2pk_state_and_tx(
            7700,
            vec![TxOutput {
                value: 10,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x99; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );
        state.has_tip = true;
        state.height = 0;

        let (mut store, dir) = create_block_store("rubin-txpool-admit-header-read");
        store
            .set_canonical_tip(0, [0x42; 32])
            .expect("set canonical tip");

        let err = TxPool::new()
            .admit(&raw, &state, Some(&store), [0u8; 32])
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("read header"));
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn admit_rejects_coinbase_semantics_as_non_coinbase() {
        let raw = genesis_coinbase_bytes();
        let err = TxPool::new()
            .admit(&raw, &ChainState::new(), None, devnet_genesis_chain_id())
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("transaction rejected"));
    }

    #[test]
    fn admit_rejects_non_coinbase_anchor_outputs_by_policy() {
        // RUB-162 Phase A migration rationale (per controller Path A
        // approval 2026-05-03 + RESP P1 #2 reorder fix 2026-05-03):
        //   - old assumption: input=10, sum-outputs=9 → fee=1 reaches
        //     the anchor-output policy guard which returns Rejected.
        //   - new invariant: apply_policy (which evaluates the
        //     anchor-output policy) precedes the rolling-floor check
        //     in admit_with_metadata; mirrors Go addToMempoolLocked
        //     calling applyPolicyAgainstState then
        //     validateCapacityAdmissionLocked → validateFeeFloorLocked
        //     at clients/go/node/mempool.go (`addEntryLockedWithFloor`; renamed from `addToMempoolLocked` in wave-9 file split).
        //   - Proof assertion: `assert_eq!(err.kind, Rejected)` plus
        //     `err.message.contains("CORE_ANCHOR")` below pin the class
        //     winner against the alternative `Unavailable("mempool fee
        //     below rolling minimum")` outcome.
        //   - reachability: tx is well-formed; the anchor-output
        //     policy guard is the test goal. fee/weight headroom is
        //     defensive (apply_policy rejects regardless), but kept
        //     for parity with the cross-pollination matrix below.
        //   - replacement coverage: input bumped to 20000 so
        //     fee = 20000 - 9 = 19991 ≥ weight ⇒ candidate passes
        //     floor (the two-output tx with anchor + p2pk has a
        //     larger weight than single-output P2PK because of the
        //     anchor covenant + p2pk witness; 20000 covers it with
        //     headroom) and reaches the anchor-output policy guard.
        let (state, raw) = signed_p2pk_state_and_tx(
            20_000,
            vec![
                TxOutput {
                    value: 0,
                    covenant_type: COV_TYPE_ANCHOR,
                    covenant_data: vec![0x99],
                },
                TxOutput {
                    value: 9,
                    covenant_type: COV_TYPE_P2PK,
                    covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x22; 2592]),
                },
            ],
            0x00,
            None,
            Vec::new(),
        );
        let err = TxPool::new()
            .admit(&raw, &state, None, [0u8; 32])
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("CORE_ANCHOR"));
    }

    #[test]
    fn admit_rejects_da_tx_below_policy_stage_c_floor() {
        // RUB-162 Phase A migration rationale (per controller Path A
        // approval 2026-05-03 + RESP P1 #2 reorder fix 2026-05-03):
        //   - old assumption: input=10/output=9 fee=1, weight≈7653
        //     → reaches DA-helper which returns Rejected because
        //     fee < relay-floor-or-DA-required (whichever was higher).
        //   - new invariant: apply_policy (containing the DA helper
        //     with cfg-zero override) runs BEFORE
        //     validate_fee_floor. With cfg-zero the DA helper
        //     checks fee >= max(0, da_required) = da_required, leaving
        //     DA-side classification intact while the rolling relay
        //     floor is enforced separately AFTER apply_policy. To pin
        //     the DA-helper Stage C rejection path, fee must be
        //     < DA-required so apply_policy rejects with Rejected.
        //   - replacement coverage: input bumped to 8000 + fee=7991;
        //     weight≈7653; surcharge bumped to 200 so DA-required ≈
        //     da_bytes(70)*200 = 14000 > fee → DA-helper rejects with
        //     the canonical Stage C error message + named-term debug
        //     fields. The fee/weight headroom (≈1.04 ≥ default floor)
        //     is defensive — apply_policy rejects regardless, but the
        //     headroom keeps the fixture meaningful for the
        //     cross-pollination ordering matrix below. The test's
        //     purpose (verifying Stage C error message includes all
        //     named debug fields) remains under test.
        let (state, raw) = signed_p2pk_state_and_tx(
            8000,
            vec![TxOutput {
                value: 9,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x23; 2592]),
            }],
            0x02,
            Some(DaChunkCore {
                da_id: [0x55; 32],
                chunk_index: 0,
                chunk_hash: [0x66; 32],
            }),
            vec![0x77; 64],
        );
        let mut pool = TxPool::new_with_config(TxPoolConfig::default());
        pool.cfg.policy_da_surcharge_per_byte = 200;
        let err = pool.admit(&raw, &state, None, [0u8; 32]).unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(
            err.message.contains("DA fee below Stage C floor"),
            "expected Stage C error, got: {}",
            err.message
        );
        assert!(
            err.message.contains("relay_fee_floor=")
                && err.message.contains("da_fee_floor=")
                && err.message.contains("da_surcharge=")
                && err.message.contains("weight=")
                && err.message.contains("da_payload_len="),
            "expected debug fields in error, got: {}",
            err.message
        );
    }

    #[test]
    fn admit_rejects_core_ext_output_as_unsupported_runtime_before_floor() {
        // The candidate is floor-compliant; the test pins CORE_EXT
        // unsupported-runtime classification independent of fee floor.
        let (state, raw) = signed_p2pk_state_and_tx(
            7700,
            vec![TxOutput {
                value: 9,
                covenant_type: COV_TYPE_CORE_EXT,
                covenant_data: empty_core_ext_covenant_data(7),
            }],
            0x00,
            None,
            Vec::new(),
        );
        let err = TxPool::new()
            .admit(&raw, &state, None, [0u8; 32])
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("TX_ERR_COVENANT_TYPE_INVALID"));
    }

    #[test]
    fn admit_rejects_core_ext_spend_as_unsupported_runtime_before_floor() {
        // Spend-side twin of the output test. The candidate is made
        // floor-compliant so the unsupported-runtime policy is the winner.
        let (mut state, raw) = core_ext_spend_state_and_tx(9);
        for entry in state.utxos.values_mut() {
            entry.value = 7700;
        }
        let err = TxPool::new()
            .admit(&raw, &state, None, [0u8; 32])
            .unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Rejected);
        assert!(err.message.contains("TX_ERR_COVENANT_TYPE_INVALID"));
    }

    #[test]
    fn helper_errors_preserve_kind_and_message() {
        let conflict_err = conflict("conflict");
        assert_eq!(conflict_err.kind, TxPoolAdmitErrorKind::Conflict);
        assert_eq!(conflict_err.to_string(), "conflict");

        let rejected_err = rejected("rejected");
        assert_eq!(rejected_err.kind, TxPoolAdmitErrorKind::Rejected);
        assert_eq!(rejected_err.to_string(), "rejected");

        let unavailable_err = unavailable("unavailable");
        assert_eq!(unavailable_err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert_eq!(unavailable_err.to_string(), "unavailable");

        let owner_internal = super::owner_error(super::PendingOutpointError {
            kind: super::PendingOutpointErrorKind::Internal,
            message: "owner internal".to_string(),
            existing_txid: None,
        });
        assert_eq!(owner_internal.kind, TxPoolAdmitErrorKind::Unavailable);
        assert_eq!(owner_internal.message, "owner internal");
    }

    #[test]
    fn standard_delta_shape_and_invalid_victim_fail_before_mutation() {
        let state = ChainState::new();
        let owner = super::PendingOutpointOwnerHandle::new(
            super::PendingOutpointTip::from_chain_state(&state),
        );
        let context = owner.admission_context().expect("owner context");
        let mut pool = TxPool::new_bound_with_config(TxPoolConfig::default(), owner.clone());
        let entry = |tag| TxPoolEntry {
            inputs: vec![Outpoint {
                txid: [tag; 32],
                vout: 0,
            }],
            ..test_entry(1, 1, 1, TxSource::Local)
        };
        let (valid, invalid, candidate_txid) = ([0x31; 32], [0x32; 32], [0x33; 32]);
        pool.insert_entry(valid, entry(0x41));
        pool.insert_entry(invalid, entry(0x42));
        pool.owner_tokens.remove(&invalid);
        let high_water = owner.test_high_water();
        let mut candidate = Some(entry(0x43));
        let inputs = candidate.as_ref().unwrap().inputs.clone();
        let mut token = Some(
            owner
                .reserve(context, candidate_txid, inputs)
                .expect("candidate reservation"),
        );
        let snapshot = |pool: &TxPool| {
            (
                pool.txs.clone(),
                pool.owner_tokens.clone(),
                pool.heap_seqs.clone(),
                pool.worst_heap.clone().into_sorted_vec(),
                pool.next_heap_id,
                pool.used_bytes,
                owner.test_row_count().expect("rows"),
            )
        };
        let before = snapshot(&pool);
        let candidate_as_victim = [candidate_txid];
        let duplicate_victims = [valid, valid];
        for victims in [&candidate_as_victim[..], &duplicate_victims[..]] {
            assert_eq!(
                pool.commit_standard_delta(
                    &owner,
                    context,
                    candidate_txid,
                    &mut candidate,
                    &mut token,
                    victims,
                )
                .expect_err("invalid delta shape")
                .kind,
                TxPoolAdmitErrorKind::Unavailable
            );
        }
        let error = pool
            .commit_standard_delta(
                &owner,
                context,
                candidate_txid,
                &mut candidate,
                &mut token,
                &[valid, invalid],
            )
            .expect_err("invalid final victim");
        assert_eq!(error.kind, TxPoolAdmitErrorKind::Unavailable);
        assert_eq!(error.message, "missing pending-outpoint token");
        assert_eq!(
            snapshot(&pool),
            before,
            "invalid deltas leave records, tokens, rows, and heap state unchanged"
        );
        owner
            .release(token.as_ref().expect("candidate token"))
            .expect("candidate release");
        assert_eq!(owner.test_high_water(), high_water + 1);
        assert_eq!(owner.test_row_count().expect("rows"), 2);
    }

    // ----- Stage C DA fee policy parity tests (Linear RUB-122) -----
    //
    // These tests pin the Stage C admission predicate
    //
    //   required_fee = max(relay_fee_floor, da_fee_floor + da_surcharge)
    //
    // bit-for-bit against the merged Go reference
    // (clients/go/node/policy_da_anchor.go::RejectDaAnchorTxPolicy) and prove
    // that prior surcharge-only behavior, zero-surcharge bypass, relay/DA
    // dominance, overflow fail-closed, and non-DA short-circuit all behave
    // identically in Rust. Tests build a real signed DA transaction, then
    // call `reject_da_anchor_tx_policy` directly with crafted rate inputs to
    // exercise both accepted and rejected cases against the actual fee
    // computed by `compute_fee_no_verify`.

    fn build_signed_da_tx_with_fee(
        fee: u64,
        da_payload: Vec<u8>,
    ) -> (ChainState, Vec<u8>, u64, u64) {
        let output_value = 100u64;
        let input_value = output_value
            .checked_add(fee)
            .expect("test fee + output_value must not overflow");
        let (state, raw) = signed_p2pk_state_and_tx(
            input_value,
            vec![TxOutput {
                value: output_value,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x23; 2592]),
            }],
            0x02,
            Some(DaChunkCore {
                da_id: [0x55; 32],
                chunk_index: 0,
                chunk_hash: [0x66; 32],
            }),
            da_payload,
        );
        let (tx, _, _, _) = parse_tx(&raw).expect("parse signed DA tx");
        let (weight, da_bytes, _) =
            tx_weight_and_stats_public(&tx).expect("tx_weight_and_stats_public");
        assert!(da_bytes > 0, "test setup must produce DA tx");
        assert!(weight > 0, "test setup must produce non-zero weight");
        (state, raw, weight, da_bytes)
    }

    fn run_da_policy(
        raw: &[u8],
        state: &ChainState,
        current_min: u64,
        min_da: u64,
        surcharge: u64,
    ) -> Result<(), String> {
        let (tx, _, _, _) = parse_tx(raw).expect("parse tx");
        // RUB-167 single-walk invariant: tests mirror production callers
        // by passing pre-computed weight/da_bytes from one
        // tx_weight_and_stats_public call into reject_da_anchor_tx_policy.
        let (weight, da_bytes, _) =
            tx_weight_and_stats_public(&tx).expect("tx_weight_and_stats_public");
        reject_da_anchor_tx_policy(
            &tx,
            weight,
            da_bytes,
            &state.utxos,
            current_min,
            min_da,
            surcharge,
        )
    }

    #[test]
    fn stage_c_da_admit_with_fee_equal_required_da_dominant() {
        // current_min=0 zeroes the relay term; min_da=1, surcharge=1 makes
        // da_required = 2 * da_bytes the dominant term. Fee = 2*da_bytes
        // exactly admits (Stage C accepts fee == required).
        let (_, _, _, da_bytes) = build_signed_da_tx_with_fee(0, vec![0x77; 64]);
        let required = da_bytes
            .checked_mul(2)
            .expect("test setup u64 mul not overflow");
        let (state, raw, weight, da_bytes2) = build_signed_da_tx_with_fee(required, vec![0x77; 64]);
        // weight/da_bytes are deterministic across reruns with the same shape.
        assert_eq!(da_bytes2, da_bytes, "DA bytes deterministic");
        assert!(weight > 0);
        run_da_policy(&raw, &state, 0, 1, 1).expect("fee == required should admit");
    }

    #[test]
    fn stage_c_da_reject_with_fee_one_below_required_da_dominant() {
        let (_, _, _, da_bytes) = build_signed_da_tx_with_fee(0, vec![0x77; 64]);
        let required = da_bytes
            .checked_mul(2)
            .expect("test setup u64 mul not overflow");
        let (state, raw, _, _) = build_signed_da_tx_with_fee(required - 1, vec![0x77; 64]);
        let err =
            run_da_policy(&raw, &state, 0, 1, 1).expect_err("fee == required - 1 must reject");
        assert!(
            err.starts_with("DA fee below Stage C floor"),
            "expected Stage C error, got: {err}"
        );
        assert!(err.contains(&format!("required_fee={required}")));
        assert!(err.contains(&format!("da_fee_floor={da_bytes}")));
        assert!(err.contains(&format!("da_surcharge={da_bytes}")));
        assert!(err.contains(&format!("da_payload_len={da_bytes}")));
    }

    #[test]
    fn stage_c_da_zero_surcharge_still_enforces_min_da_fee_rate() {
        // Surcharge-zero regression: prior surcharge-only Rust returned Ok
        // unconditionally when da_surcharge_per_byte == 0. Stage C requires
        // da_floor (= da_bytes * min_da_fee_rate) to still bind.
        let (_, _, _, da_bytes) = build_signed_da_tx_with_fee(0, vec![0x77; 64]);
        let min_da_rate = 7u64;
        let required = da_bytes
            .checked_mul(min_da_rate)
            .expect("test setup u64 mul not overflow");
        // accept at exact floor
        let (state_ok, raw_ok, _, _) = build_signed_da_tx_with_fee(required, vec![0x77; 64]);
        run_da_policy(&raw_ok, &state_ok, 0, min_da_rate, 0)
            .expect("fee == da_floor with surcharge=0 admits");
        // reject one below
        let (state_bad, raw_bad, _, _) = build_signed_da_tx_with_fee(required - 1, vec![0x77; 64]);
        let err = run_da_policy(&raw_bad, &state_bad, 0, min_da_rate, 0)
            .expect_err("fee == da_floor - 1 with surcharge=0 must reject");
        assert!(err.contains("DA fee below Stage C floor"), "got: {err}");
        assert!(err.contains("da_surcharge=0"));
    }

    #[test]
    fn stage_c_da_zero_min_rate_still_enforces_surcharge() {
        // Symmetric regression: prior Rust short-circuited when
        // da_surcharge_per_byte == 0 but did not run da_floor at all. Stage C
        // separates the two terms; zeroing min_da_fee_rate must still leave
        // surcharge enforced.
        let (_, _, _, da_bytes) = build_signed_da_tx_with_fee(0, vec![0x77; 64]);
        let surcharge_rate = 5u64;
        let required = da_bytes
            .checked_mul(surcharge_rate)
            .expect("test setup u64 mul not overflow");
        let (state_bad, raw_bad, _, _) = build_signed_da_tx_with_fee(required - 1, vec![0x77; 64]);
        let err = run_da_policy(&raw_bad, &state_bad, 0, 0, surcharge_rate)
            .expect_err("fee == surcharge - 1 with min_da_fee_rate=0 must reject");
        assert!(err.contains("DA fee below Stage C floor"), "got: {err}");
        assert!(err.contains("da_fee_floor=0"));
        assert!(err.contains(&format!("da_surcharge={required}")));
    }

    #[test]
    fn stage_c_da_relay_floor_dominates_when_higher() {
        let (_, _, weight, da_bytes) = build_signed_da_tx_with_fee(0, vec![0x77; 8]);
        // Pick current_min so weight*current_min strictly > da_bytes*1 (min_da=1, surcharge=0).
        // For typical signed P2PK + 8-byte DA payload, weight is several thousand and
        // da_bytes is small (~10), so current_min=2 makes relay dominate.
        let current_min = 2u64;
        let relay_floor = weight
            .checked_mul(current_min)
            .expect("test setup u64 mul not overflow");
        let da_required = da_bytes;
        assert!(
            relay_floor > da_required,
            "test premise: relay_floor={relay_floor} must dominate da_required={da_required}"
        );
        let (state_bad, raw_bad, _, _) =
            build_signed_da_tx_with_fee(relay_floor - 1, vec![0x77; 8]);
        let err = run_da_policy(&raw_bad, &state_bad, current_min, 1, 0)
            .expect_err("fee == relay_floor - 1 must reject when relay dominates");
        assert!(err.contains("DA fee below Stage C floor"));
        assert!(err.contains(&format!("required_fee={relay_floor}")));
        assert!(err.contains(&format!("relay_fee_floor={relay_floor}")));
        // exact-floor accept
        let (state_ok, raw_ok, _, _) = build_signed_da_tx_with_fee(relay_floor, vec![0x77; 8]);
        run_da_policy(&raw_ok, &state_ok, current_min, 1, 0).expect("fee == relay_floor admits");
    }

    #[test]
    fn stage_c_da_floor_dominates_when_higher() {
        let (_, _, weight, da_bytes) = build_signed_da_tx_with_fee(0, vec![0x77; 64]);
        // Make da_required strictly larger than relay_floor: pick min_da_rate
        // so da_bytes*min_da_rate > weight (with current_min=1, surcharge=0).
        let min_da = (weight / da_bytes) + 2;
        let da_required = da_bytes
            .checked_mul(min_da)
            .expect("test setup u64 mul not overflow");
        assert!(
            da_required > weight,
            "test premise: da_required={da_required} must dominate weight={weight}"
        );
        let (state_bad, raw_bad, _, _) =
            build_signed_da_tx_with_fee(da_required - 1, vec![0x77; 64]);
        let err = run_da_policy(&raw_bad, &state_bad, 1, min_da, 0)
            .expect_err("fee == da_required - 1 must reject when DA dominates");
        assert!(err.contains("DA fee below Stage C floor"));
        assert!(err.contains(&format!("required_fee={da_required}")));
        assert!(err.contains(&format!("da_fee_floor={da_required}")));
    }

    /// Pins that a weight*current_mempool_min_fee_rate product above u64 is
    /// computed exactly rather than rejected. Before RUB-1127 this product
    /// went through `weight.checked_mul(..)`, so any rate that overflowed the
    /// u64 product rejected every DA transaction regardless of its fee -- a
    /// product-overflow rejection the contract forbids, and one that already
    /// did not exist in `fee_below_rate` or in either consensus-CLI policy
    /// model. Mirrors Go
    /// `TestRejectDaAnchorTxPolicy_RelayFloorProductAboveU64AdmitsClearingFee`.
    #[test]
    fn stage_c_relay_floor_product_above_u64_admits_clearing_fee() {
        // One u64-max input paying nearly everything as fee, plus a second
        // u64-max input: sum_in crosses u64, so fee ~= 2^65 and only an exact
        // u128 floor can decide the comparison.
        let (state, raw, _, _) = build_signed_da_tx_with_fee(u64::MAX - 100, vec![0x77; 8]);
        let (mut tx, _, _, _) = parse_tx(&raw).expect("parse tx");
        let second = Outpoint {
            txid: [0x17; 32],
            vout: 0,
        };
        tx.inputs.push(TxInput {
            prev_txid: second.txid,
            prev_vout: second.vout,
            script_sig: Vec::new(),
            sequence: 0,
        });
        let mut utxos = state.utxos.clone();
        utxos.insert(
            second,
            UtxoEntry {
                value: u64::MAX,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x23; 2592]),
                creation_height: 0,
                created_by_coinbase: false,
            },
        );
        let (weight, da_bytes, _) =
            tx_weight_and_stats_public(&tx).expect("tx_weight_and_stats_public");

        // Smallest rate whose product with weight exceeds u64.
        let current_min = (u64::MAX / weight) + 1;
        let exact_floor = u128::from(weight) * u128::from(current_min);
        assert!(
            exact_floor > u128::from(u64::MAX),
            "test premise: relay floor {exact_floor} must exceed u64"
        );
        let fee = compute_fee_no_verify(&tx, &utxos).expect("fee must derive above u64");
        assert!(
            fee >= exact_floor,
            "test premise: fee {fee} must clear the exact floor {exact_floor}"
        );

        reject_da_anchor_tx_policy(&tx, weight, da_bytes, &utxos, current_min, 1, 0)
            .expect("fee clearing the exact >u64 relay floor must be admitted");

        // One unit below the exact floor still rejects, with the honest Stage
        // C comparison reason rather than an arithmetic-capability error.
        let shortfall = fee - (exact_floor - 1);
        assert!(
            shortfall <= u128::from(u64::MAX),
            "test premise: shortfall {shortfall} must fit an output value"
        );
        tx.outputs[0].value += shortfall as u64;
        let err = reject_da_anchor_tx_policy(&tx, weight, da_bytes, &utxos, current_min, 1, 0)
            .expect_err("fee one below the exact relay floor must reject");
        assert!(err.contains("DA fee below Stage C floor"), "got: {err}");
        assert!(
            !err.contains("overflow"),
            "no overflow rejection may remain on the relay-floor term: {err}"
        );
    }

    /// Two u64-max inputs sum to 2^64: a u64 accumulator returned
    /// `sum_in overflow` here while Go derived the fee exactly, so the two
    /// clients disagreed on admission without any fee above u64 being
    /// involved. Mirrors the Go coverage in
    /// `clients/go/node/coverage_hotspots_test.go`.
    #[test]
    fn compute_fee_no_verify_sums_two_u64_max_inputs_exactly() {
        let (state, raw, _, _) = build_signed_da_tx_with_fee(u64::MAX - 100, vec![0x77; 8]);
        let (mut tx, _, _, _) = parse_tx(&raw).expect("parse tx");
        let second = Outpoint {
            txid: [0x18; 32],
            vout: 0,
        };
        tx.inputs.push(TxInput {
            prev_txid: second.txid,
            prev_vout: second.vout,
            script_sig: Vec::new(),
            sequence: 0,
        });
        let mut utxos = state.utxos.clone();
        utxos.insert(
            second,
            UtxoEntry {
                value: u64::MAX,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x23; 2592]),
                creation_height: 0,
                created_by_coinbase: false,
            },
        );
        // sum_in = 2*(2^64-1), sum_out = 100 (the single output).
        let fee = compute_fee_no_verify(&tx, &utxos).expect("two u64-max inputs must derive a fee");
        assert_eq!(fee, 2 * u128::from(u64::MAX) - 100);
        assert!(fee > u128::from(u64::MAX), "fee {fee} must exceed u64");
    }

    /// RUB-1127: the maximum structural transaction fee. MAX_TX_INPUTS=1024
    /// inputs of u64-max each with zero outputs is the largest fee any
    /// transaction can carry, and the summation path must return it exactly.
    /// The row drives compute_fee_no_verify rather than a signed consensus
    /// transaction: 1024 ML-DSA-87 signatures is not a proportionate test cost,
    /// and the accumulator is the code the fee width is claimed for.
    #[test]
    fn compute_fee_no_verify_sums_the_maximum_structural_fee_exactly() {
        let outpoint = Outpoint {
            txid: [0x01; 32],
            vout: 0,
        };
        let tx = Tx {
            version: TX_WIRE_VERSION,
            tx_kind: 0,
            tx_nonce: 0,
            inputs: vec![
                TxInput {
                    prev_txid: outpoint.txid,
                    prev_vout: outpoint.vout,
                    script_sig: Vec::new(),
                    sequence: 0,
                };
                MAX_TX_INPUTS as usize
            ],
            outputs: Vec::new(),
            locktime: 0,
            da_commit_core: None,
            da_chunk_core: None,
            witness: Vec::new(),
            da_payload: Vec::new(),
        };
        let mut utxos = HashMap::new();
        utxos.insert(
            outpoint,
            UtxoEntry {
                value: u64::MAX,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x23; 2592]),
                creation_height: 0,
                created_by_coinbase: false,
            },
        );
        let fee = compute_fee_no_verify(&tx, &utxos).expect("maximum structural fee must derive");
        // 1024*(2^64-1) in canonical decimal.
        assert_eq!(fee.to_string(), "18889465931478580853760");
    }

    #[test]
    fn stage_c_da_overflow_da_floor_rejects_fail_closed() {
        let (state, raw, _, da_bytes) = build_signed_da_tx_with_fee(1_000_000, vec![0x77; 8]);
        let err = run_da_policy(&raw, &state, 1, u64::MAX, 1)
            .expect_err("u64::MAX * da_bytes must overflow");
        assert!(err.starts_with("DA fee floor overflow"), "got: {err}");
        assert!(err.contains(&format!("da_payload_len={da_bytes}")));
    }

    #[test]
    fn stage_c_da_overflow_da_surcharge_rejects_fail_closed() {
        let (state, raw, _, da_bytes) = build_signed_da_tx_with_fee(1_000_000, vec![0x77; 8]);
        let err = run_da_policy(&raw, &state, 1, 1, u64::MAX)
            .expect_err("u64::MAX * da_bytes must overflow");
        assert!(err.starts_with("DA surcharge overflow"), "got: {err}");
        assert!(err.contains(&format!("da_payload_len={da_bytes}")));
    }

    #[test]
    fn stage_c_da_overflow_da_required_addition_rejects_fail_closed() {
        // Pick min_da and surcharge each just below u64::MAX/da_bytes so each
        // mul stays in range, but their sum (da_floor + da_surcharge) overflows.
        let (_, _, _, da_bytes) = build_signed_da_tx_with_fee(0, vec![0x77; 8]);
        let half = u64::MAX / da_bytes;
        let min_da = half;
        let surcharge = half;
        // da_floor = da_bytes * half; da_surcharge = da_bytes * half.
        // Each fits. Sum = 2 * da_bytes * half ~ 2*u64::MAX/2 -> overflow on add.
        let da_floor = da_bytes
            .checked_mul(min_da)
            .expect("test premise: each mul fits");
        let da_surcharge = da_bytes
            .checked_mul(surcharge)
            .expect("test premise: each mul fits");
        assert!(
            da_floor.checked_add(da_surcharge).is_none(),
            "test premise: addition must overflow"
        );
        let (state, raw, _, _) = build_signed_da_tx_with_fee(1_000_000, vec![0x77; 8]);
        let err =
            run_da_policy(&raw, &state, 0, min_da, surcharge).expect_err("addition must overflow");
        assert!(err.starts_with("DA required fee overflow"), "got: {err}");
    }

    #[test]
    fn stage_c_non_da_tx_short_circuits_admit() {
        // Non-DA tx (tx_kind=0, no DaChunkCore, no da_payload) must be admitted
        // by reject_da_anchor_tx_policy without applying any DA term, even with
        // aggressive rate config. Non-DA relay-fee floor enforcement lives in
        // the free validate_fee_floor predicate. Admission calls it after the
        // Go-compatible duplicate/conflict boundary; relay calls it through
        // apply_post_consensus_policy_with_floor. This helper only validates
        // the DA half of the Stage C contract and intentionally short-circuits
        // for non-DA inputs.
        let (state, raw) = signed_p2pk_state_and_tx(
            10,
            vec![TxOutput {
                value: 9,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x23; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );
        run_da_policy(&raw, &state, u64::MAX, u64::MAX, u64::MAX)
            .expect("non-DA tx admits regardless of DA rate config");
    }

    #[test]
    fn stage_c_da_zero_rates_admit_without_fee_compute() {
        // DA tx with all three rates = 0: relay_floor=0, da_floor=0,
        // da_surcharge=0, required=0. Stage C admits without computing fee.
        //
        // Proof assertion: state.utxos is cleared before run_da_policy is
        // called, so compute_fee_no_verify (which dereferences each input
        // outpoint via utxos.get(...) and returns Err("missing utxo") on
        // None) would propagate that error. The test asserts run_da_policy
        // returns Ok(()), which is reachable only through the
        // `if required == 0 { return Ok(()); }` short-circuit branch in
        // reject_da_anchor_tx_policy that bypasses compute_fee_no_verify
        // entirely.
        let (mut state, raw, _, _) = build_signed_da_tx_with_fee(1_000_000, vec![0x77; 8]);
        state.utxos.clear(); // strip utxos so any fee compute would error
        run_da_policy(&raw, &state, 0, 0, 0)
            .expect("required=0 admits without compute_fee_no_verify");
    }

    // ====================================================================
    // RUB-162 Phase A regression tests — section header
    //
    // Tests in this section exercise the four Phase A invariants from
    // GitHub #1407 through the public TxPool::admit / admit_with_metadata
    // entrypoint (NOT helper-only direct calls — that was the reviewer
    // concern). Each individual test below carries its own Proof
    // assertion in the doc-comment naming the exact assertion mechanism;
    // see the `rub162_*` test functions for specifics.
    // ====================================================================

    use super::{fee_rate_below_floor, validate_fee_floor};

    /// P1 #4 fix — DA tx whose DA fee passes but rolling relay floor fails
    /// must return Unavailable from validate_fee_floor, NOT Rejected
    /// from reject_da_anchor_tx_policy. Mirrors Go applyPolicyAgainstState
    /// behaviour (clients/go/node/mempool.go (`validateFeeFloorLocked` wrapper)
    /// and `validateFeeFloorLockedWithFloor`): mempool admit
    /// passes currentMempoolMinFeeRate=0 to the DA helper so the
    /// relay-floor classification is owned uniformly by
    /// validateFeeFloorLocked (Unavailable, transient/retryable).
    ///
    /// Reachability: tx is well-formed (parses, basic+canonical checks pass,
    /// inputs not double-spent), DA-side floor configured to 0 so DA helper
    /// short-circuits Ok, then validate_fee_floor rejects on the
    /// rolling floor. The TxPoolConfig used has policy_min_da_fee_rate=0 and
    /// policy_da_surcharge_per_byte=0, so the DA helper's required term
    /// is 0 — the only failing path is the new validate_fee_floor.
    #[test]
    fn rub162_admit_da_below_rolling_floor_returns_unavailable_not_rejected() {
        // Build a DA tx with low fee_rate (fee=1, weight ≈ 7653 from ML-DSA
        // witness + 64-byte da_payload).
        let (state, raw, _weight, _da_bytes) = build_signed_da_tx_with_fee(1, vec![0x77; 64]);

        // cfg with relay floor = DEFAULT (1) and DA-side terms zeroed so
        // the DA helper's `required` collapses to 0 (short-circuit Ok).
        let mut pool = TxPool::new_with_config(TxPoolConfig {
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: 0,
            policy_da_surcharge_per_byte: 0,
            ..TxPoolConfig::default()
        });

        let err = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect_err("DA tx with fee=1, weight≈7653 must fail relay floor");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Unavailable,
            "P1 #4 fix: relay-floor failure returns Unavailable (transient/retryable), NOT Rejected (terminal)"
        );
        assert!(
            err.message.contains("mempool fee below rolling minimum"),
            "error message must come from validate_fee_floor, not DA helper; got: {}",
            err.message
        );
    }

    /// P1 #4 baseline preservation — DA tx whose DA fee fails the DA-side
    /// floor (independent of the rolling relay floor) still returns
    /// Rejected from reject_da_anchor_tx_policy. The Phase A change must
    /// NOT regress DA-floor classification.
    ///
    /// Reachability: tx is well-formed; DA-side terms force a non-zero
    /// required fee that the tx fails. apply_policy returns Err mapped
    /// to Rejected before validate_fee_floor is reached.
    #[test]
    fn rub162_admit_da_below_da_floor_returns_rejected() {
        // Per RUB-162 RESP P1 #4 ordering: apply_policy (containing the
        // DA helper with cfg-zero override) runs BEFORE
        // validate_fee_floor. To pin the DA-side rejection path,
        // build a fixture that FAILS the DA-side terms
        // (fee < da_bytes * (min_da + surcharge)). The fixture also
        // PASSES the relay floor (fee >= weight) defensively — under
        // WIP #6 ordering apply_policy rejects first regardless of
        // floor compliance, but the headroom keeps the fixture stable
        // if a future change reorders the validators again.
        let (state, raw, weight, da_bytes) = build_signed_da_tx_with_fee(8_000, vec![0x77; 64]);
        assert!(
            8_000 >= weight,
            "fee 8000 must >= weight {weight} to pass relay-floor (defensive headroom)"
        );
        // DA-required = da_bytes * (min_da + surcharge). With ~70 bytes
        // and 200+200, da_required ≈ 28000 ≫ fee=8000.
        let min_da = 200u64;
        let surcharge = 200u64;
        let required_da = da_bytes
            .checked_mul(min_da + surcharge)
            .expect("test arithmetic");
        assert!(
            8_000 < required_da,
            "fee 8000 must be below DA-required {required_da} so DA helper rejects"
        );
        let mut pool = TxPool::new_with_config(TxPoolConfig {
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: min_da,
            policy_da_surcharge_per_byte: surcharge,
            ..TxPoolConfig::default()
        });
        let err = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect_err("DA tx with fee below DA-required must fail DA Stage C floor");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Rejected,
            "DA-side failure must remain Rejected (terminal); preserved RUB-122 behaviour"
        );
        assert!(
            err.message.contains("DA fee below Stage C floor"),
            "error must come from DA helper, not floor check; got: {}",
            err.message
        );
    }

    /// P1 #4 happy path — DA tx that passes BOTH the DA-side floor AND
    /// the rolling relay floor admits cleanly.
    #[test]
    fn rub162_admit_da_above_both_floors_admits_with_indexes() {
        // weight ≈ 7653 from ML-DSA witness + 64-byte da_payload. Pick
        // fee that is comfortably above both floors:
        //   relay floor: weight * DEFAULT (1) ≈ 7653
        //   DA floor: da_bytes * 1 + da_bytes * 1 = 2 * da_bytes
        // Use fee = 30000 (well above both for da_bytes around 70-100).
        let (state, raw, weight, da_bytes) = build_signed_da_tx_with_fee(30_000, vec![0x77; 64]);
        assert!(
            weight <= 30_000,
            "fee must cover relay floor (weight={weight})"
        );
        assert!(
            2 * da_bytes <= 30_000,
            "fee must cover DA term ({da_bytes} bytes)"
        );

        let mut pool = TxPool::new_with_config(TxPoolConfig {
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: 1,
            policy_da_surcharge_per_byte: 1,
            ..TxPoolConfig::default()
        });
        let txid = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect("DA tx above both floors must admit");
        assert_eq!(pool.len(), 1);
        assert!(pool.txs.contains_key(&txid));
        assert_eq!(
            pool.owner_row_count(),
            1,
            "single-input tx must populate exactly one owner claim row"
        );
        assert_eq!(
            pool.heap_seqs.len(),
            1,
            "tx must be tracked in heap_seqs after insert_entry"
        );
        assert_eq!(
            pool.worst_heap.len(),
            1,
            "tx must be pushed to worst_heap after insert_entry"
        );
    }

    /// P1 #2 atomicity through public admit — Unavailable on relay-floor
    /// failure leaves pool records, owner claim rows, and lazy heap filtering
    /// (current_worst_txid) unchanged. This proves admit_with_metadata
    /// performs the rolling-floor classification BEFORE any insert_entry
    /// call.
    ///
    /// Proof assertion: pre-populate via the production `insert_entry`
    /// path (so resident record/claim and next_heap_id receive the snapshot);
    /// rolling-floor rejection through the public admit entry; assert
    /// record/claim state, worst_heap length, next_heap_id, and
    /// current_worst_txid lookup are all unchanged.
    #[test]
    fn rub162_admit_atomicity_unavailable_floor_leaves_no_partial_state() {
        let (state, raw, _weight, _da_bytes) = build_signed_da_tx_with_fee(1, vec![0x77; 64]);
        let mut pool = TxPool::new_with_config(TxPoolConfig {
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: 0,
            policy_da_surcharge_per_byte: 0,
            ..TxPoolConfig::default()
        });
        let resident_txid = [0xAB; 32];
        let resident_input = Outpoint {
            txid: [0x77; 32],
            vout: 0,
        };
        pool.insert_entry(
            resident_txid,
            TxPoolEntry {
                raw: vec![0x01],
                inputs: vec![resident_input.clone()],
                fee: 100,
                weight: 1,
                size: 1,
                source: TxSource::Local,
            },
        );
        let pool_len_before = pool.len();
        let owner_rows_before = pool.owner_row_count();
        let heap_seqs_before = pool.heap_seqs.len();
        let worst_heap_before = pool.worst_heap.len();
        let next_heap_id_before = pool.next_heap_id;
        let worst_txid_before = pool.current_worst_txid();
        let owner_high_water_before = pool.test_owner().test_high_water();

        let err = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect_err("must fail rolling floor");
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);

        assert_eq!(pool.len(), pool_len_before, "txs unchanged on Err");
        assert_eq!(
            pool.test_owner().test_high_water(),
            owner_high_water_before + 1,
            "final-floor failure releases its claim but retains token high-water"
        );
        assert_eq!(
            pool.owner_row_count(),
            owner_rows_before,
            "owner claim rows unchanged on Err"
        );
        assert_eq!(
            pool.heap_seqs.len(),
            heap_seqs_before,
            "heap_seqs unchanged on Err"
        );
        assert_eq!(
            pool.worst_heap.len(),
            worst_heap_before,
            "worst_heap unchanged on Err (no spurious WorstEntryKey push)"
        );
        assert_eq!(
            pool.next_heap_id, next_heap_id_before,
            "next_heap_id unchanged (no partial WorstEntryKey reservation)"
        );
        assert_eq!(
            pool.current_worst_txid(),
            worst_txid_before,
            "current_worst_txid lookup unchanged (resident still observable)"
        );
        assert!(pool.txs.contains_key(&resident_txid), "resident untouched");
        assert_eq!(
            pool.owner_txid_for_outpoint(&resident_input),
            Some(resident_txid),
            "resident owner claim row untouched"
        );
    }

    #[test]
    fn rub162_evict_txids_clears_all_indexes_added_by_insert_entry() {
        let mut pool = TxPool::new_with_config(TxPoolConfig {
            policy_current_mempool_min_fee_rate: 0,
            ..TxPoolConfig::default()
        });
        let txid = [0xC1; 32];
        let outpoint = Outpoint {
            txid: [0xAA; 32],
            vout: 0,
        };
        // Insert via the same private path admit_with_metadata uses so
        // the test shares the production insert side.
        pool.insert_entry(
            txid,
            TxPoolEntry {
                raw: vec![0xff],
                inputs: vec![outpoint.clone()],
                fee: 100,
                weight: 1,
                size: 1,
                source: TxSource::Local,
            },
        );
        assert!(pool.txs.contains_key(&txid));
        assert_eq!(pool.owner_txid_for_outpoint(&outpoint), Some(txid));
        assert!(pool.heap_seqs.contains_key(&txid));

        pool.evict_txids(&[txid]);
        assert!(!pool.txs.contains_key(&txid), "txs cleared");
        assert!(
            pool.owner_txid_for_outpoint(&outpoint).is_none(),
            "owner claim rows cleared"
        );
        assert!(!pool.heap_seqs.contains_key(&txid), "heap_seqs cleared");
        // Lazy worst_heap behaviour: the physical heap entry MAY still
        // exist (compaction is delayed via 2x threshold), but the
        // public lookup must filter it out via the heap_seqs guard.
        assert!(
            pool.current_worst_txid().is_none(),
            "current_worst_txid hides the evicted entry via the heap_seqs guard"
        );
    }

    /// P1 #3 cleanup state symmetry — `remove_conflicting_outpoints`
    /// (the public path used by reorg/conflict cleanup) routes through
    /// `evict_txids` and clears the same coordinated record/claim state.
    #[test]
    fn rub162_remove_conflicting_outpoints_clears_all_indexes() {
        let mut pool = TxPool::new_with_config(TxPoolConfig {
            policy_current_mempool_min_fee_rate: 0,
            ..TxPoolConfig::default()
        });
        let txid = [0xC2; 32];
        let outpoint = Outpoint {
            txid: [0xBB; 32],
            vout: 0,
        };
        pool.insert_entry(
            txid,
            TxPoolEntry {
                raw: vec![0xff],
                inputs: vec![outpoint.clone()],
                fee: 100,
                weight: 1,
                size: 1,
                source: TxSource::Local,
            },
        );

        let unrelated = Outpoint {
            txid: [0xBC; 32],
            vout: 1,
        };
        pool.remove_conflicting_outpoints(&[unrelated, outpoint.clone()]);
        assert!(
            !pool.txs.contains_key(&txid),
            "resident removed on conflict"
        );
        assert!(
            pool.owner_txid_for_outpoint(&outpoint).is_none(),
            "owner claim rows cleared on conflict"
        );
        assert!(
            !pool.heap_seqs.contains_key(&txid),
            "heap_seqs cleared on conflict"
        );
    }

    /// RUB-17 remove-helper parity: removing an existing entry and then
    /// removing the same or an absent txid again must remain a no-op for
    /// unrelated resident entries. This pins Go-compatible
    /// `removeTxLocked` no-op semantics while checking Rust's extra
    /// heap/accounting indexes.
    #[test]
    fn rub17_evict_txids_is_idempotent_and_preserves_unrelated_accounting() {
        let mut pool = TxPool::new();
        let removed_txid = [0xD1; 32];
        let survivor_txid = [0xD2; 32];
        let removed_outpoint = Outpoint {
            txid: [0xE1; 32],
            vout: 0,
        };
        let survivor_outpoint = Outpoint {
            txid: [0xE2; 32],
            vout: 1,
        };

        pool.insert_entry(
            removed_txid,
            TxPoolEntry {
                raw: vec![0xD1; 3],
                inputs: vec![removed_outpoint.clone()],
                fee: 30,
                weight: 3,
                size: 3,
                source: TxSource::Remote,
            },
        );
        pool.insert_entry(
            survivor_txid,
            TxPoolEntry {
                raw: vec![0xD2; 5],
                inputs: vec![survivor_outpoint.clone()],
                fee: 50,
                weight: 5,
                size: 5,
                source: TxSource::Reorg,
            },
        );
        let next_heap_id_before = pool.next_heap_id;
        let survivor_heap_seq_before = *pool
            .heap_seqs
            .get(&survivor_txid)
            .expect("survivor heap sequence before eviction");

        pool.evict_txids(&[removed_txid]);
        pool.evict_txids(&[removed_txid, [0xFF; 32]]);

        assert!(!pool.txs.contains_key(&removed_txid), "removed tx gone");
        assert!(
            pool.owner_txid_for_outpoint(&removed_outpoint).is_none(),
            "removed owner claim row gone"
        );
        assert!(
            !pool.heap_seqs.contains_key(&removed_txid),
            "removed heap sequence gone"
        );
        assert_eq!(pool.len(), 1, "only survivor remains");
        assert_eq!(pool.used_bytes, 5, "used_bytes counts only survivor");
        assert_eq!(
            pool.owner_txid_for_outpoint(&survivor_outpoint),
            Some(survivor_txid),
            "survivor owner claim row preserved"
        );
        assert_eq!(
            pool.entry_source(&survivor_txid),
            Some(TxSource::Reorg),
            "survivor source preserved"
        );
        assert_eq!(
            pool.heap_seqs.get(&survivor_txid),
            Some(&survivor_heap_seq_before),
            "survivor heap sequence preserved before lazy worst lookup"
        );
        assert_eq!(
            pool.next_heap_id, next_heap_id_before,
            "remove/no-op paths must not allocate heap ids"
        );
        assert_eq!(
            pool.current_worst_txid(),
            Some(survivor_txid),
            "stale removed heap entries are hidden by heap_seqs"
        );
        assert_eq!(
            pool.next_heap_id, next_heap_id_before,
            "worst lookup must not repair accounting after remove/no-op"
        );
    }

    #[test]
    fn rub17_remove_entry_then_readd_same_txid_has_fresh_accounting() {
        let mut pool = TxPool::new();
        let txid = [0xD3; 32];
        let outpoint = Outpoint {
            txid: [0xE3; 32],
            vout: 2,
        };

        pool.insert_entry(
            txid,
            TxPoolEntry {
                raw: vec![0x01; 7],
                inputs: vec![outpoint.clone()],
                fee: 70,
                weight: 7,
                size: 7,
                source: TxSource::Remote,
            },
        );
        let first_heap_seq = *pool.heap_seqs.get(&txid).expect("first heap seq");

        pool.remove_entry(&txid);
        assert_eq!(pool.len(), 0, "entry removed");
        assert_eq!(pool.used_bytes, 0, "bytes cleared after remove_entry");
        assert!(
            pool.owner_txid_for_outpoint(&outpoint).is_none(),
            "owner claim row cleared"
        );
        assert!(!pool.heap_seqs.contains_key(&txid), "heap seq cleared");
        assert_eq!(pool.entry_source(&txid), None, "source cleared");
        assert_eq!(pool.current_worst_txid(), None, "no live worst entry");

        pool.insert_entry(
            txid,
            TxPoolEntry {
                raw: vec![0x02; 11],
                inputs: vec![outpoint.clone()],
                fee: 110,
                weight: 11,
                size: 11,
                source: TxSource::Reorg,
            },
        );

        assert_eq!(pool.len(), 1, "re-add succeeds");
        assert_eq!(pool.used_bytes, 11, "bytes reflect fresh entry only");
        assert_eq!(
            pool.owner_txid_for_outpoint(&outpoint),
            Some(txid),
            "owner claim row rebuilt for fresh entry"
        );
        assert_eq!(
            pool.entry_source(&txid),
            Some(TxSource::Reorg),
            "source reflects fresh entry"
        );
        assert!(
            *pool.heap_seqs.get(&txid).expect("second heap seq") > first_heap_seq,
            "re-add must allocate a fresh heap sequence"
        );
        assert_eq!(
            pool.current_worst_txid(),
            Some(txid),
            "fresh heap entry is visible"
        );
    }

    /// P1 #4 helper unit test — `fee_rate_below_floor` is a u128 cross-mul
    /// predicate matching Go `feeRateBelowFloor`
    /// (`feeRateBelowFloor` in clients/go/node/mempool.go), including the in-helper
    /// `floor < DefaultMempoolMinFeeRate` clamp at
    /// clients/go/node/mempool.go (in-helper clamp inside `feeRateBelowFloor`; grep `DefaultMempoolMinFeeRate` in fn body). Calling with floor=0
    /// promotes to DEFAULT_MEMPOOL_MIN_FEE_RATE
    /// before the cross-mul, so a fee=0 / weight=100 / floor=0 input
    /// rejects (required becomes 100*1=100 post-clamp, fee<100).
    ///
    /// Proof assertion: assertions below pin the documented Go branches:
    /// - weight=0 always returns true (uncomputable rate)
    /// - floor=0 + non-zero weight clamps to DEFAULT and uses required=weight*1
    /// - exact-floor (fee == weight*floor) returns false
    /// - one-below-floor returns true
    /// - u128 boundary (u64::MAX*u64::MAX) lossless
    #[test]
    fn rub162_fee_rate_below_floor_helper_branches() {
        // Zero weight: always below floor.
        assert!(fee_rate_below_floor(u128::from(u64::MAX), 0, 1));
        // Zero floor: clamps to DEFAULT (1). For weight=100: required=100.
        // fee=0 < 100 → true; fee=99 < 100 → true; fee=100 == 100 → false.
        assert!(fee_rate_below_floor(0, 100, 0));
        assert!(fee_rate_below_floor(99, 100, 0));
        assert!(!fee_rate_below_floor(100, 100, 0));
        // Exact floor: not below.
        assert!(!fee_rate_below_floor(100, 100, 1));
        // One below floor.
        assert!(fee_rate_below_floor(99, 100, 1));
        // u128 boundary: required = u64::MAX * u64::MAX. Fits u128, no wrap.
        assert!(fee_rate_below_floor(u128::from(u64::MAX), u64::MAX, 2));
        assert!(!fee_rate_below_floor(u128::from(u64::MAX), u64::MAX, 1));
        // Concrete 7653 pin (matches the conformance fixture's weight).
        assert!(fee_rate_below_floor(10, 7653, 1));
        assert!(!fee_rate_below_floor(7653, 7653, 1));
    }

    /// P1 #4 + clamp regression — `validate_fee_floor` propagates
    /// a cfg-seeded zero into `fee_rate_below_floor`, which itself clamps
    /// to DEFAULT_MEMPOOL_MIN_FEE_RATE per Go `feeRateBelowFloor`
    /// (`feeRateBelowFloor` in clients/go/node/mempool.go, with the in-helper clamp
    /// inside the function body — `if floor < DefaultMempoolMinFeeRate` block). The error message surfaces the post-clamp
    /// value so operators see the actual decision basis.
    #[test]
    fn rub162_validate_fee_floor_clamps_cfg_zero_to_default() {
        // fee=0 weight=1 cfg_floor=0: helper clamps floor 0 → DEFAULT (1);
        // required = 1*1 = 1; fee<1 → reject.
        let err = validate_fee_floor(0, 1, 0)
            .expect_err("helper clamp must enforce DEFAULT even when cfg=0");
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(err.message.contains("min_fee_rate=1"));
        // fee=1 weight=1 cfg_floor=0: exact floor passes after clamp.
        validate_fee_floor(1, 1, 0).expect("exact floor passes");
    }

    /// PR-1410 wave-2 fix — relay_metadata must NOT be weaker than
    /// admit_with_metadata. The cfg-zero override before apply_policy
    /// (`applyPolicyAgainstState` in clients/go/node/mempool.go) preserves DA-side error class
    /// isolation. Both Rust `relay_metadata` and Go `RelayMetadata` enforce
    /// the rolling relay floor after structural/chainstate/policy validation,
    /// so relay metadata never propagates a tx the local mempool would reject
    /// solely as below the rolling floor.
    ///
    /// Proof assertion: same DA tx, same TxPoolConfig with non-zero
    /// rolling floor + DA-side terms; admit returns Unavailable; relay
    /// metadata ALSO returns Unavailable with the same message — they
    /// must surface the same classification for the same tx so peer
    /// relay never propagates a tx the local mempool refuses to admit.
    #[test]
    fn rub162_relay_metadata_da_below_rolling_floor_returns_unavailable_matching_admit() {
        // weight ≈ 7653; DA-required = 2 * da_bytes (≈ 70 bytes ⇒ 140);
        // pick fee that beats DA-required but is well below relay floor
        // (fee=1000 < weight=7653 ⇒ relay-floor reject on both paths).
        let (state, raw, weight, da_bytes) = build_signed_da_tx_with_fee(1_000, vec![0x77; 64]);
        assert!(
            1_000 >= 2 * da_bytes,
            "test setup must let DA-required fit fee: 2*{da_bytes} <= 1000"
        );
        assert!(
            1_000 < weight,
            "test setup must put fee below relay floor: 1000 < weight={weight}"
        );
        let cfg = TxPoolConfig {
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: 1,
            policy_da_surcharge_per_byte: 1,
            ..TxPoolConfig::default()
        };
        // admit must reject with Unavailable from validate_fee_floor.
        let mut pool = TxPool::new_with_config(cfg.clone());
        let admit_err = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect_err("admit should reject below relay floor");
        assert_eq!(admit_err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(admit_err
            .message
            .contains("mempool fee below rolling minimum"));
        // relay_metadata MUST also reject with Unavailable — same
        // classification as admit so peer relay does not propagate a
        // tx the local mempool refuses.
        let relay_err = relay_metadata(&raw, &state, None, [0u8; 32], &cfg)
            .expect_err("relay_metadata must reject below rolling floor (matching admit)");
        assert_eq!(relay_err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(relay_err
            .message
            .contains("mempool fee below rolling minimum"));
    }

    /// PR-1410 wave-2 — DA tx whose DA fee fails the DA-side floor
    /// must surface as Rejected (DA helper class, terminal) from
    /// relay_metadata, NOT collapsed into the new rolling-floor
    /// Unavailable class. Preserves DA-side error class isolation
    /// while the new validate_fee_floor call sits AFTER apply_policy.
    #[test]
    fn rub162_relay_metadata_da_below_da_floor_returns_rejected_not_unavailable() {
        // weight ≈ 7653; with min_da=200 + surcharge=200 + da_bytes≈70,
        // DA-required ≈ 28000 ≫ fee=8000. The fee=8000 also passes the
        // rolling floor (fee/weight ≈ 1.04 ≥ 1) so the DA-side reject
        // is the ONLY rejection path — proves DA classification
        // survives the new floor check.
        let (state, raw, weight, da_bytes) = build_signed_da_tx_with_fee(8_000, vec![0x77; 64]);
        let min_da = 200u64;
        let surcharge = 200u64;
        let required_da = da_bytes
            .checked_mul(min_da + surcharge)
            .expect("test arithmetic");
        assert!(
            8_000 < required_da,
            "test setup must put fee below DA-required: 8000 < {required_da}"
        );
        assert!(
            8_000 >= weight,
            "test setup must put fee above rolling floor: 8000 >= weight={weight}"
        );
        let cfg = TxPoolConfig {
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: min_da,
            policy_da_surcharge_per_byte: surcharge,
            ..TxPoolConfig::default()
        };
        let err = relay_metadata(&raw, &state, None, [0u8; 32], &cfg)
            .expect_err("relay_metadata must reject DA tx below DA-side floor");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Rejected,
            "DA-side rejection must remain terminal Rejected, NOT new rolling-floor Unavailable; got {:?}",
            err
        );
        assert!(
            err.message.contains("DA fee below Stage C floor"),
            "error must come from DA helper, not floor check; got: {}",
            err.message
        );
    }

    /// PR-1410 wave-2 — DA tx whose fee passes BOTH floors must yield
    /// relay metadata (Ok). Smoke for the post-fix happy path.
    #[test]
    fn rub162_relay_metadata_da_above_both_floors_returns_ok() {
        // fee = 30000; DA-required = da_bytes(70)*1*2 = 140; weight≈7653,
        // fee/weight ≈ 3.92 ≥ 1. Both floors pass.
        let (state, raw, _weight, _da_bytes) = build_signed_da_tx_with_fee(30_000, vec![0x77; 64]);
        let cfg = TxPoolConfig {
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: 1,
            policy_da_surcharge_per_byte: 1,
            ..TxPoolConfig::default()
        };
        let meta = relay_metadata(&raw, &state, None, [0u8; 32], &cfg)
            .expect("relay_metadata should succeed when both floors pass");
        assert!(meta.fee > 0);
        assert_eq!(meta.size, raw.len());
    }

    /// P1 #2 fix — fee-floor classification runs BEFORE pool-full
    /// ordering classification (Go `validateCapacityAdmissionLocked`
    /// order at clients/go/node/mempool.go (`addEntryLockedWithFloor` capacity-eviction block)). A sub-floor
    /// candidate against a full pool must surface the floor error,
    /// not the pool-full error.
    ///
    /// Proof assertion: build a candidate with fee/weight far below
    /// DEFAULT_MEMPOOL_MIN_FEE_RATE, fill the pool to capacity, then
    /// call admit. Expected: TxPoolAdmitErrorKind::Unavailable with
    /// "mempool fee below rolling minimum" message (NOT "tx pool full").
    #[test]
    fn rub162_admit_full_pool_below_floor_returns_floor_error_not_pool_full() {
        // Sub-floor candidate: input=10 / output=8 → fee=2, weight≈7653.
        let (state, raw) = signed_p2pk_state_and_tx(
            10,
            vec![TxOutput {
                value: 8,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&vec![0x88; 2592]),
            }],
            0x00,
            None,
            Vec::new(),
        );
        let mut pool = TxPool::new();
        // Fill pool to MAX so the pool-full ordering branch IS reachable.
        for idx in 0..MAX_TX_POOL_TRANSACTIONS {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&(idx as u64 + 1).to_le_bytes());
            pool.insert_entry(
                key,
                TxPoolEntry {
                    raw: vec![0xff],
                    inputs: Vec::new(),
                    fee: 100,
                    weight: 1,
                    size: 1,
                    source: TxSource::Local,
                },
            );
        }
        let err = pool.admit(&raw, &state, None, [0u8; 32]).unwrap_err();
        assert_eq!(err.kind, TxPoolAdmitErrorKind::Unavailable);
        assert!(
            err.message.contains("mempool fee below rolling minimum"),
            "P1 #2: floor error must win over pool-full ordering; got: {}",
            err.message
        );
        // Pool unchanged (atomicity preserved through the new ordering).
        assert_eq!(pool.txs.len(), MAX_TX_POOL_TRANSACTIONS);
    }

    /// RUB-162 cross-pollination ordering PIN — a tx that is BOTH
    /// sub-floor AND fails DA-side terms must classify as Rejected
    /// (apply_policy → DA helper fires first under WIP #6 ordering),
    /// NOT Unavailable (rolling floor would fire if it ran first).
    /// Mirrors Go addToMempoolLocked at clients/go/node/mempool.go (`addEntryLockedWithFloor`; renamed from `addToMempoolLocked` in wave-9 file split)
    /// where applyPolicyAgainstState precedes validateCapacityAdmissionLocked
    /// → validateFeeFloorLocked.
    ///
    /// Proof assertion: build a DA tx whose fee is below BOTH the rolling
    /// floor AND the DA-required floor; admit returns
    /// `TxPoolAdmitErrorKind::Rejected` with the canonical
    /// `DA fee below Stage C floor` message (not the
    /// `mempool fee below rolling minimum` message).
    #[test]
    fn rub162_admit_sub_floor_da_anchor_classifies_as_da_rejected_not_floor_unavailable() {
        // weight ≈ 7600 from ML-DSA-87 witness + 64-byte da_payload
        // (same shape as `rub162_admit_da_above_both_floors_admits_with_indexes`
        // in the body of `build_signed_da_tx_with_fee` above). fee=10 ≪ weight ⇒ sub-floor under
        // DEFAULT_MEMPOOL_MIN_FEE_RATE=1. da_bytes ≈ 70 (DA chunk core
        // + 64-byte payload). DA-required = da_bytes * (min_da=200 +
        // surcharge=200) ≈ 28_000 ≫ fee=10 ⇒ sub-DA. The asserts
        // below pin only the test-relevant inequalities (10 < weight,
        // 10 < required_da) — the ≈ values are illustrative bounds
        // for the sub-floor / sub-DA classification reasoning.
        let (state, raw, weight, da_bytes) = build_signed_da_tx_with_fee(10, vec![0x55; 64]);
        assert!(
            10 < weight,
            "test setup must put fee below relay floor: 10 < weight={weight}"
        );
        let min_da = 200u64;
        let surcharge = 200u64;
        let required_da = da_bytes
            .checked_mul(min_da + surcharge)
            .expect("test arithmetic");
        assert!(
            10 < required_da,
            "test setup must put fee below DA-required: 10 < da_required={required_da}"
        );

        let mut pool = TxPool::new_with_config(TxPoolConfig {
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: min_da,
            policy_da_surcharge_per_byte: surcharge,
            ..TxPoolConfig::default()
        });
        let err = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect_err("admit must reject when both sub-floor and sub-DA");
        // Cross-pollination winner: apply_policy (DA helper) → Rejected.
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Rejected,
            "DA-rejection class must win when apply_policy runs before validate_fee_floor; got kind={:?} message={}",
            err.kind,
            err.message
        );
        assert!(
            err.message.contains("DA fee below Stage C floor"),
            "error must come from DA helper, not the rolling floor check; got: {}",
            err.message
        );
        // Atomicity: no partial insert on cross-pollination reject.
        assert_eq!(pool.len(), 0);
        assert!(pool.owner_row_count() == 0);
        assert_eq!(pool.heap_seqs.len(), 0);
    }

    /// RUB-162 cross-pollination ordering PIN — sub-floor tx with a
    /// CORE_EXT output while the node runtime does not support CORE_EXT.
    /// Same ordering invariant as the DA-anchor cross-pollination test above.
    ///
    /// Proof assertion: build a sub-floor P2PK->CORE_EXT tx;
    /// `assert_eq!(err.kind, Rejected)` plus
    /// `err.message.contains("TX_ERR_COVENANT_TYPE_INVALID")` below pin
    /// the class winner against the alternative
    /// `Unavailable("mempool fee below rolling minimum")` outcome.
    #[test]
    fn rub162_admit_sub_floor_core_ext_classifies_as_core_ext_rejected_not_floor_unavailable() {
        // input=10 / output=9 → fee=1 with weight≈7533 ⇒ sub-floor.
        // CORE_EXT output with ext_id=7 => unsupported-runtime guard rejects.
        let (state, raw) = signed_p2pk_state_and_tx(
            10,
            vec![TxOutput {
                value: 9,
                covenant_type: COV_TYPE_CORE_EXT,
                covenant_data: empty_core_ext_covenant_data(7),
            }],
            0x00,
            None,
            Vec::new(),
        );
        let mut pool = TxPool::new();
        let err = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect_err("admit must reject when both sub-floor and unsupported CORE_EXT");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Rejected,
            "CORE_EXT unsupported class must win when apply_policy runs before validate_fee_floor; got kind={:?} message={}",
            err.kind,
            err.message
        );
        assert!(
            err.message.contains("TX_ERR_COVENANT_TYPE_INVALID"),
            "error must come from the consensus 0x0102 unassigned-reject (RUB-585), not the rolling floor check; got: {}",
            err.message
        );
        // Atomicity: no partial insert on cross-pollination reject.
        assert_eq!(pool.len(), 0);
        assert!(pool.owner_row_count() == 0);
        assert_eq!(pool.heap_seqs.len(), 0);
    }

    // ====================================================================
    // RUB-166 fast-reject regression tests
    // Mirrors Go RUB-165 (PR #1415, merge SHA ed3be97) tests pattern from
    // clients/go/node/mempool_test.go::TestMempoolCheapFeeFloorPrecheck*.
    // ====================================================================

    /// Plain P2PK below the rolling floor: the cheap precheck rejects
    /// with `Unavailable` carrying the verbatim
    /// `mempool fee below rolling minimum: fee=X weight=Y min_fee_rate=Z`
    /// message, matching Go's `cheapFeeFloorPrecheck` and the existing
    /// `validate_fee_floor` format. Pool stays empty.
    ///
    /// Proof of fast-reject ordering: this test deliberately passes
    /// `chain_id = [0u8; 32]` while the fixture signs with
    /// `devnet_genesis_chain_id()`. If the precheck did NOT fire before
    /// `apply_non_coinbase_tx_basic_update_*`, the chain_id mismatch
    /// would surface inside ML-DSA signature verification and the
    /// returned error would be `Rejected` with a `TX_ERR_SIG_INVALID`
    /// substring. The assertion that the error is `Unavailable` AND
    /// the message does NOT contain "sig" empirically pins the
    /// pre-signature-verify ordering of the precheck.
    #[test]
    fn rub166_admit_below_floor_p2pk_returns_unavailable_with_floor_message() {
        // Fee = 20 - 10 = 10; weight ≈ 7.6KB (ML-DSA witness); fee_rate
        // ≪ DEFAULT_MEMPOOL_MIN_FEE_RATE (1).
        let (mut state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        state.has_tip = true;
        state.height = 1;
        let mut pool = TxPool::new();
        let err = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect_err("plain P2PK below the rolling floor must fast-reject");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Unavailable,
            "below-floor reject must be transient/retryable Unavailable"
        );
        assert!(
            err.message.contains("mempool fee below rolling minimum"),
            "message must match the Go-parity precheck format; got: {}",
            err.message
        );
        assert!(
            !err.message.to_lowercase().contains("sig"),
            "precheck must reject BEFORE signature verification (mirror of Go test \
             assertion `!strings.Contains(txErr.Message, TX_ERR_SIG_INVALID)`); got: {}",
            err.message
        );
        assert_eq!(pool.len(), 0);
        assert!(pool.owner_row_count() == 0);
    }

    /// RUB-18 ordering parity: Go's `addTxWithSource` runs the plain-P2PK
    /// cheap fee-floor precheck before locked duplicate-txid admission checks.
    /// A duplicate that is also below the current rolling floor must therefore
    /// return `Unavailable`, not `Conflict`.
    #[test]
    fn rub18_admit_below_floor_duplicate_fast_rejects_before_conflict() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let (tx, _txid, _wtxid, consumed) = parse_tx(&raw).expect("parse tx");
        assert_eq!(consumed, raw.len());
        let (weight, _da_bytes, _) = tx_weight_and_stats_public(&tx).expect("weight");
        assert!(
            7690u128 < (weight as u128) * 2,
            "test fixture must become below-floor after raising floor to 2"
        );

        let mut pool = TxPool::new();
        pool.admit(&raw, &state, None, devnet_genesis_chain_id())
            .expect("first floor-compliant admit");
        pool.cfg.policy_current_mempool_min_fee_rate = 2;

        let err = pool
            .admit(&raw, &state, None, devnet_genesis_chain_id())
            .expect_err("below-floor duplicate must hit fast floor precheck first");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Unavailable,
            "Go-compatible cheap precheck must win over duplicate Conflict"
        );
        assert!(
            err.message.contains("mempool fee below rolling minimum"),
            "expected rolling-floor message, got: {}",
            err.message
        );
        assert!(
            !err.message.contains("already in mempool"),
            "duplicate conflict must not win this ordering case: {}",
            err.message
        );
        assert_eq!(pool.len(), 1, "failed duplicate retry must not mutate pool");
    }

    #[test]
    fn rub18_admit_below_floor_spender_conflict_fast_rejects_before_conflict() {
        let (state, resident, conflicting) = signed_conflicting_p2pk_state_and_txs(7700, 10, 7699);
        let (tx, _txid, _wtxid, consumed) = parse_tx(&conflicting).expect("parse tx");
        assert_eq!(consumed, conflicting.len());
        let (weight, _da_bytes, _) = tx_weight_and_stats_public(&tx).expect("weight");
        assert!(
            1u128 < weight as u128,
            "conflicting fixture must be below the default floor"
        );

        let mut pool = TxPool::new();
        pool.admit(&resident, &state, None, devnet_genesis_chain_id())
            .expect("resident admit");

        let err = pool
            .admit(&conflicting, &state, None, devnet_genesis_chain_id())
            .expect_err("below-floor spender conflict must hit fast floor precheck first");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Unavailable,
            "Go-compatible cheap precheck must win over spender Conflict"
        );
        assert!(
            err.message.contains("mempool fee below rolling minimum"),
            "expected rolling-floor message, got: {}",
            err.message
        );
        assert!(
            !err.message.contains("double-spend conflict"),
            "spender conflict must not win this ordering case: {}",
            err.message
        );
        assert_eq!(
            pool.len(),
            1,
            "failed conflicting candidate must not mutate pool"
        );
    }

    /// Plain P2PK at or above the rolling floor: precheck returns
    /// Ok(()), the expensive admission path runs, and the tx admits.
    /// Proves the precheck does not false-positive on floor-compliant
    /// transactions. Uses `devnet_genesis_chain_id()` because the
    /// fixture signs with that chain id (so the post-precheck signature
    /// validation step actually succeeds — the precheck-skipped path is
    /// what proves the precheck is conservative for compliant tx).
    #[test]
    fn rub166_admit_at_or_above_floor_p2pk_admits() {
        // Fee = 7700 - 10 = 7690; weight ≈ 7653; fee_rate ≈ 1.005 ≥ 1.
        let (mut state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        state.has_tip = true;
        state.height = 1;
        let mut pool = TxPool::new();
        let (_txid, _meta) = pool
            .admit_with_metadata(&raw, &state, None, devnet_genesis_chain_id())
            .expect("floor-compliant signed P2PK must admit");
        assert_eq!(pool.len(), 1);
    }

    /// DA-bearing transaction below the DA-side floor: the cheap
    /// precheck must short-circuit on `tx_kind != 0x00` /
    /// `da_payload non-empty` so the expensive admission path keeps its
    /// DA-vs-rolling-floor classification ordering (RUB-162 / RUB-122).
    /// With the default config the DA-side term dominates and the
    /// expensive path returns `Rejected` with `DA fee below Stage C
    /// floor` — the same outcome as before RUB-166. This test pins
    /// that the precheck did NOT fast-reject the DA tx as a generic
    /// rolling-floor `Unavailable` (which would have masked the DA
    /// classification).
    #[test]
    fn rub166_admit_da_tx_skips_precheck_keeps_da_side_classification() {
        // fee=10 ≪ da_required (= da_bytes * (min_da + surcharge) with
        // default DEFAULT_MIN_DA_FEE_RATE=1, surcharge=0 ⇒ ~70 with
        // da_bytes ≈ 70). DA-side fails first inside apply_policy.
        let (state, raw, _weight, _da_bytes) = build_signed_da_tx_with_fee(10, vec![0x55; 64]);
        let mut pool = TxPool::new();
        let err = pool
            .admit(&raw, &state, None, [0u8; 32])
            .expect_err("DA tx below DA-side floor must reject");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Rejected,
            "DA-side failure remains Rejected (terminal) per RUB-122 / RUB-162; \
             precheck must not have masked it as rolling-floor Unavailable"
        );
        assert!(
            err.message.contains("DA fee below Stage C floor"),
            "DA-side classification message preserved; precheck did not fast-reject; got: {}",
            err.message
        );
    }

    /// Missing-UTXO defer: when the input UTXO is missing from
    /// chainstate, `fee_precheck_p2pk_input_value` returns `None` and
    /// the precheck defers to `Ok(())` so the expensive admission path
    /// runs and surfaces the missing-UTXO failure as `Rejected`. The
    /// precheck must NOT mask the missing-UTXO failure as a
    /// rolling-floor `Unavailable`. Mirrors Go's
    /// `TestMempoolCheapFeeFloorPrecheckPreservesMissingUTXOReject`
    /// in clients/go/node/mempool_test.go. The relay-path
    /// no-precheck boundary is covered by the dedicated
    /// `rub162_relay_metadata_*` tests further up — this test scope is
    /// strictly the missing-UTXO defer branch on the admit path.
    #[test]
    fn rub166_admit_below_floor_with_missing_utxo_keeps_expensive_reject_class() {
        let (mut state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        state.has_tip = true;
        state.height = 1;
        // Drop the input UTXO so fee_precheck_p2pk_input_value returns
        // None and the precheck defers to the expensive path.
        state.utxos.clear();
        let mut pool = TxPool::new();
        let err = pool
            .admit(&raw, &state, None, devnet_genesis_chain_id())
            .expect_err("missing-UTXO must reject through the expensive path");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Rejected,
            "missing-UTXO failure must remain Rejected (terminal); precheck must not have \
             masked it as rolling-floor Unavailable"
        );
        assert!(
            !err.message.contains("mempool fee below rolling minimum"),
            "missing-UTXO path was stolen by fee precheck: {}",
            err.message
        );
        assert_eq!(pool.len(), 0);
    }

    /// Conservatism guard `tx.inputs.len() != 1` in
    /// `fee_precheck_p2pk_input_value` defers BOTH `len == 0` and
    /// `len > 1` cases. Direct-helper exercise of both branches with
    /// the single-input fixture mutated in-place. Building a real
    /// multi-input signed P2PK fixture requires a second-keypair +
    /// second-UTXO setup that is heavier than the helper-direct branch
    /// pin and adds no additional safety beyond what this test
    /// already proves: any non-1 input count returns None, so the
    /// precheck defers and the expensive path classifies the tx.
    #[test]
    fn rub166_precheck_defers_when_input_count_not_exactly_one() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        // Branch 1: zero inputs.
        let mut zero_inputs = parsed.clone();
        zero_inputs.inputs.clear();
        let zero_result = fee_precheck_p2pk_input_value(
            &zero_inputs,
            &state.utxos,
            /* next_height */ 1,
            None,
            None,
        );
        assert!(
            zero_result.is_none(),
            "len == 0 must return None so precheck defers; got {:?}",
            zero_result
        );
        // Branch 2: two inputs (duplicate the single input). The second
        // outpoint may not resolve in utxos but that is irrelevant —
        // the `len != 1` guard short-circuits before any lookup.
        let mut two_inputs = parsed.clone();
        let dup = two_inputs.inputs[0].clone();
        two_inputs.inputs.push(dup);
        let multi_result = fee_precheck_p2pk_input_value(
            &two_inputs,
            &state.utxos,
            /* next_height */ 1,
            None,
            None,
        );
        assert!(
            multi_result.is_none(),
            "len > 1 must return None so precheck defers; got {:?}",
            multi_result
        );
    }

    /// Output sum overflow defer: `fee_precheck_p2pk_output_value` uses
    /// `checked_add` and returns `None` when summing P2PK outputs
    /// overflows `u64`. The precheck must defer (`Ok(())`) so the
    /// expensive admission path classifies the tx — fast-rejecting an
    /// overflow as "below floor" would be wrong (the tx may be
    /// massively over-funded by an attacker-controlled sum). Direct-
    /// helper exercise constructs two P2PK outputs whose sum exceeds
    /// `u64::MAX`.
    #[test]
    fn rub166_precheck_defers_on_output_sum_overflow() {
        use rubin_consensus::constants::{COV_TYPE_P2PK, SUITE_ID_ML_DSA_87};
        use rubin_consensus::TxOutput;
        // Wave-4: outputs need a valid 33-byte P2PK covenant_data (suite +
        // 32-byte payload) so the new wave-4 conservatism guards
        // (cov_data_len, suite_id) do NOT fire before the overflow branch.
        // This keeps the test scoped to the `checked_add` overflow path.
        let mut cov_data = vec![SUITE_ID_ML_DSA_87];
        cov_data.extend_from_slice(&[0u8; 32]);
        let outputs = vec![
            TxOutput {
                value: u64::MAX,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: cov_data.clone(),
            },
            TxOutput {
                value: 1,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: cov_data,
            },
        ];
        let result = fee_precheck_p2pk_output_value(&outputs, /* next_height */ 1, None);
        assert!(
            result.is_none(),
            "u64-overflow output sum must return None so precheck defers; got {:?}",
            result
        );
    }

    /// Defensive `weight == 0` guard inside `cheap_fee_floor_precheck`:
    /// production callers always pass non-zero weight from
    /// `tx_weight_and_stats_public` (zero-weight tx is structurally
    /// impossible), but the guard mirrors Go's
    /// `if err != nil || weight == 0 { return nil }` defensive check
    /// for future-proofing. Direct-helper exercise pins the branch.
    #[test]
    fn rub166_precheck_defers_when_weight_is_zero() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        let result = cheap_fee_floor_precheck(
            &parsed,
            &state.utxos,
            /* weight */ 0,
            /* min_fee_rate */ 1,
            /* next_height */ 1,
            /* rotation */ None,
            /* registry */ None,
        );
        assert!(
            result.is_ok(),
            "weight==0 must defer (Ok); got {:?}",
            result
        );
    }

    /// Output sum > input value: precheck must defer (overspend belongs
    /// to the expensive path's overspend error class, not to the cheap
    /// rolling-floor reject). Exercises the `output_value > input_value`
    /// branch of `cheap_fee_floor_precheck`.
    #[test]
    fn rub166_precheck_defers_when_output_exceeds_input() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        // Inject an extra utxo so fee_precheck_p2pk_input_value returns
        // a SMALLER value than the output sum. We rebuild state with
        // input_value = 5 (less than the original 10-value output).
        let mut state = state;
        let outpoint = Outpoint {
            txid: parsed.inputs[0].prev_txid,
            vout: parsed.inputs[0].prev_vout,
        };
        if let Some(entry) = state.utxos.get_mut(&outpoint) {
            entry.value = 5;
        }
        // The signed tx still has output_value=10, so output > input.
        // Precheck path: fee_precheck_p2pk_input_value returns Some(5);
        // fee_precheck_p2pk_output_value returns Some(10); 10 > 5 →
        // return Ok(()).
        let result = cheap_fee_floor_precheck(
            &parsed,
            &state.utxos,
            /* weight */ 1,
            /* min_fee_rate */ 1,
            /* next_height */ 1,
            /* rotation */ None,
            /* registry */ None,
        );
        assert!(
            result.is_ok(),
            "overspend must defer (Ok); got {:?}",
            result
        );
    }

    /// Wave-4 class-closure conservatism guard: `tx_nonce == 0` for a
    /// non-coinbase tx is permanently rejected by
    /// `apply_non_coinbase_tx_basic_update_*` (slow path) at
    /// `clients/rust/crates/rubin-consensus/src/utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`
    /// with `TxErrTxNonceInvalid`. Without the wave-4 early-defer
    /// guard, a below-floor tx with `tx_nonce == 0` would be
    /// fast-rejected as transient `Unavailable("mempool fee below
    /// rolling minimum")`, masking the permanent reject class.
    /// Direct-helper exercise pins the early-defer branch.
    #[test]
    fn rub166_precheck_defers_when_tx_nonce_is_zero() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (mut parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        // Mutate parsed tx to set tx_nonce = 0 while keeping shape valid.
        parsed.tx_nonce = 0;
        let result = cheap_fee_floor_precheck(
            &parsed,
            &state.utxos,
            /* weight */ 1,
            /* min_fee_rate */ 1,
            /* next_height */ 1,
            /* rotation */ None,
            /* registry */ None,
        );
        assert!(
            result.is_ok(),
            "tx_nonce==0 must defer (Ok) so the slow path returns the \
             permanent TxErrTxNonceInvalid; got {:?}",
            result
        );
    }

    /// Wave-4 class-closure conservatism guard: a P2PK output with
    /// `value == 0` is permanently rejected by
    /// `validate_tx_covenants_genesis` (slow path) at
    /// `clients/rust/crates/rubin-consensus/src/covenant_genesis.rs` `validate_tx_covenants_genesis`
    /// with `"CORE_P2PK value must be > 0"`. Without the wave-4 defer
    /// guard, a below-floor tx with a zero-value P2PK output would be
    /// fast-rejected as transient `Unavailable`, masking the
    /// permanent reject class. Direct-helper exercise on
    /// `fee_precheck_p2pk_output_value` pins the value==0 branch.
    #[test]
    fn rub166_precheck_defers_when_p2pk_output_value_is_zero() {
        use rubin_consensus::constants::{COV_TYPE_P2PK, SUITE_ID_ML_DSA_87};
        use rubin_consensus::TxOutput;
        let mut cov_data = vec![SUITE_ID_ML_DSA_87];
        cov_data.extend_from_slice(&[0u8; 32]);
        let outputs = vec![TxOutput {
            value: 0,
            covenant_type: COV_TYPE_P2PK,
            covenant_data: cov_data,
        }];
        let result = fee_precheck_p2pk_output_value(&outputs, /* next_height */ 1, None);
        assert!(
            result.is_none(),
            "P2PK output value==0 must return None so precheck defers; got {:?}",
            result
        );
    }

    /// Wave-4 class-closure conservatism guard: a P2PK output whose
    /// `covenant_data` length is not exactly `MAX_P2PK_COVENANT_DATA`
    /// (33 = 1-byte suite_id + 32-byte payload) is permanently
    /// rejected by `validate_tx_covenants_genesis` at
    /// `clients/rust/crates/rubin-consensus/src/covenant_genesis.rs` `validate_tx_covenants_genesis`
    /// with `"invalid CORE_P2PK covenant_data length"`. Without the
    /// wave-4 defer guard, a below-floor tx with a wrong-length P2PK
    /// covenant_data would be fast-rejected as transient
    /// `Unavailable`, masking the permanent reject class. Both `len <
    /// 33` (empty) and `len > 33` (oversized) branches pinned.
    #[test]
    fn rub166_precheck_defers_when_p2pk_covenant_data_length_invalid() {
        use rubin_consensus::constants::COV_TYPE_P2PK;
        use rubin_consensus::TxOutput;
        // Branch 1: empty covenant_data (len == 0, < 33).
        let outputs_empty = vec![TxOutput {
            value: 100,
            covenant_type: COV_TYPE_P2PK,
            covenant_data: vec![],
        }];
        let result_empty =
            fee_precheck_p2pk_output_value(&outputs_empty, /* next_height */ 1, None);
        assert!(
            result_empty.is_none(),
            "empty covenant_data must return None so precheck defers; got {:?}",
            result_empty
        );
        // Branch 2: oversized covenant_data (len > 33).
        let outputs_oversized = vec![TxOutput {
            value: 100,
            covenant_type: COV_TYPE_P2PK,
            covenant_data: vec![0u8; 64],
        }];
        let result_oversized =
            fee_precheck_p2pk_output_value(&outputs_oversized, /* next_height */ 1, None);
        assert!(
            result_oversized.is_none(),
            "oversized covenant_data must return None so precheck defers; got {:?}",
            result_oversized
        );
    }

    /// Wave-4 class-closure conservatism guard: a P2PK output whose
    /// `covenant_data[0]` (suite_id) is not in the active
    /// `native_create_suites(next_height)` set is permanently rejected
    /// by `validate_tx_covenants_genesis` at
    /// `clients/rust/crates/rubin-consensus/src/covenant_genesis.rs` `validate_tx_covenants_genesis`
    /// with `"CORE_P2PK suite not in native create set"`. Without the
    /// wave-4 defer guard, a below-floor tx with an invalid suite
    /// would be fast-rejected as transient `Unavailable`, masking the
    /// permanent reject class. Default rotation provider
    /// (`DefaultRotationProvider`) accepts only `SUITE_ID_ML_DSA_87`
    /// (0x01); any other byte value triggers the defer.
    #[test]
    fn rub166_precheck_defers_when_p2pk_suite_not_in_native_create_set() {
        use rubin_consensus::constants::{COV_TYPE_P2PK, SUITE_ID_ML_DSA_87};
        use rubin_consensus::TxOutput;
        // Use suite_id = 0xFE (non-native, definitely not in default set).
        // Sanity: 0xFE != SUITE_ID_ML_DSA_87 (0x01).
        assert_ne!(0xFEu8, SUITE_ID_ML_DSA_87);
        let mut cov_data = vec![0xFEu8];
        cov_data.extend_from_slice(&[0u8; 32]);
        let outputs = vec![TxOutput {
            value: 100,
            covenant_type: COV_TYPE_P2PK,
            covenant_data: cov_data,
        }];
        let result = fee_precheck_p2pk_output_value(&outputs, /* next_height */ 1, None);
        assert!(
            result.is_none(),
            "non-native suite_id must return None so precheck defers; got {:?}",
            result
        );
    }

    /// Wave-4 input-side class-closure: a non-empty `script_sig` on a
    /// P2PK input is rejected by the slow path
    /// (`apply_non_coinbase_tx_basic_update_*` at
    /// `clients/rust/crates/rubin-consensus/src/utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`)
    /// with `TxErrParse "script_sig must be empty under genesis
    /// covenant set"`. Without the wave-4 defer guard, a below-floor
    /// tx with non-empty `script_sig` would be misclassified as
    /// transient `Unavailable` instead of `Rejected` (terminal).
    #[test]
    fn rub166_precheck_defers_when_input_script_sig_non_empty() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (mut parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        parsed.inputs[0].script_sig = vec![0x01];
        let result = fee_precheck_p2pk_input_value(
            &parsed,
            &state.utxos,
            /* next_height */ 1,
            None,
            None,
        );
        assert!(
            result.is_none(),
            "non-empty script_sig must return None so precheck defers; got {:?}",
            result
        );
    }

    /// Wave-4 input-side class-closure: a `sequence > 0x7fffffff` on a
    /// P2PK input is rejected by the slow path at
    /// `clients/rust/crates/rubin-consensus/src/utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`
    /// with `TxErrSequenceInvalid "sequence exceeds 0x7fffffff"`.
    /// Without the wave-4 defer guard, a below-floor tx with
    /// out-of-range sequence would be misclassified as transient
    /// `Unavailable` instead of `Rejected` (terminal).
    #[test]
    fn rub166_precheck_defers_when_input_sequence_out_of_range() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (mut parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        parsed.inputs[0].sequence = 0x8000_0000;
        let result = fee_precheck_p2pk_input_value(
            &parsed,
            &state.utxos,
            /* next_height */ 1,
            None,
            None,
        );
        assert!(
            result.is_none(),
            "sequence > 0x7fffffff must return None so precheck defers; got {:?}",
            result
        );
    }

    /// Wave-4 input-side class-closure: a tx with `witness.len() != 1`
    /// (zero or more than one witness slot) on a single-P2PK-input tx
    /// is rejected by the slow path at
    /// `clients/rust/crates/rubin-consensus/src/utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`
    /// with `TxErrParse` ("invalid witness slots" / "witness
    /// underflow"). Without the wave-4 defer guard, a below-floor tx
    /// with mismatched witness count would be misclassified as
    /// transient `Unavailable` instead of `Rejected` (terminal). Both
    /// `len == 0` and `len == 2` branches pinned.
    #[test]
    fn rub166_precheck_defers_when_witness_count_not_exactly_one() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        // Branch 1: zero witness slots.
        let mut zero_witness = parsed.clone();
        zero_witness.witness.clear();
        let zero_result = fee_precheck_p2pk_input_value(
            &zero_witness,
            &state.utxos,
            /* next_height */ 1,
            None,
            None,
        );
        assert!(
            zero_result.is_none(),
            "witness.len() == 0 must return None so precheck defers; got {:?}",
            zero_result
        );
        // Branch 2: two witness slots.
        let mut two_witness = parsed.clone();
        let dup = two_witness.witness[0].clone();
        two_witness.witness.push(dup);
        let two_result = fee_precheck_p2pk_input_value(
            &two_witness,
            &state.utxos,
            /* next_height */ 1,
            None,
            None,
        );
        assert!(
            two_result.is_none(),
            "witness.len() == 2 must return None so precheck defers; got {:?}",
            two_result
        );
    }

    /// Wave-4 input-side class-closure: a non-coinbase tx whose input
    /// uses the coinbase-prevout marker (`prev_txid == [0u8; 32]` AND
    /// `prev_vout == 0xffff_ffff`) is rejected by the slow path at
    /// `clients/rust/crates/rubin-consensus/src/utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`
    /// with `TxErrParse "coinbase prevout encoding forbidden in
    /// non-coinbase"`. Without the wave-4 defer guard, a below-floor
    /// tx with the coinbase marker would be misclassified as transient
    /// `Unavailable` instead of `Rejected` (terminal).
    #[test]
    fn rub166_precheck_defers_when_input_uses_coinbase_prevout_marker() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (mut parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        parsed.inputs[0].prev_txid = [0u8; 32];
        parsed.inputs[0].prev_vout = 0xffff_ffff;
        let result = fee_precheck_p2pk_input_value(
            &parsed,
            &state.utxos,
            /* next_height */ 1,
            None,
            None,
        );
        assert!(
            result.is_none(),
            "coinbase-prevout marker on non-coinbase input must return None so precheck defers; got {:?}",
            result
        );
    }

    /// Wave-5 input-side class-closure: an immature coinbase P2PK
    /// spend (`entry.created_by_coinbase && next_height -
    /// entry.creation_height < COINBASE_MATURITY`) is rejected by the
    /// slow path at
    /// `clients/rust/crates/rubin-consensus/src/utxo_basic.rs` `apply_non_coinbase_tx_basic_update_*`
    /// with `TxErrCoinbaseImmature` "coinbase immature". Without the
    /// wave-5 defer guard, a below-floor immature-coinbase spend
    /// would be misclassified as transient `Unavailable("mempool fee
    /// below rolling minimum")`, signalling caller to retry-with-
    /// higher-fee when the actual remedy is to wait for
    /// COINBASE_MATURITY blocks. Different caller action means real
    /// class-leak (P1) — the wave-4 class-leak-auditor severity
    /// rating of P2 ("transient -> transient") was wrong.
    #[test]
    fn rub166_precheck_defers_when_p2pk_input_is_immature_coinbase() {
        use rubin_consensus::constants::COINBASE_MATURITY;
        let (mut state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        // Mark the input UTXO as a coinbase output created at height 0.
        let outpoint = Outpoint {
            txid: parsed.inputs[0].prev_txid,
            vout: parsed.inputs[0].prev_vout,
        };
        let entry = state.utxos.get_mut(&outpoint).expect("test utxo present");
        entry.created_by_coinbase = true;
        entry.creation_height = 0;
        // next_height < COINBASE_MATURITY threshold => immature.
        let immature_height = COINBASE_MATURITY - 1;
        let result =
            fee_precheck_p2pk_input_value(&parsed, &state.utxos, immature_height, None, None);
        assert!(
            result.is_none(),
            "immature coinbase spend must return None so precheck defers; got {:?}",
            result
        );
        // At maturity threshold the defer no longer fires (sanity
        // pin for the negative branch).
        let mature_height = COINBASE_MATURITY;
        let result_mature =
            fee_precheck_p2pk_input_value(&parsed, &state.utxos, mature_height, None, None);
        assert!(
            result_mature.is_some(),
            "mature coinbase spend must NOT defer (precheck returns Some); got {:?}",
            result_mature
        );
    }

    /// Wave-15 panic-safety: defer when `entry.covenant_data.len()`
    /// is not exactly `MAX_P2PK_COVENANT_DATA` (33). Without this
    /// guard a corrupted on-disk UTXO entry (chainstate accepts
    /// arbitrary covenant_data bytes per chain_state_from_disk)
    /// would panic the admission loop on the `[0]` / `[1..33]`
    /// indexing. Mirrors slow-path covenant_data length check at
    /// `clients/rust/crates/rubin-consensus/src/spend_verify.rs` `validate_p2pk_spend_q`.
    /// Direct-helper exercise pins both `len < 33` and `len > 33`.
    #[test]
    fn rub166_precheck_defers_when_input_utxo_covenant_data_length_invalid() {
        let (mut state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        let outpoint = Outpoint {
            txid: parsed.inputs[0].prev_txid,
            vout: parsed.inputs[0].prev_vout,
        };
        // Branch 1: empty covenant_data (len 0 < 33) — panic-safety case.
        {
            let entry = state.utxos.get_mut(&outpoint).expect("test utxo present");
            entry.covenant_data = vec![];
            let result = fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, None, None);
            assert!(
                result.is_none(),
                "empty covenant_data must defer (None) — guards [0] index against panic; got {:?}",
                result
            );
        }
        // Branch 2: oversized covenant_data (len 64 > 33).
        {
            let entry = state.utxos.get_mut(&outpoint).expect("test utxo present");
            entry.covenant_data = vec![0u8; 64];
            let result = fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, None, None);
            assert!(
                result.is_none(),
                "oversized covenant_data must defer (None); got {:?}",
                result
            );
        }
        // Sanity (parity with Go test): valid 33-byte covenant_data with
        // SHA3 binding matching the parsed witness pubkey must NOT defer.
        {
            use sha3::{Digest, Sha3_256};
            let entry = state.utxos.get_mut(&outpoint).expect("test utxo present");
            let mut hasher = Sha3_256::new();
            hasher.update(&parsed.witness[0].pubkey);
            let pubkey_hash: [u8; 32] = hasher.finalize().into();
            let mut cov = vec![rubin_consensus::constants::SUITE_ID_ML_DSA_87];
            cov.extend_from_slice(&pubkey_hash);
            entry.covenant_data = cov;
            let result = fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, None, None);
            assert!(
                result.is_some(),
                "valid 33-byte covenant_data with matching SHA3 binding must NOT defer; got {:?}",
                result
            );
        }
    }

    /// Wave-16 sighash trailer: defer when `signature[-1]` is NOT one
    /// of the six valid sighash types accepted by `is_valid_sighash_type`
    /// (`SIGHASH_ALL/NONE/SINGLE × ANYONECANPAY`). Mirrors slow-path
    /// reject at `clients/rust/crates/rubin-consensus/src/sighash.rs` `is_valid_sighash_type`.
    /// Verifies BOTH (a) invalid trailer (e.g. 0x05) defers and
    /// (b) valid non-ALL trailer (e.g. SIGHASH_NONE = 0x02) is
    /// accepted by the precheck (closes wave-15 over-defer P1).
    #[test]
    fn rub166_precheck_defers_only_on_invalid_sighash_trailer() {
        use rubin_consensus::constants::{SIGHASH_ALL, SIGHASH_NONE};
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        // Branch 1: invalid trailer 0x05 (not in 6-element accept set) →
        // precheck defers so slow path returns terminal SighashType.
        let mut bad_trailer = parsed.clone();
        let sig_len = bad_trailer.witness[0].signature.len();
        bad_trailer.witness[0].signature[sig_len - 1] = 0x05;
        let bad_result = fee_precheck_p2pk_input_value(&bad_trailer, &state.utxos, 1, None, None);
        assert!(
            bad_result.is_none(),
            "invalid sighash trailer 0x05 must defer (None); got {:?}",
            bad_result
        );
        // Branch 2: valid SIGHASH_NONE trailer (0x02) — precheck must
        // ACCEPT (return Some), not defer. Wave-15 incorrectly hard-coded
        // SIGHASH_ALL only and over-deferred 5 of 6 valid trailers.
        let mut none_trailer = parsed.clone();
        let sig_len = none_trailer.witness[0].signature.len();
        none_trailer.witness[0].signature[sig_len - 1] = SIGHASH_NONE;
        let none_result = fee_precheck_p2pk_input_value(&none_trailer, &state.utxos, 1, None, None);
        // Sanity: SIGHASH_ALL trailer (default) must also accept.
        assert_ne!(SIGHASH_NONE, SIGHASH_ALL);
        // The valid non-ALL trailer should NOT defer due to sighash check
        // alone. Other guards (key-binding sha3) may still defer for the
        // synthetic fixture, which is fine — the test asserts the
        // sighash branch does NOT cause a spurious defer when value
        // 0x02 is also a valid trailer.
        // Specifically: if ANY guard fires, it should not be sighash.
        // We rely on the wave-15 valid signed fixture passing all other
        // guards, so SIGHASH_NONE must keep returning Some.
        assert!(
            none_result.is_some(),
            "valid SIGHASH_NONE trailer must NOT defer; got {:?}",
            none_result
        );
    }

    /// Wave-15 key-binding: defer when `SHA3(witness.pubkey)` does
    /// not match `entry.covenant_data[1..33]`. Mirrors slow-path key-
    /// binding check at
    /// `clients/rust/crates/rubin-consensus/src/spend_verify.rs` `validate_p2pk_spend_q`.
    /// Direct-helper exercise mutates the witness pubkey first byte so
    /// the SHA3 hash diverges from the covenant_data binding.
    #[test]
    fn rub166_precheck_defers_when_pubkey_key_binding_mismatch() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let (mut parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        // Mutate first byte of pubkey so SHA3(pubkey) differs from
        // covenant_data[1..33] (which still binds the original pubkey).
        parsed.witness[0].pubkey[0] ^= 0xFF;
        let result = fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, None, None);
        assert!(
            result.is_none(),
            "pubkey key-binding mismatch must defer (None); got {:?}",
            result
        );
    }

    /// Wave-14 input-side rotation guard: defer when the witness
    /// `suite_id` is not in the active `native_spend_suites(next_height)`
    /// set. Mirrors slow-path reject at
    /// `clients/rust/crates/rubin-consensus/src/spend_verify.rs` `validate_p2pk_spend_q`
    /// (`SigAlgInvalid`). Without this guard, a below-floor tx whose
    /// suite is consensus-rejected at spend time would be misclassified
    /// as transient `Unavailable` instead of terminal `Rejected`.
    /// Closes Copilot wave-17 P1 (`rub166_*` regression coverage gap).
    #[test]
    fn rub166_precheck_defers_when_input_suite_not_in_native_spend_set() {
        use rubin_consensus::constants::SUITE_ID_ML_DSA_87;
        use rubin_consensus::{NativeSuiteSet, RotationProvider};
        struct EmptySpendRotation;
        impl RotationProvider for EmptySpendRotation {
            fn native_create_suites(&self, _h: u64) -> NativeSuiteSet {
                NativeSuiteSet::new(&[SUITE_ID_ML_DSA_87])
            }
            fn native_spend_suites(&self, _h: u64) -> NativeSuiteSet {
                // Empty set rejects ALL spend suites.
                NativeSuiteSet::new(&[])
            }
        }
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        let rotation = EmptySpendRotation;
        let result = fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, Some(&rotation), None);
        assert!(
            result.is_none(),
            "rotation with empty native_spend_suites must defer (None); got {:?}",
            result
        );
        // Sanity (negative-branch pin): default rotation accepts
        // SUITE_ID_ML_DSA_87 in the spend set, so the same fixture with
        // `rotation=None` must NOT defer. Confirms the rotation
        // provider is the only difference exercised by this test.
        let baseline = fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, None, None);
        assert!(
            baseline.is_some(),
            "default rotation must NOT defer on signed valid P2PK fixture; got {:?}",
            baseline
        );
    }

    /// Wave-14 input-side registry guard: defer when the
    /// `SuiteRegistry::lookup(suite_id)` returns `None` (the active
    /// registry has no entry for the witness suite). Mirrors slow-path
    /// reject at
    /// `clients/rust/crates/rubin-consensus/src/spend_verify.rs` `validate_p2pk_spend_q`
    /// (`SigAlgInvalid`). Without this guard, a below-floor tx with an
    /// unregistered suite would be misclassified as transient
    /// `Unavailable` instead of terminal `Rejected`. Closes Copilot
    /// wave-17 P1 (`rub166_*` regression coverage gap).
    #[test]
    fn rub166_precheck_defers_when_input_suite_registry_lookup_misses() {
        use rubin_consensus::SuiteRegistry;
        use std::collections::BTreeMap;
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        // Empty registry: lookup of any suite_id (including
        // SUITE_ID_ML_DSA_87 carried by the signed fixture witness)
        // returns None, so the wave-14 registry guard fires.
        let empty_registry = SuiteRegistry::with_suites(BTreeMap::new());
        let result =
            fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, None, Some(&empty_registry));
        assert!(
            result.is_none(),
            "empty registry must defer (lookup miss returns None); got {:?}",
            result
        );
        // Sanity (negative-branch pin): default registry has
        // SUITE_ID_ML_DSA_87 with canonical params, so the same fixture
        // with `registry=None` must NOT defer. Confirms the registry
        // provider is the only difference exercised by this test.
        let baseline = fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, None, None);
        assert!(
            baseline.is_some(),
            "default registry must NOT defer on signed valid P2PK fixture; got {:?}",
            baseline
        );
    }

    /// Wave-14 input-side witness-length guard: defer when
    /// `witness.pubkey.len()` does not match `params.pubkey_len` OR
    /// `witness.signature.len()` does not match `params.sig_len + 1`
    /// (the +1 is the trailing sighash byte). Mirrors slow-path reject
    /// at `clients/rust/crates/rubin-consensus/src/spend_verify.rs` `validate_p2pk_spend_q`
    /// (`SigNoncanonical`). Without this guard a below-floor tx with a
    /// malformed witness (truncated pubkey or oversized signature)
    /// would be misclassified as transient `Unavailable` instead of
    /// terminal `Rejected`. Closes Copilot wave-19 P1.
    #[test]
    fn rub166_precheck_defers_when_witness_pubkey_or_signature_length_noncanonical() {
        let (state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        // Branch 1: pubkey too short (truncated by 1 byte).
        {
            let mut p = parsed.clone();
            p.witness[0].pubkey.pop().expect("witness pubkey non-empty");
            let result = fee_precheck_p2pk_input_value(&p, &state.utxos, 1, None, None);
            assert!(
                result.is_none(),
                "truncated pubkey must defer (None); got {:?}",
                result
            );
        }
        // Branch 2: signature too long (extended by 1 byte beyond sig_len+1).
        {
            let mut p = parsed.clone();
            p.witness[0].signature.push(0u8);
            let result = fee_precheck_p2pk_input_value(&p, &state.utxos, 1, None, None);
            assert!(
                result.is_none(),
                "oversized signature must defer (None); got {:?}",
                result
            );
        }
        // Sanity (negative-branch pin): canonical lengths must NOT defer.
        let baseline = fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, None, None);
        assert!(
            baseline.is_some(),
            "canonical pubkey/signature lengths must NOT defer; got {:?}",
            baseline
        );
    }

    /// Wave-15 panic-safety + suite-consistency guard (the
    /// `covenant_data[0] != witness.suite_id` defer in
    /// `fee_precheck_p2pk_input_value`): defer when the on-disk
    /// covenant_data prefix byte does not match the witness-declared
    /// suite_id, even when length is canonical 33 bytes. Mirrors
    /// slow-path `CovenantTypeInvalid` in
    /// `clients/rust/crates/rubin-consensus/src/spend_verify.rs`
    /// (`validate_p2pk_spend_q` covenant_data length+suite check).
    /// Combined-coverage rationale via the length test
    /// (`rub166_precheck_defers_when_input_utxo_covenant_data_length_invalid`)
    /// was insufficient — wave-20 pre-push-reviewer LAYER 4.1 caught
    /// the gap. Closes pre-push-reviewer wave-20 P1 finding #1.
    #[test]
    fn rub166_precheck_defers_when_input_utxo_covenant_data_suite_mismatch() {
        use rubin_consensus::constants::MAX_P2PK_COVENANT_DATA;
        let (mut state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let (parsed, _txid, _wtxid, _consumed) = parse_tx(&raw).expect("parse tx");
        let outpoint = Outpoint {
            txid: parsed.inputs[0].prev_txid,
            vout: parsed.inputs[0].prev_vout,
        };
        // Mutate covenant_data[0] to a value != witness.suite_id while
        // keeping length canonical (33 bytes) and SHA3-binding intact for
        // bytes [1..33]. The wave-15 covenant_data[0]==suite_id guard
        // in `fee_precheck_p2pk_input_value` must defer.
        let entry = state.utxos.get_mut(&outpoint).expect("test utxo present");
        // Witness suite is SUITE_ID_ML_DSA_87 (0x01); set covenant_data[0] = 0xFE.
        let mut new_cov = entry.covenant_data.clone();
        assert_eq!(
            new_cov.len(),
            MAX_P2PK_COVENANT_DATA as usize,
            "fixture covenant_data must be canonical length"
        );
        new_cov[0] = 0xFE;
        entry.covenant_data = new_cov;
        let result = fee_precheck_p2pk_input_value(&parsed, &state.utxos, 1, None, None);
        assert!(
            result.is_none(),
            "covenant_data[0] != witness.suite_id must defer (None); got {:?}",
            result
        );
    }

    /// Wave-22 cache identity + canonical-manifest pin (Copilot
    /// wave-21 P2 #1+#2 follow-up). Asserts: (a) repeated calls to
    /// `cached_default_registry()` return the SAME pointer (no
    /// rebuild), (b) cached value matches `is_canonical_default_live_manifest`,
    /// (c) cached registry contains `SUITE_ID_ML_DSA_87` lookup with
    /// canonical params. Future change that swaps the cache to a
    /// non-canonical builder fails this test.
    #[test]
    fn rub166_cached_default_registry_identity_and_canonical_manifest() {
        use rubin_consensus::constants::SUITE_ID_ML_DSA_87;
        let r1 = super::cached_default_registry();
        let r2 = super::cached_default_registry();
        assert!(
            std::ptr::eq(r1, r2),
            "cached_default_registry must return identical pointer (no rebuild per call)"
        );
        assert!(
            r1.is_canonical_default_live_manifest(),
            "cached_default_registry must satisfy IsCanonicalDefaultLiveManifest"
        );
        let params = r1
            .lookup(SUITE_ID_ML_DSA_87)
            .expect("ML-DSA-87 must be present in cached default registry");
        assert_eq!(params.suite_id, SUITE_ID_ML_DSA_87);
        assert_eq!(params.alg_name, "ML-DSA-87");
    }

    /// Wave-22 cache identity for native_spend / native_create sets.
    /// Same rationale as registry-identity test: ensure the cached
    /// `NativeSuiteSet` is reused, not rebuilt per call.
    #[test]
    fn rub166_cached_default_native_sets_identity() {
        use rubin_consensus::constants::SUITE_ID_ML_DSA_87;
        let s1 = super::cached_default_native_spend_set();
        let s2 = super::cached_default_native_spend_set();
        assert!(std::ptr::eq(s1, s2), "spend set cache identity");
        assert!(s1.contains(SUITE_ID_ML_DSA_87));

        let c1 = super::cached_default_native_create_set();
        let c2 = super::cached_default_native_create_set();
        assert!(std::ptr::eq(c1, c2), "create set cache identity");
        assert!(c1.contains(SUITE_ID_ML_DSA_87));
    }

    #[test]
    fn rub166_relay_metadata_below_floor_p2pk_still_returns_unavailable_matching_admit() {
        let (mut state, raw, _conflict) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        state.has_tip = true;
        state.height = 1;
        let cfg = TxPoolConfig::default();
        let err = relay_metadata(&raw, &state, None, devnet_genesis_chain_id(), &cfg)
            .expect_err("plain P2PK below the rolling floor must reject on the relay path too");
        assert_eq!(
            err.kind,
            TxPoolAdmitErrorKind::Unavailable,
            "relay-side rolling-floor failure must remain Unavailable to match admit"
        );
        assert!(
            err.message.contains("mempool fee below rolling minimum"),
            "relay-side message format must match admit; got: {}",
            err.message
        );
    }

    // -------------------------------------------------------------------
    // RUB-174: source-aware admission API tests
    //
    // These tests cover the `add_tx_with_source` canonical entry mirroring
    // Go's `addTxWithSource` (clients/go/node/mempool.go), and verify that
    // the legacy `admit()` / `admit_with_metadata()` wrappers preserve
    // backward-compat by defaulting `TxSource::Local`. The parity-invariant
    // hostile case (`source_does_not_affect_admission_ordering`) asserts
    // that admission is source-blind — recording source is observability
    // metadata only and must not influence eviction or mining selection.
    // -------------------------------------------------------------------

    /// `add_tx_with_source(_, TxSource::Local)` admits successfully and
    /// records `Local` on the resulting `TxPoolEntry.source`. Mirrors Go
    /// `Mempool.AddTx` → `addTxWithSource(_, mempoolTxSourceLocal)`.
    #[test]
    fn add_tx_with_source_records_local_provenance() {
        let (state, admitted_raw, _block_raw) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let mut pool = TxPool::new();
        let (txid, _) = pool
            .add_tx_with_source(
                &admitted_raw,
                &state,
                None,
                devnet_genesis_chain_id(),
                TxSource::Local,
            )
            .expect("add_tx_with_source Local");
        let entry = pool
            .txs
            .get(&txid)
            .expect("admitted entry must exist in pool");
        assert_eq!(
            entry.source,
            TxSource::Local,
            "Local source must be recorded on entry"
        );
        assert!(pool.owner_claim_is_finalized(&txid));
        assert_eq!(pool.owner_row_count(), entry.inputs.len());
    }

    /// `add_tx_with_source(_, TxSource::Remote)` admits successfully and
    /// records `Remote` on the resulting `TxPoolEntry.source`. Mirrors Go
    /// `Mempool.AddRemoteTx` → `addTxWithSource(_, mempoolTxSourceRemote)`.
    #[test]
    fn add_tx_with_source_records_remote_provenance() {
        let (state, admitted_raw, _block_raw) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let mut pool = TxPool::new();
        let (txid, _) = pool
            .add_tx_with_source(
                &admitted_raw,
                &state,
                None,
                devnet_genesis_chain_id(),
                TxSource::Remote,
            )
            .expect("add_tx_with_source Remote");
        let entry = pool
            .txs
            .get(&txid)
            .expect("admitted entry must exist in pool");
        assert_eq!(
            entry.source,
            TxSource::Remote,
            "Remote source must be recorded on entry"
        );
        assert!(pool.owner_claim_is_finalized(&txid));
        assert_eq!(pool.owner_row_count(), entry.inputs.len());
    }

    /// `add_tx_with_source(_, TxSource::Reorg)` admits successfully and
    /// records `Reorg` on the resulting `TxPoolEntry.source`. Mirrors Go
    /// `Mempool.AddReorgTx` → `addTxWithSource(_, mempoolTxSourceReorg)`.
    #[test]
    fn add_tx_with_source_records_reorg_provenance() {
        let (state, admitted_raw, _block_raw) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let mut pool = TxPool::new();
        let (txid, _) = pool
            .add_tx_with_source(
                &admitted_raw,
                &state,
                None,
                devnet_genesis_chain_id(),
                TxSource::Reorg,
            )
            .expect("add_tx_with_source Reorg");
        let entry = pool
            .txs
            .get(&txid)
            .expect("admitted entry must exist in pool");
        assert_eq!(
            entry.source,
            TxSource::Reorg,
            "Reorg source must be recorded on entry"
        );
        assert!(pool.owner_claim_is_finalized(&txid));
        assert_eq!(pool.owner_row_count(), entry.inputs.len());
    }

    /// Backward-compat: `admit()` defaults the recorded source to `Local`
    /// (Go parity: `AddTx` is the legacy entry that maps to
    /// `mempoolTxSourceLocal`). Existing producers calling `admit()`
    /// continue to work and their admissions are recorded as `Local`.
    #[test]
    fn admit_records_local_source_for_backward_compat() {
        let (state, admitted_raw, _block_raw) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        let mut pool = TxPool::new();
        let txid = pool
            .admit(&admitted_raw, &state, None, devnet_genesis_chain_id())
            .expect("legacy admit");
        let entry = pool
            .txs
            .get(&txid)
            .expect("admitted entry must exist in pool");
        assert_eq!(
            entry.source,
            TxSource::Local,
            "legacy admit() must record Local source for backward compat"
        );
    }

    /// Hostile case: source variant must NOT influence admission
    /// ordering, eviction priority, or mining selection. Mirrors Go
    /// invariant where `mempoolEntry.source` is observability metadata
    /// only and is never consulted by `validateCapacityAdmissionLocked`,
    /// `capacityEvictionPlanLocked`, or block-template selection.
    ///
    /// Coverage axes:
    ///   1. Admission produces source-independent txid (same tx bytes
    ///      under different sources → identical txid; recorded source
    ///      varies).
    ///   2. Selection ordering across MULTIPLE entries with equal
    ///      (fee, weight) but different txids must be byte-identical
    ///      between two pools that differ ONLY in source assignment
    ///      (Pool A: ent1=Local + ent2=Reorg; Pool B: ent1=Reorg +
    ///      ent2=Local). This catches regressions where source
    ///      accidentally enters the comparator (Copilot-2026-05-04 P2:
    ///      single-entry selection is trivially identical, so the
    ///      multi-entry leg is the load-bearing assertion).
    ///
    /// Test design note: `signed_conflicting_p2pk_state_and_txs`
    /// produces fresh ML-DSA signatures per call, so the admit-path
    /// leg admits the SAME `admitted_raw` to two pools to isolate the
    /// source-difference variable. The selection-ordering leg uses
    /// `insert_entry` to side-step admission's input-conflict /
    /// fee-floor / signature-verify pipeline (which is source-blind by
    /// construction at the admission API level — see leg 1) and
    /// directly exercise the selection-time comparator
    /// (`compare_entries_for_mining`, sorting `self.txs` in
    /// `select_transactions`) with equal-priority distinct-txid entries
    /// that production admission would not normally produce together
    /// (because they would double-spend the same input). The injected
    /// Worst-heap source-blindness is exercised by the capacity tests
    /// elsewhere in this module; this leg focuses on the selection
    /// comparator.
    #[test]
    fn source_does_not_affect_admission_ordering() {
        // Leg 1: admit-path source-independence (same tx bytes, two pools,
        // different recorded source on entry, identical txid).
        let (state, admitted_raw, _block_raw) = signed_conflicting_p2pk_state_and_txs(7700, 10, 9);

        let mut pool_local = TxPool::new();
        let (txid_local, _) = pool_local
            .add_tx_with_source(
                &admitted_raw,
                &state,
                None,
                devnet_genesis_chain_id(),
                TxSource::Local,
            )
            .expect("local admit");

        let mut pool_reorg = TxPool::new();
        let (txid_reorg, _) = pool_reorg
            .add_tx_with_source(
                &admitted_raw,
                &state,
                None,
                devnet_genesis_chain_id(),
                TxSource::Reorg,
            )
            .expect("reorg admit");

        assert_eq!(
            txid_local, txid_reorg,
            "txid must be source-independent (computed from tx bytes only)"
        );

        assert_eq!(
            pool_local.txs.get(&txid_local).expect("local entry").source,
            TxSource::Local
        );
        assert_eq!(
            pool_reorg.txs.get(&txid_reorg).expect("reorg entry").source,
            TxSource::Reorg
        );

        // Leg 2: multi-entry selection-ordering source-blindness
        // (Pool A: ent1=Local + ent2=Reorg; Pool B: ent1=Reorg + ent2=Local).
        // Equal (fee, weight) but distinct txids → comparator must produce
        // identical order between pools that differ ONLY in source labels.
        // Use insert_entry to bypass the admission pipeline (signature
        // verify, fee floor, input-conflict check) and directly exercise
        // the selection-time comparator (compare_entries_for_mining
        // sorting self.txs in select_transactions). The worst-heap path
        // is exercised separately by the capacity tests above; this leg
        // focuses on the selection comparator's source-blindness.
        let txid_aaa = [0xaa; 32];
        let txid_bbb = [0xbb; 32];
        let make_entry = |raw: Vec<u8>, source: TxSource| TxPoolEntry {
            raw,
            inputs: Vec::new(),
            fee: 100,
            weight: 1,
            size: 1,
            source,
        };

        let mut pool_a_b = TxPool::new();
        pool_a_b.insert_entry(txid_aaa, make_entry(vec![0x01], TxSource::Local));
        pool_a_b.insert_entry(txid_bbb, make_entry(vec![0x02], TxSource::Reorg));

        let mut pool_b_a = TxPool::new();
        pool_b_a.insert_entry(txid_aaa, make_entry(vec![0x01], TxSource::Reorg));
        pool_b_a.insert_entry(txid_bbb, make_entry(vec![0x02], TxSource::Local));

        let selected_a_b = pool_a_b.select_transactions(10, usize::MAX);
        let selected_b_a = pool_b_a.select_transactions(10, usize::MAX);
        assert_eq!(
            selected_a_b.len(),
            2,
            "both injected entries must be selected (size budget non-binding)"
        );
        assert_eq!(
            selected_a_b, selected_b_a,
            "multi-entry selection order must be byte-identical between pools \
             that differ ONLY in source assignment (comparator must be source-blind)"
        );
    }
}
