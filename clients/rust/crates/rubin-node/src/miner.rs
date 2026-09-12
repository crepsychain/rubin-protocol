use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use rubin_consensus::constants::{
    COV_TYPE_DA_COMMIT, MAX_BLOCK_WEIGHT, MAX_DA_BATCHES_PER_BLOCK, MAX_DA_BYTES_PER_BLOCK,
    MAX_DA_CHUNK_COUNT, MAX_FUTURE_DRIFT, POW_LIMIT,
};
use rubin_consensus::merkle::{witness_commitment_hash, witness_merkle_root_wtxids};
use rubin_consensus::{
    apply_non_coinbase_tx_basic_update_with_mtp_and_suite_context as apply_basic_non_coinbase_update,
    encode_compact_size, merkle_root_txids, parse_tx, pow_check, tx_weight_and_stats_public,
    Outpoint, Tx, UtxoEntry,
};
use sha3::{Digest, Sha3_256};

use crate::coinbase::{
    build_coinbase_tx, default_mine_address, normalize_mine_address, parse_mine_address,
};
use crate::da_relay::{CompleteDaSetCandidate, CompleteDaSetProvider};
use crate::sync::{target_schedule_error_message, SyncEngine};
use crate::txpool::{
    apply_policy, TxPool, TxPoolConfig, DEFAULT_MEMPOOL_MIN_FEE_RATE, DEFAULT_MIN_DA_FEE_RATE,
};

fn current_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Clone, Debug)]
pub struct MinerConfig {
    pub timestamp_source: fn() -> u64,
    pub max_tx_per_block: usize,
    pub target: [u8; 32],
    pub mine_address: Vec<u8>,
    /// Master switch for the whole DA/anchor anti-abuse miner-template
    /// policy package. When false,
    /// `policy_reject_non_coinbase_anchor_outputs` is ignored. This is
    /// policy-only and does not change consensus validity.
    pub policy_da_anchor_anti_abuse: bool,
    /// Sub-flag for non-coinbase CORE_ANCHOR rejection. It is effective
    /// only when `policy_da_anchor_anti_abuse` is true.
    pub policy_reject_non_coinbase_anchor_outputs: bool,
    pub policy_max_da_bytes_per_block: u64,
    pub policy_da_surcharge_per_byte: u64,
    /// Stage C `current_mempool_min_fee_rate` input forwarded to the DA
    /// fee policy when `policy_da_anchor_anti_abuse` is on. Defaults to
    /// `DEFAULT_MEMPOOL_MIN_FEE_RATE` (mirrors Go's documented pattern
    /// for callers without a live rolling-floor source).
    pub policy_current_mempool_min_fee_rate: u64,
    /// Stage C `min_da_fee_rate` input forwarded to the DA fee policy
    /// when `policy_da_anchor_anti_abuse` is on. Defaults to
    /// `DEFAULT_MIN_DA_FEE_RATE`, kept separate from
    /// `DEFAULT_MEMPOOL_MIN_FEE_RATE` so a future change to the relay
    /// floor cannot silently change the DA floor.
    pub policy_min_da_fee_rate: u64,
    /// Mirror of Go `MinerConfig.PolicyRejectSimplicityPreActivation`:
    /// when true, the miner excludes CORE_SIMPLICITY (0x0106)
    /// create/spend candidates until the rotation provider reports the
    /// Simplicity deployment active at the candidate height. NOT nested
    /// under `policy_da_anchor_anti_abuse` (Go pins this with
    /// `TestMinerSimplicityPolicyStillRunsWhenDaAnchorMasterOff`).
    /// Policy-only; consensus validity is unaffected.
    pub policy_reject_simplicity_pre_activation: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinedBlock {
    pub height: u64,
    pub hash: [u8; 32],
    pub timestamp: u64,
    pub nonce: u64,
    pub tx_count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct MinedCandidate {
    raw: Vec<u8>,
    tx: Tx,
    txid: [u8; 32],
    wtxid: [u8; 32],
    weight: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct CompleteDaSetMiningCandidate {
    txs: Vec<MinedCandidate>,
    da_bytes: u64,
}

#[allow(dead_code)]
pub(crate) struct CompleteDaSetGroupProjection<'a> {
    pub(crate) selected_nonces: &'a HashSet<u64>,
    pub(crate) selected_inputs: &'a HashSet<Outpoint>,
    pub(crate) next_height: u64,
    pub(crate) block_mtp: u64,
    pub(crate) selected_weight: u64,
    pub(crate) remaining_weight: u64,
    pub(crate) policy_da_included: u64,
}

/// RUB-1186: checkpoints before their next mutation boundary. `PreBootstrap`
/// precedes genesis bootstrap; `NonceQuantum` and `PreApply` may follow a
/// completed bootstrap. Named so a test can isolate one checkpoint
/// deterministically; the production poll ignores the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MineCheckpoint {
    PreBootstrap,
    NonceQuantum,
    PreApply,
}

/// RUB-1186: a borrowed, thread-free lifecycle observer over a caller-owned
/// cancellation source, carrying the latch its caller classifies from.
///
/// Postconditions the code cannot show on its own:
/// - `check` latches ONLY on an ACTUAL checkpoint observation. Callers must
///   classify cancellation from `observed()`, never by reading the source
///   again after the mine returned: a post-return read would remap a mining
///   failure that formed before any observation into a cancellation.
/// - the latch is monotone; once set it stays set for the rest of the mine.
/// - it is local control flow, not a registry: it owns no thread, no channel
///   and no state beyond the borrow and the latch bit.
pub(crate) struct MineCancelObserver<'a> {
    poll: &'a dyn Fn(MineCheckpoint) -> bool,
    observed: bool,
}

impl<'a> MineCancelObserver<'a> {
    pub(crate) fn new(poll: &'a dyn Fn(MineCheckpoint) -> bool) -> Self {
        Self {
            poll,
            observed: false,
        }
    }

    fn check(&mut self, at: MineCheckpoint) -> bool {
        if !self.observed && (self.poll)(at) {
            self.observed = true;
        }
        self.observed
    }

    pub(crate) fn observed(&self) -> bool {
        self.observed
    }
}

/// The `Err` a cancelled mine returns. It is never surfaced to a client: the
/// caller answers a latched observation with its own route-shaped 503.
pub(crate) const MINE_CANCELLED: &str = "mining cancelled by rpc shutdown";

/// Nonces tried between two lifecycle checkpoints in `mine_header_nonce`. The
/// search itself is unchanged — nonces are still tried in order from 0 and the
/// first one satisfying `target` still wins — so the header/nonce this returns
/// for a given prefix and target is byte-identical to the baseline unbounded
/// loop. The quantum only bounds how long the search runs between two checks.
const MINE_NONCE_QUANTUM: u64 = 1024;

pub struct Miner<'a> {
    sync: &'a mut SyncEngine,
    tx_pool: Option<&'a mut TxPool>,
    complete_da_set_provider: Option<&'a dyn CompleteDaSetProvider>,
    cfg: MinerConfig,
}

impl Default for MinerConfig {
    fn default() -> Self {
        Self {
            timestamp_source: current_unix,
            max_tx_per_block: 1024,
            target: POW_LIMIT,
            mine_address: default_mine_address(),
            policy_da_anchor_anti_abuse: true,
            policy_reject_non_coinbase_anchor_outputs: true,
            policy_max_da_bytes_per_block: MAX_DA_BYTES_PER_BLOCK / 4,
            policy_da_surcharge_per_byte: 0,
            policy_current_mempool_min_fee_rate: DEFAULT_MEMPOOL_MIN_FEE_RATE,
            policy_min_da_fee_rate: DEFAULT_MIN_DA_FEE_RATE,
            policy_reject_simplicity_pre_activation: true,
        }
    }
}

impl<'a> Miner<'a> {
    pub fn new(
        sync: &'a mut SyncEngine,
        tx_pool: Option<&'a mut TxPool>,
        mut cfg: MinerConfig,
    ) -> Result<Self, String> {
        cfg.mine_address = normalize_mine_address(&cfg.mine_address)?;
        if cfg.max_tx_per_block == 0 {
            cfg.max_tx_per_block = 1024;
        }
        let complete_da_set_provider = None;
        #[rustfmt::skip]
        let miner = Self { sync, tx_pool, complete_da_set_provider, cfg };
        Ok(miner)
    }

    pub fn set_complete_da_set_provider(&mut self, provider: &'a dyn CompleteDaSetProvider) {
        self.complete_da_set_provider = Some(provider);
    }

    pub fn mine_n(&mut self, blocks: usize, txs: &[Vec<u8>]) -> Result<Vec<MinedBlock>, String> {
        let mut out = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            out.push(self.mine_one(txs)?);
        }
        Ok(out)
    }

    pub fn mine_one(&mut self, txs: &[Vec<u8>]) -> Result<MinedBlock, String> {
        // RUB-1186: ordinary mining for every existing caller (`mine_n`, the
        // `--mine-blocks` bin path, `da_txgen`, benches, tests). A
        // never-observing observer keeps the baseline behaviour and the exact
        // public signature; only devnet RPC `/mine_next` passes a live one.
        let never = |_: MineCheckpoint| false;
        self.mine_one_observed(txs, &mut MineCancelObserver::new(&never))
    }

    /// RUB-1186: `mine_one` with the three lifecycle checkpoints devnet RPC
    /// `/mine_next` needs to answer a graceful shutdown without mutating the
    /// chain. Mirrors the merged Go reference observables — `Miner.MineOne`
    /// entry (`clients/go/node/miner.go:197`), the `mineHeaderNonce` quantum
    /// polls (`clients/go/node/miner.go:853`) and the `executeMineOne`
    /// pre-apply check (`clients/go/node/miner_mine_helpers.go:32`) — not its
    /// implementation shape.
    ///
    /// Postcondition, in the Go reference's own wording (`clients/go/node/miner.go:188-190`):
    /// on `Err` with `MineCancelObserver::observed`, there was NO WRITE PAST
    /// THE CHECK THAT LATCHED. It is not "nothing was mutated": a
    /// `PreBootstrap` latch leaves an empty canonical store empty, but a
    /// `NonceQuantum` or `PreApply` latch means the earlier pre-bootstrap check
    /// was clean, so the idempotent genesis bootstrap below already ran and its
    /// effect stands. What every latch does guarantee is that the mine stops at
    /// that checkpoint: it never reaches `apply_block` and never rolls back
    /// what an earlier stage already did. Once `apply_block` is entered the
    /// result stands and is never rolled back or remapped by a later
    /// cancellation.
    pub(crate) fn mine_one_observed(
        &mut self,
        txs: &[Vec<u8>],
        cancel: &mut MineCancelObserver<'_>,
    ) -> Result<MinedBlock, String> {
        // Checkpoint 1: before the bootstrap MUTATION below, so an observing
        // check leaves an empty canonical store empty.
        if cancel.check(MineCheckpoint::PreBootstrap) {
            return Err(MINE_CANCELLED.to_string());
        }
        // Ensure the chain is bootstrapped at the canonical published genesis
        // before any candidate is built, mirroring the Go reference's
        // `bootstrapGenesisIfNeeded` call position in `Miner.MineOne`
        // (`clients/go/node/miner.go`). The height-0 genesis-identity guard
        // rejects miner-synthesized height-0 blocks under a devnet ChainID, so
        // empty-chain mining must start from the published bytes. The call is
        // idempotent: a no-op once the chain has a tip and a no-op for ChainIDs
        // without a published canonical genesis.
        self.sync.bootstrap_canonical_genesis_if_empty()?;
        let next_height = if self.sync.chain_state.has_tip {
            self.sync
                .chain_state
                .height
                .checked_add(1)
                .ok_or_else(|| "height overflow".to_string())?
        } else {
            0
        };
        let prev_hash = if self.sync.chain_state.has_tip {
            self.sync.chain_state.tip_hash
        } else {
            [0u8; 32]
        };
        // RUB-655: resolved BEFORE the MTP window, and used for BOTH header
        // construction and the nonce search below.
        let target = self.template_target(prev_hash, next_height)?;
        let remaining_weight = self.remaining_weight_budget(next_height)?;
        let candidates = self.candidate_transactions(txs);
        let prev_timestamps = self.sync.prev_timestamps_for_next_block()?;
        let timestamp = choose_valid_timestamp(
            next_height,
            prev_timestamps.as_deref().unwrap_or(&[]),
            (self.cfg.timestamp_source)(),
        );
        let block_mtp = prev_timestamps
            .as_deref()
            .filter(|timestamps| !timestamps.is_empty())
            .map_or(timestamp, |timestamps| mtp_median(next_height, timestamps));
        #[rustfmt::skip]
        let parsed = self.select_candidate_transactions(candidates, next_height, remaining_weight, block_mtp)?;
        let witness_commitment = build_witness_commitment(&parsed)?;
        let coinbase = build_coinbase_tx(
            next_height,
            self.sync.chain_state.already_generated,
            &self.cfg.mine_address,
            witness_commitment,
        )?;
        let (_, coinbase_txid, _, consumed) = parse_tx(&coinbase).map_err(|e| e.to_string())?;
        if consumed != coinbase.len() {
            return Err("coinbase serialization is non-canonical".to_string());
        }

        let mut txids = Vec::with_capacity(1 + parsed.len());
        txids.push(coinbase_txid);
        for candidate in &parsed {
            txids.push(candidate.txid);
        }
        let merkle_root = merkle_root_txids(&txids).map_err(|e| e.to_string())?;
        let block_without_nonce = make_header_prefix(prev_hash, merkle_root, timestamp, target);
        // Checkpoint 2 lives inside the search, at every quantum entry.
        let (header_bytes, nonce) = mine_header_nonce(&block_without_nonce, target, cancel)?;
        let block_bytes = assemble_block_bytes(&header_bytes, &coinbase, &parsed);
        // Checkpoint 3: the last instant before the canonical apply MUTATION.
        // Nothing observes cancellation inside or after `apply_block` — once
        // entered, apply/postcommit truth is final.
        if cancel.check(MineCheckpoint::PreApply) {
            return Err(MINE_CANCELLED.to_string());
        }
        let summary = self
            .sync
            .apply_block(&block_bytes, prev_timestamps.as_deref())?;
        self.evict_confirmed_from_pool(&parsed);
        Ok(MinedBlock {
            height: summary.block_height,
            hash: summary.block_hash,
            timestamp,
            nonce,
            tx_count: 1 + parsed.len(),
        })
    }

    /// The target this post-genesis template commits to, for BOTH header
    /// construction and the nonce search — the miner must never mine against a
    /// target the apply path will then reject. `prev_hash` is the canonical tip
    /// `mine_one` snapshotted (this template's authoritative selected parent)
    /// and `next_height` the miner's own candidate height. On a stock Phase-0
    /// devnet chain the derived target REPLACES the static `MinerConfig.target`,
    /// so that field carries no authority there; everywhere else it is returned
    /// unchanged, including `next_height == 0` (empty chain, no schedule target).
    /// Mirror of Go `templateTarget`, whose nil-`m.sync` arm has no counterpart
    /// here (the Rust `Miner` holds `&mut SyncEngine`).
    fn template_target(&self, prev_hash: [u8; 32], next_height: u64) -> Result<[u8; 32], String> {
        if next_height == 0 || !self.sync.stock_devnet_target_schedule() {
            return Ok(self.cfg.target);
        }
        self.sync
            .expected_target_for_candidate(prev_hash, next_height)
            .map_err(target_schedule_error_message)
    }

    /// Flat candidate selection, mirroring Go `candidateTransactions`: individual DA
    /// commit/chunk txs (tx_kind 0x01/0x02) are skipped unconditionally from both the
    /// explicit `txs` and the txpool snapshot — DA enters a block only through the
    /// complete-set provider group, never as a flat candidate.
    fn candidate_transactions(&self, txs: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let max_selected = self.cfg.max_tx_per_block.saturating_sub(1);
        if max_selected == 0 {
            return Vec::new();
        }
        if !txs.is_empty() {
            return pick_flat_candidate_raw(txs, max_selected);
        }
        let Some(pool) = self.tx_pool.as_deref() else {
            return Vec::new();
        };
        pool.select_transactions_with_filter(max_selected, MAX_BLOCK_WEIGHT as usize, |raw| {
            is_mining_da_tx_raw(raw)
        })
    }

    fn evict_confirmed_from_pool(&mut self, parsed: &[MinedCandidate]) {
        let Some(pool) = self.tx_pool.as_deref_mut() else {
            return;
        };
        let txids: Vec<[u8; 32]> = parsed.iter().map(|candidate| candidate.txid).collect();
        pool.evict_txids(&txids);
        let block_txs: Vec<Tx> = parsed
            .iter()
            .map(|candidate| candidate.tx.clone())
            .collect();
        pool.remove_conflicting_inputs(&block_txs);
    }

    fn remaining_weight_budget(&self, next_height: u64) -> Result<u64, String> {
        let coinbase = build_coinbase_tx(
            next_height,
            self.sync.chain_state.already_generated,
            &self.cfg.mine_address,
            [0u8; 32],
        )?;
        let weight = canonical_tx_weight(&coinbase, "coinbase serialization is non-canonical")?;
        MAX_BLOCK_WEIGHT
            .checked_sub(weight)
            .ok_or_else(|| "coinbase exceeds block weight budget".to_string())
    }

    fn select_candidate_transactions(
        &self,
        candidate_txs: Vec<Vec<u8>>,
        next_height: u64,
        remaining_weight: u64,
        block_mtp: u64,
    ) -> Result<Vec<MinedCandidate>, String> {
        let max_selected = self.cfg.max_tx_per_block.saturating_sub(1);
        let mut parsed = Vec::with_capacity(candidate_txs.len().min(max_selected));
        let mut selected_weight = 0u64;
        let mut policy_da_included = 0u64;
        let mut selected_da_batches = 0u64;
        let mut selected_da_ids = HashSet::new();
        let mut selected_nonces = HashSet::new();
        let mut selected_inputs = HashSet::new();
        let provider_enabled = self.complete_da_set_provider.is_some();
        for raw in candidate_txs {
            if parsed.len() >= max_selected {
                break;
            }
            let candidate = parse_mining_candidate(&raw)?;
            if provider_enabled && matches!(candidate.tx.tx_kind, 0x01 | 0x02) {
                continue;
            }
            let (reject, next_da_included) =
                self.reject_candidate(&candidate.tx, next_height, policy_da_included)?;
            if reject {
                continue;
            }
            let candidate_slice = std::slice::from_ref(&candidate);
            #[rustfmt::skip]
            let Some(candidate_inputs) = collect_complete_da_set_group_inputs(candidate_slice, &selected_nonces, &selected_inputs) else { continue; };
            if candidate.weight > remaining_weight.saturating_sub(selected_weight) {
                continue;
            }
            selected_weight = selected_weight
                .checked_add(candidate.weight)
                .ok_or_else(|| "selected transaction weight overflow".to_string())?;
            policy_da_included = next_da_included;
            selected_nonces.insert(candidate.tx.tx_nonce);
            selected_inputs.extend(candidate_inputs);
            parsed.push(candidate);
        }
        let Some(provider) = self.complete_da_set_provider else {
            return Ok(parsed);
        };
        if max_selected == 0 || parsed.len() >= max_selected || selected_weight >= remaining_weight
        {
            return Ok(parsed);
        }
        let mut provider_budget = MAX_DA_BYTES_PER_BLOCK;
        if self.cfg.policy_da_anchor_anti_abuse
            && self.cfg.policy_max_da_bytes_per_block > 0
            && self.cfg.policy_max_da_bytes_per_block < provider_budget
        {
            provider_budget = self.cfg.policy_max_da_bytes_per_block;
        }
        let mut provider_da_included = 0u64;
        for set in provider.complete_da_set_candidates(provider_budget) {
            if parsed.len() >= max_selected || selected_da_batches >= MAX_DA_BATCHES_PER_BLOCK {
                break;
            }
            if selected_da_ids.contains(&set.da_id) {
                continue;
            }
            let group_len = set.chunks.len().saturating_add(1);
            if group_len > max_selected - parsed.len() {
                continue;
            }
            // RUB-423 decision: a parse error here ABORTS template construction
            // (mirror of Go miner.go selectCandidateTransactions returning the error),
            // it is not skipped — the complete-DA-set provider is the node's own
            // trusted source, so a non-canonical candidate is an internal invariant
            // violation handled fail-closed, not a provider-induced template DoS. An
            // invalid *shape* with no parse error (Ok(None)) is still skipped below.
            let group = parse_complete_da_set_candidate(&set)?;
            #[rustfmt::skip]
            let Some(group) = group else { continue; };
            #[rustfmt::skip]
            let next_da = updated_policy_da_bytes(provider_da_included, group.da_bytes, provider_budget);
            #[rustfmt::skip]
            let Some(next_provider_da_included) = next_da else { continue; };
            #[rustfmt::skip]
            let projection = CompleteDaSetGroupProjection { selected_nonces: &selected_nonces, selected_inputs: &selected_inputs, next_height, block_mtp, selected_weight, remaining_weight, policy_da_included };
            let projected = self.project_complete_da_set_group(&group.txs, projection)?;
            #[rustfmt::skip]
            let Some((group_weight, next_da_included)) = projected else { continue; };
            selected_weight = selected_weight
                .checked_add(group_weight)
                .ok_or_else(|| "selected transaction weight overflow".to_string())?;
            policy_da_included = next_da_included;
            provider_da_included = next_provider_da_included;
            selected_da_batches += 1;
            selected_da_ids.insert(set.da_id);
            for candidate in group.txs {
                selected_nonces.insert(candidate.tx.tx_nonce);
                selected_inputs.extend(candidate.tx.inputs.iter().map(|input| Outpoint {
                    txid: input.prev_txid,
                    vout: input.prev_vout,
                }));
                parsed.push(candidate);
            }
        }
        Ok(parsed)
    }

    fn reject_candidate(
        &self,
        tx: &Tx,
        next_height: u64,
        policy_da_included: u64,
    ) -> Result<(bool, u64), String> {
        let utxos = &self.sync.chain_state.utxos;
        self.reject_candidate_with_utxos(tx, utxos, next_height, policy_da_included)
    }

    fn reject_candidate_with_utxos(
        &self,
        tx: &Tx,
        utxos: &HashMap<Outpoint, UtxoEntry>,
        next_height: u64,
        policy_da_included: u64,
    ) -> Result<(bool, u64), String> {
        let policy_cfg = TxPoolConfig {
            policy_da_surcharge_per_byte: if self.cfg.policy_da_anchor_anti_abuse {
                self.cfg.policy_da_surcharge_per_byte
            } else {
                0
            },
            // Anchor rejection is nested under the DA/anchor master switch
            // by policy contract; this does not change consensus validity.
            policy_reject_non_coinbase_anchor_outputs: self.cfg.policy_da_anchor_anti_abuse
                && self.cfg.policy_reject_non_coinbase_anchor_outputs,
            // NOT nested under the DA/anchor master switch: mirrors Go,
            // where the Simplicity pre-activation guardrail is its own
            // top-level miner policy flag (rotation comes from the sync
            // engine's suite context, like Go's `m.sync.cfg.RotationProvider`).
            policy_reject_simplicity_pre_activation: self
                .cfg
                .policy_reject_simplicity_pre_activation,
            suite_context: self.sync.cfg.suite_context.clone(),
            policy_current_mempool_min_fee_rate: if self.cfg.policy_da_anchor_anti_abuse {
                self.cfg.policy_current_mempool_min_fee_rate
            } else {
                0
            },
            policy_min_da_fee_rate: if self.cfg.policy_da_anchor_anti_abuse {
                self.cfg.policy_min_da_fee_rate
            } else {
                0
            },
        };
        // RUB-167 single-walk invariant: extract weight + da_bytes once
        // here and reuse via `apply_policy` (which forwards into
        // `reject_da_anchor_tx_policy`) AND in the policy_da_included
        // budget update below. Avoids the previous double walk where
        // `apply_policy` recomputed weight/da_bytes internally and the
        // miner then walked again to read `da_bytes`.
        let (weight, da_bytes, _) = tx_weight_and_stats_public(tx).map_err(|e| e.to_string())?;
        let policy_result = apply_policy(tx, weight, da_bytes, utxos, next_height, &policy_cfg);
        if policy_result.is_err() {
            return Ok((true, policy_da_included));
        }
        if self.cfg.policy_da_anchor_anti_abuse {
            let next_da = updated_policy_da_bytes(
                policy_da_included,
                da_bytes,
                self.cfg.policy_max_da_bytes_per_block,
            );
            if next_da.is_none() {
                return Ok((true, policy_da_included));
            }
            return Ok((false, next_da.unwrap_or(policy_da_included)));
        }
        Ok((false, policy_da_included))
    }

    #[allow(dead_code)]
    pub(crate) fn project_complete_da_set_group(
        &self,
        group: &[MinedCandidate],
        projection: CompleteDaSetGroupProjection<'_>,
    ) -> Result<Option<(u64, u64)>, String> {
        if group.is_empty() || projection.selected_weight > projection.remaining_weight {
            return Ok(None);
        }
        let Some(group_inputs) = collect_complete_da_set_group_inputs(
            group,
            projection.selected_nonces,
            projection.selected_inputs,
        ) else {
            return Ok(None);
        };
        let available_weight = projection.remaining_weight - projection.selected_weight;
        let mut group_weight = 0u64;
        let mut next_da_included = projection.policy_da_included;
        let valid = self.validate_complete_da_set_group_consensus(
            group,
            &group_inputs,
            projection.next_height,
            projection.block_mtp,
            |candidate, _work_utxos| {
                let (reject, updated_da_included) = self
                    .reject_candidate_with_utxos(
                        &candidate.tx,
                        &self.sync.chain_state.utxos,
                        projection.next_height,
                        next_da_included,
                    )
                    .unwrap_or((true, next_da_included));
                #[rustfmt::skip]
                let Some(next_group_weight) = group_weight.checked_add(candidate.weight) else { return Ok(false); };
                if reject || next_group_weight > available_weight {
                    return Ok(false);
                };
                group_weight = next_group_weight;
                next_da_included = updated_da_included;
                Ok(true)
            },
        )?;
        if !valid {
            return Ok(None);
        }
        Ok(Some((group_weight, next_da_included)))
    }

    #[allow(dead_code)]
    pub(crate) fn validate_complete_da_set_group_consensus(
        &self,
        group: &[MinedCandidate],
        group_inputs: &[Outpoint],
        next_height: u64,
        block_mtp: u64,
        mut before_apply: impl FnMut(
            &MinedCandidate,
            &HashMap<Outpoint, UtxoEntry>,
        ) -> Result<bool, String>,
    ) -> Result<bool, String> {
        let (rotation, registry) = self.sync.suite_context();
        let mut work_utxos = copy_selected_utxo_set(&self.sync.chain_state.utxos, group_inputs);
        for candidate in group {
            if !before_apply(candidate, &work_utxos)? {
                return Ok(false);
            }
            let next = apply_basic_non_coinbase_update(
                &candidate.tx,
                candidate.txid,
                &work_utxos,
                next_height,
                block_mtp,
                block_mtp,
                self.sync.cfg.chain_id,
                rotation,
                registry,
            );
            let Ok((next_utxos, _summary)) = next else {
                return Ok(false);
            };
            work_utxos = next_utxos;
        }
        Ok(true)
    }
}

#[allow(dead_code)]
pub(crate) fn collect_complete_da_set_group_inputs(
    group: &[MinedCandidate],
    selected_nonces: &HashSet<u64>,
    selected_inputs: &HashSet<Outpoint>,
) -> Option<Vec<Outpoint>> {
    let mut group_nonces = HashSet::new();
    let mut group_inputs = HashSet::new();
    let mut ordered_inputs = Vec::new();
    for candidate in group {
        let nonce = candidate.tx.tx_nonce;
        if nonce == 0 || selected_nonces.contains(&nonce) || !group_nonces.insert(nonce) {
            return None;
        }
        for input in &candidate.tx.inputs {
            let outpoint = Outpoint {
                txid: input.prev_txid,
                vout: input.prev_vout,
            };
            if selected_inputs.contains(&outpoint) || !group_inputs.insert(outpoint.clone()) {
                return None;
            }
            ordered_inputs.push(outpoint);
        }
    }
    Some(ordered_inputs)
}

#[allow(dead_code)]
pub(crate) fn copy_selected_utxo_set(
    utxos: &HashMap<Outpoint, UtxoEntry>,
    inputs: &[Outpoint],
) -> HashMap<Outpoint, UtxoEntry> {
    let mut selected = HashMap::new();
    for input in inputs {
        if let Some(entry) = utxos.get(input) {
            selected.insert(input.clone(), entry.clone());
        }
    }
    selected
}

fn parse_mining_candidate(raw: &[u8]) -> Result<MinedCandidate, String> {
    let (tx, txid, wtxid, consumed) = parse_tx(raw).map_err(|e| e.to_string())?;
    if consumed != raw.len() {
        return Err("non-canonical tx bytes in miner input".to_string());
    }
    let (weight, _, _) = tx_weight_and_stats_public(&tx).map_err(|e| e.to_string())?;
    Ok(MinedCandidate {
        raw: raw.to_vec(),
        tx,
        txid,
        wtxid,
        weight,
    })
}

pub fn validate_complete_da_set_candidate_shape(
    set: &CompleteDaSetCandidate,
) -> Result<bool, String> {
    parse_complete_da_set_candidate(set).map(|candidate| candidate.is_some())
}

pub(crate) fn parse_complete_da_set_candidate(
    set: &CompleteDaSetCandidate,
) -> Result<Option<CompleteDaSetMiningCandidate>, String> {
    let commit = parse_mining_candidate(&set.commit_tx)?;
    let Some(core) = commit.tx.da_commit_core.as_ref() else {
        return Ok(None);
    };
    if core.da_id != set.da_id
        || core.chunk_count == 0
        || u64::from(core.chunk_count) > MAX_DA_CHUNK_COUNT
        || set.chunks.len() != usize::from(core.chunk_count)
        || commit.tx.inputs.is_empty()
    {
        return Ok(None);
    }

    let mut group = CompleteDaSetMiningCandidate {
        da_bytes: commit.tx.da_payload.len() as u64,
        txs: vec![commit],
    };
    let mut payload_hasher = Sha3_256::new();
    for (index, chunk) in set.chunks.iter().enumerate() {
        let want_index = u16::try_from(index).map_err(|_| "DA chunk index overflow".to_string())?;
        if chunk.index != want_index {
            return Ok(None);
        }
        let candidate = parse_mining_candidate(&chunk.tx)?;
        let Some(chunk_core) = candidate.tx.da_chunk_core.as_ref() else {
            return Ok(None);
        };
        let chunk_hash: [u8; 32] = Sha3_256::digest(&candidate.tx.da_payload).into();
        if chunk_core.da_id != set.da_id
            || chunk_core.chunk_index != want_index
            || candidate.tx.inputs.is_empty()
            || chunk_hash != chunk_core.chunk_hash
        {
            return Ok(None);
        }
        let Some(next_da_bytes) = group
            .da_bytes
            .checked_add(candidate.tx.da_payload.len() as u64)
        else {
            return Ok(None);
        };
        group.da_bytes = next_da_bytes;
        payload_hasher.update(&candidate.tx.da_payload);
        group.txs.push(candidate);
    }
    let payload_commitment: [u8; 32] = payload_hasher.finalize().into();
    let Some(commit_candidate) = group.txs.first() else {
        return Ok(None);
    };
    if !complete_da_commitment_matches(&commit_candidate.tx, &payload_commitment) {
        return Ok(None);
    }
    Ok(Some(group))
}

fn complete_da_commitment_matches(tx: &Tx, commitment: &[u8; 32]) -> bool {
    let mut count = 0usize;
    for output in &tx.outputs {
        if output.covenant_type != COV_TYPE_DA_COMMIT {
            continue;
        }
        if count != 0
            || output.covenant_data.len() != 32
            || output.covenant_data.as_slice() != commitment
        {
            return false;
        }
        count += 1;
    }
    count == 1
}

fn canonical_tx_weight(raw: &[u8], msg: &str) -> Result<u64, String> {
    let (tx, _, _, consumed) = parse_tx(raw).map_err(|e| e.to_string())?;
    if consumed != raw.len() {
        return Err(msg.to_string());
    }
    let (weight, _, _) = tx_weight_and_stats_public(&tx).map_err(|e| e.to_string())?;
    Ok(weight)
}

fn build_witness_commitment(parsed: &[MinedCandidate]) -> Result<[u8; 32], String> {
    let mut wtxids = Vec::with_capacity(1 + parsed.len());
    wtxids.push([0u8; 32]);
    for candidate in parsed {
        wtxids.push(candidate.wtxid);
    }
    let root = witness_merkle_root_wtxids(&wtxids).map_err(|e| e.to_string())?;
    Ok(witness_commitment_hash(root))
}

fn updated_policy_da_bytes(current: u64, da_bytes: u64, max_per_block: u64) -> Option<u64> {
    if da_bytes == 0 || max_per_block == 0 {
        return Some(current);
    }
    let next = current.checked_add(da_bytes)?;
    (next <= max_per_block).then_some(next)
}

fn is_mining_da_tx_raw(raw: &[u8]) -> bool {
    matches!(raw.get(4).copied(), Some(0x01) | Some(0x02))
        && parse_tx(raw).is_ok_and(|(tx, _, _, consumed)| {
            consumed == raw.len() && matches!(tx.tx_kind, 0x01 | 0x02)
        })
}

/// Select flat candidate raw txs, skipping individual DA commit/chunk txs before
/// the count cap is reached (mirror of Go `pickFlatCandidateRaw`): DA txs may only
/// enter a block through the complete-set provider group, never as flat candidates.
/// Count-only, no byte cap (matches Go).
fn pick_flat_candidate_raw(txs: &[Vec<u8>], max_count: usize) -> Vec<Vec<u8>> {
    if max_count == 0 {
        return Vec::new();
    }
    let mut selected = Vec::with_capacity(txs.len().min(max_count));
    for raw in txs {
        if is_mining_da_tx_raw(raw) {
            continue;
        }
        selected.push(raw.clone());
        if selected.len() >= max_count {
            break;
        }
    }
    selected
}
fn choose_valid_timestamp(next_height: u64, prev_timestamps: &[u64], now: u64) -> u64 {
    if next_height == 0 || prev_timestamps.is_empty() {
        return if now == 0 { 1 } else { now };
    }
    let median = mtp_median(next_height, prev_timestamps);
    if now > median && now <= median.saturating_add(MAX_FUTURE_DRIFT) {
        return now;
    }
    median.saturating_add(1)
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

fn make_header_prefix(
    prev_hash: [u8; 32],
    merkle_root: [u8; 32],
    timestamp: u64,
    target: [u8; 32],
) -> Vec<u8> {
    let mut header = Vec::with_capacity(108);
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&prev_hash);
    header.extend_from_slice(&merkle_root);
    header.extend_from_slice(&timestamp.to_le_bytes());
    header.extend_from_slice(&target);
    header
}

fn mine_header_nonce(
    block_without_nonce: &[u8],
    target: [u8; 32],
    cancel: &mut MineCancelObserver<'_>,
) -> Result<(Vec<u8>, u64), String> {
    let mut nonce = 0u64;
    loop {
        // RUB-1186 checkpoint 2: at the entry of every finite quantum, so an
        // arbitrarily hard target cannot pin a draining node. Polling only
        // before or after the whole search would not bound it.
        if cancel.check(MineCheckpoint::NonceQuantum) {
            return Err(MINE_CANCELLED.to_string());
        }
        for _ in 0..MINE_NONCE_QUANTUM {
            let mut header = block_without_nonce.to_vec();
            header.extend_from_slice(&nonce.to_le_bytes());
            if pow_check(&header, target).is_ok() {
                return Ok((header, nonce));
            }
            nonce = nonce.wrapping_add(1);
            if nonce == 0 {
                return Err("exhausted nonce space without valid header".to_string());
            }
        }
    }
}

fn assemble_block_bytes(
    header_bytes: &[u8],
    coinbase: &[u8],
    parsed: &[MinedCandidate],
) -> Vec<u8> {
    let mut block = Vec::with_capacity(
        header_bytes.len()
            + coinbase.len()
            + parsed.iter().map(|p| p.raw.len()).sum::<usize>()
            + 16,
    );
    block.extend_from_slice(header_bytes);
    encode_compact_size((1 + parsed.len()) as u64, &mut block);
    block.extend_from_slice(coinbase);
    for candidate in parsed {
        block.extend_from_slice(&candidate.raw);
    }
    block
}

pub fn parse_mine_address_arg(value: &str) -> Result<Option<Vec<u8>>, String> {
    parse_mine_address(value)
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::PathBuf;

    use rubin_consensus::constants::{
        COV_TYPE_ANCHOR, COV_TYPE_CORE_EXT, COV_TYPE_CORE_SIMPLICITY, COV_TYPE_DA_COMMIT,
        COV_TYPE_P2PK, MAX_BLOCK_WEIGHT, MAX_DA_BATCHES_PER_BLOCK as MDB, TX_WIRE_VERSION,
    };
    use rubin_consensus::merkle::{witness_commitment_hash, witness_merkle_root_wtxids};
    use rubin_consensus::{
        encode_compact_size, marshal_tx, p2pk_covenant_data_for_pubkey, parse_tx, sign_transaction,
        tx_weight_and_stats_public, DaChunkCore, DaCommitCore, Mldsa87Keypair, Outpoint, Tx,
        TxInput, TxOutput, UtxoEntry,
    };
    use sha3::{Digest, Sha3_256};

    use crate::da_relay::{CompleteDaSetCandidate, CompleteDaSetChunkCandidate};
    use crate::txpool::TxSource;
    use crate::{
        block_store_path, chain_state_path, default_sync_config, devnet_genesis_chain_id,
        test_helpers::signed_conflicting_p2pk_state_and_txs, BlockStore, ChainState, SyncEngine,
        TxPool,
    };
    type ProviderSet = CompleteDaSetCandidate;

    use super::{
        assemble_block_bytes, build_witness_commitment, canonical_tx_weight,
        choose_valid_timestamp, default_mine_address, make_header_prefix, mine_header_nonce,
        mtp_median, parse_complete_da_set_candidate, parse_mine_address_arg,
        parse_mining_candidate, pick_flat_candidate_raw, updated_policy_da_bytes,
        validate_complete_da_set_candidate_shape, MineCancelObserver, MineCheckpoint, Miner,
        MinerConfig, MINE_CANCELLED,
    };

    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Per-call sequence so two test_sync() dirs never collide. Keying only on
    // pid let tests sharing a `prefix` (the `s` helper is called many times by
    // miner_da_provider_selection_bounds_matrix, plus any parallel test on the
    // same prefix) reuse one `{prefix}-{pid}` directory under `cargo test`; one
    // call's start-of-setup `remove_dir_all` then raced another call's
    // chainstate save, flaking with
    // `rename ... chainstate.json: No such file or directory`.
    static TEST_SYNC_SEQ: AtomicU64 = AtomicU64::new(0);

    /// RAII scratch directory: removes its tree on drop so the per-call unique
    /// `test_sync` directories cannot accumulate in TMPDIR, even for callers
    /// that bind it as `_dir`. Derefs/`AsRef`s to `Path` so existing `&dir`
    /// uses (and the explicit end-of-test `remove_dir_all(&dir)` calls) keep
    /// working; drop is best-effort and ignores errors.
    struct TempDirGuard(PathBuf);
    impl std::ops::Deref for TempDirGuard {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }
    impl AsRef<Path> for TempDirGuard {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn test_sync(prefix: &str) -> (TempDirGuard, BlockStore, SyncEngine) {
        let seq = TEST_SYNC_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("{prefix}-{}-{seq}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");
        let chain_state_file = chain_state_path(&dir);
        let block_store = BlockStore::create(block_store_path(&dir)).expect("blockstore");
        let chain_state = ChainState::new();
        chain_state.save(&chain_state_file).expect("save");
        let sync = SyncEngine::new(
            chain_state,
            Some(block_store.clone()),
            default_sync_config(None, devnet_genesis_chain_id(), Some(chain_state_file)),
        )
        .expect("sync");
        (TempDirGuard(dir), block_store, sync)
    }

    fn coinbase_bytes(height: u64) -> Vec<u8> {
        super::build_coinbase_tx(height, 0, &default_mine_address(), [0u8; 32]).expect("coinbase")
    }

    fn p2pk_utxos(marker: u8, value: u64) -> ([u8; 32], HashMap<Outpoint, UtxoEntry>) {
        let prev = [marker; 32];
        let mut utxos = HashMap::new();
        utxos.insert(
            Outpoint {
                txid: prev,
                vout: 0,
            },
            UtxoEntry {
                value,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&[marker; 32]),
                creation_height: 0,
                created_by_coinbase: false,
            },
        );
        (prev, utxos)
    }

    fn one_input_policy_tx(
        marker: u8,
        out_value: u64,
        covenant_type: u16,
        covenant_data: Vec<u8>,
    ) -> (Tx, HashMap<Outpoint, UtxoEntry>) {
        let (prev, utxos) = p2pk_utxos(marker, 100);
        let tx = Tx {
            version: TX_WIRE_VERSION,
            tx_kind: 0x00,
            tx_nonce: marker as u64,
            inputs: vec![TxInput {
                prev_txid: prev,
                prev_vout: 0,
                script_sig: Vec::new(),
                sequence: 0,
            }],
            outputs: vec![TxOutput {
                value: out_value,
                covenant_type,
                covenant_data,
            }],
            locktime: 0,
            da_commit_core: None,
            da_chunk_core: None,
            witness: Vec::new(),
            da_payload: Vec::new(),
        };
        tx_weight_and_stats_public(&tx).expect("policy tx weight");
        (tx, utxos)
    }

    fn anchor_policy_tx(marker: u8) -> (Tx, HashMap<Outpoint, UtxoEntry>) {
        one_input_policy_tx(marker, 0, COV_TYPE_ANCHOR, vec![marker; 32])
    }

    fn core_simplicity_policy_tx(marker: u8) -> (Tx, HashMap<Outpoint, UtxoEntry>) {
        // Mirror of Go `simplicityCovenantDataForNodeTest`: 32-byte CMR +
        // compactSize(0) empty state.
        let mut cov = vec![marker; 32];
        encode_compact_size(0, &mut cov);
        one_input_policy_tx(marker, 1, COV_TYPE_CORE_SIMPLICITY, cov)
    }

    fn da_budget_policy_tx(marker: u8) -> (Tx, HashMap<Outpoint, UtxoEntry>) {
        let (prev, utxos) = p2pk_utxos(marker, 1_000_000);
        let tx = Tx {
            version: TX_WIRE_VERSION,
            tx_kind: 0x01,
            tx_nonce: marker as u64,
            inputs: vec![TxInput {
                prev_txid: prev,
                prev_vout: 0,
                script_sig: Vec::new(),
                sequence: 0,
            }],
            outputs: vec![TxOutput {
                value: 100_000,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: p2pk_covenant_data_for_pubkey(&[marker; 32]),
            }],
            locktime: 0,
            da_commit_core: Some(DaCommitCore {
                da_id: [marker; 32],
                chunk_count: 1,
                retl_domain_id: [marker.wrapping_add(1); 32],
                batch_number: 1,
                tx_data_root: [marker.wrapping_add(2); 32],
                state_root: [marker.wrapping_add(3); 32],
                withdrawals_root: [marker.wrapping_add(4); 32],
                batch_sig_suite: 0,
                batch_sig: Vec::new(),
            }),
            da_chunk_core: None,
            witness: Vec::new(),
            da_payload: vec![marker; 11],
        };
        tx_weight_and_stats_public(&tx).expect("DA policy tx weight");
        (tx, utxos)
    }

    fn provider_shape_tx(tx_kind: u8, seed: &[u8]) -> Tx {
        let digest: [u8; 32] = Sha3_256::digest(seed).into();
        Tx {
            version: TX_WIRE_VERSION,
            tx_kind,
            tx_nonce: u64::from_le_bytes(digest[..8].try_into().expect("digest prefix")),
            inputs: vec![TxInput {
                prev_txid: digest,
                prev_vout: 0,
                script_sig: Vec::new(),
                sequence: 0,
            }],
            outputs: Vec::new(),
            locktime: 0,
            da_commit_core: None,
            da_chunk_core: None,
            witness: Vec::new(),
            da_payload: Vec::new(),
        }
    }

    fn miner_da_provider_shape_set(da_id: [u8; 32], payloads: &[&[u8]]) -> ProviderSet {
        let mut hasher = Sha3_256::new();
        let mut chunks = Vec::with_capacity(payloads.len());
        for (index, payload) in payloads.iter().enumerate() {
            hasher.update(payload);
            let index = u16::try_from(index).expect("test chunk index");
            let mut tx = provider_shape_tx(0x02, payload);
            tx.da_payload = payload.to_vec();
            tx.da_chunk_core = Some(DaChunkCore {
                da_id,
                chunk_index: index,
                chunk_hash: Sha3_256::digest(payload).into(),
            });
            chunks.push(CompleteDaSetChunkCandidate {
                index,
                tx: marshal_tx(&tx).expect("marshal provider tx"),
            });
        }
        let commitment: [u8; 32] = hasher.finalize().into();
        let mut commit = provider_shape_tx(0x01, &commitment);
        commit.outputs.push(TxOutput {
            value: 0,
            covenant_type: COV_TYPE_DA_COMMIT,
            covenant_data: commitment.to_vec(),
        });
        commit.da_commit_core = Some(DaCommitCore {
            da_id,
            chunk_count: payloads.len() as u16,
            retl_domain_id: [0x10; 32],
            batch_number: 1,
            tx_data_root: [0x11; 32],
            state_root: [0x12; 32],
            withdrawals_root: [0x13; 32],
            batch_sig_suite: 0,
            batch_sig: Vec::new(),
        });
        commit.da_payload = vec![0xa1];
        CompleteDaSetCandidate {
            da_id,
            payload_bytes: payloads.iter().map(|payload| payload.len() as u64).sum(),
            commit_tx: marshal_tx(&commit).expect("marshal provider tx"),
            chunks,
        }
    }

    fn mutate_provider_tx(raw: &[u8], mutate: impl FnOnce(&mut Tx)) -> Vec<u8> {
        let (mut tx, _, _, consumed) = parse_tx(raw).expect("parse provider tx");
        assert_eq!(consumed, raw.len());
        mutate(&mut tx);
        marshal_tx(&tx).expect("marshal provider tx")
    }

    fn mutate_set(base: &ProviderSet, mutate: impl FnOnce(&mut ProviderSet)) -> ProviderSet {
        let mut set = base.clone();
        mutate(&mut set);
        set
    }

    fn signed_miner_da_provider_set(
        da_id: [u8; 32],
        input_marker: u8,
        payload: &[u8],
    ) -> (ProviderSet, HashMap<Outpoint, UtxoEntry>) {
        let mut set = miner_da_provider_shape_set(da_id, &[payload]);
        let keypair = Mldsa87Keypair::generate().expect("provider keypair");
        let covenant_data = p2pk_covenant_data_for_pubkey(&keypair.pubkey_bytes());
        let commit_input = Outpoint {
            txid: [input_marker; 32],
            vout: 0,
        };
        let chunk_input = Outpoint {
            txid: [input_marker.wrapping_add(1); 32],
            vout: 0,
        };
        set.commit_tx = mutate_provider_tx(&set.commit_tx, |tx| {
            tx.inputs[0].prev_txid = commit_input.txid;
            tx.outputs.push(TxOutput {
                value: 1_000_000,
                covenant_type: COV_TYPE_P2PK,
                covenant_data: covenant_data.clone(),
            });
        });
        let entry = |value| UtxoEntry {
            value,
            covenant_type: COV_TYPE_P2PK,
            covenant_data: covenant_data.clone(),
            creation_height: 0,
            created_by_coinbase: false,
        };
        let utxos = HashMap::from([
            (commit_input, entry(2_000_000)),
            (chunk_input.clone(), entry(1_000_000)),
        ]);
        let sign_raw = |raw: &mut Vec<u8>, utxos: &HashMap<Outpoint, UtxoEntry>| {
            let (mut tx, _, _, _) = parse_tx(raw).expect("parse provider tx");
            sign_transaction(&mut tx, utxos, devnet_genesis_chain_id(), &keypair)
                .expect("sign provider tx");
            *raw = marshal_tx(&tx).expect("marshal provider tx");
        };
        sign_raw(&mut set.commit_tx, &utxos);
        let chunk = &mut set.chunks[0];
        chunk.tx = mutate_provider_tx(&chunk.tx, |tx| {
            tx.inputs[0].prev_txid = chunk_input.txid;
        });
        sign_raw(&mut chunk.tx, &utxos);
        (set, utxos)
    }
    impl crate::da_relay::CompleteDaSetProvider for Vec<ProviderSet> {
        fn complete_da_set_candidates(&self, _: u64) -> Vec<ProviderSet> {
            self.clone()
        }
    }
    fn s(
        sets: Vec<ProviderSet>,
        utxos: HashMap<Outpoint, UtxoEntry>,
        cfg: MinerConfig,
    ) -> Result<usize, String> {
        let (_dir, _block_store, mut sync) = test_sync("rubin-rust-miner-da-provider-selection");
        sync.chain_state.utxos = utxos;
        let mut miner = Miner::new(&mut sync, None, cfg).expect("miner");
        miner.set_complete_da_set_provider(&sets);
        let selected = miner.select_candidate_transactions(vec![], 1, MAX_BLOCK_WEIGHT, 0);
        selected.map(|selected| selected.len())
    }
    fn expect_bad(set: ProviderSet) {
        assert!(!validate_complete_da_set_candidate_shape(&set).expect("shape validation"));
    }

    #[test]
    fn miner_da_provider_shape_matrix() {
        let da_id = [0x71; 32];
        let base = miner_da_provider_shape_set(da_id, &[b"chunk-0", b"chunk-1"]);
        assert!(validate_complete_da_set_candidate_shape(&base).expect("valid provider shape"));

        expect_bad(mutate_set(&base, |set| {
            set.chunks.pop();
        }));
        expect_bad(mutate_set(&base, |set| set.chunks.swap(0, 1)));
        expect_bad(mutate_set(&base, |set| {
            set.chunks[0].tx = mutate_provider_tx(&set.chunks[0].tx, |tx| {
                tx.da_chunk_core.as_mut().expect("chunk core").chunk_hash = [0xe1; 32];
            });
        }));
        expect_bad(mutate_set(&base, |set| {
            set.commit_tx = mutate_provider_tx(&set.commit_tx, |tx| {
                tx.outputs[0].covenant_data[0] ^= 1;
            });
        }));
        expect_bad(mutate_set(&base, |set| {
            set.commit_tx = mutate_provider_tx(&set.commit_tx, |tx| {
                tx.da_commit_core.as_mut().expect("commit core").da_id = [0x72; 32];
            });
        }));
        expect_bad(mutate_set(&base, |set| {
            set.chunks[1].tx = mutate_provider_tx(&set.chunks[1].tx, |tx| {
                tx.da_chunk_core.as_mut().expect("chunk core").da_id = [0x73; 32];
            });
        }));
        expect_bad(mutate_set(&base, |set| {
            set.commit_tx = mutate_provider_tx(&set.commit_tx, |tx| tx.inputs.clear());
        }));
        expect_bad(mutate_set(&base, |set| {
            set.chunks[0].tx = mutate_provider_tx(&set.chunks[0].tx, |tx| tx.inputs.clear());
        }));
    }

    #[test]
    fn miner_da_provider_shape_malformed_raw_errors() {
        let mut malformed_commit = miner_da_provider_shape_set([0x81; 32], &[b"chunk-0"]);
        malformed_commit.commit_tx.push(0);
        assert_eq!(
            parse_complete_da_set_candidate(&malformed_commit).unwrap_err(),
            "non-canonical tx bytes in miner input"
        );
    }

    #[test]
    fn miner_provider_parse_error_aborts_selection() {
        // RUB-423 decision: a provider COMPLETE_SET parse error ABORTS the whole
        // mining selection (mirror of Go selectCandidateTransactions returning the
        // error), it is not skipped — provider raw is trusted internal input, so a
        // non-canonical candidate is fail-closed, not a provider-induced template DoS.
        let mut malformed = miner_da_provider_shape_set([0x82; 32], &[b"chunk-0"]);
        malformed.commit_tx.push(0); // trailing byte -> non-canonical parse error
        assert_eq!(
            s(vec![malformed], HashMap::new(), MinerConfig::default()).unwrap_err(),
            "non-canonical tx bytes in miner input",
        );
    }

    #[test]
    fn miner_da_provider_group_projection_matrix() {
        let (set, provider_utxos) = signed_miner_da_provider_set([0x91; 32], 0x91, b"chunk");
        let provider_group = parse_complete_da_set_candidate(&set)
            .expect("parse provider group")
            .expect("provider group");
        let (_dir, _block_store, mut sync) = test_sync("rubin-rust-miner-da-provider-group");
        sync.chain_state.utxos = provider_utxos;
        let cfg = MinerConfig {
            policy_max_da_bytes_per_block: 10_000,
            policy_current_mempool_min_fee_rate: 0,
            policy_min_da_fee_rate: 0,
            ..MinerConfig::default()
        };
        let mut miner = Miner::new(&mut sync, None, cfg).expect("miner");
        fn projection_context<'a>(
            selected_nonces: &'a HashSet<u64>,
            selected_inputs: &'a HashSet<Outpoint>,
        ) -> super::CompleteDaSetGroupProjection<'a> {
            super::CompleteDaSetGroupProjection {
                selected_nonces,
                selected_inputs,
                next_height: 1,
                block_mtp: 1,
                selected_weight: 0,
                remaining_weight: MAX_BLOCK_WEIGHT,
                policy_da_included: 0,
            }
        }
        macro_rules! expect_no_projection {
            ($group:expr, $projection:expr) => {
                assert!(miner
                    .project_complete_da_set_group($group, $projection)
                    .expect("project")
                    .is_none());
            };
        }
        let empty_nonces = HashSet::new();
        let empty_inputs = HashSet::new();
        let original_da_cap = miner.cfg.policy_max_da_bytes_per_block;
        miner.cfg.policy_max_da_bytes_per_block = provider_group.da_bytes - 1;
        expect_no_projection!(
            &provider_group.txs,
            projection_context(&empty_nonces, &empty_inputs)
        );
        miner.cfg.policy_max_da_bytes_per_block = original_da_cap;
        let before_utxos = miner.sync.chain_state.utxos.clone();
        let group_weight = provider_group.txs[0].weight + provider_group.txs[1].weight;
        let projection = miner
            .project_complete_da_set_group(
                &provider_group.txs,
                projection_context(&empty_nonces, &empty_inputs),
            )
            .expect("project")
            .expect("valid group");

        assert_eq!(projection, (group_weight, provider_group.da_bytes));
        assert_eq!(miner.sync.chain_state.utxos, before_utxos);
    }

    #[test]
    fn miner_da_provider_selection_bounds_matrix() {
        let (set, utxos) = signed_miner_da_provider_set([0xa1; 32], 0xa1, b"chunk");
        let group = parse_complete_da_set_candidate(&set).unwrap().unwrap();
        let provider_cfg = |max_tx_per_block| MinerConfig {
            max_tx_per_block,
            ..MinerConfig::default()
        };
        let mut cfg = provider_cfg(2 * (MDB as usize + 1) + 1);
        cfg.policy_max_da_bytes_per_block = group.da_bytes * (MDB + 1);
        let selected_len = |sets, cfg| s(sets, utxos.clone(), cfg).unwrap();
        assert_eq!(selected_len(vec![set.clone()], cfg.clone()), 2);
        assert_eq!(selected_len(vec![set.clone()], provider_cfg(2)), 0);
        let mut low_provider_budget = cfg.clone();
        low_provider_budget.policy_max_da_bytes_per_block = group.da_bytes - 1;
        assert_eq!(selected_len(vec![set.clone()], low_provider_budget), 0);
        assert_eq!(selected_len(vec![set.clone(), set.clone()], cfg.clone()), 2);
        let mut capped_sets = Vec::new();
        let mut capped_utxos = HashMap::new();
        for i in 0..=MDB as u8 {
            let marker = i.wrapping_mul(2).wrapping_add(1);
            let (candidate, candidate_utxos) =
                signed_miner_da_provider_set([marker; 32], marker, &[marker]);
            capped_sets.push(candidate);
            capped_utxos.extend(candidate_utxos);
        }
        assert_eq!(s(capped_sets, capped_utxos, cfg).unwrap(), 2 * MDB as usize);
    }
    #[test]
    fn miner_da_anchor_master_switch_off_ignores_anchor_subflag() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-da-anchor-master-off");
        let (tx, utxos) = anchor_policy_tx(0x71);
        sync.chain_state.utxos = utxos;
        let cfg = MinerConfig {
            policy_da_anchor_anti_abuse: false,
            policy_reject_non_coinbase_anchor_outputs: true,
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, None, cfg).expect("miner");

        let (reject, next_da) = miner.reject_candidate(&tx, 0, 0).expect("reject candidate");
        assert!(
            !reject,
            "master=false must ignore policy_reject_non_coinbase_anchor_outputs=true"
        );
        assert_eq!(next_da, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn miner_da_anchor_master_switch_on_rejects_non_coinbase_anchor_when_subflag_on() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-da-anchor-subflag-on");
        let (tx, utxos) = anchor_policy_tx(0x72);
        sync.chain_state.utxos = utxos;
        let cfg = MinerConfig {
            policy_da_anchor_anti_abuse: true,
            policy_reject_non_coinbase_anchor_outputs: true,
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, None, cfg).expect("miner");

        let (reject, next_da) = miner.reject_candidate(&tx, 0, 0).expect("reject candidate");
        assert!(
            reject,
            "master=true and subflag=true must reject non-coinbase CORE_ANCHOR"
        );
        assert_eq!(next_da, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn miner_da_anchor_master_switch_on_allows_anchor_when_subflag_off() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-da-anchor-subflag-off");
        let (tx, utxos) = anchor_policy_tx(0x73);
        sync.chain_state.utxos = utxos;
        let cfg = MinerConfig {
            policy_da_anchor_anti_abuse: true,
            policy_reject_non_coinbase_anchor_outputs: false,
            policy_max_da_bytes_per_block: 10,
            policy_da_surcharge_per_byte: 0,
            policy_current_mempool_min_fee_rate: 0,
            policy_min_da_fee_rate: 0,
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, None, cfg).expect("miner");

        let (reject, next_da) = miner.reject_candidate(&tx, 0, 0).expect("anchor candidate");
        assert!(
            !reject,
            "master=true and subflag=false must allow CORE_ANCHOR through anchor policy"
        );
        assert_eq!(next_da, 0);

        let (da_tx, da_utxos) = da_budget_policy_tx(0x74);
        miner.sync.chain_state.utxos = da_utxos;
        let (reject, next_da) = miner.reject_candidate(&da_tx, 0, 0).expect("DA candidate");
        assert!(
            reject,
            "master=true must keep DA byte-budget policy active when anchor subflag=false"
        );
        assert_eq!(next_da, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mine_one_unassigned_covenant_failure_image() {
        let (_dir, _block_store, mut sync) = test_sync("rub1335-mine-one-unassigned");
        sync.bootstrap_canonical_genesis_if_empty()
            .expect("bootstrap");
        let (tx, utxos) = one_input_policy_tx(0x77, 1, COV_TYPE_CORE_EXT, vec![0x01]);
        sync.chain_state.utxos.extend(utxos);
        let state_before = sync.chain_state.clone();
        let sync_tip_before = sync.tip().expect("sync tip before");
        let store = sync.block_store.as_ref().expect("block store");
        let store_tip_before = store.tip().expect("store tip before");
        let canonical_len_before = store.canonical_len();
        let err = Miner::new(&mut sync, None, MinerConfig::default())
            .expect("miner")
            .mine_one(&[marshal_tx(&tx).expect("marshal candidate")])
            .expect_err("unassigned candidate must fail");
        assert_eq!(err, "TX_ERR_COVENANT_TYPE_INVALID: unknown covenant_type");
        assert_eq!(sync.chain_state, state_before);
        assert_eq!(sync.tip().expect("sync tip after"), sync_tip_before);
        let store = sync.block_store.as_ref().expect("block store");
        assert_eq!(store.tip().expect("store tip after"), store_tip_before);
        assert_eq!(store.canonical_len(), canonical_len_before);
    }

    #[test]
    fn miner_simplicity_policy_still_runs_when_da_anchor_master_off() {
        // Mirror of Go `TestMinerSimplicityPolicyStillRunsWhenDaAnchorMasterOff`:
        // the CORE_SIMPLICITY pre-activation guardrail is NOT nested
        // under the DA/anchor master switch.
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-simplicity-master-off");
        let (tx, utxos) = core_simplicity_policy_tx(0x76);
        sync.chain_state.utxos = utxos;
        let cfg = MinerConfig {
            policy_da_anchor_anti_abuse: false,
            policy_reject_simplicity_pre_activation: true,
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, None, cfg).expect("miner");

        let (reject, next_da) = miner
            .reject_candidate(&tx, 0, 0)
            .expect("CORE_SIMPLICITY candidate");
        assert!(
            reject,
            "CORE_SIMPLICITY pre-activation policy must still run when DA/anchor master is off"
        );
        assert_eq!(next_da, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mine_one_from_empty_state_updates_tip() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner");
        let cfg = MinerConfig {
            timestamp_source: || 1_777_000_000,
            ..MinerConfig::default()
        };
        let mut miner = Miner::new(&mut sync, None, cfg).expect("miner");

        // RUB-1137 migration (bootstrap-first): `mine_one` now adopts the
        // published devnet genesis at height 0 before building a candidate, so
        // the first mined block is height 1. Original property — mining from an
        // empty state publishes a tip equal to the mined block — unchanged.
        let mined = miner.mine_one(&[]).expect("mine one");
        assert_eq!(mined.height, 1);
        assert_eq!(mined.tx_count, 1);
        let tip = miner.sync.tip().expect("tip").expect("some tip");
        assert_eq!(tip.0, 1);
        assert_eq!(tip.1, mined.hash);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mine_one_observes_supply_above_u64_without_narrowing() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-u128-supply");
        sync.bootstrap_canonical_genesis_if_empty()
            .expect("bootstrap genesis");
        let supply = u128::from(u64::MAX) + 1;
        sync.chain_state.already_generated = supply;
        let expected = supply + u128::from(rubin_consensus::subsidy::block_subsidy(1, supply));
        let cfg = MinerConfig {
            timestamp_source: || 1_777_000_001,
            ..MinerConfig::default()
        };
        let mut miner = Miner::new(&mut sync, None, cfg).expect("miner");

        let mined = miner.mine_one(&[]).expect("mine above u64");
        assert_eq!(mined.height, 1);
        assert_eq!(miner.sync.chain_state.already_generated, expected);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Mirror of Go `TestMinerMineOne_PropagatesBootstrapError`
    /// (clients/go/node/sync_genesis_identity_test.go): when the bootstrap
    /// fails, `mine_one` surfaces that error instead of continuing into
    /// candidate construction on an engine that cannot persist. Go triggers the
    /// failure by nilling the engine's `chainState`; the Rust engine owns its
    /// chainstate by value, so the equivalent construction is a latched
    /// persistence fault — the only failure `bootstrap_canonical_genesis_if_empty`
    /// can report before it applies anything.
    ///
    /// Asserting the returned message alone would be vacuous: a swallowed
    /// bootstrap error is rediscovered with the SAME message when the candidate
    /// reaches `apply_block`. The counting timestamp source isolates it — it is
    /// consulted only during candidate construction.
    #[test]
    fn mine_one_propagates_bootstrap_error() {
        use crate::io_utils::{atomic_write_error_after, AtomicWriteOperation};
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-bootstrap-err");
        let _ = sync.handle_persistence_error(
            atomic_write_error_after(&dir, AtomicWriteOperation::Overwrite, "postcommit"),
            false,
            false,
        );
        let cfg = MinerConfig {
            timestamp_source: bootstrap_error_timestamp_source,
            ..MinerConfig::default()
        };
        let mut miner = Miner::new(&mut sync, None, cfg).expect("miner");

        assert_eq!(
            miner
                .mine_one(&[])
                .expect_err("a halted engine must not mine"),
            "storage persistence fault; restart required"
        );
        assert_eq!(
            BOOTSTRAP_ERROR_TIMESTAMP_CALLS.load(Ordering::Relaxed),
            0,
            "mine_one must return at the bootstrap call, before building a candidate"
        );
        assert!(
            !miner.sync.chain_state.has_tip,
            "the failed bootstrap must not leave a tip behind"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Owned by `mine_one_propagates_bootstrap_error`; no other test installs it.
    static BOOTSTRAP_ERROR_TIMESTAMP_CALLS: AtomicU64 = AtomicU64::new(0);
    fn bootstrap_error_timestamp_source() -> u64 {
        BOOTSTRAP_ERROR_TIMESTAMP_CALLS.fetch_add(1, Ordering::Relaxed);
        1
    }

    /// RUB-1137 accepted row: a fresh empty devnet datadir mined N times ends
    /// with the published devnet genesis at canonical index row 0 — the same
    /// row 0 the Go reference produces — mined heights 1..=N, and a restart
    /// that reconciles cleanly onto the mined tip.
    #[test]
    fn mine_n_from_empty_devnet_chain_bootstraps_published_genesis_at_row_zero() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-bootstrap-row0");
        let cfg = MinerConfig {
            timestamp_source: || 1,
            ..MinerConfig::default()
        };
        let mut miner = Miner::new(&mut sync, None, cfg).expect("miner");

        let mined = miner.mine_n(2, &[]).expect("mine n");
        assert_eq!(mined[0].height, 1);
        assert_eq!(mined[1].height, 2);

        let mut store =
            BlockStore::open(crate::blockstore::block_store_path(&dir)).expect("reopen store");
        assert_eq!(
            store.canonical_hash(0).expect("row 0"),
            Some(crate::genesis::devnet_genesis_hash()),
            "row 0 must be the published devnet genesis, not a miner-synthesized block"
        );
        assert_eq!(store.tip().expect("tip"), Some((2, mined[1].hash)));

        // Restart leg: the datadir reconciles cleanly onto the mined tip.
        let cfg = miner.sync.cfg.clone();
        let mut state = ChainState::new();
        crate::reconcile_chain_state_with_block_store(&mut state, &mut store, &cfg)
            .expect("reconcile mined datadir");
        assert_eq!(state.height, 2);
        assert_eq!(state.tip_hash, mined[1].hash);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mine_n_produces_height_and_timestamp_progression() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-n");
        let cfg = MinerConfig {
            timestamp_source: || 1,
            ..MinerConfig::default()
        };
        let mut pool = TxPool::new();
        let mut miner = Miner::new(&mut sync, Some(&mut pool), cfg).expect("miner");

        // RUB-1137 migration (bootstrap-first): heights shift by one because
        // height 0 is now the published devnet genesis. Original property —
        // consecutive heights with non-decreasing timestamps — unchanged.
        let mined = miner.mine_n(3, &[]).expect("mine n");
        assert_eq!(mined.len(), 3);
        assert_eq!(mined[0].height, 1);
        assert_eq!(mined[1].height, 2);
        assert_eq!(mined[2].height, 3);
        // Timestamps stay non-decreasing and advance across the run. The
        // pre-migration strict `>` between the first two blocks was an artifact
        // of the height-0 candidate taking the raw clock: every post-genesis
        // height clamps to MTP+1, and the even-window MTP median repeats one
        // value, so heights 1 and 2 may legitimately share a timestamp.
        assert!(mined[1].timestamp >= mined[0].timestamp);
        assert!(mined[2].timestamp >= mined[1].timestamp);
        assert!(mined[2].timestamp > mined[0].timestamp);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn miner_new_normalizes_defaults() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-defaults");
        let cfg = MinerConfig {
            max_tx_per_block: 0,
            mine_address: Vec::new(),
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, None, cfg).expect("miner");
        assert_eq!(miner.cfg.max_tx_per_block, 1024);
        assert_eq!(miner.cfg.mine_address, default_mine_address());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_transactions_prefers_explicit_input_and_caps_selection() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-explicit");
        let cfg = MinerConfig {
            max_tx_per_block: 2,
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, None, cfg).expect("miner");
        let selected = miner.candidate_transactions(&[vec![0x01], vec![0x02], vec![0x03]]);
        assert_eq!(selected, vec![vec![0x01]]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_transactions_filters_provider_da_before_pool_cap() {
        let (_dir, _block_store, mut sync) = test_sync("rubin-rust-miner-pool-provider-da");
        let mut pool = TxPool::new();
        let da_raw = miner_da_provider_shape_set([0x65; 32], &[b"chunk"]).commit_tx;
        let non_da = vec![0xee; da_raw.len() + 1];
        pool.inject_test_entry([0x01; 32], da_raw.clone());
        pool.inject_test_entry([0x03; 32], non_da.clone());
        let cfg = MinerConfig {
            max_tx_per_block: 2,
            ..MinerConfig::default()
        };
        let sets = Vec::new();
        let mut miner = Miner::new(&mut sync, Some(&mut pool), cfg).expect("miner");
        miner.set_complete_da_set_provider(&sets);
        assert_eq!(miner.candidate_transactions(&[]), vec![non_da.clone()]);
        let explicit = miner.candidate_transactions(&[da_raw, non_da.clone(), vec![0xdd]]);
        assert_eq!(explicit, vec![non_da]);
    }

    #[test]
    fn candidate_transactions_skips_explicit_da_before_count_cap() {
        // No provider wired: the flat filter is unconditional (mirror Go
        // pickFlatCandidateRaw), so individual DA commit/chunk txs are skipped before
        // the count cap; a trailing-byte commit is not canonical DA, so it is kept.
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-flat-da-explicit");
        let set = miner_da_provider_shape_set([0x65; 32], &[b"chunk"]);
        let commit = set.commit_tx;
        let chunk = set.chunks[0].tx.clone();
        let ordinary = vec![0xee; commit.len() + 1];
        let mut trailing = commit.clone();
        trailing.push(0);
        let cfg = MinerConfig {
            max_tx_per_block: 2,
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, None, cfg).expect("miner");
        assert_eq!(
            miner.candidate_transactions(&[commit.clone(), chunk.clone(), ordinary.clone()]),
            vec![ordinary.clone()],
            "both DA txs skipped before the count cap of 1"
        );
        assert_eq!(
            miner.candidate_transactions(&[commit, chunk, trailing.clone(), ordinary]),
            vec![trailing],
            "trailing-byte commit is not classified as DA and fills the slot"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_transactions_skips_pool_da_before_count_cap() {
        // Same shape as the provider test but with NO provider wired: the txpool DA
        // filter is unconditional (mirror Go pickMinerCandidateEntries), so pool DA
        // commit and chunk are skipped before the count/byte caps; the ordinary mines.
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-flat-da-pool");
        let mut pool = TxPool::new();
        let set = miner_da_provider_shape_set([0x66; 32], &[b"chunk"]);
        let non_da = vec![0xee; set.commit_tx.len() + 1];
        pool.inject_test_entry([0x01; 32], set.commit_tx.clone());
        pool.inject_test_entry([0x02; 32], set.chunks[0].tx.clone());
        pool.inject_test_entry([0x03; 32], non_da.clone());
        let cfg = MinerConfig {
            max_tx_per_block: 2,
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, Some(&mut pool), cfg).expect("miner");
        assert_eq!(
            miner.candidate_transactions(&[]),
            vec![non_da],
            "pool DA commit and chunk skipped; only the ordinary tx selected"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn flat_candidate_selection_handles_zero_and_empty_guards() {
        // Defensive guards: zero caps and an empty/absent candidate source.
        assert!(pick_flat_candidate_raw(&[vec![0x00]], 0).is_empty());
        let pool = TxPool::new();
        assert!(pool
            .select_transactions_with_filter(0, 100, |_| false)
            .is_empty());
        assert!(pool
            .select_transactions_with_filter(1, 0, |_| false)
            .is_empty());
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-flat-da-guards");
        // max_tx_per_block == 1 -> max_selected == 0 -> no candidates.
        {
            let cfg = MinerConfig {
                max_tx_per_block: 1,
                ..MinerConfig::default()
            };
            let miner = Miner::new(&mut sync, None, cfg).expect("miner");
            assert!(miner.candidate_transactions(&[vec![0x00]]).is_empty());
        }
        // No txpool wired and no explicit txs -> no candidates.
        let cfg = MinerConfig {
            max_tx_per_block: 2,
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, None, cfg).expect("miner");
        assert!(miner.candidate_transactions(&[]).is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn helper_branches_cover_timestamp_and_da_budget_edges() {
        assert_eq!(updated_policy_da_bytes(7, 0, 10), Some(7));
        assert_eq!(updated_policy_da_bytes(7, 2, 0), Some(7));
        assert_eq!(updated_policy_da_bytes(5, 6, 10), None);
        assert_eq!(updated_policy_da_bytes(u64::MAX, 1, u64::MAX), None);
        assert_eq!(choose_valid_timestamp(0, &[], 0), 1);
        assert_eq!(choose_valid_timestamp(1, &[10, 11, 12], 12), 12);
        let future = 12 + super::MAX_FUTURE_DRIFT + 1;
        assert_eq!(choose_valid_timestamp(1, &[10, 11, 12], future), 11);
    }

    #[test]
    fn helper_parsers_cover_noncanonical_and_address_edges() {
        let tx = coinbase_bytes(0);
        let candidate = parse_mining_candidate(&tx).expect("candidate");
        assert_eq!(candidate.raw, tx);
        assert!(candidate.weight > 0);

        let mut noncanonical = candidate.raw.clone();
        noncanonical.push(0);
        assert_eq!(
            parse_mining_candidate(&noncanonical).unwrap_err(),
            "non-canonical tx bytes in miner input"
        );
        assert_eq!(
            canonical_tx_weight(&noncanonical, "bad tx").unwrap_err(),
            "bad tx"
        );

        let default_hex = hex::encode(default_mine_address());
        assert_eq!(
            parse_mine_address_arg(&default_hex).expect("address"),
            Some(default_mine_address())
        );
        assert_eq!(parse_mine_address_arg("").expect("empty"), None);
        assert!(parse_mine_address_arg("zz").is_err());
    }

    #[test]
    fn helper_commitment_and_block_assembly_match_wire_helpers() {
        let coinbase0 = coinbase_bytes(0);
        let coinbase1 = coinbase_bytes(1);
        let first = parse_mining_candidate(&coinbase0).expect("first");
        let second = parse_mining_candidate(&coinbase1).expect("second");

        let expected_root =
            witness_merkle_root_wtxids(&[[0u8; 32], first.wtxid, second.wtxid]).expect("root");
        assert_eq!(
            build_witness_commitment(&[first.clone(), second.clone()]).expect("commitment"),
            witness_commitment_hash(expected_root)
        );

        let header = make_header_prefix([0x11; 32], [0x22; 32], 7, [0x33; 32]);
        assert_eq!(header.len(), 108);
        assert_eq!(&header[..4], &1u32.to_le_bytes());
        assert_eq!(&header[4..36], &[0x11; 32]);
        assert_eq!(&header[36..68], &[0x22; 32]);
        assert_eq!(&header[68..76], &7u64.to_le_bytes());
        assert_eq!(&header[76..108], &[0x33; 32]);

        let block = assemble_block_bytes(&header, &coinbase0, std::slice::from_ref(&second));
        let mut expected = header.clone();
        rubin_consensus::encode_compact_size(2, &mut expected);
        expected.extend_from_slice(&coinbase0);
        expected.extend_from_slice(&second.raw);
        assert_eq!(block, expected);
    }

    #[test]
    fn helper_medians_and_candidate_limits_cover_extra_branches() {
        assert_eq!(mtp_median(3, &[9, 3, 7]), 7);

        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-one-slot");
        let cfg = MinerConfig {
            max_tx_per_block: 1,
            ..MinerConfig::default()
        };
        let miner = Miner::new(&mut sync, None, cfg).expect("miner");
        let selected = miner.candidate_transactions(&[coinbase_bytes(0)]);
        assert!(selected.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mine_one_rejects_explicit_noncanonical_tx_bytes() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-bad-explicit");
        let mut miner = Miner::new(&mut sync, None, MinerConfig::default()).expect("miner");
        let mut bad = coinbase_bytes(0);
        bad.push(0);
        let err = miner.mine_one(&[bad]).unwrap_err();
        assert!(
            err.contains("non-canonical tx bytes in miner input"),
            "{err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    fn signed_p2pk_state_and_tx(input_value: u64, output_value: u64) -> (ChainState, Vec<u8>) {
        let (state, raw, _conflict) =
            signed_conflicting_p2pk_state_and_txs(input_value, output_value, output_value - 1);
        (state, raw)
    }

    fn admit_setup_pool_tx(
        pool: &mut TxPool,
        raw: &[u8],
        state: &ChainState,
        source: TxSource,
    ) -> [u8; 32] {
        let (txid, _) = pool
            .add_tx_with_source(raw, state, None, devnet_genesis_chain_id(), source)
            .expect("setup admit");
        assert_eq!(
            pool.entry_source(&txid),
            Some(source),
            "setup admission records its declared source before miner cleanup"
        );
        txid
    }

    #[test]
    fn mine_one_includes_valid_explicit_tx() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-explicit-valid");
        let (state, raw) = signed_p2pk_state_and_tx(20, 10);
        sync.chain_state.utxos = state.utxos;
        let cfg = MinerConfig {
            timestamp_source: || 1_777_000_123,
            ..MinerConfig::default()
        };
        let mut miner = Miner::new(&mut sync, None, cfg).expect("miner");

        // RUB-1137 migration (bootstrap-first): height shifts 0 -> 1. Original
        // property — the explicit transaction is included alongside the
        // coinbase and a tip is published — unchanged.
        let mined = miner.mine_one(&[raw]).expect("mine one");
        assert_eq!(mined.height, 1);
        assert_eq!(mined.tx_count, 2);
        assert!(miner.sync.chain_state.has_tip);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mine_n_evicts_confirmed_pool_transactions_between_blocks() {
        // RUB-162 Phase A migration rationale (per controller Q2 / Path A
        // approval 2026-05-03):
        //   - old assumption: signed_p2pk_state_and_tx(20, 10) → fee=10
        //     with weight ≈ 7653 admits because pre-RUB-162
        //     admit_with_metadata did not enforce the rolling fee floor.
        //   - new invariant: admit_with_metadata enforces the rolling fee
        //     floor (DEFAULT=1) via validate_fee_floor.
        //   - reachability: tx is well-formed; setup admission reaches the
        //     txpool admission path. Mine_n then confirms blocks and
        //     evicts confirmed txs from the pool.
        //   - producer boundary: the admission below is setup-only; miner
        //     production code only selects and cleans up existing pool txs.
        //   - replacement coverage: input bumped to 7700 so fee = 7700 - 10
        //     = 7690 ≥ weight (≈7653). The mine-then-evict invariant
        //     remains under test.
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-pool-evict");
        let (state, raw) = signed_p2pk_state_and_tx(7700, 10);
        sync.chain_state.utxos = state.utxos.clone();

        let mut pool = TxPool::new();
        let confirmed_txid = admit_setup_pool_tx(&mut pool, &raw, &state, TxSource::Remote);

        let mined = {
            let cfg = MinerConfig {
                timestamp_source: || 1_777_000_321,
                ..MinerConfig::default()
            };
            let mut miner = Miner::new(&mut sync, Some(&mut pool), cfg).expect("miner");
            miner.mine_n(2, &[]).expect("mine n")
        };

        assert_eq!(mined.len(), 2);
        assert_eq!(mined[0].tx_count, 2);
        assert_eq!(mined[1].tx_count, 1);
        assert!(pool.is_empty());
        assert!(!pool.contains(&confirmed_txid));
        assert_eq!(pool.entry_source(&confirmed_txid), None);
        // Re-admit against the original setup state only as an index
        // cleanup probe; this does not model post-confirmation runtime
        // validity or a miner producer path.
        let reaccepted_txid = admit_setup_pool_tx(&mut pool, &raw, &state, TxSource::Reorg);
        assert_eq!(
            reaccepted_txid, confirmed_txid,
            "confirmed cleanup must remove txid/index state, not leave a stale producer marker"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mine_one_evicts_conflicting_pool_transaction_for_explicit_candidate() {
        // RUB-162 Phase A migration rationale (per controller Q2 / Path A
        // approval 2026-05-03):
        //   - old assumption: signed_conflicting_p2pk_state_and_txs(20,10,9)
        //     produced two txs with fee=10/fee=11 that admitted because
        //     pre-RUB-162 admit_with_metadata did not enforce the rolling
        //     fee floor.
        //   - new invariant: admit_with_metadata enforces the rolling fee
        //     floor.
        //   - reachability: setup admission on the conflicting_raw reaches
        //     the txpool admission path; mine_one then includes the
        //     explicit candidate which conflicts with the pool entry,
        //     evicting it.
        //   - producer boundary: the setup source is not a miner producer
        //     claim; miner cleanup must remove it source-neutrally.
        //   - replacement coverage: input bumped to 7700 so both txs have
        //     fees ≥ weight (~7653). The conflicting-eviction-on-explicit-
        //     candidate invariant remains under test.
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-explicit-conflict");
        let (state, explicit_raw, conflicting_raw) =
            signed_conflicting_p2pk_state_and_txs(7700, 10, 9);
        sync.chain_state.utxos = state.utxos.clone();

        let mut pool = TxPool::new();
        let conflicting_txid =
            admit_setup_pool_tx(&mut pool, &conflicting_raw, &state, TxSource::Reorg);

        {
            let cfg = MinerConfig {
                timestamp_source: || 1_777_000_555,
                ..MinerConfig::default()
            };
            let mut miner = Miner::new(&mut sync, Some(&mut pool), cfg).expect("miner");
            let mined = miner.mine_one(&[explicit_raw]).expect("mine explicit");
            assert_eq!(mined.tx_count, 2);
        }

        assert!(pool.is_empty());
        assert!(!pool.contains(&conflicting_txid));
        assert_eq!(pool.entry_source(&conflicting_txid), None);
        // Re-admit against the original setup state only as a spender-index
        // cleanup probe; the miner remains a cleanup consumer, not a tx
        // producer.
        let reaccepted_txid =
            admit_setup_pool_tx(&mut pool, &conflicting_raw, &state, TxSource::Remote);
        assert_eq!(
            reaccepted_txid, conflicting_txid,
            "conflict cleanup must clear spender/index state before a new producer can admit"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mine_one_rejects_height_overflow_before_assembly() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-height-overflow");
        sync.chain_state.has_tip = true;
        sync.chain_state.height = u64::MAX;
        let mut miner = Miner::new(&mut sync, None, MinerConfig::default()).expect("miner");
        let err = miner.mine_one(&[]).unwrap_err();
        assert!(err.contains("height overflow"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn select_candidate_transactions_covers_reject_skip_and_accept_paths() {
        let (dir, _block_store, mut sync) = test_sync("rubin-rust-miner-selection");
        let (state, raw, conflicting_raw) = signed_conflicting_p2pk_state_and_txs(20, 10, 9);
        sync.chain_state.utxos = state.utxos;
        let cfg = MinerConfig {
            max_tx_per_block: 2,
            ..MinerConfig::default()
        };
        let sets = Vec::new();
        let mut miner = Miner::new(&mut sync, None, cfg).expect("miner");

        let rejected = miner
            .select_candidate_transactions(vec![coinbase_bytes(0)], 0, MAX_BLOCK_WEIGHT, 0)
            .expect("reject branch");
        assert!(rejected.is_empty());

        let overweight = miner
            .select_candidate_transactions(vec![raw.clone()], 0, 0, 0)
            .expect("weight skip");
        assert!(overweight.is_empty());

        miner.set_complete_da_set_provider(&sets);
        let da_raw = miner_da_provider_shape_set([0x66; 32], &[b"chunk"]).commit_tx;
        let accepted = miner
            .select_candidate_transactions(
                vec![da_raw, raw.clone(), conflicting_raw.clone()],
                0,
                MAX_BLOCK_WEIGHT,
                0,
            )
            .expect("accept branch");
        assert_eq!(accepted[0].raw, raw);
        miner.cfg.max_tx_per_block = 3;
        let conflict_skipped = miner
            .select_candidate_transactions(
                vec![raw.clone(), conflicting_raw],
                0,
                MAX_BLOCK_WEIGHT,
                0,
            )
            .expect("conflict branch");
        assert_eq!(conflict_skipped.len(), 1);
        assert_eq!(conflict_skipped[0].raw, raw);
        let _ = fs::remove_dir_all(&dir);
    }

    // ---- RUB-655 target-schedule runtime rows ----

    /// On a stock devnet chain the template IGNORES a deliberately wrong static
    /// `MinerConfig.target` for BOTH header construction and the nonce search —
    /// otherwise `mine_one`'s own `apply_block` would reject the block the miner
    /// just mined. At a retarget BOUNDARY it commits to the retargeted value; on
    /// a non-stock configuration the static target is verbatim.
    #[test]
    fn target_schedule_runtime_miner_uses_derived_target() {
        use crate::sync::{boundary_chain_for_test, stock_devnet_engine_for_test};
        use rubin_consensus::constants::{POW_LIMIT, WINDOW_SIZE};

        // Two leading zero bytes: far HARDER than the derived POW_LIMIT, so a
        // nonce searched against it lands nowhere near the first nonce that
        // satisfies the derived target.
        let mut wrong = POW_LIMIT;
        wrong[0] = 0x00;
        wrong[1] = 0x00;
        #[rustfmt::skip]
        let cfg = MinerConfig { target: wrong, ..Default::default() };

        // End to end: `mine_one` mines AND applies, so a template that kept the
        // wrong static target could not have produced an accepted block.
        let (mut sync, dir) = stock_devnet_engine_for_test("rubin-ts-miner", |_| {});
        let mined = {
            let mut miner = Miner::new(&mut sync, None, cfg.clone()).expect("miner");
            miner.mine_one(&[]).expect("mine one")
        };
        let store = sync.block_store_snapshot().expect("store");
        let header = store.get_header_by_hash(mined.hash).expect("mined header");
        let parsed = rubin_consensus::parse_block_header_bytes(&header).expect("parse header");
        assert_eq!(parsed.target, POW_LIMIT, "template kept the static target");
        // The NONCE SEARCH must take the derived target too, not just the
        // header field: the mined nonce has to be the FIRST that satisfies the
        // derived value, which a search against the static one would skip.
        let never = |_: MineCheckpoint| false;
        let (_, first) = mine_header_nonce(
            &header[..108],
            POW_LIMIT,
            &mut MineCancelObserver::new(&never),
        )
        .expect("first nonce");
        assert_eq!(
            mined.nonce, first,
            "nonce searched against the static target"
        );
        fs::remove_dir_all(&dir).expect("cleanup");

        // One height below the boundary the parent's own target binds, so the
        // boundary row is not a tautology; height 0 is the empty-chain path and
        // keeps the configured value; a non-stock chain keeps it too.
        let (mut boundary, boundary_dir, times) = boundary_chain_for_test("rubin-ts-mnbound");
        let want = rubin_consensus::retarget_v1_clamped(POW_LIMIT, &times).expect("reference");
        let parent = boundary.chain_state.tip_hash;
        {
            let miner = Miner::new(&mut boundary, None, cfg.clone()).expect("boundary miner");
            #[rustfmt::skip]
            let rows = [(WINDOW_SIZE, want), (WINDOW_SIZE - 1, POW_LIMIT), (0, wrong)];
            for (height, expected) in rows {
                let got = miner.template_target(parent, height);
                assert_eq!(got, Ok(expected), "{height}");
            }
        }
        drop(boundary);
        fs::remove_dir_all(&boundary_dir).expect("cleanup");

        let (mut regtest, regtest_dir) =
            stock_devnet_engine_for_test("rubin-ts-mnregtest", |c| c.network = "regtest".into());
        let regtest_parent = regtest.chain_state.tip_hash;
        {
            let miner = Miner::new(&mut regtest, None, cfg).expect("regtest miner");
            assert_eq!(miner.template_target(regtest_parent, 1), Ok(wrong));
        }
        drop(regtest);
        fs::remove_dir_all(&regtest_dir).expect("cleanup");
    }

    // ---- RUB-1186 lifecycle checkpoint rows ----

    /// RUB-1186 checkpoint 3 in isolation. `/mine_next` has no in-process seam
    /// between the last nonce quantum and `apply_block`, so the pre-apply
    /// observation is isolated here instead: the named checkpoint makes the
    /// earlier two provably clean without a thread or a timing window. Removing
    /// only the pre-apply check lets the staged candidate reach `apply_block`
    /// and move the tip, which these assertions forbid.
    #[test]
    fn mine_one_observed_pre_apply_checkpoint_leaves_tip_unchanged() {
        use crate::sync::stock_devnet_engine_for_test;

        let (mut sync, dir) = stock_devnet_engine_for_test("rubin-mn-preapply", |_| {});
        let tip_before = sync.chain_state.tip_hash;
        let height_before = sync.chain_state.height;
        let at_pre_apply = |at: MineCheckpoint| at == MineCheckpoint::PreApply;
        let mut cancel = MineCancelObserver::new(&at_pre_apply);
        let err = {
            let mut miner = Miner::new(&mut sync, None, MinerConfig::default()).expect("miner");
            miner
                .mine_one_observed(&[], &mut cancel)
                .expect_err("pre-apply cancellation")
        };
        assert_eq!(err, MINE_CANCELLED, "cancelled, not a mining failure");
        assert!(cancel.observed(), "an actual checkpoint must latch");
        assert_eq!(sync.chain_state.tip_hash, tip_before, "apply must not run");
        assert_eq!(sync.chain_state.height, height_before, "tip unchanged");
        drop(sync);
        fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// RUB-1186: the quantum restructure is search-neutral. Against a target
    /// hard enough that the winner lies past the first quantum, the bounded
    /// loop returns the byte-identical header and nonce a plain unbounded scan
    /// finds — the property the checkpoint may not buy at the cost of.
    #[test]
    fn mine_header_nonce_quantum_preserves_the_first_valid_nonce() {
        use rubin_consensus::constants::POW_LIMIT;
        use rubin_consensus::pow_check;

        let mut hard = POW_LIMIT;
        hard[0] = 0x00;
        hard[1] = 0x00;
        let prefix = make_header_prefix([7u8; 32], [9u8; 32], 1_777_000_000, hard);
        let never = |_: MineCheckpoint| false;
        let (header, nonce) =
            mine_header_nonce(&prefix, hard, &mut MineCancelObserver::new(&never))
                .expect("bounded search");

        // Oracle: the baseline unbounded scan, inlined.
        let mut want = 0u64;
        let want_header = loop {
            let mut candidate = prefix.clone();
            candidate.extend_from_slice(&want.to_le_bytes());
            if pow_check(&candidate, hard).is_ok() {
                break candidate;
            }
            want = want.wrapping_add(1);
            assert_ne!(want, 0, "reference scan exhausted the nonce space");
        };
        assert_eq!(nonce, want, "same first valid nonce");
        assert_eq!(header, want_header, "byte-identical header");
        assert!(
            nonce >= super::MINE_NONCE_QUANTUM,
            "the fixture must cross a quantum boundary to be meaningful: {nonce}"
        );
    }

    /// RUB-1186 checkpoint 2, proved INSIDE the search rather than around it.
    /// The poll answers `false` on its first invocation and `true` afterwards,
    /// so only a checkpoint at the entry of the SECOND quantum can end the
    /// search. The exact cancellation result plus `polls == 2` makes that
    /// second-quantum entry non-vacuous. A search polled only once, at entry,
    /// cannot latch at all and runs to a mined header instead. The
    /// bounded `recv_timeout` keeps a mutant that removes the in-loop poll from
    /// pinning the suite when its target is unsatisfiable.
    #[test]
    fn mine_header_nonce_polls_at_every_quantum_not_only_at_search_entry() {
        use rubin_consensus::constants::POW_LIMIT;

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let searching = std::thread::spawn(move || {
            let mut hard = POW_LIMIT;
            hard[0] = 0x00;
            hard[1] = 0x00;
            let prefix = make_header_prefix([3u8; 32], [4u8; 32], 1_777_000_000, hard);
            let polls = AtomicU64::new(0);
            let poll = |_: MineCheckpoint| polls.fetch_add(1, Ordering::SeqCst) > 0;
            let mut cancel = MineCancelObserver::new(&poll);
            let outcome = mine_header_nonce(&prefix, hard, &mut cancel);
            let _ = done_tx.send(());
            (outcome, polls.load(Ordering::SeqCst), cancel.observed())
        });
        done_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("an in-search checkpoint must end the search");
        let (outcome, polls, observed) = searching.join().expect("search thread");
        assert_eq!(
            outcome.expect_err("the second quantum entry cancels"),
            MINE_CANCELLED
        );
        assert!(observed, "the latch records an ACTUAL checkpoint");
        assert_eq!(polls, 2, "polled at every quantum entry, not once");
    }

    /// RUB-1186: a never-observing observer keeps ordinary `mine_one` behaviour
    /// for every existing caller, and the latch stays clear when no checkpoint
    /// ever observed — the precondition `/mine_next` classification relies on.
    #[test]
    fn mine_one_without_observation_keeps_baseline_behaviour() {
        use crate::sync::stock_devnet_engine_for_test;

        let (mut sync, dir) = stock_devnet_engine_for_test("rubin-mn-nocancel", |_| {});
        let height_before = sync.chain_state.height;
        let never = |_: MineCheckpoint| false;
        let mut cancel = MineCancelObserver::new(&never);
        let mined = {
            let mut miner = Miner::new(&mut sync, None, MinerConfig::default()).expect("miner");
            miner.mine_one_observed(&[], &mut cancel).expect("mine one")
        };
        assert!(!cancel.observed(), "no checkpoint observed");
        assert_eq!(mined.height, height_before + 1, "apply advanced the tip");
        assert_eq!(
            sync.chain_state.tip_hash, mined.hash,
            "tip is the new block"
        );
        drop(sync);
        fs::remove_dir_all(&dir).expect("cleanup");
    }
}
