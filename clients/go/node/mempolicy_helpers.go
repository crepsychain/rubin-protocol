package node

import (
	"errors"
	"fmt"

	"github.com/2tbmz9y2xt-lang/rubin-protocol/clients/go/consensus"
)

// validateChainSnapshot validates chain state snapshot and extracts next height
func validateChainSnapshot(snapshot *chainStateAdmissionSnapshot) (uint64, error) {
	if snapshot == nil {
		return 0, selectRelayDisposition(txAdmitUnavailable("nil chainstate"), RelayAdmissionUnavailable)
	}
	nextHeight, _, err := nextBlockContextFromFields(snapshot.hasTip, snapshot.height, snapshot.tipHash)
	if err != nil {
		return 0, selectRelayDisposition(txAdmitUnavailable(err.Error()), RelayAdmissionUnavailable)
	}
	return nextHeight, nil
}

// buildPolicyInputSnapshotIfNeeded (mempool_precheck.go) replaced the old preparePolicyUtxos; callers use it directly.

// validateTransactionWithConsensus performs consensus validation with configured profiles
func (m *Mempool) validateTransactionWithConsensus(
	txBytes []byte,
	tx *consensus.Tx,
	txid [32]byte,
	wtxid [32]byte,
	snapshot *chainStateAdmissionSnapshot,
	nextHeight uint64,
	blockMTP uint64,
	policy MempoolConfig,
) (*consensus.CheckedTransaction, error) {
	checked, err := consensus.CheckParsedTransactionWithOwnedUtxoSetAndSuiteContext(
		txBytes,
		tx,
		consensus.ParsedTxIDs{TxID: txid, WTxID: wtxid},
		snapshot.utxos,
		nextHeight,
		blockMTP,
		m.chainID,
		consensus.SuiteValidationContext{
			Rotation: policy.RotationProvider,
			Registry: policy.SuiteRegistry,
			SigCache: m.sigCache,
		},
	)
	if err != nil {
		// Classified from the ORIGINAL consensus error, before the public wrapper
		// drops its typed code and cause, exactly as the standard producer does.
		return nil, selectRelayDisposition(txAdmitRejected(err.Error()), relayDispositionForConsensusError(err, policy))
	}
	return checked, nil
}

// extractTxInputs extracts outpoints from checked transaction
func extractTxInputs(checked *consensus.CheckedTransaction) []consensus.Outpoint {
	inputs := make([]consensus.Outpoint, 0, len(checked.Tx.Inputs))
	for _, in := range checked.Tx.Inputs {
		inputs = append(inputs, consensus.Outpoint{Txid: in.PrevTxid, Vout: in.PrevVout})
	}
	return inputs
}

// applyPolicyAgainstStateDA handles DA fee policy application
func applyPolicyAgainstStateDA(checked *consensus.CheckedTransaction, policy MempoolConfig, utxos map[consensus.Outpoint]consensus.UtxoEntry) error {
	if err := rejectDaCommitDeclaredBudget(checked.Tx, policy.PolicyMaxDaBytesPerBlock); err != nil {
		return err
	}
	// Stage C DA fee policy: only enter the helper for DA-bearing tx when
	// the DA-side floor is configured (MinDaFeeRate > 0) or a per-byte
	// surcharge applies. Non-DA tx skip the helper entirely.
	//
	// The zero currentMempoolMinFeeRate below collapses
	// max(relay_fee_floor, da_required_fee) to da_required_fee, so this
	// helper enforces the DA-specific terms only and never reports a
	// rolling-floor verdict. That split still matters for every caller that
	// reaches it with a DA transaction: standard Add* admission now refuses
	// a policy-valid DA at rejectNonStandardKindLocked, downstream of these
	// terms, while RelayMetadata and the miner's rejectCandidate keep their
	// own unchanged floor checks against the live rolling floor.
	if checked.DaBytes > 0 && (policy.MinDaFeeRate > 0 || policy.PolicyDaSurchargePerByte > 0) {
		reject, _, reason, err := RejectDaAnchorTxPolicy(
			checked.Tx,
			utxos,
			0,
			policy.MinDaFeeRate,
			policy.PolicyDaSurchargePerByte,
		)
		if err != nil {
			return txAdmitRejected(fmt.Sprintf("%s: %v", reason, err))
		}
		if reject {
			return txAdmitRejected(reason)
		}
	}
	return nil
}

func rejectDaCommitDeclaredBudget(tx *consensus.Tx, maxDaBytesPerBlock uint64) error {
	if tx == nil || tx.TxKind != 0x01 || tx.DaCommitCore == nil {
		return nil
	}
	declaredDaBytes, err := mulU64NoOverflow(uint64(tx.DaCommitCore.ChunkCount), consensus.CHUNK_BYTES)
	if err != nil {
		return fmt.Errorf("DA declared chunk budget overflow (chunk_count=%d chunk_bytes=%d): %w", tx.DaCommitCore.ChunkCount, consensus.CHUNK_BYTES, err)
	}
	if maxDaBytesPerBlock > 0 && declaredDaBytes > maxDaBytesPerBlock {
		return fmt.Errorf("DA declared chunk budget exceeded (declared_da_bytes=%d max_da_bytes=%d chunk_count=%d chunk_bytes=%d)", declaredDaBytes, maxDaBytesPerBlock, tx.DaCommitCore.ChunkCount, consensus.CHUNK_BYTES)
	}
	return nil
}

// policyImpossibleInvariantError marks an applyPolicyAgainstState outcome that
// no candidate property can produce — a record shape the live admission path
// never builds. Error() returns the wrapped message verbatim, so every caller
// that renders, wraps or compares this error stays byte-identical to baseline;
// the wrapper is a TYPE signal read only by the relay classifier, never parsed
// from text. It is the policy fan-out's form of the impossible-invariant tagging
// the locked admission helpers do inline with selectRelayDisposition.
type policyImpossibleInvariantError struct{ err error }

func (e *policyImpossibleInvariantError) Error() string { return e.err.Error() }

func (e *policyImpossibleInvariantError) Unwrap() error { return e.err }

// simplicityPreActivationRetryableError marks a CORE_SIMPLICITY pre-activation
// policy outcome DECIDED FROM a deployment provider, which the published
// admission context cannot pin. Error() returns the wrapped message
// verbatim, so every caller that renders, wraps or compares this error stays
// byte-identical to baseline; the wrapper is a TYPE signal read only by the
// relay classifier, never parsed from text.
type simplicityPreActivationRetryableError struct{ err error }

func (e *simplicityPreActivationRetryableError) Error() string { return e.err.Error() }

func (e *simplicityPreActivationRetryableError) Unwrap() error { return e.err }

// markSimplicityPreActivation wraps err ONLY when this outcome is the retryable
// provider-decided one. reject is the SAME tuple-shape discriminator the
// precheck call site applies (relayDispositionForSimplicityPreActivationOutcome):
// reject == true is the half a deployment provider governs, while the
// reject == false error is the covenant well-formedness verdict, which reads no
// deployment set and is context-complete. The no-provider verdict rests on the
// constructor-frozen configuration. Both stay unwrapped, hence stable terminal.
func markSimplicityPreActivation(err error, reject bool, rotation consensus.RotationProvider) error {
	if relayDispositionForSimplicityPreActivationOutcome(reject, rotation) != RelayAdmissionUnavailable {
		return err
	}
	return &simplicityPreActivationRetryableError{err: err}
}

func applyPolicyAgainstStateSimplicity(checked *consensus.CheckedTransaction, utxos map[consensus.Outpoint]consensus.UtxoEntry, chainID [32]byte, nextHeight uint64, policy MempoolConfig) error {
	if policy.PolicyRejectSimplicityPreActivation {
		reject, reason, err := rejectCoreSimplicityPreActivation(checked.Tx, utxos, chainID, nextHeight, policy.RotationProvider)
		if err != nil {
			return markSimplicityPreActivation(err, reject, policy.RotationProvider)
		}
		if reject {
			return markSimplicityPreActivation(errors.New(reason), true, policy.RotationProvider)
		}
	}
	return nil
}

// applyPolicyAgainstStateAnchor handles non-coinbase anchor output policy application
func applyPolicyAgainstStateAnchor(checked *consensus.CheckedTransaction, policy MempoolConfig) error {
	if policy.PolicyRejectNonCoinbaseAnchorOutputs {
		reject, reason, err := RejectNonCoinbaseAnchorOutputs(checked.Tx)
		if err != nil {
			return err
		}
		if reject {
			return errors.New(reason)
		}
	}
	return nil
}
