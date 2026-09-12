package node

import (
	"errors"

	"github.com/2tbmz9y2xt-lang/rubin-protocol/clients/go/consensus"
)

func nativeSuiteRelayContextTrusted(policy MempoolConfig) bool {
	switch rotation := policy.RotationProvider.(type) {
	case nil:
		return policy.SuiteRegistry == nil
	case consensus.DefaultRotationProvider:
		return policy.SuiteRegistry == nil || policy.SuiteRegistry.IsCanonicalDefaultLiveManifest()
	case *consensus.DefaultRotationProvider:
		return rotation != nil && (policy.SuiteRegistry == nil || policy.SuiteRegistry.IsCanonicalDefaultLiveManifest())
	case consensus.DescriptorRotationProvider:
		return rotation.Descriptor.Validate(policy.SuiteRegistry) == nil
	default:
		return false
	}
}

func relayDispositionForConsensusError(err error, policy MempoolConfig) RelayAdmissionDisposition {
	disposition := relayDispositionForInputError(err, RelayAdmissionStableTerminalReject)
	var txErr *consensus.TxError
	if disposition == RelayAdmissionStableTerminalReject && errors.As(err, &txErr) && txErr.Cause() == consensus.TxErrorCauseUnspecified && txErr.Code == consensus.TX_ERR_SIG_ALG_INVALID && !nativeSuiteRelayContextTrusted(policy) {
		return RelayAdmissionUnavailable
	}
	return disposition
}

// buildPolicyInputSnapshotIfNeeded returns the immutable pre-validation
// snapshot of only the transaction inputs that policy lanes inspect.
// Built before CheckTransaction*WithOwnedUtxoSet; that helper owns supplied utxos and removes spent inputs during validation. Extracted from checkTransactionWithSnapshot to keep
// cyclomatic complexity within the repository's lint budget.
func buildPolicyInputSnapshotIfNeeded(parsedTx *consensus.Tx, snapshot *chainStateAdmissionSnapshot, policy MempoolConfig) (map[consensus.Outpoint]consensus.UtxoEntry, error) {
	needs, err := policyNeedsInputSnapshotForTx(parsedTx, policy)
	if err != nil {
		// Only a nil parsed transaction reaches here, which the callers cannot
		// produce: an impossible invariant, never a candidate property.
		return nil, selectRelayDisposition(txAdmitRejected(err.Error()), RelayAdmissionInternal)
	}
	if !needs {
		return nil, nil
	}
	policyUtxos, err := policyInputSnapshot(parsedTx, snapshot.utxos)
	if err != nil {
		// An input the snapshot does not carry is a retryable dependency gap;
		// the remaining shapes (nil tx, nil utxo set) are impossible invariants.
		return nil, selectRelayDisposition(txAdmitRejected(err.Error()), relayDispositionForInputError(err, RelayAdmissionInternal))
	}
	return policyUtxos, nil
}

// checkTxParseAndContext resolves the chain-context inputs every
// admission needs (next block height + MTP) and parses the candidate
// transaction in canonical-bytes mode. Extracted from
// checkTransactionWithSnapshot to keep cyclomatic complexity within
// the repository's lint budget. Returns the parsed Tx, next-block
// height, next-block MTP, or a typed admission error if any step
// fails (Unavailable for chain-context failure, Rejected for parse
// failure / trailing bytes).
func (m *Mempool) checkTxParseAndContext(txBytes []byte, snapshot *chainStateAdmissionSnapshot, probe *relayAdmissionProbe) (*consensus.Tx, uint64, uint64, error) {
	if snapshot == nil {
		return nil, 0, 0, selectRelayDisposition(txAdmitUnavailable("nil chainstate"), RelayAdmissionUnavailable)
	}
	nextHeight, _, err := nextBlockContextFromFields(snapshot.hasTip, snapshot.height, snapshot.tipHash)
	if err != nil {
		return nil, 0, 0, selectRelayDisposition(txAdmitUnavailable(err.Error()), RelayAdmissionUnavailable)
	}
	blockMTP, err := m.nextBlockMTP(nextHeight)
	if err != nil {
		return nil, 0, 0, selectRelayDisposition(txAdmitUnavailable(err.Error()), RelayAdmissionUnavailable)
	}
	parsedTx, txid, wtxid, consumed, err := consensus.ParseTx(txBytes)
	if err != nil {
		return nil, 0, 0, selectRelayDisposition(txAdmitRejected(err.Error()), RelayAdmissionStableTerminalReject)
	}
	if consumed != len(txBytes) {
		return nil, 0, 0, selectRelayDisposition(txAdmitRejected("trailing bytes after canonical tx"), RelayAdmissionStableTerminalReject)
	}
	// The canonical identity exists from here on, so every later outcome
	// reports the real candidate and an unparseable candidate reports none.
	probe.noteIdentity(txid, wtxid)
	return parsedTx, nextHeight, blockMTP, nil
}

// checkTransactionWithSnapshot validates a transaction against a consistent
// owned admission snapshot plus an immutable mempool policy snapshot.
//
// `snappedFloor` is the rolling-relay-floor value snapped ONCE in the
// caller (`addTxWithSource`) before either the precheck or the locked
// admission path runs. The precheck uses this snapped value directly
// (wave-6); the locked admission path enforces
// `max(snappedFloor, m.currentMinFeeRateLocked())` (wave-8) so newer
// HIGHER floors raised by `raiseMinFeeRateAfterEvictionLocked` win. The admission
// guard excludes canonical-transition publication until this call returns.
func (m *Mempool) checkTransactionWithSnapshot(txBytes []byte, snapshot *chainStateAdmissionSnapshot, policy MempoolConfig, snappedFloor uint64, probe *relayAdmissionProbe) (*consensus.CheckedTransaction, []consensus.Outpoint, error) {
	parsedTx, nextHeight, blockMTP, err := m.checkTxParseAndContext(txBytes, snapshot, probe)
	if err != nil {
		return nil, nil, err
	}
	// Only plain P2PK candidates use the cheap floor reject. Transactions
	// that may hit DA, CORE_ANCHOR, CORE_SIMPLICITY, or missing-UTXO policy lanes
	// keep the existing validation and policy-error precedence below.
	// Wave-4 (PR #1422): pass nextHeight + policy.RotationProvider so the
	// precheck can defer on consensus-invalid P2PK output shapes
	// (value==0, wrong covenant_data length, suite outside native create
	// set) that ValidateTxCovenantsGenesis would return Rejected
	// (terminal). Without these passes a below-floor + malformed tx
	// would be misclassified as transient Unavailable instead of
	// Rejected (terminal).
	if err := cheapFeeFloorPrecheck(parsedTx, snapshot, snappedFloor, nextHeight, policy.RotationProvider, policy.SuiteRegistry); err != nil {
		// The cheap precheck produces exactly one failure: below the rolling
		// floor. Every shape it cannot decide defers to the slow path above.
		return nil, nil, selectRelayDisposition(err, RelayAdmissionRollingFloor)
	}
	policyUtxos, err := buildPolicyInputSnapshotIfNeeded(parsedTx, snapshot, policy)
	if err != nil {
		return nil, nil, err
	}
	if policy.PolicyRejectSimplicityPreActivation {
		// This lane has TWO error exits and they are discriminated by the SHAPE of
		// the tuple the helper already returns, never by its message text.
		//
		// reject == true (with or without err) is the deployment-set-dependent
		// half: the lookup failure and the pre-ACTIVE verdict alike are decided
		// from what a provider published, and a {StableTip, Generation} context
		// does not pin that set — so with a provider present they are UNAVAILABLE,
		// never cache-authorizing, and with no provider they rest on the
		// constructor-frozen configuration and stay stable terminal.
		//
		// reject == false with a non-nil err is the covenant well-formedness
		// rejection, evaluated against a rotation shim that forces Simplicity
		// active: it consults no deployment set at all, only the candidate, the
		// chain id, the pinned height and the frozen live artifact hashes. Its
		// context is complete, so it stays stable terminal even with a provider
		// configured.
		//
		// The returned error, its message and its kind are unchanged in every case.
		reject, reason, err := rejectCoreSimplicityPreActivation(parsedTx, policyUtxos, m.chainID, nextHeight, policy.RotationProvider)
		if err != nil {
			return nil, nil, selectRelayDisposition(txAdmitRejected(err.Error()), relayDispositionForSimplicityPreActivationOutcome(reject, policy.RotationProvider))
		}
		if reject {
			return nil, nil, selectRelayDisposition(txAdmitRejected(reason), relayDispositionForSimplicityPreActivationOutcome(true, policy.RotationProvider))
		}
	}
	checked, err := consensus.CheckTransactionWithOwnedUtxoSetAndValidationContext(
		txBytes,
		snapshot.utxos,
		nextHeight,
		blockMTP,
		m.chainID,
		// Same validation, same order, same errors: the only difference from
		// the positional wrapper is that the live AddTx path carries this
		// Mempool's positive signature cache into the suite-aware seam.
		consensus.SuiteValidationContext{
			Rotation: policy.RotationProvider,
			Registry: policy.SuiteRegistry,
			SigCache: m.sigCache,
		},
	)
	if err != nil {
		// An input that is absent, still immature, or not yet unlocked becomes
		// spendable at a later height and stays retryable; every other
		// consensus failure is stable terminal invalidity of these exact bytes
		// unless its typed cause or untrusted native-suite policy selects otherwise.
		return nil, nil, selectRelayDisposition(txAdmitRejected(err.Error()), relayDispositionForConsensusError(err, policy))
	}
	if err := m.applyPolicyAgainstState(checked, nextHeight, policyUtxos, policy); err != nil {
		// Constructor-frozen, context-bound static policy: anchor outputs, the
		// DA fee/budget terms and Simplicity
		// pre-activation. The one state-availability outcome this fan-out can
		// produce — an undetermined CORE_SIMPLICITY deployment state — is typed,
		// and is UNAVAILABLE rather than stable terminal.
		return nil, nil, selectRelayDisposition(txAdmitRejected(err.Error()), relayDispositionForPolicyError(err))
	}
	inputs := extractTxInputs(checked)
	return checked, inputs, nil
}
