package node

import (
	"errors"

	"github.com/2tbmz9y2xt-lang/rubin-protocol/clients/go/consensus"
)

// RelayAdmissionDisposition is the closed classification of ONE remote-mempool
// admission outcome, published for relay-side download and representation
// state. The enum is exactly the eleven values below, and every production exit
// of Mempool.AddRemoteTxForRelay selects one of them AT the branch that decided
// the outcome, before any compatibility mapping onto TxAdmitErrorKind — except
// for the fail-closed RelayAdmissionInternal override a failed REQUIRED
// post-decision cleanup applies over that selection (see RelayAdmissionInternal).
//
// A consumer MUST NOT infer a disposition from TxAdmitError.Kind, from Error(),
// from message text, from the HTTP status mapping, or from a fallback default:
// those are lossy compatibility views. TxAdmitConflict alone carries both the
// DUPLICATE and the CONFLICT branch, and TxAdmitUnavailable carries the floor,
// capacity, sequence, dependency-free unavailability and internal branches.
//
// The zero value is deliberately NOT a member of the enum, so a value that no
// branch selected can never be mistaken for a classification.
type RelayAdmissionDisposition uint8

const (
	// RelayAdmissionRetained means the exact candidate representation is
	// resident in the mempool after the call.
	RelayAdmissionRetained RelayAdmissionDisposition = iota + 1
	// RelayAdmissionStableTerminalReject is an explicit canonical, structural,
	// consensus, or constructor-frozen context-bound static-policy rejection.
	// It is the ONLY disposition that may carry cache-authorizing context
	// evidence, and only when that exact context was proven for this call.
	//
	// A policy lane OUTCOME whose verdict depends on state the published context
	// does NOT pin is therefore not one of these, however static its
	// configuration looks: a CORE_SIMPLICITY pre-activation outcome decided from
	// a deployment provider reads a published deployment set that a
	// {StableTip, Generation} context does not bind, so it is UNAVAILABLE. The
	// classification is per outcome, not per lane — the same lane's
	// well-formedness rejection reads no deployment set and stays here
	// (see relayDispositionForSimplicityPreActivationOutcome).
	RelayAdmissionStableTerminalReject
	// RelayAdmissionDuplicate means a resident txid or wtxid duplicate.
	RelayAdmissionDuplicate
	// RelayAdmissionConflict means an exact existing pending-outpoint owner
	// conflict.
	RelayAdmissionConflict
	// RelayAdmissionMissingDependency means a referenced input is absent or not
	// spendable yet. It is retryable and is never stable-terminal: the same
	// bytes can admit at a later height.
	RelayAdmissionMissingDependency
	// RelayAdmissionRollingFloor is the retryable rolling-relay-fee-floor
	// outcome.
	RelayAdmissionRollingFloor
	// RelayAdmissionCapacity is the retryable capacity outcome.
	RelayAdmissionCapacity
	// RelayAdmissionAdmissionSequence is the admission-sequence outcome.
	RelayAdmissionAdmissionSequence
	// RelayAdmissionUnavailable covers an absent chain, owner or admission
	// context, an active canonical transition, a stale expected context,
	// retryable runtime unavailability, and a CORE_SIMPLICITY deployment state
	// this node could not determine from the admission inputs it has — a
	// consensus error carrying TxErrorCauseSimplicityDeployment{Unavailable,
	// EvidenceInvalid, InactiveProvider}, whose public code and message say only
	// "not active" while the deployment set the answer came from is not pinned
	// by the published context. An uncaused TX_ERR_SIG_ALG_INVALID under an untrusted native-suite policy snapshot is also unavailable.
	// The frozen no-provider state is NOT one of these: it reads no set and stays a stable terminal rejection.
	RelayAdmissionUnavailable
	// RelayAdmissionInternal covers an impossible invariant, a retained
	// identity mismatch, accounting corruption, an adapter contract violation,
	// and a process-local crypto-backend fault — a consensus error carrying
	// consensus.TxErrorCauseLocalCryptoBackendFault, whose public code says
	// nothing about whether this node could decide at all.
	//
	// That last class is INTERNAL rather than UNAVAILABLE because it is THIS
	// process failing closed, not a missing admission input the node is waiting
	// on; either way it is not stable invalidity of these bytes, so neither is
	// cache-authorizing.
	//
	// It is also the ONE disposition a branch does not have to select: a
	// REQUIRED post-decision cleanup step that reports a fault — pending-outpoint
	// owner accounting corruption, a claim inconsistency — publishes INTERNAL and
	// OUTRANKS the disposition the deciding branch already selected, because that
	// branch's verdict is no longer the whole truth about the call. The public
	// error still reports the branch's outcome unchanged, so a consumer that
	// wants the corruption signal MUST read the disposition.
	RelayAdmissionInternal
	// RelayAdmissionCancelled is a closed enum value ONLY. This surface
	// implements no cancellation framework and no production branch selects it.
	RelayAdmissionCancelled
)

var relayAdmissionDispositionNames = [...]string{
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

// String renders the closed enum name. A value outside the enum — including the
// zero value that means "no branch selected" — renders as UNSELECTED and is
// never a classification.
func (d RelayAdmissionDisposition) String() string {
	if int(d) < len(relayAdmissionDispositionNames) && relayAdmissionDispositionNames[d] != "" {
		return relayAdmissionDispositionNames[d]
	}
	return "UNSELECTED"
}

// RelayAdmissionResult is the immutable result of one relay admission. It is
// returned by value and holds no reference the caller can mutate back into the
// mempool.
//
// TxID and WTxID are the identities the producer PARSED from the candidate
// bytes; they are the zero hash when the candidate never parsed canonically.
//
// AdmissionContext is meaningful ONLY when HasAdmissionContext is true, and is
// the zero context otherwise. HasAdmissionContext is set only for
// RelayAdmissionStableTerminalReject, and only when the caller-supplied
// expected context was proven equal to the exact complete stable admission
// context this call validated against. Nothing else authorizes a relay
// representation cache.
//
// Disposition is the deciding branch's own selection, EXCEPT when a REQUIRED
// post-decision cleanup step reported a fault: that publishes
// RelayAdmissionInternal and outranks the branch's selection, while Err keeps
// reporting the branch's outcome byte-identically. Disposition is therefore the
// only field that can report an accounting corruption the cleanup layer found.
//
// Err is the unchanged public admission error the equivalent AddRemoteTx call
// returns: same type, same TxAdmitErrorKind, same message. It is nil exactly on
// the success exit, whose disposition is RelayAdmissionRetained — or, if the
// published record somehow failed to be the exact candidate, the
// retained-identity-mismatch RelayAdmissionInternal that a nil error can also
// carry.
type RelayAdmissionResult struct {
	Disposition         RelayAdmissionDisposition
	TxID                [32]byte
	WTxID               [32]byte
	AdmissionContext    PendingOutpointAdmissionContext
	HasAdmissionContext bool
	Err                 error
}

// AddRemoteTxForRelay admits a transaction received from a peer through EXACTLY
// the AddRemoteTx implementation, and additionally publishes the
// producer-selected relay disposition, the parsed candidate identity, and — for
// a proven stable terminal rejection only — the exact stable admission context
// a relay representation cache needs.
//
// expected is the caller's own observed pending-outpoint admission context. A
// nil expected context admits exactly as AddRemoteTx does, never sets
// HasAdmissionContext, and never authorizes caching. A NON-nil expected context
// is COPIED BY VALUE at this boundary and never dereferenced again: this call
// compares against, and publishes evidence for, the snapshot taken here, so a
// caller that mutates the pointee afterwards — from this goroutine or any
// other — can change neither this call's admission behavior nor its published
// evidence. The pointer is a presence flag only; the memory stays the caller's.
//
// The snapshot is validated at the existing owner seam
// (PendingOutpointOwner.Reserve), after the baseline parse, consensus, policy,
// cheap-floor and duplicate precedence: a stale, superseded or mismatched
// context is UNAVAILABLE there, before the owner scans for a conflict and
// before it consumes a sequence.
//
// It classifies without mutating: every classification input is either the
// producing branch's own decision or a pure read. Public errors, messages,
// admission counters, validation order and mutation order are untouched.
//
// Nil-safe like every other exported Mempool accessor: a nil receiver returns
// RelayAdmissionUnavailable carrying the legacy "nil mempool" TxAdmitUnavailable
// error, publishes no identity and no context, and moves no counter.
func (m *Mempool) AddRemoteTxForRelay(txBytes []byte, expected *PendingOutpointAdmissionContext) RelayAdmissionResult {
	probe := newRelayAdmissionProbe(expected)
	err := m.addTxWithSource(txBytes, mempoolTxSourceRemote, probe)
	return probe.result(err)
}

// newRelayAdmissionProbe is the ONLY constructor of a probe carrying an expected
// context. It copies the caller-owned context by value, so the pointer never
// escapes this function and no later read of the probe can observe a mutation
// the caller made after entry. A nil expected context leaves hasExpected false,
// which is the legacy binding every seam already keys off.
func newRelayAdmissionProbe(expected *PendingOutpointAdmissionContext) *relayAdmissionProbe {
	probe := &relayAdmissionProbe{}
	if expected != nil {
		probe.expected = *expected
		probe.hasExpected = true
	}
	return probe
}

// relayAdmissionProbe is the per-call producer sink for the relay evidence a
// plain error cannot carry: the candidate identity the producer parsed, the
// exact stable admission context this call ran against, and the selection made
// on the success exit. Exactly one admission call owns one probe.
//
// A nil probe is the legacy AddTx / AddRemoteTx / AddReorgTx path: every method
// here is nil-safe, so that path allocates nothing and does nothing extra.
//
// The disposition of a FAILED admission deliberately does not travel here. It is
// selected at its originating branch and carried by the admission error itself
// (selectRelayDisposition), so no helper signature, error, message, counter or
// ordering had to change to make a branch classifiable.
//
// internalFailure is the ONE exception, and it is not a branch selection: a
// REQUIRED post-decision cleanup step ran AFTER its branch already tagged the
// error, so first-selection-wins leaves the cleanup layer no way to reclassify.
// It records the fault here instead and result() applies it as a fail-closed
// override.
//
// expected is a VALUE snapshot taken once by newRelayAdmissionProbe, never a
// caller-owned pointer: every seam below reads memory this call owns, so the
// context proven, compared and published stays the one supplied at entry.
// hasExpected — not a sentinel value — distinguishes "no expected context"
// from a legitimately zero one.
type relayAdmissionProbe struct {
	expected      PendingOutpointAdmissionContext
	hasExpected   bool
	observed      PendingOutpointAdmissionContext
	contextProven bool
	// success is the disposition of the nil-error exit only.
	success RelayAdmissionDisposition
	txid    [32]byte
	wtxid   [32]byte
	// internalFailure records that a REQUIRED post-decision cleanup step
	// reported a fault. It is never a branch selection; result() turns it into
	// the fail-closed INTERNAL override.
	internalFailure bool
}

// noteCleanupFailure records a REQUIRED post-decision cleanup fault — an owner
// accounting inconsistency, a claim inconsistency, or an impossible invariant
// that only the cleanup step can observe. err is the cleanup step's own result:
// a nil err records nothing, so the ordinary successful-cleanup path is
// untouched and keeps the disposition its branch selected.
//
// It changes no error, message, kind, counter, ordering or mutation: the caller
// still returns its unchanged admission cause. A nil probe is the legacy
// AddTx / AddRemoteTx / AddReorgTx path and this is a no-op there.
func (p *relayAdmissionProbe) noteCleanupFailure(err error) {
	if p == nil || err == nil {
		return
	}
	p.internalFailure = true
}

// noteIdentity records the identity the producer parsed from the candidate
// bytes. It is called once, immediately after the canonical parse succeeds, so
// every outcome decided from that point on reports the real candidate identity
// and an unparseable candidate reports none.
func (p *relayAdmissionProbe) noteIdentity(txid, wtxid [32]byte) {
	if p == nil {
		return
	}
	p.txid = txid
	p.wtxid = wtxid
}

// checkExpectedContext validates a caller-supplied expected admission context
// against the owner AT the seam, without creating an owner and without any
// other mutation. It exists for the candidate shape that claims no outpoint and
// therefore never reaches Reserve, so no candidate shape can bypass the check.
//
// tip is the ChainState tip the caller's admission read guard is pinning. It
// performs three checks, in this order, each of them UNAVAILABLE:
//
//  1. the owner publishes a complete admission context at all;
//  2. that context's stable tip equals tip;
//  3. that context equals the caller's expected context.
//
// Checks 1 and 2 are the SAME two the input-bearing sibling
// (reserveEntryInputsLocked) runs before Reserve, with the same messages in the
// same order. Check 3 stands in for the two refusals Reserve would raise from
// the supplied context — "pending-outpoint expected tip mismatch" and
// "pending-outpoint expected generation mismatch" — but reports both through ONE
// combined message, so a consumer cannot tell those two apart here.
//
// Reserve's remaining refusals do not apply to a candidate that claims no
// outpoint: it reserves nothing and takes no token, so the token-sequence
// exhaustion refusal cannot bear on its outcome; the conflict scan has no
// outpoint of its own to find a live claim for; and Reserve's empty-input-set
// rejection describes exactly this candidate shape, which is why it must not be
// routed through Reserve at all. A transition in progress is not missing either:
// check 1 already reports unavailable while one is active, and the whole
// admission call holds the ChainState admission read guard a transition needs
// for write, so none can begin between this seam and the decision it feeds.
//
// It is a no-op for the legacy path (nil probe) and for a nil expected context,
// so an input-less candidate keeps its exact baseline behavior there.
func (p *relayAdmissionProbe) checkExpectedContext(owner *PendingOutpointOwner, tip PendingOutpointTip) error {
	if p == nil || !p.hasExpected {
		return nil
	}
	observed, ok := owner.AdmissionContext()
	if !ok {
		return selectRelayDisposition(txAdmitUnavailable("pending-outpoint owner admission context unavailable"), RelayAdmissionUnavailable)
	}
	if observed.StableTip != tip {
		return selectRelayDisposition(txAdmitUnavailable("pending-outpoint owner tip does not match the guarded chainstate tip"), RelayAdmissionUnavailable)
	}
	if observed != p.expected {
		return selectRelayDisposition(txAdmitUnavailable("pending-outpoint expected admission context mismatch"), RelayAdmissionUnavailable)
	}
	return nil
}

// result assembles the immutable published result from the producer's own
// selections. It reads no error message and no error kind.
//
// A recorded REQUIRED-cleanup fault OUTRANKS the branch-selected disposition:
// the branch decided the outcome before the cleanup ran, and first-selection-wins
// keeps it from reclassifying, so the override lives here. It publishes INTERNAL
// and, since INTERNAL is not a stable terminal rejection, no cache-authorizing
// context.
func (p *relayAdmissionProbe) result(err error) RelayAdmissionResult {
	if p == nil {
		// The legacy path allocates no probe. It publishes no identity and no
		// context, and a nil error there is an unselected outcome, which
		// relayDispositionOf reports as INTERNAL, fail closed.
		return RelayAdmissionResult{Disposition: relayDispositionOf(err), Err: err}
	}
	out := RelayAdmissionResult{TxID: p.txid, WTxID: p.wtxid, Err: err}
	switch {
	case err != nil:
		out.Disposition = relayDispositionOf(err)
	case p.success != 0:
		out.Disposition = p.success
	default:
		// Unreachable: the success exit always selects. Fail closed rather than
		// publish an unselected value as a classification.
		out.Disposition = RelayAdmissionInternal
	}
	if p.internalFailure {
		// Fail-closed override, applied AFTER the selection it outranks: the
		// branch's verdict is no longer the whole truth about this call.
		out.Disposition = RelayAdmissionInternal
	}
	// Only an explicit stable terminal rejection carries cache-authorizing
	// context, and only when that exact context was proven for this call.
	if out.Disposition == RelayAdmissionStableTerminalReject && p.contextProven {
		out.AdmissionContext = p.observed
		out.HasAdmissionContext = true
	}
	return out
}

// bindRelayAdmissionContext proves ONCE per relay call that the owner's exact
// complete admission context is available, agrees with the guarded chainstate
// tip, and equals the caller-supplied expected context.
//
// It is a pure read: it returns no error, decides no admission outcome, and
// mutates nothing — in particular it never creates an owner. A context it
// cannot prove simply leaves contextProven false, so no branch's error, message
// or ordering can depend on it.
//
// The caller holds ChainState.admissionMu for read. Every owner generation and
// stable-tip move happens inside a canonical transition that holds the SAME
// guard for write (beginCanonicalTransition through finish/abort), and the
// chainstate tip only moves under that guard too. The context proven here is
// therefore the exact one every later decision of this call runs against,
// including the decisions that return before the owner seam.
func (m *Mempool) bindRelayAdmissionContext(probe *relayAdmissionProbe) {
	if probe == nil || !probe.hasExpected {
		return
	}
	m.mu.RLock()
	owner := m.pendingOutpoints
	m.mu.RUnlock()
	observed, ok := owner.AdmissionContext()
	if !ok || observed.StableTip != pendingOutpointTipOf(m.chainState) || observed != probe.expected {
		return
	}
	probe.observed = observed
	probe.contextProven = true
}

// noteRetainedLocked selects the success-exit disposition. RETAINED is the
// contract's exact statement — the exact candidate representation is resident
// after the call — so the residency is READ from the live index rather than
// assumed from a nil error. A mismatch is the retained-identity-mismatch
// invariant and is INTERNAL. The read mutates nothing; the caller holds m.mu.
func (m *Mempool) noteRetainedLocked(entry *mempoolEntry, probe *relayAdmissionProbe) {
	if probe == nil {
		return
	}
	probe.noteIdentity(entry.txid, entry.wtxid)
	if resident, ok := m.txs[entry.txid]; ok && resident == entry {
		probe.success = RelayAdmissionRetained
		return
	}
	probe.success = RelayAdmissionInternal
}

// selectRelayDisposition records, ON the admission error the branch just built,
// the relay disposition that SAME branch selected for it. It changes no error
// type, kind, message, HTTP mapping or counter behavior: the field is
// unexported and no public mapping reads it.
//
// The FIRST selection wins, so an outer caller that classifies a whole failure
// class cannot overwrite the more precise disposition an inner branch already
// selected. err is returned unchanged — including nil and non-TxAdmitError
// values — so a branch wraps its existing return expression in place.
func selectRelayDisposition(err error, disposition RelayAdmissionDisposition) error {
	var admitErr *TxAdmitError
	if errors.As(err, &admitErr) && admitErr.disposition == 0 {
		admitErr.disposition = disposition
	}
	return err
}

// relayDispositionOf reads the disposition the PRODUCING branch stored on err.
// It never inspects Kind, Error(), message text or an HTTP mapping. An
// admission error no branch classified is an impossible invariant and is
// reported INTERNAL, fail closed — never as a stable terminal rejection.
func relayDispositionOf(err error) RelayAdmissionDisposition {
	var admitErr *TxAdmitError
	if errors.As(err, &admitErr) && admitErr.disposition != 0 {
		return admitErr.disposition
	}
	return RelayAdmissionInternal
}

// relayDispositionForOwnerError selects the disposition of a pending-outpoint
// owner failure from the OWNER's own typed kind, not from the compatibility
// TxAdmitErrorKind that txAdmitFromPendingOutpointError folds the owner's
// Unavailable and Internal kinds together into.
func relayDispositionForOwnerError(err error) RelayAdmissionDisposition {
	var ownerErr *PendingOutpointError
	if !errors.As(err, &ownerErr) {
		return RelayAdmissionInternal
	}
	switch ownerErr.Kind {
	case PendingOutpointConflict:
		return RelayAdmissionConflict
	case PendingOutpointUnavailable:
		return RelayAdmissionUnavailable
	default:
		return RelayAdmissionInternal
	}
}

// relayDispositionForSimplicityPreActivation selects the CORE_SIMPLICITY
// pre-activation disposition from the ONE thing that decides whether the
// published admission context is complete for this lane: whether a deployment
// provider governs the verdict at all.
//
// With a provider present EVERY outcome — the lookup failure, an unobtainable
// or only partially known set, and an inactive one alike — is UNAVAILABLE and
// never cache-authorizing. consensus.SimplicityActiveAtHeight reads the
// provider by documented pure I/O, and consensus.SimplicityDeploymentSetAnchor
// binds that set to (chain_id, descriptors) and to no block or tip, while a
// PendingOutpointAdmissionContext pins only {StableTip, Generation}. The set can
// therefore differ at the same tip and generation, so the context is not
// complete with respect to this decision. A descriptor's ActivationHeight makes
// a pre-ACTIVE rejection flip to valid by advancing height anyway — the same
// property for which TX_ERR_TIMELOCK_NOT_MET and TX_ERR_COINBASE_IMMATURE are
// classified retryable rather than stable terminal.
//
// With no provider the rotation implements no deployment surface: the lane does
// no I/O and reads no descriptor set, so the verdict rests entirely on the
// constructor-frozen configuration the context does pin, and stays stable
// terminal. That is the default node wiring, which publishes no provider.
func relayDispositionForSimplicityPreActivation(rotation consensus.RotationProvider) RelayAdmissionDisposition {
	if _, ok := rotation.(consensus.SimplicityDeploymentProvider); ok {
		return RelayAdmissionUnavailable
	}
	return RelayAdmissionStableTerminalReject
}

// relayDispositionForSimplicityPreActivationOutcome narrows the lane-wide rule
// above to the ONE outcome that actually consulted the deployment set, using the
// (reject, err) tuple rejectCoreSimplicityPreActivation already returns as the
// discriminator — never its message text, and never a second probe of the
// provider.
//
// reject == true is the deployment-set-dependent half of the lane, and it is
// exactly the half the provider governs: the lookup failure returns
// (true, reason, err) and the pre-ACTIVE verdict returns (true, reason, nil).
// Both are decided from what the provider published — or, with no provider,
// from the constructor-frozen configuration — so they take the lane-wide rule.
//
// reject == false with a non-nil err is the covenant well-formedness rejection
// consensus.ValidateTxCovenantsGenesis returned against a rotation shim that
// FORCES Simplicity active. That verdict never reads the provider's set: it is a
// function of the candidate bytes, the chain id, the pinned next-block height
// and the frozen live artifact hashes — every one of which the published
// {StableTip, Generation} context does pin or the candidate itself carries. It
// is therefore context-complete and stays stable terminal whether or not a
// provider is configured, so the same malformed bytes are cacheable exactly as
// they are on the default no-provider wiring.
//
// The remaining tuple, (false, "", nil), is the accept exit and reaches no
// caller of this function.
func relayDispositionForSimplicityPreActivationOutcome(reject bool, rotation consensus.RotationProvider) RelayAdmissionDisposition {
	if !reject {
		return RelayAdmissionStableTerminalReject
	}
	return relayDispositionForSimplicityPreActivation(rotation)
}

// relayDispositionForPolicyError classifies the static-policy fan-out
// (applyPolicyAgainstState). Every term it applies is a constructor-frozen
// decision over pinned inputs — anchor outputs and the DA fee/budget terms —
// EXCEPT two outcomes that arrive as typed wrappers
// and are read by TYPE, never by message:
//
//   - the CORE_SIMPLICITY pre-activation lane's provider-DECIDED outcomes, which
//     are UNAVAILABLE because the published context does not pin the deployment
//     set they depend on. That lane's well-formedness rejection depends on no
//     such set, arrives unwrapped, and classifies with the pinned-input terms.
//   - the fan-out's own impossible invariant, the nil checked transaction no
//     caller can produce, which is INTERNAL and so never cache-authorizing.
func relayDispositionForPolicyError(err error) RelayAdmissionDisposition {
	var retryable *simplicityPreActivationRetryableError
	if errors.As(err, &retryable) {
		return RelayAdmissionUnavailable
	}
	var impossible *policyImpossibleInvariantError
	if errors.As(err, &impossible) {
		return RelayAdmissionInternal
	}
	return RelayAdmissionStableTerminalReject
}

// relayDispositionForInputError separates the outcomes that are NOT stable
// invalidity of these exact bytes from the branch's own non-dependency
// selection, reading only typed values the producer set — never Error(), Msg,
// backend text, or an OpenSSL error string.
//
// It reads the typed CAUSE first, because the cause outranks the code. Three families need it:
//
//   - A process-local crypto-backend fault is published under a public code that
//     also carries genuine candidate invalidity (TX_ERR_PARSE for a latched
//     initialization failure, TX_ERR_SIG_INVALID for a verifier that could not
//     decide, TX_ERR_SIG_ALG_INVALID for a binding this node could not resolve
//     from its own configuration). That error is evidence about THIS process, so
//     it is INTERNAL — never a stable terminal rejection, and therefore never
//     cache-authorizing. A candidate that selected an unsupported or
//     unauthorized suite carries no such cause and keeps its baseline
//     classification here; relayDispositionForConsensusError may downgrade an uncaused native-suite failure under an untrusted snapshot.
//   - A CORE_SIMPLICITY deployment rejection collapses five distinct states onto
//     TX_ERR_COVENANT_TYPE_INVALID and two fixed messages. Three of them —
//     provider unavailability, invalid provider evidence, and inactivity
//     DECIDED FROM a published deployment set — rest on an admission input the
//     {StableTip, Generation} context does not pin, so they are UNAVAILABLE and
//     never cache-authorizing: the same bytes admit unchanged once that set
//     answers differently. Corrupted evidence is UNAVAILABLE, not INTERNAL,
//     because the activation state simply cannot be computed from a required
//     admission input; INTERNAL stays reserved for an impossible internal state,
//     an unknown non-zero cause, or a producer-result contract violation. A
//     cause a WRAPPER lost is deliberately NOT in that set: losing a cause is
//     observationally TxErrorCauseUnspecified, and an unspecified cause
//     preserves every existing ErrorCode-based mapping. Only the
//     no-provider-configured state rests entirely on constructor-frozen
//     configuration and stays a stable terminal rejection, with cache authority
//     still governed by the caller's own exact-context proof.
//   - A non-0xF0 CORE_SIMPLICITY witness is candidate-intrinsic and policy-independent, so it stays STABLE_TERMINAL_REJECT.
//
// ORDERING NOTE — structural, and deliberately NOT claimed by any test. Reading
// the cause before the code switch is a PAIRWISE fact, and on today's inputs the
// two orders are observationally identical: the closed cause set is
// {LocalCryptoBackendFault, SimplicityDeploymentUnavailable,
// SimplicityDeploymentEvidenceInvalid, SimplicityDeploymentInactiveProvider,
// SimplicityDeploymentInactiveFrozen, SimplicityWitnessSuiteInvalid}, every producer of those causes emits
// TX_ERR_PARSE, TX_ERR_SIG_INVALID, TX_ERR_SIG_ALG_INVALID or
// TX_ERR_COVENANT_TYPE_INVALID, and none of those codes is a member of the
// switch set below. No error can therefore carry a cause AND a switch-set code,
// so no test can separate the two orders — the order is load-bearing only for a
// FUTURE producer whose cause and code disagree, and none exists today.
//
// The cause switch is closed in BOTH directions. TxErrorCauseUnspecified is an
// explicit arm that falls through to the code fallback below; every other value
// this consumer does not enumerate takes the default arm and becomes INTERNAL,
// so a cause added upstream without its arm here fails closed instead of
// publishing a cache-authorizing stable terminal rejection. Nothing reaches that
// arm today — the cause set is closed and every member has an arm.
//
// Only then does the CODE fallback apply: an input that is absent, still
// immature, or not yet unlocked becomes spendable at a later height, so it must
// never be published as a stable terminal rejection either.
//
// nonDependency is the disposition the calling branch selects for everything
// else it can produce: stable terminal invalidity for the consensus validation
// seam, an impossible invariant for the policy input-snapshot seam.
func relayDispositionForInputError(err error, nonDependency RelayAdmissionDisposition) RelayAdmissionDisposition {
	var txErr *consensus.TxError
	if errors.As(err, &txErr) {
		switch txErr.Cause() {
		case consensus.TxErrorCauseUnspecified:
			// No producing branch selected a cause (or a wrapper lost one):
			// the code fallback below decides, exactly as it did before.
		case consensus.TxErrorCauseLocalCryptoBackendFault:
			return RelayAdmissionInternal
		case consensus.TxErrorCauseSimplicityDeploymentUnavailable,
			consensus.TxErrorCauseSimplicityDeploymentEvidenceInvalid,
			consensus.TxErrorCauseSimplicityDeploymentInactiveProvider:
			return RelayAdmissionUnavailable
		case consensus.TxErrorCauseSimplicityDeploymentInactiveFrozen, consensus.TxErrorCauseSimplicityWitnessSuiteInvalid:
			return RelayAdmissionStableTerminalReject
		default:
			return RelayAdmissionInternal
		}
		switch txErr.Code {
		case consensus.TX_ERR_MISSING_UTXO, consensus.TX_ERR_COINBASE_IMMATURE, consensus.TX_ERR_TIMELOCK_NOT_MET:
			return RelayAdmissionMissingDependency
		}
	}
	return nonDependency
}
