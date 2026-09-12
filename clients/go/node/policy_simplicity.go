package node

import (
	"fmt"

	"github.com/2tbmz9y2xt-lang/rubin-protocol/clients/go/consensus"
)

func rejectCoreSimplicityPreActivation(
	tx *consensus.Tx,
	utxos map[consensus.Outpoint]consensus.UtxoEntry,
	chainID [32]byte,
	height uint64,
	rotation consensus.RotationProvider,
) (reject bool, reason string, err error) {
	kind := covenantPolicyKind(tx, utxos, consensus.COV_TYPE_CORE_SIMPLICITY)
	if kind == "" {
		return false, "", nil
	}
	active := false
	if provider, ok := rotation.(consensus.SimplicityDeploymentProvider); ok {
		active, err = consensus.SimplicityActiveAtHeight(chainID, height, provider)
		if err != nil {
			return true, "CORE_SIMPLICITY deployment lookup failure",
				fmt.Errorf("CORE_SIMPLICITY deployment lookup failure: %w", err)
		}
	}
	if active {
		return false, "", nil
	}
	if rotation == nil {
		rotation = consensus.DefaultRotationProvider{}
	}
	if err := consensus.ValidateTxCovenantsGenesis(tx, chainID, height, activeSimplicityGenesisRotation{chainID: chainID, RotationProvider: rotation}); err != nil {
		return false, "", err
	}
	return true, fmt.Sprintf("CORE_SIMPLICITY %s pre-ACTIVE", kind), nil
}

func covenantPolicyKind(tx *consensus.Tx, utxos map[consensus.Outpoint]consensus.UtxoEntry, covenantType uint16) string {
	if tx == nil {
		return ""
	}
	for _, out := range tx.Outputs {
		if out.CovenantType == covenantType {
			return "output"
		}
	}
	if utxos == nil {
		return ""
	}
	for _, in := range tx.Inputs {
		op := consensus.Outpoint{Txid: in.PrevTxid, Vout: in.PrevVout}
		if utxos[op].CovenantType == covenantType {
			return "spend"
		}
	}
	return ""
}

type activeSimplicityGenesisRotation struct {
	chainID [32]byte
	consensus.RotationProvider
}

// PublishedSimplicityDeployments forces CORE_SIMPLICITY active by publishing the
// live valid single-descriptor set for the node's chain, so the pre-activation
// well-formedness check passes the deployment-active gate. This is a policy shim,
// not a real descriptor source (the live descriptor set is wired later).
func (r activeSimplicityGenesisRotation) PublishedSimplicityDeployments() ([]consensus.SimplicityDeploymentDescriptor, [32]byte, bool, error) {
	set := []consensus.SimplicityDeploymentDescriptor{consensus.LiveSimplicityDeploymentDescriptor(r.chainID)}
	anchor, err := consensus.SimplicityDeploymentSetAnchor(r.chainID, set)
	if err != nil {
		return nil, [32]byte{}, false, err
	}
	return set, anchor, true, nil
}
