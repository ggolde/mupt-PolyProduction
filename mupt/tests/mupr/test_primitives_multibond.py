"""Unit tests for Primitives whose children are joined by multiple bonds (ladders)"""

import pytest

from mupt.mupr.primitives import Primitive, PrimitiveHandle, BijectionError
from mupt.mupr.connection import ConnectorHandle
from mupt.tests.pim_units import TTSBI_SMILES, TFTPN_SMILES, unit_with_ports


@pytest.fixture(scope="module")
def ttsbi() -> tuple[Primitive, dict[int, ConnectorHandle]]:
    """TTSBI repeat unit and the map from its port numbers to Connector handles"""
    return unit_with_ports(TTSBI_SMILES)


@pytest.fixture(scope="module")
def tftpn() -> tuple[Primitive, dict[int, ConnectorHandle]]:
    """TFTPN repeat unit and the map from its port numbers to Connector handles"""
    return unit_with_ports(TFTPN_SMILES)


@pytest.fixture
def ladder_trimer(ttsbi, tftpn) -> tuple[Primitive, list[PrimitiveHandle]]:
    """TTSBI-TFTPN-TTSBI chain, with two bonds across each junction"""
    A, ports_A = ttsbi
    B, ports_B = tftpn

    chain = Primitive(label="pim")
    handles = [chain.attach_child(unit.copy()) for unit in (A, B, A)]
    chain.connect_children(handles[0], ports_A[3], handles[1], ports_B[5])
    chain.connect_children(handles[0], ports_A[4], handles[1], ports_B[6])
    chain.connect_children(handles[1], ports_B[7], handles[2], ports_A[1])
    chain.connect_children(handles[1], ports_B[8], handles[2], ports_A[2])

    return chain, handles


@pytest.fixture
def single_bond_dimer(ttsbi, tftpn) -> tuple[Primitive, list[PrimitiveHandle]]:
    """TTSBI-TFTPN chain, with only one bond across the junction"""
    A, ports_A = ttsbi
    B, ports_B = tftpn

    chain = Primitive(label="dimer")
    handles = [chain.attach_child(unit.copy()) for unit in (A, B)]
    chain.connect_children(handles[0], ports_A[3], handles[1], ports_B[5])

    return chain, handles


# consistency and counting
def test_ladder_self_consistent(ladder_trimer) -> None:
    """Test a ladder trimer passes all self-consistency checks"""
    chain, _ = ladder_trimer
    chain.check_self_consistent()


def test_ladder_counts(ladder_trimer) -> None:
    """Test a ladder trimer has one edge but two bonds per junction"""
    chain, _ = ladder_trimer
    assert chain.topology.number_of_edges() == 2
    assert chain.num_internal_connections == 4
    assert chain.functionality == 4


def test_ladder_neighbors(ladder_trimer) -> None:
    """Test neighbors are counted once each, while bonds are counted individually"""
    chain, handles = ladder_trimer
    assert chain.neighbor_handles(handles[1]) == {handles[0], handles[2]}
    assert chain.num_internal_connections_on_child(handles[1]) == 4


def test_ladder_connections_by_pairs(ladder_trimer) -> None:
    """Test both bonds of each junction are grouped under the pair of children"""
    chain, handles = ladder_trimer
    by_pairs = chain.internal_connections_by_pairs
    assert set(by_pairs.keys()) == {
        frozenset((handles[0], handles[1])),
        frozenset((handles[1], handles[2])),
    }
    assert all(len(conn_pairs) == 2 for conn_pairs in by_pairs.values())


def test_ladder_copy_self_consistent(ladder_trimer) -> None:
    """Test copies of a ladder remain self-consistent"""
    chain, _ = ladder_trimer
    chain_copy = chain.copy()
    chain_copy.check_self_consistent()
    assert chain_copy.num_internal_connections == 4
    assert chain_copy.topology.number_of_edges() == 2


def test_edge_without_connection_inconsistent(ttsbi, tftpn) -> None:
    """Edges in the topology must still be backed by at least one connection"""
    A, _ = ttsbi
    B, _ = tftpn
    chain = Primitive(label="unbonded")
    handles = [chain.attach_child(unit.copy()) for unit in (A, B)]
    chain.adjoin_child_nodes(*handles)
    with pytest.raises(BijectionError):
        chain.check_self_consistent()


# fetching connections between children
def test_internal_connections_between(ladder_trimer, ttsbi, tftpn) -> None:
    """Test all bonds between two children are returned, oriented by argument order"""
    chain, handles = ladder_trimer
    _, ports_A = ttsbi
    _, ports_B = tftpn

    forward = chain.internal_connections_between(handles[0], handles[1])
    assert set(forward) == {(ports_A[3], ports_B[5]), (ports_A[4], ports_B[6])}

    backward = chain.internal_connections_between(handles[1], handles[0])
    assert set(backward) == {(ports_B[5], ports_A[3]), (ports_B[6], ports_A[4])}


def test_internal_connections_between_unconnected(ladder_trimer) -> None:
    """Test unconnected children have no connections between them"""
    chain, handles = ladder_trimer
    assert chain.internal_connections_between(handles[0], handles[2]) == []


def test_internal_connection_between_raises_for_multibond(ladder_trimer) -> None:
    """Test the singular lookup refuses to pick one of several bonds"""
    chain, handles = ladder_trimer
    with pytest.raises(ValueError):
        chain.internal_connection_between(handles[0], handles[1])


def test_internal_connection_between_single_bond(
    single_bond_dimer, ttsbi, tftpn
) -> None:
    """Test the singular lookup is unchanged for single bonds"""
    chain, handles = single_bond_dimer
    _, ports_A = ttsbi
    _, ports_B = tftpn
    chain.check_self_consistent()
    assert chain.internal_connection_between(handles[0], handles[1]) == (
        ports_A[3],
        ports_B[5],
    )
    assert chain.internal_connection_between(handles[1], handles[0]) == (
        ports_B[5],
        ports_A[3],
    )


def test_internal_connection_between_unconnected(ladder_trimer) -> None:
    """Test the singular lookup returns an empty tuple for unconnected children"""
    chain, handles = ladder_trimer
    assert chain.internal_connection_between(handles[0], handles[2]) == tuple()


# disconnection and detachment
def test_disconnect_multibond_pair(ladder_trimer) -> None:
    """Test disconnecting a junction frees both bonds and removes the edge"""
    chain, handles = ladder_trimer
    chain.disconnect_children(handles[0], handles[1])

    assert not chain.topology.has_edge(handles[0], handles[1])
    assert chain.internal_connections_between(handles[0], handles[1]) == []
    assert chain.num_internal_connections == 2
    assert chain.num_external_connectors_on_child(handles[0]) == 4
    assert chain.num_external_connectors_on_child(handles[1]) == 2
    assert chain.functionality == 8
    chain.check_self_consistent()


def test_disconnect_single_bond_pair(single_bond_dimer) -> None:
    """Test disconnecting a single bond removes the edge too"""
    chain, handles = single_bond_dimer
    chain.disconnect_children(handles[0], handles[1])

    assert chain.topology.number_of_edges() == 0
    assert chain.num_internal_connections == 0
    chain.check_self_consistent()


def test_detach_multibonded_child(ladder_trimer) -> None:
    """Test detaching a doubly-bonded child frees all of its bonds"""
    chain, handles = ladder_trimer
    chain.detach_child(handles[1])

    assert chain.num_children == 2
    assert chain.num_internal_connections == 0
    assert chain.topology.number_of_edges() == 0
    assert chain.functionality == 8
    chain.check_self_consistent()


# expansion
def test_expand_middle_unit(ladder_trimer, tftpn) -> None:
    """Expanding one side of a multi-bond junction splits it into single-bond edges"""
    chain, handles = ladder_trimer
    B, _ = tftpn
    chain.expand(handles[1])

    chain.check_self_consistent()
    assert chain.num_children == 2 + B.num_children
    assert chain.num_internal_connections == 4 + B.num_internal_connections
    assert chain.topology.number_of_edges() == 4 + B.num_internal_connections


def test_expand_ladder_to_atoms(ladder_trimer, ttsbi, tftpn) -> None:
    """Test expanding a ladder to atoms re-forms every bond between atoms"""
    chain, handles = ladder_trimer
    A, _ = ttsbi
    B, _ = tftpn
    for handle in handles:
        chain.expand(handle)

    chain.check_self_consistent()
    num_bonds_expected = 2 * A.num_internal_connections + B.num_internal_connections + 4
    assert chain.num_children == 2 * A.num_children + B.num_children
    assert chain.num_internal_connections == num_bonds_expected
    assert chain.topology.number_of_edges() == num_bonds_expected
    assert chain.functionality == 4
