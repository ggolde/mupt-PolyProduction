"""Unit tests for next-unit placement strategies, independent of any chain builder"""

import pytest

import numpy as np
from scipy.spatial.transform import Rotation

from mupt.builders.random_walk import SelfAvoidingWalk
from mupt.builders.unit_placement import (
    Junction,
    SingleBondPlacer,
    TwoBondPlacer,
    UnitPlacement,
)
from mupt.mupr.primitives import Primitive
from mupt.tests.pim_units import (
    TFTPN_SMILES,
    TTSBI_SMILES,
    pim_ladder_chain,
    unit_with_ports,
)

from .random_walk_golden import GOLDEN_PLACEMENTS
from .walk_helpers import (
    GOLDEN_CHAIN_SEED,
    GOLDEN_SCENARIOS,
    synthetic_chains,
)


def chain_layout(primitive: Primitive) -> tuple[list, list[Junction]]:
    """Path and junctions of the single chain in a Primitive"""
    (path,) = SelfAvoidingWalk.chain_paths(primitive)
    return path, SelfAvoidingWalk.chain_junctions(primitive, path)


def place_chain(
    placer, primitive: Primitive, initial_point, rng
) -> list[UnitPlacement]:
    """Drive a placer through a whole chain, accepting every proposal"""
    path, junctions = chain_layout(primitive)
    placer.check_preconditions(primitive, path, junctions)
    placer.prepare(primitive, path, junctions)
    placements = [placer.place_first(np.asarray(initial_point, dtype=float), rng=rng)]
    for depth in range(1, len(path)):
        placements.append(placer.place_next(depth, placements[-1], rng=rng))
    return placements


def junction_bond_lengths(
    primitive: Primitive, placements: list[UnitPlacement]
) -> np.ndarray:
    """Anchor-anchor distances across every bond between consecutive placed units"""
    transforms = {placement.handle: placement.transform for placement in placements}
    path, junctions = chain_layout(primitive)
    lengths = []
    for junction in junctions:
        for from_conn_handle, to_conn_handle in junction.connector_pairs:
            from_anchor = transforms[junction.from_handle].apply(
                primitive.fetch_connector_on_child(
                    junction.from_handle, from_conn_handle
                ).anchor.position
            )
            to_anchor = transforms[junction.to_handle].apply(
                primitive.fetch_connector_on_child(
                    junction.to_handle, to_conn_handle
                ).anchor.position
            )
            lengths.append(np.linalg.norm(from_anchor - to_anchor))
    return np.array(lengths)


@pytest.fixture(scope="module")
def pim_trimer() -> Primitive:
    """TTSBI-TFTPN-TTSBI ladder trimer with embedded positions"""
    chain, _ = pim_ladder_chain(3)
    return chain


# SingleBondPlacer
def test_single_bond_placer_matches_golden() -> None:
    """Test stepping the placer by hand reproduces pre-patch placements"""
    walk_kwargs, rng_seed, chain_lengths = GOLDEN_SCENARIOS["generator_rng"]
    primitive, _ = synthetic_chains(chain_lengths, seed=GOLDEN_CHAIN_SEED)
    placer = SingleBondPlacer(
        bond_length=walk_kwargs["bond_length"],
        angle_max_rad=walk_kwargs["angle_max_rad"],
    )
    placements = place_chain(
        placer, primitive, np.zeros(3), np.random.default_rng(rng_seed)
    )

    golden = GOLDEN_PLACEMENTS["generator_rng"]
    assert [placement.handle for placement in placements] == [
        handle for handle, _ in golden
    ]
    for placement, (_, matrix) in zip(placements, golden):
        assert np.allclose(placement.transform.as_matrix(), matrix, atol=1e-12)


def test_single_bond_placer_steps_within_cone() -> None:
    """Test consecutive walk steps stay within the angle cone"""
    primitive, _ = synthetic_chains((8,), seed=0)
    placer = SingleBondPlacer(angle_max_rad=np.pi / 8)
    placements = place_chain(placer, primitive, np.zeros(3), np.random.default_rng(0))
    directions = [placement.state[1] for placement in placements]
    for previous, current in zip(directions[:-1], directions[1:]):
        assert np.dot(previous, current) >= np.cos(np.pi / 8) - 1e-12


def test_single_bond_placer_rejects_two_bond_junctions(pim_trimer) -> None:
    """Test the single-bond placer rejects ladder junctions"""
    path, junctions = chain_layout(pim_trimer)
    with pytest.raises(ValueError, match="exactly 1 bond"):
        SingleBondPlacer().check_preconditions(pim_trimer, path, junctions)


def test_placer_requires_shapes() -> None:
    """Test placers require every unit to have a shape"""
    chain, _ = pim_ladder_chain(2, embed_positions=False)
    for handle in chain.children_by_handle:
        chain.fetch_child(handle)._shape = None
    path, junctions = chain_layout(chain)
    with pytest.raises(TypeError):
        TwoBondPlacer().check_preconditions(chain, path, junctions)


@pytest.mark.parametrize("angle_max_rad", [-4.0, 4.0])
def test_single_bond_placer_invalid_angle(angle_max_rad: float) -> None:
    """Test maximum step angles outside [-pi, pi] are rejected"""
    with pytest.raises(ValueError):
        SingleBondPlacer(angle_max_rad=angle_max_rad)


@pytest.mark.parametrize("bond_length", [0.0, -1.0, np.inf])
def test_placer_invalid_bond_length(bond_length: float) -> None:
    """Test non-positive or infinite bond lengths are rejected"""
    with pytest.raises(ValueError):
        SingleBondPlacer(bond_length=bond_length)
    with pytest.raises(ValueError):
        TwoBondPlacer(bond_length=bond_length)


# TwoBondPlacer
def test_two_bond_placer_exact_bond_lengths(pim_trimer) -> None:
    """Test docked junctions have both bonds at exactly the target length"""
    placements = place_chain(
        TwoBondPlacer(bond_length=1.37),
        pim_trimer,
        np.zeros(3),
        np.random.default_rng(0),
    )
    assert np.allclose(junction_bond_lengths(pim_trimer, placements), 1.37, atol=1e-10)


def test_two_bond_placer_first_unit_centered(pim_trimer) -> None:
    """Test the first unit is centered on the initial point"""
    placer = TwoBondPlacer()
    path, junctions = chain_layout(pim_trimer)
    placer.prepare(pim_trimer, path, junctions)
    initial_point = np.array([3.0, -1.0, 2.0])
    first = placer.place_first(initial_point, rng=np.random.default_rng(1))
    centroid = pim_trimer.fetch_child(path[0]).shape.centroid
    assert np.allclose(first.transform.apply(centroid), initial_point)


def test_two_bond_placer_fixed_initial_rotation(pim_trimer) -> None:
    """Test a fixed initial rotation makes rigid placement deterministic"""
    rotation = Rotation.from_rotvec([0.1, 0.2, 0.3])
    placer = TwoBondPlacer(initial_rotation=rotation)
    runs = [
        place_chain(placer, pim_trimer, np.zeros(3), np.random.default_rng(seed))
        for seed in (0, 1)
    ]
    for placement_1, placement_2 in zip(*runs):
        assert np.allclose(
            placement_1.transform.as_matrix(), placement_2.transform.as_matrix()
        )


def test_two_bond_placer_flex_preserves_bonds(pim_trimer) -> None:
    """Test hinge flex changes placements without stretching bonds"""
    rotation = Rotation.identity()
    rigid = place_chain(
        TwoBondPlacer(initial_rotation=rotation), pim_trimer, np.zeros(3), None
    )
    flexed = place_chain(
        TwoBondPlacer(flex_angle_rad=0.3, initial_rotation=rotation),
        pim_trimer,
        np.zeros(3),
        np.random.default_rng(0),
    )
    assert np.allclose(junction_bond_lengths(pim_trimer, flexed), 1.37, atol=1e-10)
    assert not np.allclose(
        rigid[1].transform.as_matrix(), flexed[1].transform.as_matrix()
    )


def test_two_bond_placer_is_stochastic() -> None:
    """Test which depths are reported as worth retrying"""
    assert TwoBondPlacer().is_stochastic(0)
    assert not TwoBondPlacer().is_stochastic(1)
    assert TwoBondPlacer(flex_angle_rad=0.1).is_stochastic(1)
    assert not TwoBondPlacer(initial_rotation=Rotation.identity()).is_stochastic(0)


def test_two_bond_placer_rejects_single_bond_junctions() -> None:
    """Test the two-bond placer rejects single-bond junctions"""
    primitive, _ = synthetic_chains((3,))
    path, junctions = chain_layout(primitive)
    with pytest.raises(ValueError, match="exactly 2 bond"):
        TwoBondPlacer().check_preconditions(primitive, path, junctions)


def test_two_bond_placer_rejects_degenerate_connectors() -> None:
    """Test Connectors without bond vectors cannot be docked"""
    chain, handles = pim_ladder_chain(2)
    path, junctions = chain_layout(chain)
    conn_handle = junctions[0].connector_pairs[0][0]
    conn = chain.fetch_connector_on_child(handles[0], conn_handle)
    conn.linker.position = conn.anchor.position.copy()
    with pytest.raises(ValueError, match="distinct anchor and linker"):
        TwoBondPlacer().check_preconditions(chain, path, junctions)


def test_mixed_junctions_rejected_by_both_placers() -> None:
    """Test chains mixing single and double junctions are rejected"""
    ttsbi, ports_ttsbi = unit_with_ports(TTSBI_SMILES, embed_positions=True)
    tftpn, ports_tftpn = unit_with_ports(TFTPN_SMILES, embed_positions=True)
    chain = Primitive(label="mixed")
    handles = [chain.attach_child(unit.copy()) for unit in (ttsbi, tftpn, ttsbi)]
    chain.connect_children(handles[0], ports_ttsbi[3], handles[1], ports_tftpn[5])
    chain.connect_children(handles[0], ports_ttsbi[4], handles[1], ports_tftpn[6])
    chain.connect_children(handles[1], ports_tftpn[7], handles[2], ports_ttsbi[1])

    path, junctions = chain_layout(chain)
    for placer in (SingleBondPlacer(), TwoBondPlacer()):
        with pytest.raises(ValueError, match="mixing junction types"):
            placer.check_preconditions(chain, path, junctions)
