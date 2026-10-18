"""Unit tests for sequential (optionally self-avoiding) random walk chain builders"""

import pytest

from typing import Optional

import numpy as np
import networkx as nx
from scipy.spatial.transform import RigidTransform
from rdkit import Chem

from mupt.builders.random_walk import (
    AngleConstrainedRandomWalk,
    SelfAvoidanceError,
    SelfAvoidingWalk,
)
from mupt.builders.unit_placement import (
    SingleBondPlacer,
    TwoBondPlacer,
    UnitPlacement,
    UnitPlacer,
)
from mupt.geometry.collision import SpatialHashIndex, minimum_image_displacement
from mupt.interfaces.rdkit import primitive_to_rdkit
from mupt.mupr.primitives import Primitive
from mupt.tests.pim_units import pim_ladder_chain

from .random_walk_golden import GOLDEN_PLACEMENTS
from .walk_helpers import (
    GOLDEN_CHAIN_SEED,
    GOLDEN_GLOBAL_SEED,
    GOLDEN_SCENARIOS,
    HEAD_CONN,
    TAIL_CONN,
    synthetic_chains,
    synthetic_unit,
)


def apply_placements(primitive: Primitive, placements) -> None:
    """Rigidly transform each child of a Primitive by its placement"""
    for handle, transform in placements:
        primitive.fetch_child(handle).rigidly_transform(transform)


def leaf_sites(primitive: Primitive) -> np.ndarray:
    """Centroids of all leaves beneath a Primitive"""
    return np.array([leaf.shape.centroid for leaf in primitive.leaves])


def min_pair_distance(sites: np.ndarray, box_lengths=None) -> float:
    """Smallest distance between any two distinct sites (optionally periodic)"""
    displacements = sites[:, None, :] - sites[None, :, :]
    if box_lengths is not None:
        displacements = minimum_image_displacement(
            displacements, np.asarray(box_lengths)
        )
    distances = np.linalg.norm(displacements, axis=-1)
    return distances[np.triu_indices(len(sites), k=1)].min()


def junction_bond_lengths(chain: Primitive, handles) -> np.ndarray:
    """Anchor-anchor distances across all bonds between consecutive units"""
    return np.array([
        np.linalg.norm(
            chain.fetch_connector_on_child(handle_1, conn_handle_1).anchor.position
            - chain.fetch_connector_on_child(handle_2, conn_handle_2).anchor.position
        )
        for handle_1, handle_2 in zip(handles[:-1], handles[1:])
        for conn_handle_1, conn_handle_2 in chain.internal_connections_between(
            handle_1, handle_2
        )
    ])


# regression against pre-patch behavior
@pytest.mark.parametrize("scenario", sorted(GOLDEN_SCENARIOS))
def test_angle_constrained_walk_matches_golden(scenario: str) -> None:
    """Test legacy-signature walks reproduce pre-patch placements"""
    walk_kwargs, rng_seed, chain_lengths = GOLDEN_SCENARIOS[scenario]
    primitive, _ = synthetic_chains(chain_lengths, seed=GOLDEN_CHAIN_SEED)
    np.random.seed(GOLDEN_GLOBAL_SEED)
    rng = None if (rng_seed is None) else np.random.default_rng(rng_seed)
    placements = list(
        AngleConstrainedRandomWalk(**walk_kwargs, rng=rng).generate_placements(
            primitive
        )
    )

    golden = GOLDEN_PLACEMENTS[scenario]
    assert [handle for handle, _ in placements] == [handle for handle, _ in golden]
    for (_, transform), (_, matrix) in zip(placements, golden):
        # NOTE: not bit-identical, as the pre-patch walker perturbed anchor
        # positions by rounding error when realigning linkers before placement
        assert np.allclose(transform.as_matrix(), matrix, rtol=0.0, atol=1e-12)


def test_angle_constrained_walk_legacy_attributes() -> None:
    """Test the legacy walker exposes its pre-patch attributes"""
    direction = np.array([0.0, 1.0, 0.0])
    walker = AngleConstrainedRandomWalk(
        bond_length=2.0, angle_max_rad=0.5, initial_direction=direction
    )
    assert isinstance(walker.placer, SingleBondPlacer)
    assert walker.bond_length == 2.0
    assert walker.angle_max_rad == 0.5
    assert walker.initial_direction is direction
    assert not walker.is_self_avoiding


def test_walk_does_not_modify_primitive() -> None:
    """Test that generating placements leaves Connectors untouched"""
    primitive, handles = synthetic_chains((4,))
    before = {
        (handle, conn_handle): (
            conn.anchor.position.copy(),
            conn.linker.position.copy(),
        )
        for handle in handles[0]
        for conn_handle, conn in primitive.fetch_child(handle).connectors.items()
    }
    list(
        AngleConstrainedRandomWalk(rng=np.random.default_rng(0)).generate_placements(
            primitive
        )
    )
    for (handle, conn_handle), (anchor, linker) in before.items():
        conn = primitive.fetch_connector_on_child(handle, conn_handle)
        assert np.array_equal(conn.anchor.position, anchor)
        assert np.array_equal(conn.linker.position, linker)


# chain traversal
def test_chain_paths_follow_insertion_order() -> None:
    """Test chains are traversed in the order their units were added"""
    primitive, handles = synthetic_chains((4, 3, 1))
    assert SelfAvoidingWalk.chain_paths(primitive) == handles


def test_chain_paths_start_from_first_inserted_terminus() -> None:
    """Test traversal starts from the terminus added to the topology first"""
    primitive = Primitive(label="reversed")
    rng = np.random.default_rng(0)
    handles = [primitive.attach_child(synthetic_unit(rng)) for _ in range(3)]
    # bond middle-to-last first, so the topology lists the middle unit before the first
    primitive.connect_children(handles[1], TAIL_CONN, handles[2], HEAD_CONN)
    primitive.connect_children(handles[0], TAIL_CONN, handles[1], HEAD_CONN)
    assert SelfAvoidingWalk.chain_paths(primitive) == [handles]


def test_chain_paths_reject_branched() -> None:
    """Test that branched topologies are rejected"""
    primitive = Primitive(label="branched")
    primitive._topology = nx.star_graph(3)
    with pytest.raises(ValueError, match="branched"):
        SelfAvoidingWalk.chain_paths(primitive)


def test_chain_paths_reject_cyclic() -> None:
    """Test that cyclic topologies are rejected"""
    primitive = Primitive(label="cyclic")
    primitive._topology = nx.cycle_graph(4)
    with pytest.raises(ValueError, match="cyclic"):
        SelfAvoidingWalk.chain_paths(primitive)


def test_single_unit_chain() -> None:
    """Previously raised, since an isolated unit has no terminal (degree-1) node"""
    primitive, (handles,) = synthetic_chains((1,))
    initial_point = np.array([1.0, 2.0, 3.0])
    ((handle, transform),) = AngleConstrainedRandomWalk(
        initial_point=initial_point, rng=np.random.default_rng(0)
    ).generate_placements(primitive)
    assert handle == handles[0]
    assert np.all(np.isfinite(transform.as_matrix()))


# ladder (two-bond) chains
@pytest.mark.parametrize("linkage", ["1-3/2-4", "1-4/2-3"])
def test_ladder_walk_exact_junctions(linkage: str) -> None:
    """Test every bond of a walked ladder has exactly the target length"""
    chain, handles = pim_ladder_chain(6, linkage)
    placements = list(
        SelfAvoidingWalk(
            TwoBondPlacer(bond_length=1.37), rng=np.random.default_rng(0)
        ).generate_placements(chain)
    )
    assert sorted(handle for handle, _ in placements) == sorted(handles)

    apply_placements(chain, placements)
    lengths = junction_bond_lengths(chain, handles)
    assert len(lengths) == 10
    assert np.allclose(lengths, 1.37, atol=1e-8)


def test_ladder_walk_is_rigid_without_flex() -> None:
    """Without flex, only the position and orientation of the whole chain is random"""
    shapes = []
    for seed in (0, 1):
        chain, handles = pim_ladder_chain(5)
        apply_placements(
            chain,
            SelfAvoidingWalk(
                TwoBondPlacer(), rng=np.random.default_rng(seed)
            ).generate_placements(chain),
        )
        sites = leaf_sites(chain)
        shapes.append(np.linalg.norm(sites[:, None] - sites[None], axis=-1))
    assert np.allclose(shapes[0], shapes[1], atol=1e-8)


def test_ladder_walk_reproducible() -> None:
    """Test seeded ladder walks are reproducible"""
    runs = []
    for _ in range(2):
        chain, _ = pim_ladder_chain(4)
        runs.append([
            transform.as_matrix()
            for _, transform in SelfAvoidingWalk(
                TwoBondPlacer(flex_angle_rad=0.2), rng=np.random.default_rng(5)
            ).generate_placements(chain)
        ])
    assert np.allclose(runs[0], runs[1])


def test_ladder_walk_exports_reasonable_bonds() -> None:
    """Test all bonds of an expanded, exported ladder (incl. junctions) are sane"""
    chain, handles = pim_ladder_chain(4)
    apply_placements(
        chain,
        SelfAvoidingWalk(
            TwoBondPlacer(bond_length=1.37),
            rng=np.random.default_rng(0),
            minimum_distance=1.0,
        ).generate_placements(chain),
    )
    for handle in handles:
        chain.expand(handle)

    with pytest.warns(DeprecationWarning):
        mol = primitive_to_rdkit(chain)
    positions = mol.GetConformer().GetPositions()
    heavy_bond_lengths = [
        np.linalg.norm(
            positions[bond.GetBeginAtomIdx()] - positions[bond.GetEndAtomIdx()]
        )
        for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() > 1
        and bond.GetEndAtom().GetAtomicNum() > 1
    ]
    assert 1.0 < min(heavy_bond_lengths) and max(heavy_bond_lengths) < 1.7
    assert Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False)).count(".") == 0


# self-avoidance
def test_self_avoiding_chains_respect_minimum_distance() -> None:
    """Test chains sharing a periodic index never come within the cutoff"""
    box_lengths = (12.0, 12.0, 12.0)
    index = SpatialHashIndex(cutoff=1.0, box_lengths=box_lengths)
    primitive, _ = synthetic_chains((10, 10, 10), seed=3)
    walker = AngleConstrainedRandomWalk(
        bond_length=1.5, rng=np.random.default_rng(0), collision_index=index
    )
    placements = list(walker.generate_placements(primitive))
    apply_placements(primitive, placements)

    assert index.num_points == len(placements) == 30
    assert len(walker.placed_sites) == 30
    assert min_pair_distance(leaf_sites(primitive), box_lengths) >= 1.0


def test_self_avoiding_ladder_respects_minimum_distance() -> None:
    """Test atoms of a self-avoiding ladder never come within the cutoff"""
    chain, _ = pim_ladder_chain(8)
    box_lengths = (30.0, 30.0, 30.0)
    apply_placements(
        chain,
        SelfAvoidingWalk(
            TwoBondPlacer(flex_angle_rad=0.2),
            rng=np.random.default_rng(0),
            minimum_distance=0.9,
            box_lengths=box_lengths,
        ).generate_placements(chain),
    )
    assert min_pair_distance(leaf_sites(chain), box_lengths) >= 0.9


def test_occupied_sites_respected() -> None:
    """Test fixed obstacles are avoided"""
    obstacles = np.stack(
        np.meshgrid(*3 * [np.arange(-8.0, 8.1, 4.0)]), axis=-1
    ).reshape(-1, 3)
    primitive, _ = synthetic_chains((8,), seed=4)
    walker = AngleConstrainedRandomWalk(
        bond_length=1.5,
        initial_point=np.array([2.0, 2.0, 2.0]),  # as far from obstacles as possible
        rng=np.random.default_rng(1),
        minimum_distance=1.0,
        occupied_sites=obstacles,
    )
    apply_placements(primitive, walker.generate_placements(primitive))
    sites = leaf_sites(primitive)
    distances = np.linalg.norm(sites[:, None] - obstacles[None], axis=-1)
    assert distances.min() >= 1.0


def test_failed_placement_rolls_back_index() -> None:
    """Test a failed walk leaves the shared collision index as it was"""
    box_lengths = (2.5, 2.5, 2.5)
    index = SpatialHashIndex(cutoff=1.0, box_lengths=box_lengths)
    index.add(np.array([[0.0, 0.0, 0.0]]))
    primitive, _ = synthetic_chains(
        (3, 30), seed=5
    )  # first chain may fit, second can't
    walker = AngleConstrainedRandomWalk(
        bond_length=1.5,
        rng=np.random.default_rng(0),
        collision_index=index,
        max_attempts_per_unit=5,
        max_backtracks=20,
    )
    with pytest.raises(SelfAvoidanceError):
        list(walker.generate_placements(primitive))
    assert index.num_points == 1


# search logic, exercised through a scripted placer with no chemistry involved
class ScriptedPlacer(UnitPlacer):
    """
    Places the unit at each depth at successive scripted points, i.e. the k-th
    proposal at a given depth translates that unit's centroid to the point
    scripts[depth][k % len(scripts[depth])]
    """

    bonds_per_junction = 1

    def __init__(
        self, scripts: dict[int, list], stochastic_depths: Optional[set] = None
    ):
        super().__init__(bond_length=1.0)
        self.scripts = {
            depth: np.asarray(points, dtype=float) for depth, points in scripts.items()
        }
        self.stochastic_depths = stochastic_depths
        self.calls: list[int] = []

    def prepare(self, primitive, path, junctions):
        """Record the path and template centroids of the chain"""
        self._path = path
        self._centroids = [
            primitive.fetch_child(handle).shape.centroid for handle in path
        ]
        self._proposals = {depth: 0 for depth in self.scripts}

    def _propose(self, depth: int) -> UnitPlacement:
        self.calls.append(depth)
        points = self.scripts[depth]
        point = points[self._proposals[depth] % len(points)]
        self._proposals[depth] += 1
        return UnitPlacement(
            handle=self._path[depth],
            transform=RigidTransform.from_translation(point - self._centroids[depth]),
        )

    def place_first(self, initial_point, rng=None):
        """Propose the next scripted placement of the first unit"""
        return self._propose(0)

    def place_next(self, depth, previous, rng=None):
        """Propose the next scripted placement of the unit at this depth"""
        return self._propose(depth)

    def is_stochastic(self, depth):
        """Whether retrying the unit at this depth is worthwhile"""
        return (self.stochastic_depths is None) or (depth in self.stochastic_depths)


def point_primitive(num_units: int) -> Primitive:
    """Chain of single-point units, so each unit has exactly one collision site"""
    primitive = Primitive(label="points")
    rng = np.random.default_rng(0)
    handles = []
    for _ in range(num_units):
        unit = synthetic_unit(rng, n_points=2)
        unit.shape.positions = unit.shape.positions.mean(axis=0, keepdims=True)
        handles.append(primitive.attach_child(unit))
    for handle_1, handle_2 in zip(handles[:-1], handles[1:]):
        primitive.connect_children(handle_1, TAIL_CONN, handle_2, HEAD_CONN)
    return primitive


def test_backtracking_retries_previous_unit() -> None:
    """Test a dead end triggers a retry of the preceding unit"""
    placer = ScriptedPlacer({
        0: [[0.0, 0.0, 0.0]],
        1: [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        # clashes with unit 1's first placement, but not its second
        2: [[2.5, 0.0, 0.0]],
    })
    walker = SelfAvoidingWalk(placer, minimum_distance=1.0, max_attempts_per_unit=2)
    placements = dict(walker.generate_placements(point_primitive(3)))

    assert placer.calls == [0, 1, 2, 2, 1, 2]
    final_positions = [transform.translation for transform in placements.values()]
    assert len(final_positions) == 3
    assert walker.collision_index.num_points == 3


def test_backtracking_skips_deterministic_units() -> None:
    """Test backtracking skips past units with only one possible placement"""
    placer = ScriptedPlacer(
        {
            0: [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            1: [[2.0, 0.0, 0.0]],
            2: [[4.0, 0.0, 0.0]],  # always clashes with the obstacle at 4.5
        },
        stochastic_depths={0},
    )
    walker = SelfAvoidingWalk(
        placer,
        minimum_distance=1.0,
        max_backtracks=1,
        occupied_sites=np.array([[4.5, 0.0, 0.0]]),
    )
    with pytest.raises(SelfAvoidanceError):
        # depth 2 always clashes with the obstacle, so after a single permitted
        # backtrack (straight to depth 0, skipping deterministic depth 1) we give up
        list(walker.generate_placements(point_primitive(3)))
    assert placer.calls == [0, 1, 2, 0, 1, 2]


def test_no_backtracks_allowed() -> None:
    """Test the walk gives up at the first dead end when backtracking is disabled"""
    placer = ScriptedPlacer({0: [[0.0, 0.0, 0.0]], 1: [[0.5, 0.0, 0.0]]})
    walker = SelfAvoidingWalk(
        placer, minimum_distance=1.0, max_attempts_per_unit=3, max_backtracks=0
    )
    with pytest.raises(SelfAvoidanceError):
        list(walker.generate_placements(point_primitive(2)))
    assert placer.calls == [0, 1, 1, 1]
    assert walker.collision_index.num_points == 0


def test_without_self_avoidance_first_proposals_accepted() -> None:
    """Test every first proposal is accepted without self-avoidance"""
    placer = ScriptedPlacer({0: [[0.0, 0.0, 0.0]], 1: [[0.0, 0.0, 0.0]]})
    placements = list(SelfAvoidingWalk(placer).generate_placements(point_primitive(2)))
    assert len(placements) == 2
    assert placer.calls == [0, 1]


# constructor validation
def test_minimum_distance_cannot_exceed_bond_length() -> None:
    """Test cutoffs which would make bonded sites collide are rejected"""
    with pytest.raises(ValueError, match="cannot exceed"):
        SelfAvoidingWalk(SingleBondPlacer(bond_length=1.0), minimum_distance=1.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(collision_index=SpatialHashIndex(1.0), minimum_distance=1.0),
        dict(box_lengths=(5.0, 5.0, 5.0)),
        dict(occupied_sites=np.zeros((1, 3))),
        dict(max_attempts_per_unit=0),
        dict(max_backtracks=-1),
        dict(initial_point=np.zeros(2)),
    ],
)
def test_invalid_walk_arguments(kwargs) -> None:
    """Test inconsistent or invalid walk arguments are rejected"""
    with pytest.raises(ValueError):
        SelfAvoidingWalk(SingleBondPlacer(bond_length=2.0), **kwargs)


def test_placer_type_checked() -> None:
    """Test that the placer must be a UnitPlacer"""
    with pytest.raises(TypeError):
        SelfAvoidingWalk(placer="not a placer")


def test_self_avoidance_requires_leaf_shapes() -> None:
    """Test self-avoidance requires shapes on all leaves"""
    chain, _ = pim_ladder_chain(2)
    next(iter(chain.leaves))._shape = None
    walker = SelfAvoidingWalk(TwoBondPlacer(), minimum_distance=1.0)
    with pytest.raises(TypeError):
        list(walker.generate_placements(chain))
