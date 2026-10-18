"""Random walk coordinate generation, and builders which place chains along walks"""

import logging

LOGGER = logging.getLogger(__name__)

from typing import (
    Generator,
    Iterable,
    Optional,
    Sized,
    Union,
)
from numbers import Number

import numpy as np
from scipy.spatial.transform import RigidTransform
from networkx import connected_components

from .base import PlacementGenerator
from .unit_placement import Junction, UnitPlacement, UnitPlacer, SingleBondPlacer
from ..mutils.iteration import flexible_iterator, sliding_window
from ..graphs.invariants import is_branched

from ..geometry.arraytypes import ArrayNx3, Shape, Dims, NumericNP, Vector3
from ..geometry.collision import SpatialHashIndex
from ..geometry.measure import normalized
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.coordinates.reference import origin

from ..mupr.primitives import Primitive, PrimitiveHandle


# TB: supressing linter complexity (C901) warning for now,
# but in the future this should be refactored to be more modular
# and contain less branched business logic in one place
def random_walk_jointed_chain(  # noqa: C901
    step_size: Union[Number, Iterable[Number], Generator[Number, None, None]],
    n_steps_max: Optional[int] = None,
    initial_point: Optional[np.ndarray[Shape[Dims], NumericNP]] = None,
    initial_direction: Optional[np.ndarray[Shape[Dims], NumericNP]] = None,
    clip_angle: float = np.pi / 4,
    dimension: Dims = 3,
    rng: Optional[np.random.Generator] = None,
) -> Generator[np.ndarray[Shape[Dims], NumericNP], None, None]:
    """
    Generate consecutive points from a non-self-avoiding random walk
    in continuous N-dimensional space with arbitrary step sizes that
    are constrained within a prescribed angle between consecutive steps

    Parameters
    ----------
    step_size : float or Generator[float, None, None]
        Step size for each step, either as a float for uniform step size
        or as a generator the size of each subsequent step
    initial_point : ndarray[Shape[dimension], float] = None
        Starting point of the random walk. If None, the walk starts at the origin.
    initial_direction : ndarray[Shape[dimension], float] = None
        Initial direction of the first step. If None, a random direction is chosen.
    n_steps_max : int
        Maximum number of steps in the random walk

        Will result in (n_steps_max + 1) points being
        generated (+1 to include the initial point)
    clip_angle : float = pi/4
        Maximum angle allowed between directions of subsequent steps
        Smaller values will result in walks more along the "same" direction
        Angle should be passed in radians, as a value from [-pi, pi]
    dimension : int, default 3
        Dimension of the space in which the random walk is performed

        If no start point is provided, the inferred origin
        used as the start will have this many dimensions
    rng : numpy.random.Generator, optional
        Random number generator used for directions. If None, preserves the legacy
        module-level NumPy random behavior.

    Returns
    -------
    Generator[ndarray[Shape[dimension], float], None, None]
        A generator yielding conseuctive positions
        along the walk, starting with initial_position
    """
    # validate preconditions
    if (
        not -np.pi <= clip_angle <= np.pi
    ):  # negative angles are allowed since cosine is even
        raise ValueError("Clip_angle must be in the range [-pi, pi]")
    cos_max: float = np.cos(clip_angle)

    if initial_point is None:
        initial_point = origin(dimension=dimension)

    # NOTE: check user-provided start shape or the shape
    # of the auto-assigned start (redundant, but safer)
    if initial_point.shape != (dimension,):
        raise ValueError(
            f"Random walk starting point must be a {dimension}-dimensional vector"
        )

    if initial_direction is None:
        initial_direction = random_unit_vector(dimension=dimension, rng=rng)
    assert initial_direction.shape == (
        dimension,
    )  # NOTE: check user-provided start direction shape

    if n_steps_max is None:
        if not isinstance(step_size, Iterable):
            LOGGER.warning(
                "No upper bound supplied for number of random walk steps; "
                "singly-valued step size will produce steps indefinitely!"
            )
        elif not isinstance(step_size, Sized):
            LOGGER.warning(
                "No upper bound supplied for number of random walk steps; "
                "unbounded step size iterator *MAY* produce steps indefinitely!"
            )

    # generate walk points
    n_steps_taken: int = 0
    ## make mutable copy of (possibly-immutable) initial point
    net_position: np.ndarray = np.array(initial_point)
    prev_direction: np.ndarray = normalized(initial_direction)

    yield initial_point  # always yielded, consider as "step #0"
    for step_size in flexible_iterator(step_size, allowed_types=(Number,)):
        # draw new step within cone of movement by rejection sampling (simple and quick)
        step_direction: np.ndarray = random_unit_vector(dimension=dimension, rng=rng)

        # NOTE: over |x| in [0, pi], cos(x) is monotonically decreasing,
        # so overly-large steps will have cosine BELOW the cutoff
        while np.dot(step_direction, prev_direction) < cos_max:
            step_direction: np.ndarray = random_unit_vector(
                dimension=dimension, rng=rng
            )
        step = step_size * step_direction

        # DEV: resist the urge to just yield net_position after incrementing it.
        # That yields the same REFERENCE to the underlying array at each step
        # I.e. any previously-yielded steps would then point to the most recent
        # step's value (rather than retaining their at-the-time values as expected)
        yield net_position + step

        # NOTE: only increment internal state AFTER we've actually yielded
        n_steps_taken += 1
        net_position += step
        prev_direction: np.ndarray = step_direction
        if (n_steps_max is not None) and (n_steps_taken >= n_steps_max):
            break


class SelfAvoidanceError(RuntimeError):
    """Raised when a chain cannot be placed without collisions within search limits"""


class SelfAvoidingWalk(PlacementGenerator):
    """
    Places the children of a Primitive one at a time along each of its linear chains,
    optionally rejecting placements which collide with previously-placed units

    How each unit is connected to and oriented relative to the previous one is
    delegated entirely to the provided UnitPlacer; this class is responsible only
    for traversing chains, collision checking, and backtracking out of dead ends.

    Self-avoidance is enabled by providing a minimum_distance (or a pre-existing
    collision_index, e.g. one shared between several walks). Collisions are
    checked between the centroids of the leaves beneath each unit, all of which
    must then have shapes. If box_lengths are also given, distances are measured
    periodically in an orthorhombic box centered on the origin.
    Without self-avoidance, every proposed placement is accepted
    """

    def __init__(
        self,
        placer: UnitPlacer,
        initial_point: Optional[Vector3] = None,
        rng: Optional[np.random.Generator] = None,
        minimum_distance: Optional[float] = None,
        box_lengths: Optional[tuple[float, float, float]] = None,
        max_attempts_per_unit: int = 100,
        max_backtracks: int = 1_000,
        collision_index: Optional[SpatialHashIndex] = None,
        occupied_sites: Optional[ArrayNx3] = None,
    ) -> None:
        if not isinstance(placer, UnitPlacer):
            raise TypeError(f"Placer must be a UnitPlacer, not {type(placer)}")
        self.placer = placer

        if initial_point is not None:
            initial_point = np.asarray(initial_point, dtype=float)
            if initial_point.shape != (3,):
                raise ValueError("Initial point must be a 3-dimensional vector")
        self.initial_point = initial_point
        self.rng = rng

        self.max_attempts_per_unit = int(max_attempts_per_unit)
        self.max_backtracks = int(max_backtracks)
        if self.max_attempts_per_unit < 1:
            raise ValueError("Must allow at least one placement attempt per unit")
        if self.max_backtracks < 0:
            raise ValueError("Maximum number of backtracks must be non-negative")

        self.collision_index = self._resolve_collision_index(
            collision_index, minimum_distance, box_lengths
        )
        if self.is_self_avoiding:
            if self.collision_index.cutoff > self.placer.bond_length:
                raise ValueError(
                    f"Minimum distance ({self.collision_index.cutoff}) cannot exceed "
                    f"the bond length ({self.placer.bond_length}), "
                    "as bonded sites would then collide"
                )
            if occupied_sites is not None:
                self.collision_index.add(occupied_sites)
        elif occupied_sites is not None:
            raise ValueError("Occupied sites require self-avoidance to be enabled")

        self.placed_sites: ArrayNx3 = np.empty((0, 3), dtype=float)

    @staticmethod
    def _resolve_collision_index(
        collision_index: Optional[SpatialHashIndex],
        minimum_distance: Optional[float],
        box_lengths: Optional[tuple[float, float, float]],
    ) -> Optional[SpatialHashIndex]:
        """The collision index to use (if any) given the self-avoidance arguments"""
        if collision_index is not None:
            if (minimum_distance is not None) or (box_lengths is not None):
                raise ValueError(
                    "Pass either a collision_index or "
                    "minimum_distance/box_lengths, not both"
                )
            return collision_index

        if minimum_distance is not None:
            return SpatialHashIndex(cutoff=minimum_distance, box_lengths=box_lengths)

        if box_lengths is not None:
            raise ValueError(
                "Periodic box_lengths require a minimum_distance to be set"
            )
        return None

    @property
    def is_self_avoiding(self) -> bool:
        """Whether placements are checked for collisions"""
        return self.collision_index is not None

    # chain layout
    @staticmethod
    def chain_paths(primitive: Primitive) -> list[list[PrimitiveHandle]]:
        """
        Ordered handles of the children along each linear chain in a Primitive's
        topology

        Chains are walked starting from whichever terminal child was added to the
        topology first, so that the direction of traversal is reproducible
        """
        topology = primitive.topology
        if is_branched(topology):
            raise ValueError("Chain builder behavior undefined for branched topologies")

        insertion_order = {handle: idx for idx, handle in enumerate(topology.nodes)}
        paths = []
        for component_nodes in connected_components(topology):
            ordered_nodes = sorted(component_nodes, key=insertion_order.__getitem__)
            chain_termini = [
                handle for handle in ordered_nodes if topology.degree[handle] <= 1
            ]
            if not chain_termini:
                raise ValueError(
                    "Chain builder behavior undefined for cyclic topologies"
                )

            path = [chain_termini[0]]
            while len(path) < len(ordered_nodes):
                (next_handle,) = (
                    nb_handle
                    for nb_handle in topology.neighbors(path[-1])
                    if (len(path) == 1) or (nb_handle != path[-2])
                )
                path.append(next_handle)
            paths.append(path)

        return paths

    @staticmethod
    def chain_junctions(
        primitive: Primitive,
        path: list[PrimitiveHandle],
    ) -> list[Junction]:
        """The bonds between each consecutive pair of children along a path"""
        return [
            Junction(
                from_handle=from_handle,
                to_handle=to_handle,
                # sorted, as connections are stored in (unordered) sets
                connector_pairs=tuple(
                    sorted(
                        primitive.internal_connections_between(from_handle, to_handle)
                    )
                ),
            )
            for from_handle, to_handle in sliding_window(path, 2)
        ]

    @staticmethod
    def unit_sites(unit: Primitive) -> ArrayNx3:
        """Positions of all leaves beneath (or at) a unit, used for collision checks"""
        leaves = [unit] if unit.is_leaf else unit.leaves
        if any(leaf.shape is None for leaf in leaves):
            raise TypeError(
                f"All leaves beneath Primitive {unit._repr_brief()} must "
                "have shapes to be checked for collisions"
            )
        return np.array([leaf.shape.centroid for leaf in leaves], dtype=float)

    # implementing builder contracts
    def check_preconditions(self, primitive: Primitive) -> None:
        """Enforce linear chains which the placer is able to place"""
        for path in self.chain_paths(primitive):
            self.placer.check_preconditions(
                primitive, path, self.chain_junctions(primitive, path)
            )
            if self.is_self_avoiding:
                for handle in path:
                    self.unit_sites(primitive.fetch_child(handle))

    def _propose_initial_point(self) -> Vector3:
        """
        Starting point for the first unit of a chain; if none was specified,
        drawn uniformly within the periodic box (if any), else the origin
        """
        if self.initial_point is not None:
            return self.initial_point

        box_lengths = (
            self.collision_index.box_lengths if self.is_self_avoiding else None
        )
        if box_lengths is not None:
            uniform = np.random.uniform if (self.rng is None) else self.rng.uniform
            return uniform(-box_lengths / 2.0, box_lengths / 2.0)

        return origin(dimension=3)

    def _max_attempts(self, depth: int) -> int:
        """Number of proposals worth making for the unit at a given depth"""
        return self.max_attempts_per_unit if self.placer.is_stochastic(depth) else 1

    def _solve_chain(
        self,
        primitive: Primitive,
        path: list[PrimitiveHandle],
        tokens: list[int],
        site_batches: list[ArrayNx3],
    ) -> list[UnitPlacement]:
        """
        Find accepted placements for every unit along one chain

        Collision index tokens and sites of accepted units are appended to the
        passed lists, which the caller is responsible for rolling back on failure
        """
        self.placer.prepare(primitive, path, self.chain_junctions(primitive, path))
        template_sites: dict[PrimitiveHandle, ArrayNx3] = {}
        if self.is_self_avoiding:
            template_sites = {
                handle: self.unit_sites(primitive.fetch_child(handle))
                for handle in path
            }
        accepted: list[UnitPlacement] = []
        num_tokens_prior = len(tokens)
        attempts: list[int] = [0] * len(path)
        num_backtracks: int = 0
        while len(accepted) < len(path):
            depth = len(accepted)
            if attempts[depth] >= self._max_attempts(depth):
                # dead end; unwind to nearest prior unit with proposals left to try
                if num_backtracks >= self.max_backtracks:
                    raise SelfAvoidanceError(
                        f"Unable to place chain in {primitive._repr_brief()}; gave "
                        f"up at unit {depth} of {len(path)} after {num_backtracks} "
                        "backtracks"
                    )
                while attempts[depth] >= self._max_attempts(depth):
                    attempts[depth] = 0
                    if depth == 0:
                        raise SelfAvoidanceError(
                            f"Unable to place chain in {primitive._repr_brief()}; "
                            "exhausted all placements of its first unit"
                        )
                    depth -= 1
                    accepted.pop()
                    self.collision_index.remove(tokens.pop())
                    site_batches.pop()
                num_backtracks += 1
                LOGGER.debug(f"Backtracked to unit {depth} of {len(path)}")
                continue

            attempts[depth] += 1
            if depth == 0:
                candidate = self.placer.place_first(
                    self._propose_initial_point(), rng=self.rng
                )
            else:
                candidate = self.placer.place_next(depth, accepted[-1], rng=self.rng)

            if self.is_self_avoiding:
                sites = candidate.transform.apply(template_sites[candidate.handle])
                if self.collision_index.collides(sites):
                    continue
                tokens.append(self.collision_index.add(sites))
                site_batches.append(sites)
            accepted.append(candidate)
        assert (not self.is_self_avoiding) or (
            len(tokens) - num_tokens_prior == len(path)
        ), "Collision bookkeeping out of sync with accepted placements"

        return accepted

    def _generate_placements(
        self, primitive: Primitive
    ) -> Generator[tuple[PrimitiveHandle, RigidTransform], None, None]:
        """
        Place every chain in the Primitive before yielding any placements, so that
        a failure to place any one chain leaves the collision index untouched
        """
        tokens: list[int] = []
        site_batches: list[ArrayNx3] = []
        chain_placements: list[list[UnitPlacement]] = []
        try:
            for path in self.chain_paths(primitive):
                chain_placements.append(
                    self._solve_chain(primitive, path, tokens, site_batches)
                )
        except BaseException:
            for token in tokens:
                self.collision_index.remove(token)
            raise

        self.placed_sites = (
            np.vstack(site_batches) if site_batches else np.empty((0, 3), dtype=float)
        )
        for placements in chain_placements:
            self.placer.finalize(primitive, placements)
            for placement in placements:
                yield placement.handle, placement.transform


class AngleConstrainedRandomWalk(SelfAvoidingWalk):
    """
    Places children of a Primitive joined by single bonds along an angle-constrained
    random walk; non-self-avoiding unless a minimum_distance or collision_index is
    passed (see SelfAvoidingWalk for the self-avoidance keyword arguments)
    """

    def __init__(
        self,
        bond_length: float = 1.0,
        angle_max_rad: float = np.pi / 4,
        initial_point: Optional[Vector3] = None,
        initial_direction: Optional[Vector3] = None,
        rng: Optional[np.random.Generator] = None,
        **walk_kwargs,
    ) -> None:
        super().__init__(
            placer=SingleBondPlacer(
                bond_length=bond_length,
                angle_max_rad=angle_max_rad,
                initial_direction=initial_direction,
            ),
            initial_point=initial_point,
            rng=rng,
            **walk_kwargs,
        )

    @property
    def bond_length(self) -> float:
        """Target distance between the anchors of bonded units"""
        return self.placer.bond_length

    @property
    def angle_max_rad(self) -> float:
        """Maximum angle between consecutive walk steps"""
        return self.placer.angle_max_rad

    @property
    def initial_direction(self) -> Optional[Vector3]:
        """Direction about which the first walk step is drawn (random if None)"""
        return self.placer.initial_direction
