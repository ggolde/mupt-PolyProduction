"""
Strategies for placing the next unit of a linear chain relative to the previous one

Used by sequential chain builders (e.g. random walks), which own the traversal of a
chain and any acceptance/rejection logic, and delegate to a UnitPlacer the questions of
how consecutive units connect and how the orientation of each new unit is sampled
"""

import logging

LOGGER = logging.getLogger(__name__)

from typing import Any, ClassVar, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from ..geometry.arraytypes import Vector3
from ..geometry.measure import normalized
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.transforms.rigid import (
    rigid_vector_coalignment,
    rigid_junction_docking,
    hinge_rotation,
)

from ..mupr.connection import ConnectorHandle
from ..mupr.primitives import Primitive, PrimitiveHandle


@dataclass(frozen=True)
class Junction:
    """
    All bonds between two consecutive units along a chain, each given as
    a pair of Connector handles ordered in the direction of traversal
    I.e. (Connector handle on from-unit, Connector handle on to-unit)
    """

    from_handle: PrimitiveHandle
    to_handle: PrimitiveHandle
    connector_pairs: tuple[tuple[ConnectorHandle, ConnectorHandle], ...]

    @property
    def num_bonds(self) -> int:
        """Number of bonds making up this junction"""
        return len(self.connector_pairs)


@dataclass(frozen=True)
class UnitPlacement:
    """
    Placement of one unit, as a rigid transformation of its current (template)
    coordinates, along with any state the placer needs to place the next unit

    The state is private to the UnitPlacer which produced the placement
    """

    handle: PrimitiveHandle
    transform: RigidTransform
    state: Any = None


class UnitPlacer(ABC):
    """
    Abstract strategy for placing the units of a linear chain one at a time,
    each relative to the previously-placed unit

    The chain builder calls prepare() once per chain, then place_first() for the
    first unit and place_next() for each subsequent one, possibly calling either
    repeatedly at the same position to obtain alternative candidate placements.
    Implementations must not modify the Primitive except in finalize()
    """

    bonds_per_junction: ClassVar[int]

    def __init__(self, bond_length: float) -> None:
        self.bond_length = float(bond_length)
        if not np.isfinite(self.bond_length) or self.bond_length <= 0.0:
            raise ValueError("Bond length must be positive and finite")

    def check_preconditions(
        self,
        primitive: Primitive,
        path: list[PrimitiveHandle],
        junctions: list[Junction],
    ) -> None:
        """
        Check that a chain (given as a path of child handles and the junctions
        between consecutive children) can be placed by this placer
        """
        for junction in junctions:
            if junction.num_bonds != self.bonds_per_junction:
                raise ValueError(
                    f"{type(self).__name__} requires exactly "
                    f"{self.bonds_per_junction} bond(s) between consecutive units, "
                    f"but units '{junction.from_handle}' and '{junction.to_handle}' "
                    f"are joined by {junction.num_bonds}; chains mixing junction "
                    "types are not supported"
                )

        for handle in path:
            if primitive.fetch_child(handle).shape is None:
                raise TypeError(
                    f"{type(self).__name__} requires all units to have a shape, "
                    f"but unit '{handle}' has none"
                )

    @abstractmethod
    def prepare(
        self,
        primitive: Primitive,
        path: list[PrimitiveHandle],
        junctions: list[Junction],
    ) -> None:
        """Cache any template geometry needed to place the units of one chain"""
        ...

    @abstractmethod
    def place_first(
        self,
        initial_point: Vector3,
        rng: Optional[np.random.Generator] = None,
    ) -> UnitPlacement:
        """Propose a placement for the first unit of the chain"""
        ...

    @abstractmethod
    def place_next(
        self,
        depth: int,
        previous: UnitPlacement,
        rng: Optional[np.random.Generator] = None,
    ) -> UnitPlacement:
        """
        Propose a placement for the unit at position "depth" along the chain,
        given the accepted placement of the unit immediately before it
        """
        ...

    def is_stochastic(self, depth: int) -> bool:
        """
        Whether repeated proposals for the unit at position "depth"
        along the chain can differ (i.e. whether retrying is worthwhile)
        """
        return True

    def finalize(
        self,
        primitive: Primitive,
        placements: list[UnitPlacement],
    ) -> None:
        """Hook called once all units of a chain have been accepted"""
        ...


class SingleBondPlacer(UnitPlacer):
    """
    Places units joined by single bonds along an angle-constrained random walk

    Each unit is a step of the walk; its anchor-to-anchor axis (or centroid, for
    terminal units) is aligned with a step whose direction is drawn uniformly within
    a cone of half-angle angle_max_rad about the direction of the previous step.
    Rotation of each unit about its own axis (i.e. torsion) is left unconstrained
    """

    bonds_per_junction: ClassVar[int] = 1

    def __init__(
        self,
        bond_length: float = 1.0,
        angle_max_rad: float = np.pi / 4,
        initial_direction: Optional[Vector3] = None,
    ) -> None:
        super().__init__(bond_length)
        # NOTE: negative angles are allowed since cosine is even
        if not -np.pi <= angle_max_rad <= np.pi:
            raise ValueError("Maximum step angle must be in the range [-pi, pi]")
        self.angle_max_rad = angle_max_rad
        self.initial_direction = initial_direction
        if (initial_direction is not None) and (np.shape(initial_direction) != (3,)):
            raise ValueError("Initial direction must be a 3-dimensional vector")

        self._path: list[PrimitiveHandle] = []
        self._connection_points: dict[PrimitiveHandle, list[np.ndarray]] = {}
        self._step_sizes: list[float] = []

    def prepare(
        self,
        primitive: Primitive,
        path: list[PrimitiveHandle],
        junctions: list[Junction],
    ) -> None:
        """Determine the pair of points per unit which are aligned to each walk step"""
        head_handle, tail_handle = path[0], path[-1]
        connection_points: dict[PrimitiveHandle, list[np.ndarray]] = {
            handle: [] for handle in path
        }
        connection_points[head_handle].append(
            primitive.fetch_child(head_handle).shape.centroid
        )
        for junction in junctions:
            ((conn_handle_outgoing, conn_handle_incoming),) = junction.connector_pairs
            # will raise Exception if anchor positions are unset
            connection_points[junction.from_handle].append(
                primitive.fetch_connector_on_child(
                    junction.from_handle, conn_handle_outgoing
                ).anchor.position
            )
            connection_points[junction.to_handle].append(
                primitive.fetch_connector_on_child(
                    junction.to_handle, conn_handle_incoming
                ).anchor.position
            )
        # NOTE: order is critical here; only placing tail point
        # AFTER its incoming connection point is inserted
        connection_points[tail_handle].append(
            primitive.fetch_child(tail_handle).shape.centroid
        )

        self._path = path
        self._connection_points = connection_points
        # step longer than the body to account for target bond length;
        # by design, makes no reference to the shape of the body
        self._step_sizes = [
            np.linalg.norm(conn_end - conn_start) + self.bond_length
            for conn_start, conn_end in (connection_points[handle] for handle in path)
        ]

    def _sample_direction(
        self,
        previous_direction: Vector3,
        rng: Optional[np.random.Generator] = None,
    ) -> Vector3:
        """Draw a unit vector within the cone about the previous direction"""
        # rejection sampling (simple and quick for wide cones); NOTE: over |x| in
        # [0, pi], cos(x) is monotonically decreasing, so overly-large steps
        # will have cosine BELOW the cutoff
        cos_max = np.cos(self.angle_max_rad)
        step_direction = random_unit_vector(dimension=3, rng=rng)
        while np.dot(step_direction, previous_direction) < cos_max:
            step_direction = random_unit_vector(dimension=3, rng=rng)

        return step_direction

    def _place_step(
        self,
        depth: int,
        step_start: Vector3,
        previous_direction: Vector3,
        rng: Optional[np.random.Generator] = None,
    ) -> UnitPlacement:
        """Draw the next walk step and align the unit at the given depth to it"""
        step_direction = self._sample_direction(previous_direction, rng=rng)
        step_end = step_start + self._step_sizes[depth] * step_direction

        handle = self._path[depth]
        conn_start, conn_end = self._connection_points[handle]
        # NOTE: no need for special case at termini, since the step size
        # matches the half-body (e.g. center-to-anchor) step size
        t_body = 0.5

        full_step_len = np.linalg.norm(step_end - step_start)
        # scale back to account for bond length being included in step size
        step_correction = 1 / (1 + (self.bond_length / full_step_len))
        # adjust step fraction to account for bond length (will always be
        # strictly smaller than 0.5, since ratio of lengths is positive)
        t_step = 0.5 * step_correction

        if np.allclose(conn_start, conn_end):
            # isolated unit (no anchors to align); just center it along the step
            return UnitPlacement(
                handle=handle,
                transform=RigidTransform.from_translation(
                    step_start + t_step * (step_end - step_start) - conn_start
                ),
                state=(step_end, step_direction),
            )

        placement_transform = rigid_vector_coalignment(
            # vector 1: spans between anchors of connection point on body
            conn_start,
            conn_end,
            # vector 2: spans between consecutive random walk steps
            step_start,
            step_end,
            # interpolation parameters for which point on
            # respective vectors will be forced exactly-coexistent
            ## take midpoint (or end, if at termini) of body-anchoring vector
            t1=t_body,
            ## ...to midpoint of random walk step vector
            t2=t_step,
        )
        LOGGER.debug(
            f"Random walk placing body {handle} "
            f"along vector from {step_start} to {step_end}"
        )

        return UnitPlacement(
            handle=handle,
            transform=placement_transform,
            state=(step_end, step_direction),
        )

    def place_first(
        self,
        initial_point: Vector3,
        rng: Optional[np.random.Generator] = None,
    ) -> UnitPlacement:
        """Draw the first walk step from the initial point and align the head unit"""
        initial_direction = self.initial_direction
        if initial_direction is None:
            initial_direction = random_unit_vector(dimension=3, rng=rng)

        return self._place_step(
            0,
            step_start=initial_point,
            previous_direction=normalized(initial_direction),
            rng=rng,
        )

    def place_next(
        self,
        depth: int,
        previous: UnitPlacement,
        rng: Optional[np.random.Generator] = None,
    ) -> UnitPlacement:
        """Draw the next walk step, continuing from where the previous one ended"""
        step_start, previous_direction = previous.state
        return self._place_step(
            depth,
            step_start=step_start,
            previous_direction=previous_direction,
            rng=rng,
        )


class TwoBondPlacer(UnitPlacer):
    """
    Places units joined by pairs of bonds (e.g. ladder polymers) by rigid docking

    The first unit is placed with a uniformly random orientation; each subsequent
    unit is docked onto its predecessor such that both bonds of the junction between
    them have exactly bond_length (see rigid_junction_docking). Docked units may
    optionally be hinged about the axis through the predecessor's anchors by a
    random angle within +/- flex_angle_rad, which leaves both bond lengths unchanged
    """

    bonds_per_junction: ClassVar[int] = 2

    def __init__(
        self,
        bond_length: float = 1.37,
        flex_angle_rad: float = 0.0,
        initial_rotation: Optional[Rotation] = None,
    ) -> None:
        super().__init__(bond_length)
        if not 0.0 <= flex_angle_rad <= np.pi:
            raise ValueError("Flex angle must be in the range [0, pi]")
        self.flex_angle_rad = flex_angle_rad
        self.initial_rotation = initial_rotation

        self._path: list[PrimitiveHandle] = []
        self._first_centroid: Optional[np.ndarray] = None
        # per junction: (from-side anchors, from-side bond vectors,
        # to-side anchors, to-side bond vectors), each a (2, 3) array
        self._junction_geometry: list[tuple[np.ndarray, ...]] = []

    def check_preconditions(
        self,
        primitive: Primitive,
        path: list[PrimitiveHandle],
        junctions: list[Junction],
    ) -> None:
        """
        Additionally enforce that every junction Connector has a well-defined
        bond vector, and that the two bonds on either side have distinct anchors
        """
        super().check_preconditions(primitive, path, junctions)
        for junction in junctions:
            from_conn_handles, to_conn_handles = zip(*junction.connector_pairs)
            for side_handle, conn_handles in (
                (junction.from_handle, from_conn_handles),
                (junction.to_handle, to_conn_handles),
            ):
                conns = [
                    primitive.fetch_connector_on_child(side_handle, conn_handle)
                    for conn_handle in conn_handles
                ]
                for conn_handle, conn in zip(conn_handles, conns):
                    if not conn.has_bond_vector:
                        raise ValueError(
                            f"Connector {conn_handle} on unit '{side_handle}' must "
                            "have distinct anchor and linker positions to be docked"
                        )
                if np.allclose(conns[0].anchor.position, conns[1].anchor.position):
                    raise ValueError(
                        f"Both bonds of the junction on unit '{side_handle}' share an "
                        "anchor position; cannot determine junction orientation"
                    )

    def prepare(
        self,
        primitive: Primitive,
        path: list[PrimitiveHandle],
        junctions: list[Junction],
    ) -> None:
        """Cache the template anchors and bond vectors of both sides of each junction"""
        self._path = path
        self._first_centroid = np.asarray(primitive.fetch_child(path[0]).shape.centroid)

        self._junction_geometry = []
        for junction in junctions:
            from_conns = [
                primitive.fetch_connector_on_child(
                    junction.from_handle, from_conn_handle
                )
                for from_conn_handle, _ in junction.connector_pairs
            ]
            to_conns = [
                primitive.fetch_connector_on_child(junction.to_handle, to_conn_handle)
                for _, to_conn_handle in junction.connector_pairs
            ]
            self._junction_geometry.append(
                tuple(
                    np.array([get_vector(conn) for conn in conns], dtype=float)
                    for conns in (from_conns, to_conns)
                    for get_vector in (
                        lambda conn: conn.anchor.position,
                        lambda conn: conn.bond_vector,
                    )
                )
            )

    def place_first(
        self,
        initial_point: Vector3,
        rng: Optional[np.random.Generator] = None,
    ) -> UnitPlacement:
        """Center the first unit on the initial point, with a random orientation"""
        rotation = self.initial_rotation
        if rotation is None:
            rotation = Rotation.random(random_state=rng)

        return UnitPlacement(
            handle=self._path[0],
            transform=(
                RigidTransform.from_translation(initial_point)
                * RigidTransform.from_rotation(rotation)
                * RigidTransform.from_translation(-self._first_centroid)
            ),
        )

    def place_next(
        self,
        depth: int,
        previous: UnitPlacement,
        rng: Optional[np.random.Generator] = None,
    ) -> UnitPlacement:
        """Dock the unit onto its predecessor, hinged by a random flex angle if set"""
        from_anchors, from_bond_vectors, to_anchors, to_bond_vectors = (
            self._junction_geometry[depth - 1]
        )
        placed_anchors = previous.transform.apply(from_anchors)
        placed_bond_vectors = previous.transform.rotation.apply(from_bond_vectors)

        placement_transform = rigid_junction_docking(
            placed_anchors,
            placed_bond_vectors,
            to_anchors,
            to_bond_vectors,
            bond_length=self.bond_length,
        )
        if self.flex_angle_rad > 0.0:
            uniform = np.random.uniform if (rng is None) else rng.uniform
            placement_transform = (
                hinge_rotation(
                    placed_anchors[0],
                    placed_anchors[1],
                    uniform(-self.flex_angle_rad, self.flex_angle_rad),
                )
                * placement_transform
            )

        return UnitPlacement(handle=self._path[depth], transform=placement_transform)

    def is_stochastic(self, depth: int) -> bool:
        """Only the first unit's orientation and any junction flex angles are random"""
        return (depth == 0 and self.initial_rotation is None) or (
            depth > 0 and self.flex_angle_rad > 0.0
        )
