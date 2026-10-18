"""
For calculation of rigid transforms which force
two bodies to be spatially coincident in some way
"""

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from .rotations import alignment_rotation
from ...arraytypes import Array2x3, Array3x3, Vector3


def rigid_vector_coalignment(
    vector1_start: Vector3,
    vector1_end: Vector3,
    vector2_start: Vector3,
    vector2_end: Vector3,
    *,
    t1: float = 0.5,
    t2: Optional[float] = None,
) -> RigidTransform:
    """
    Compute a rigid transformation that forces "vector1" (defined by
    its end points and not necessarily emanating from the origin)
    onto the span of "vector2" (defined similarly), oriented parallel
    and with the point some fraction "t1" of the way along vector1
    to be exactly coincident with the point t2 parts along vector2
    """
    # TODO: check compatibility of shapes within AND between vectors
    if t2 is None:
        t2 = t1

    vector1 = vector1_end - vector1_start
    overlaid_point_1 = vector1_start + t1 * vector1

    vector2 = vector2_end - vector2_start
    overlaid_point_2 = vector2_start + t2 * vector2

    return (
        RigidTransform.from_translation(overlaid_point_2)
        * RigidTransform.from_rotation(alignment_rotation(vector1, vector2))
        * RigidTransform.from_translation(-overlaid_point_1)
    )


def _junction_frame(
    anchors: Array2x3,
    bond_vectors: Array2x3,
    tolerance: float = 1e-8,
) -> tuple[Vector3, Array3x3, float]:
    """
    Local orthonormal frame of one side of a two-bond junction

    Returns the midpoint of the anchors, a matrix whose columns are the pair axis
    (from first to second anchor), the outward direction (mean bond direction
    orthogonalized against the pair axis), and their cross product, and the
    distance between the anchors
    """
    anchors = np.asarray(anchors, dtype=float)
    bond_vectors = np.asarray(bond_vectors, dtype=float)
    if anchors.shape != (2, 3) or bond_vectors.shape != (2, 3):
        raise ValueError("Junction anchors and bond vectors must both be (2, 3) arrays")

    pair_axis = anchors[1] - anchors[0]
    span = float(np.linalg.norm(pair_axis))
    if span < tolerance:
        raise ValueError("Cannot define junction frame for coincident anchors")
    pair_axis /= span

    bond_norms = np.linalg.norm(bond_vectors, axis=1, keepdims=True)
    if np.any(bond_norms < tolerance):
        raise ValueError("Cannot define junction frame from zero-length bond vectors")
    outward = np.mean(bond_vectors / bond_norms, axis=0)
    outward -= np.dot(outward, pair_axis) * pair_axis
    outward_norm = np.linalg.norm(outward)
    if outward_norm < tolerance:
        raise ValueError(
            "Cannot define junction frame when bond vectors "
            "are parallel to the pair axis"
        )
    outward /= outward_norm

    return (
        anchors.mean(axis=0),
        np.column_stack([pair_axis, outward, np.cross(pair_axis, outward)]),
        span,
    )


def rigid_junction_docking(
    placed_anchors: Array2x3,
    placed_bond_vectors: Array2x3,
    free_anchors: Array2x3,
    free_bond_vectors: Array2x3,
    bond_length: float,
    tolerance: float = 1e-8,
) -> RigidTransform:
    """
    Compute a rigid transformation which docks a "free" body onto a "placed"
    body across a junction of two bonds, such that the i-th free anchor ends up
    exactly bond_length away from the i-th placed anchor, for both bonds

    Each side is described by its pair of anchor positions and the corresponding
    (not necessarily normalized) bond vectors pointing away from that body, with
    rows ordered so that row i of the placed arrays bonds to row i of the free arrays.
    The pair axes of both sides are made parallel, the outward directions antiparallel,
    and the gap between the anchor midpoints chosen to make both bonds the same length.
    The result is independent of the order in which the two bonds are listed
    """
    placed_origin, placed_frame, placed_span = _junction_frame(
        placed_anchors, placed_bond_vectors, tolerance=tolerance
    )
    free_origin, free_frame, free_span = _junction_frame(
        free_anchors, free_bond_vectors, tolerance=tolerance
    )

    half_span_mismatch = abs(placed_span - free_span) / 2.0
    if half_span_mismatch > bond_length:
        raise ValueError(
            f"Anchor spacings of junction sides ({placed_span:.3f} vs {free_span:.3f}) "
            f"differ too much to be bridged by bonds of length {bond_length}"
        )
    gap = np.sqrt(bond_length**2 - half_span_mismatch**2)

    # same pair axis, flipped outward direction (flipping normal keeps handedness)
    target_frame = placed_frame @ np.diag([1.0, -1.0, -1.0])
    rotation = Rotation.from_matrix(target_frame @ free_frame.T)

    return (
        RigidTransform.from_translation(placed_origin + gap * placed_frame[:, 1])
        * RigidTransform.from_rotation(rotation)
        * RigidTransform.from_translation(-free_origin)
    )


def hinge_rotation(
    axis_point_1: Vector3,
    axis_point_2: Vector3,
    angle_rad: float,
) -> RigidTransform:
    """
    Rigid rotation by angle_rad radians (right-handed) about
    the line passing through the two given points

    Any point lying on the axis (namely, both axis points) is left fixed
    """
    axis_point_1 = np.asarray(axis_point_1, dtype=float)
    axis = np.asarray(axis_point_2, dtype=float) - axis_point_1
    axis_length = np.linalg.norm(axis)
    if axis_length == 0.0:
        raise ValueError("Hinge axis is undefined for coincident points")

    return (
        RigidTransform.from_translation(axis_point_1)
        * RigidTransform.from_rotation(
            Rotation.from_rotvec(angle_rad * axis / axis_length)
        )
        * RigidTransform.from_translation(-axis_point_1)
    )
