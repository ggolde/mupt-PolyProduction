"""Unit tests for rigid docking across two-bond junctions and hinge rotations"""

import pytest

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from mupt.geometry.transforms.rigid import hinge_rotation, rigid_junction_docking


BOND_LENGTH = 1.37

# idealized junction sides: a catechol-like pair of anchors ~2.8 apart
# whose bonds converge, and an aryl pair of anchors ~1.4 apart
PLACED_ANCHORS = np.array([[0.0, -1.4, 0.0], [0.0, 1.4, 0.0]])
PLACED_BOND_VECTORS = np.array([[1.0, 0.4, 0.0], [1.0, -0.4, 0.0]])
FREE_ANCHORS = np.array([[0.0, -0.7, 0.0], [0.0, 0.7, 0.0]])
FREE_BOND_VECTORS = np.array([[1.0, 0.0, 0.2], [1.0, 0.0, -0.1]])


def random_free_side(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Free side of the junction under an arbitrary rigid transformation"""
    rng = np.random.default_rng(seed)
    transform = RigidTransform.from_components(
        translation=rng.uniform(-5, 5, size=3),
        rotation=Rotation.random(random_state=rng),
    )
    return transform.apply(FREE_ANCHORS), transform.rotation.apply(FREE_BOND_VECTORS)


@pytest.mark.parametrize("seed", range(5))
def test_docking_bond_lengths_exact(seed: int) -> None:
    """Test both bonds of a docked junction have exactly the target length"""
    free_anchors, free_bond_vectors = random_free_side(seed)
    transform = rigid_junction_docking(
        PLACED_ANCHORS,
        PLACED_BOND_VECTORS,
        free_anchors,
        free_bond_vectors,
        BOND_LENGTH,
    )
    bond_lengths = np.linalg.norm(
        transform.apply(free_anchors) - PLACED_ANCHORS, axis=1
    )
    assert np.allclose(bond_lengths, BOND_LENGTH, atol=1e-10)


def test_docking_rotation_is_proper() -> None:
    """Test docking never reflects the free body"""
    free_anchors, free_bond_vectors = random_free_side(0)
    transform = rigid_junction_docking(
        PLACED_ANCHORS,
        PLACED_BOND_VECTORS,
        free_anchors,
        free_bond_vectors,
        BOND_LENGTH,
    )
    assert np.isclose(np.linalg.det(transform.rotation.as_matrix()), 1.0)


def test_docking_faces_bodies_away_from_each_other() -> None:
    """The docked free body's bonds should point back towards the placed body"""
    free_anchors, free_bond_vectors = random_free_side(1)
    transform = rigid_junction_docking(
        PLACED_ANCHORS,
        PLACED_BOND_VECTORS,
        free_anchors,
        free_bond_vectors,
        BOND_LENGTH,
    )
    docked_outward = transform.rotation.apply(free_bond_vectors).mean(axis=0)
    placed_outward = PLACED_BOND_VECTORS.mean(axis=0)
    assert np.dot(docked_outward, placed_outward) < 0.0
    # docked anchors sit on the side the placed bonds point to
    assert np.all(transform.apply(free_anchors)[:, 0] > 0.0)


def test_docking_independent_of_bond_order() -> None:
    """Test the order in which bonds are listed does not matter"""
    free_anchors, free_bond_vectors = random_free_side(2)
    forward = rigid_junction_docking(
        PLACED_ANCHORS,
        PLACED_BOND_VECTORS,
        free_anchors,
        free_bond_vectors,
        BOND_LENGTH,
    )
    reverse = rigid_junction_docking(
        PLACED_ANCHORS[::-1],
        PLACED_BOND_VECTORS[::-1],
        free_anchors[::-1],
        free_bond_vectors[::-1],
        BOND_LENGTH,
    )
    assert np.allclose(forward.as_matrix(), reverse.as_matrix())


def test_docking_respects_pairing() -> None:
    """Swapping which free anchor bonds to which placed anchor flips the docked body"""
    free_anchors, free_bond_vectors = random_free_side(3)
    swapped = rigid_junction_docking(
        PLACED_ANCHORS,
        PLACED_BOND_VECTORS,
        free_anchors[::-1],
        free_bond_vectors[::-1],
        BOND_LENGTH,
    )
    docked = swapped.apply(free_anchors[::-1])
    assert np.allclose(np.linalg.norm(docked - PLACED_ANCHORS, axis=1), BOND_LENGTH)


def test_docking_coincident_anchors_raises() -> None:
    """Test junctions whose anchors coincide are rejected"""
    with pytest.raises(ValueError):
        rigid_junction_docking(
            np.zeros((2, 3)),
            PLACED_BOND_VECTORS,
            FREE_ANCHORS,
            FREE_BOND_VECTORS,
            BOND_LENGTH,
        )


def test_docking_bonds_along_pair_axis_raises() -> None:
    """Test junctions with no defined outward direction are rejected"""
    with pytest.raises(ValueError):
        rigid_junction_docking(
            PLACED_ANCHORS,
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            FREE_ANCHORS,
            FREE_BOND_VECTORS,
            BOND_LENGTH,
        )


def test_docking_unbridgeable_spans_raises() -> None:
    """Test anchor spacings too mismatched to bond are rejected"""
    with pytest.raises(ValueError):
        rigid_junction_docking(
            10 * PLACED_ANCHORS,
            PLACED_BOND_VECTORS,
            FREE_ANCHORS,
            FREE_BOND_VECTORS,
            BOND_LENGTH,
        )


def test_docking_bad_shapes_raises() -> None:
    """Test junction arrays which are not (2, 3) are rejected"""
    with pytest.raises(ValueError):
        rigid_junction_docking(
            np.zeros((3, 3)),
            PLACED_BOND_VECTORS,
            FREE_ANCHORS,
            FREE_BOND_VECTORS,
            BOND_LENGTH,
        )


def test_hinge_rotation_fixes_axis_points() -> None:
    """Test hinge rotations leave the hinge axis fixed"""
    hinge = hinge_rotation(PLACED_ANCHORS[0], PLACED_ANCHORS[1], angle_rad=0.7)
    assert np.allclose(hinge.apply(PLACED_ANCHORS), PLACED_ANCHORS)
    assert not np.allclose(hinge.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])


@pytest.mark.parametrize("angle_rad", [-0.3, 0.2, 1.0])
def test_hinge_after_docking_preserves_bond_lengths(angle_rad: float) -> None:
    """Test hinging a docked body keeps both bonds at the target length"""
    free_anchors, free_bond_vectors = random_free_side(4)
    transform = hinge_rotation(PLACED_ANCHORS[0], PLACED_ANCHORS[1], angle_rad) * (
        rigid_junction_docking(
            PLACED_ANCHORS,
            PLACED_BOND_VECTORS,
            free_anchors,
            free_bond_vectors,
            BOND_LENGTH,
        )
    )
    bond_lengths = np.linalg.norm(
        transform.apply(free_anchors) - PLACED_ANCHORS, axis=1
    )
    assert np.allclose(bond_lengths, BOND_LENGTH)


def test_hinge_degenerate_axis_raises() -> None:
    """Test hinge axes through coincident points are rejected"""
    with pytest.raises(ValueError):
        hinge_rotation(np.ones(3), np.ones(3), angle_rad=0.1)
