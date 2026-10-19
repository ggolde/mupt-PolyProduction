"""Unit tests for spatial-hash collision checking"""

import pytest

import numpy as np

from mupt.geometry.collision import (
    SpatialHashIndex,
    minimum_image_displacement,
    wrap_positions,
)


BOX = np.array([10.0, 12.0, 14.0])


def test_wrap_positions_into_centered_box() -> None:
    """Test positions are wrapped into the origin-centered box"""
    wrapped = wrap_positions(np.array([[6.0, -7.0, 0.5], [-5.0, 0.0, 21.0]]), BOX)
    assert np.all(wrapped >= -BOX / 2) and np.all(wrapped < BOX / 2)
    assert np.allclose(wrapped, [[-4.0, 5.0, 0.5], [-5.0, 0.0, -7.0]])


def test_minimum_image_displacement() -> None:
    """Test displacements are reduced to their minimum image"""
    assert np.allclose(
        minimum_image_displacement(np.array([9.0, -11.0, 1.0]), BOX), [-1.0, 1.0, 1.0]
    )


@pytest.mark.parametrize("box_lengths", [None, BOX])
def test_collision_within_cutoff(box_lengths) -> None:
    """Test only points strictly within the cutoff collide"""
    index = SpatialHashIndex(cutoff=1.0, box_lengths=box_lengths)
    index.add(np.array([[0.0, 0.0, 0.0]]))
    assert index.collides(np.array([[0.5, 0.5, 0.0]]))
    assert not index.collides(np.array([[1.5, 0.0, 0.0]]))
    assert not index.collides(np.array([[1.0, 0.0, 0.0]]))  # strictly within cutoff


def test_periodic_collision_across_box_face() -> None:
    """Test points straddling a box face collide only if periodic"""
    straddling_pair = np.array([[4.8, 0.0, 0.0], [-4.8, 0.0, 0.0]])  # 0.4 apart via PBC

    periodic = SpatialHashIndex(cutoff=1.0, box_lengths=BOX)
    periodic.add(straddling_pair[:1])
    assert periodic.collides(straddling_pair[1:])

    unbounded = SpatialHashIndex(cutoff=1.0)
    unbounded.add(straddling_pair[:1])
    assert not unbounded.collides(straddling_pair[1:])


def test_periodic_collision_of_unwrapped_positions() -> None:
    """Test periodic images outside the primary box are detected"""
    index = SpatialHashIndex(cutoff=1.0, box_lengths=BOX)
    index.add(np.array([[1.0, 1.0, 1.0]]))
    assert index.collides(np.array([[11.2, -11.0, 15.0]]))  # image of (1.2, 1, 1)


def test_unbounded_collision_across_cell_boundaries() -> None:
    """Test collisions between points in adjacent cells are detected"""
    index = SpatialHashIndex(cutoff=1.0)
    index.add(np.array([[0.99, 0.99, 0.99]]))
    assert index.collides(np.array([[1.01, 1.01, 1.01]]))
    assert index.collides(np.array([[-0.005, 0.99, 0.99]]))


def test_box_smaller_than_cutoff() -> None:
    """Test boxes narrower than the cutoff still detect collisions"""
    index = SpatialHashIndex(cutoff=5.0, box_lengths=[3.0, 3.0, 3.0])
    assert np.all(index.n_cells == 1)
    index.add(np.array([[0.0, 0.0, 0.0]]))
    assert index.collides(np.array([[1.4, 1.4, 1.4]]))


def test_add_remove_round_trip() -> None:
    """Test batches of points can be removed by their tokens"""
    index = SpatialHashIndex(cutoff=1.0, box_lengths=BOX)
    token_1 = index.add(np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]]))
    token_2 = index.add(np.array([[-3.0, 2.0, 1.0]]))
    assert token_1 != token_2
    assert index.num_points == 3

    index.remove(token_1)
    assert index.num_points == 1
    assert not index.collides(np.array([[0.1, 0.0, 0.0], [3.0, 3.1, 3.0]]))
    assert index.collides(np.array([[-3.0, 2.1, 1.0]]))

    index.remove(token_2)
    assert index.num_points == 0
    assert not index.collides(np.array([[-3.0, 2.1, 1.0]]))


def test_empty_batches() -> None:
    """Test empty batches can be added, queried and removed"""
    index = SpatialHashIndex(cutoff=1.0)
    token = index.add(np.empty((0, 3)))
    assert not index.collides(np.empty((0, 3)))
    index.remove(token)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, np.inf, np.nan])
def test_invalid_cutoff(cutoff) -> None:
    """Test non-positive or non-finite cutoffs are rejected"""
    with pytest.raises(ValueError):
        SpatialHashIndex(cutoff=cutoff)


@pytest.mark.parametrize(
    "box_lengths", [[1.0, 1.0], [1.0, 0.0, 1.0], [1.0, np.inf, 1.0]]
)
def test_invalid_box_lengths(box_lengths) -> None:
    """Test malformed box lengths are rejected"""
    with pytest.raises(ValueError):
        SpatialHashIndex(cutoff=1.0, box_lengths=box_lengths)


def test_invalid_positions_shape() -> None:
    """Test positions which are not (N, 3) are rejected"""
    with pytest.raises(ValueError):
        SpatialHashIndex(cutoff=1.0).add(np.zeros((2, 2)))
