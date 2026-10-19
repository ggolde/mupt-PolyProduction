"""Spatial hashing for fast cutoff-based collision queries, with optional periodicity"""

from typing import Iterable, Optional
from collections import defaultdict
from itertools import product

import numpy as np

from .arraytypes import ArrayNx3, Vector3


def validated_box_lengths(box_lengths: Iterable[float]) -> Vector3:
    """Cast box lengths of an orthorhombic box to an array, checking they are valid"""
    box = np.asarray(box_lengths, dtype=float)
    if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("Box lengths must be three positive, finite values")

    return box


def wrap_positions(positions: ArrayNx3, box_lengths: Vector3) -> ArrayNx3:
    """Wrap positions into an origin-centered orthorhombic box of given lengths"""
    return positions - box_lengths * np.floor(
        (positions + box_lengths / 2.0) / box_lengths
    )


def minimum_image_displacement(
    displacement: ArrayNx3, box_lengths: Vector3
) -> ArrayNx3:
    """Apply the orthorhombic minimum-image convention to displacement vector(s)"""
    return displacement - box_lengths * np.round(displacement / box_lengths)


class SpatialHashIndex:
    """
    Mutable cell list for checking whether any of a batch of points
    lies within a cutoff distance of any point previously added

    Points are added in batches, each identified by an integer
    token which can later be used to remove that batch again

    If box_lengths are provided, space is treated as an orthorhombic periodic box
    centered on the origin, and distances follow the minimum-image convention.
    Otherwise, space is unbounded and cells are hashed without wrapping
    """

    MAX_CACHED_NEIGHBORHOODS: int = 4096

    def __init__(
        self, cutoff: float, box_lengths: Optional[Iterable[float]] = None
    ) -> None:
        self.cutoff = float(cutoff)
        if not np.isfinite(self.cutoff) or self.cutoff <= 0.0:
            raise ValueError("Collision cutoff must be positive and finite")

        self.box_lengths: Optional[Vector3] = None
        if box_lengths is not None:
            self.box_lengths = validated_box_lengths(box_lengths)
            # size cells so that the immediately-adjacent cells cover the cutoff
            self.n_cells = np.maximum(
                np.floor(self.box_lengths / self.cutoff).astype(int), 1
            )
            self.cell_widths = self.box_lengths / self.n_cells

        self._cells: dict[tuple[int, int, int], dict[int, ArrayNx3]] = defaultdict(dict)
        self._token_cells: dict[int, set[tuple[int, int, int]]] = {}
        self._neighborhoods: dict[
            tuple[int, int, int], tuple[tuple[int, int, int], ...]
        ] = {}
        self._next_token: int = 0

    @property
    def is_periodic(self) -> bool:
        """Whether distances are computed in a periodic box"""
        return self.box_lengths is not None

    @property
    def num_points(self) -> int:
        """Total number of points currently stored"""
        return sum(
            len(points)
            for batches in self._cells.values()
            for points in batches.values()
        )

    def _cell_indices(self, positions: ArrayNx3) -> np.ndarray:
        """Integer cell indices for each of a collection of positions"""
        if not self.is_periodic:
            return np.floor(positions / self.cutoff).astype(int)

        shifted = (
            wrap_positions(positions, self.box_lengths) + self.box_lengths / 2.0
        ) % self.box_lengths
        return np.floor(shifted / self.cell_widths).astype(int) % self.n_cells

    def _neighborhood(
        self, cell: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int], ...]:
        """All cells (including the given one) adjacent to a cell"""
        if cell not in self._neighborhoods:
            if len(self._neighborhoods) >= self.MAX_CACHED_NEIGHBORHOODS:
                self._neighborhoods.clear()  # bound memory over long searches

            neighbors = (
                np.array(cell) + np.array(offset)
                for offset in product((-1, 0, 1), repeat=3)
            )
            if self.is_periodic:
                neighbors = (neighbor % self.n_cells for neighbor in neighbors)
            # deduplicate, as wrapped neighbors coincide in boxes 1-2 cells wide
            self._neighborhoods[cell] = tuple({
                tuple(int(i) for i in neighbor) for neighbor in neighbors
            })

        return self._neighborhoods[cell]

    def _displacements(self, position: Vector3, points: ArrayNx3) -> ArrayNx3:
        displacements = points - position
        if self.is_periodic:
            displacements = minimum_image_displacement(displacements, self.box_lengths)

        return displacements

    def collides(self, positions: ArrayNx3) -> bool:
        """Whether any given position is strictly within the cutoff of a stored point"""
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if positions.size == 0:
            return False
        cutoff_squared = self.cutoff**2
        for position, cell in zip(positions, self._cell_indices(positions)):
            for neighbor_cell in self._neighborhood(tuple(int(i) for i in cell)):
                for points in self._cells.get(neighbor_cell, {}).values():
                    displacements = self._displacements(position, points)
                    if np.any(
                        np.einsum("ij,ij->i", displacements, displacements)
                        < cutoff_squared
                    ):
                        return True

        return False

    def add(self, positions: ArrayNx3) -> int:
        """Store a batch of positions, returning a token with which to remove them"""
        positions = np.array(np.atleast_2d(positions), dtype=float, copy=True)
        if positions.size == 0:
            positions = positions.reshape(0, 3)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Positions must be an (N, 3) array")

        token = self._next_token
        self._next_token += 1

        cells = self._cell_indices(positions)
        touched = set()
        for cell in np.unique(cells, axis=0):
            cell_key = tuple(int(i) for i in cell)
            self._cells[cell_key][token] = positions[np.all(cells == cell, axis=1)]
            touched.add(cell_key)
        self._token_cells[token] = touched

        return token

    def remove(self, token: int) -> None:
        """Remove a previously-added batch of positions by its token"""
        for cell_key in self._token_cells.pop(token):
            del self._cells[cell_key][token]
            if not self._cells[cell_key]:
                del self._cells[cell_key]
