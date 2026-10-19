"""Deterministic, RDKit-free Primitives for testing random walk builders"""

from typing import Optional

import numpy as np

from mupt.geometry.shapes import PointCloud
from mupt.mupr.connection import AttachmentPoint, Connector
from mupt.mupr.primitives import Primitive, PrimitiveHandle


HEAD_CONN = ("head", 0)
TAIL_CONN = ("tail", 0)


def synthetic_unit(rng: np.random.Generator, n_points: int = 6) -> Primitive:
    """
    Leaf Primitive with a random point cloud shape and two bondable
    Connectors (head and tail) anchored on its first and last points
    """
    positions = rng.normal(scale=1.5, size=(n_points, 3))
    head_anchor, tail_anchor = positions[0], positions[-1]
    centroid = positions.mean(axis=0)

    def outward(anchor: np.ndarray) -> np.ndarray:
        direction = anchor - centroid
        return anchor + direction / np.linalg.norm(direction)

    head = Connector(
        anchor=AttachmentPoint({"h"}, position=head_anchor),
        linker=AttachmentPoint({"t"}, position=outward(head_anchor)),
        label="head",
    )
    tail = Connector(
        anchor=AttachmentPoint({"t"}, position=tail_anchor),
        linker=AttachmentPoint({"h"}, position=outward(tail_anchor)),
        label="tail",
    )
    return Primitive(shape=PointCloud(positions), connectors=[head, tail], label="unit")


def synthetic_chains(
    chain_lengths: tuple[int, ...] = (5,),
    seed: int = 0,
    parent: Optional[Primitive] = None,
) -> tuple[Primitive, list[list[PrimitiveHandle]]]:
    """
    Primitive whose children form one disconnected linear chain per entry
    of chain_lengths, each unit bonded tail-to-head to the next one
    """
    rng = np.random.default_rng(seed)
    if parent is None:
        parent = Primitive(label="chains")

    all_handles = []
    for chain_length in chain_lengths:
        handles = [
            parent.attach_child(synthetic_unit(rng)) for _ in range(chain_length)
        ]
        for handle_1, handle_2 in zip(handles[:-1], handles[1:]):
            parent.connect_children(handle_1, TAIL_CONN, handle_2, HEAD_CONN)
        all_handles.append(handles)
    parent.check_self_consistent()

    return parent, all_handles


# scenarios recorded in random_walk_golden.py from the pre-patch walker; each maps
# to (AngleConstrainedRandomWalk kwargs, rng seed or None for the legacy global
# NumPy RNG, chain lengths). Chains are always built with synthetic_chains(seed=11)
GOLDEN_SCENARIOS: dict[str, tuple[dict, Optional[int], tuple[int, ...]]] = {
    "generator_rng": (
        dict(bond_length=1.5, angle_max_rad=np.pi / 4),
        42,
        (5,),
    ),
    "fixed_start_two_chains": (
        dict(
            bond_length=0.8,
            angle_max_rad=np.pi / 3,
            initial_point=np.array([1.0, -2.0, 3.0]),
            initial_direction=np.array([0.0, 0.0, 1.0]),
        ),
        7,
        (4, 3),
    ),
    "legacy_global_rng": (
        dict(bond_length=1.2, angle_max_rad=np.pi / 6),
        None,
        (6,),
    ),
}
GOLDEN_CHAIN_SEED: int = 11
GOLDEN_GLOBAL_SEED: int = 1234  # seeds np.random before every scenario
