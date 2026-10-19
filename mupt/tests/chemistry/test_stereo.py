"""Unit tests for the tetrahedral stereocenter guard"""
# ruff: noqa: D103 ("undocumented public function")

from typing import Optional

import networkx as nx
import numpy as np
import pytest
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdmolfiles import MolFromSmiles
from scipy.spatial.transform import RigidTransform, Rotation

from mupt.chemistry.rdkit.sanitization import sanitized_mol
from mupt.chemistry.smiles import DEFAULT_SMILES_READ_PARAMS
from mupt.chemistry.stereo import (
    StereoInversionError,
    assert_stereo_preserved,
    atom_positions,
    atomic_bond_graph,
    capture_stereo,
    capture_stereo_from_arrays,
    check_stereo,
    normalized_volume,
    potential_stereocenters,
    signed_dihedral,
)
from mupt.chemistry.rdkit.rdloggers import suppress_rdkit_logs
from mupt.interfaces.rdkit import primitive_from_rdkit
from mupt.mupr.primitives import Primitive
from mupt.geometry.shapes import PointCloud

MIRROR = np.array([-1.0, 1.0, 1.0])
MENTHOL = "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O"
MESO_TARTARIC_ACID = "O[C@H](C(=O)O)[C@@H](O)C(=O)O"


def seeded_primitive(
    smiles: str, seed: int = 20261012, label: Optional[str] = None
) -> Primitive:
    """Like primitive_from_smiles(embed_positions=True), but with a seeded embedding"""
    with suppress_rdkit_logs():
        rdmol = sanitized_mol(
            MolFromSmiles(smiles, params=DEFAULT_SMILES_READ_PARAMS), add_Hs=True
        )
        conformer_idx = EmbedMolecule(rdmol, randomSeed=seed)
        return primitive_from_rdkit(rdmol, conformer_idx=conformer_idx, label=label)


def positions_of(prim: Primitive) -> np.ndarray:
    atoms, _ = atomic_bond_graph(prim)
    return atom_positions(atoms)


def center_signs(prim: Primitive, atoms: list[int]) -> list[int]:
    signs = {center.atom: center.sign for center in capture_stereo(prim).centers}
    return [signs[atom] for atom in atoms]


# geometry
def test_normalized_volume_flips_under_reflection_only() -> None:
    center = np.zeros(3)
    neighbors = np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ])
    volume = normalized_volume(center, neighbors)
    assert abs(volume) > 0.5

    assert normalized_volume(center, neighbors * MIRROR) == pytest.approx(-volume)
    rotation = Rotation.from_euler("xyz", [0.3, -1.1, 2.0]).as_matrix()
    shift = np.array([4.0, -2.0, 7.0])
    moved = neighbors @ rotation.T + shift
    assert normalized_volume(center + shift, moved) == pytest.approx(volume)
    assert normalized_volume(center, 10.0 * neighbors) == pytest.approx(volume)


def test_normalized_volume_degenerate_inputs() -> None:
    assert np.isnan(normalized_volume(np.zeros(3), np.full((4, 3), np.nan)))
    assert np.isnan(normalized_volume(np.zeros(3), np.ones((3, 3))))
    assert normalized_volume(np.zeros(3), np.ones((4, 3))) == 0.0


# capture
def test_adapter_graph_matches_rdkit() -> None:
    with suppress_rdkit_logs():
        mol = sanitized_mol(MolFromSmiles(MENTHOL), add_Hs=True)
    atoms, bonds = atomic_bond_graph(seeded_primitive(MENTHOL))
    assert [atom.element.symbol for atom in atoms] == [
        atom.GetSymbol() for atom in mol.GetAtoms()
    ]
    assert bonds == sorted(
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()
    )


def test_every_four_coordinate_atom_is_a_center() -> None:
    # 2-butanol: both CH3, the CH and the CH2, but not the hydroxyl oxygen
    reference = capture_stereo(seeded_primitive("C[C@H](O)CC"))
    assert [center.atom for center in reference.centers] == [0, 1, 3, 4]


def test_enantiomers_have_opposite_signs() -> None:
    # same atom order, opposite configuration at C2 (atom 1)
    sinister = seeded_primitive("C[C@H](O)CC")
    rectus = seeded_primitive("C[C@@H](O)CC")
    assert center_signs(sinister, [1]) == [-sign for sign in center_signs(rectus, [1])]


def test_sign_is_a_property_of_configuration_not_conformer() -> None:
    signs = {
        center_signs(seeded_primitive(MENTHOL, seed=seed), [3])[0] for seed in range(5)
    }
    assert len(signs) == 1


def test_explicit_centers_are_honored() -> None:
    reference = capture_stereo(seeded_primitive(MENTHOL), centers=[6])
    assert [center.atom for center in reference.centers] == [6]


def test_capture_rejects_invalid_centers() -> None:
    with pytest.raises(ValueError, match="neighbors"):
        capture_stereo(seeded_primitive("C[C@H](O)CC"), centers=[2])  # oxygen


def test_capture_rejects_planar_reference() -> None:
    elements = ["C", "F", "Cl", "Br", "H"]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    flat = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float
    )
    with pytest.raises(ValueError, match="near-planar"):
        capture_stereo_from_arrays(elements, bonds, flat)


# checking
@pytest.mark.parametrize("smiles", ["F[C@H](Cl)Br", MESO_TARTARIC_ACID, MENTHOL])
def test_reflection_inverts_every_center(smiles: str) -> None:
    prim = seeded_primitive(smiles)
    reference = capture_stereo(prim)
    positions = positions_of(prim)
    assert check_stereo(positions, reference).passed

    report = check_stereo(positions * MIRROR, reference)
    assert report.inverted == [center.atom for center in reference.centers]
    with pytest.raises(StereoInversionError) as excinfo:
        assert_stereo_preserved(positions * MIRROR, reference)
    assert excinfo.value.report.inverted == report.inverted


def test_proper_motion_preserves_every_center() -> None:
    prim = seeded_primitive(MENTHOL)
    reference = capture_stereo(prim)
    rotation = Rotation.from_euler("zyx", [1.0, 2.0, -0.5]).as_matrix()
    moved = positions_of(prim) @ rotation.T + np.array([10.0, -3.0, 5.0])
    assert assert_stereo_preserved(moved, reference).passed


def test_swapping_substituents_inverts_only_that_center() -> None:
    prim = seeded_primitive(MESO_TARTARIC_ACID)
    reference = capture_stereo(prim)
    positions = positions_of(prim)
    # exchange the hydroxyl oxygen (0) and carboxyl carbon (2) around C2 (atom 1)
    swapped = positions.copy()
    swapped[0], swapped[2] = positions[2], positions[0]
    report = check_stereo(swapped, reference)
    assert report.inverted == [1]
    assert len(report.with_status("preserved")) == len(reference.centers) - 1


def test_methyl_hydrogen_swap_is_reported() -> None:
    # the contract has no exceptions: a CH3 inversion is reported too
    prim = seeded_primitive("C[C@H](O)CC")
    reference = capture_stereo(prim)
    positions = positions_of(prim)
    methyl = reference.centers[0]
    first, second = methyl.neighbors[1], methyl.neighbors[2]  # two hydrogens
    swapped = positions.copy()
    swapped[first], swapped[second] = positions[second], positions[first]
    assert check_stereo(swapped, reference).inverted == [methyl.atom]


def test_flattened_and_nonfinite_centers_are_reported() -> None:
    prim = seeded_primitive("C[C@H](O)CC")
    reference = capture_stereo(prim)
    center = reference.centers[1]
    positions = positions_of(prim)

    flattened = positions.copy()
    flattened[list(center.neighbors), 2] = flattened[center.atom, 2]
    flattened[center.atom] = flattened[list(center.neighbors)].mean(axis=0)
    report = check_stereo(flattened, reference)
    assert center.atom in report.near_planar and not report.passed

    broken = positions.copy()
    broken[center.atom] = np.nan
    # every center that includes the broken atom, as center or neighbor
    assert check_stereo(broken, reference).nonfinite == [0, 1, 3]


def test_wrapped_coordinates_need_box() -> None:
    prim = seeded_primitive("C[C@H](O)CC")
    reference = capture_stereo(prim, centers=[1])
    box = np.array([12.0, 12.0, 12.0])
    wrapped = positions_of(prim)
    wrapped[reference.centers[0].neighbors[0]] += box * np.array([1.0, 0.0, -1.0])
    assert check_stereo(wrapped, reference, box_lengths=box).passed


def test_check_against_primitive() -> None:
    prim = seeded_primitive(MENTHOL)
    reference = capture_stereo(prim)
    assert check_stereo(prim, reference).passed

    atoms, _ = atomic_bond_graph(prim)
    for atom in atoms:
        atom.shape = PointCloud(positions=np.asarray(atom.shape.centroid) * MIRROR)
    assert len(check_stereo(prim, reference).inverted) == len(reference.centers)

    with pytest.raises(ValueError, match="do not match"):
        check_stereo(seeded_primitive("C[C@H](O)CC"), reference)


# centers spanning residues
RESIDUE_SMILES = {
    "head": "CC[C@H:2](C)-*",
    "mid": "*-[CH2:1][C@H:2](C)-*",
    "mid_inverted": "*-[CH2:1][C@@H:2](C)-*",
    "tail": "*-[CH2:1]C(C)C",
}


def polypropylene_chain(sequence: list[str]) -> Primitive:
    """A chain of polypropylene residue templates, bonded but not placed"""
    templates = {
        name: seeded_primitive(RESIDUE_SMILES[name], label=name)
        for name in set(sequence)
    }
    chain = Primitive(label="chain")
    handles = [chain.attach_child(templates[name].copy()) for name in sequence]
    chain.set_topology(
        nx.path_graph(handles, create_using=nx.Graph),
        max_registration_iter=100,
    )
    return chain


def backbone_methines(chain: Primitive) -> list[int]:
    """Backbone CH carbons, whose fourth neighbor is in the next residue"""
    atoms, bonds = atomic_bond_graph(chain)
    graph = nx.Graph(bonds)
    return [
        idx
        for idx, atom in enumerate(atoms)
        if atom.element.symbol == "C"
        and graph.degree[idx] == 4
        and sum(atoms[nb].element.symbol == "H" for nb in graph[idx]) == 1
        and any(atoms[nb].parent is not atom.parent for nb in graph[idx])
    ]


def test_junction_centers_are_captured_from_residue_templates() -> None:
    chain = polypropylene_chain(["head", "mid", "mid", "mid", "tail"])
    methines = backbone_methines(chain)
    assert len(methines) == 4
    # isotactic: every residue template defines the same handedness
    assert len(set(center_signs(chain, methines))) == 1


def test_junction_handedness_follows_residue_template() -> None:
    iso = polypropylene_chain(["head", "mid", "mid", "tail"])
    syndio = polypropylene_chain(["head", "mid", "mid_inverted", "tail"])
    methines = backbone_methines(iso)
    assert methines == backbone_methines(syndio)
    iso_signs, syndio_signs = (
        center_signs(iso, methines),
        center_signs(syndio, methines),
    )
    assert iso_signs[1] == syndio_signs[1]
    assert iso_signs[2] == -syndio_signs[2]


def test_moving_residues_rigidly_keeps_junction_handedness() -> None:
    chain = polypropylene_chain(["head", "mid", "mid", "mid", "tail"])
    before = capture_stereo(chain)
    rng = np.random.default_rng(7)
    for residue in chain.children:
        residue.rigidly_transform(
            RigidTransform.from_components(
                rotation=Rotation.random(random_state=rng),
                translation=rng.uniform(-20, 20, size=3),
            )
        )
    after = capture_stereo(chain)
    assert [c.sign for c in after.centers] == [c.sign for c in before.centers]


def polypropylene_universe(
    sequence: list[str],
) -> tuple[Primitive, dict[str, str]]:
    """A polypropylene chain in a SAAMR universe, with residue names for export"""
    from mupt.roles import assign_SAAMR_roles

    resname_map = {"head": "PPH", "mid": "PPM", "mid_inverted": "PPI", "tail": "PPT"}
    chain = polypropylene_chain(sequence)
    for residue in chain.children:
        residue.metadata["residue_name"] = resname_map[residue.label]
    root = Primitive(label="pp")
    root.attach_child(chain)
    assign_SAAMR_roles(root)
    return root, resname_map


# fork extensions: reference dihedrals, atom ordering, perception, summaries
def test_signed_dihedral_flips_under_reflection_only() -> None:
    points = np.random.default_rng(0).normal(size=(4, 3))
    dihedral = signed_dihedral(points)
    assert signed_dihedral(points * MIRROR) == pytest.approx(-dihedral)
    rotation = Rotation.from_euler("xyz", [0.4, 1.2, -2.2]).as_matrix()
    moved = points @ rotation.T + np.array([1.0, -5.0, 2.0])
    assert signed_dihedral(moved) == pytest.approx(dihedral)


def test_signed_dihedral_rejects_degenerate_points() -> None:
    with pytest.raises(ValueError, match="undefined"):
        signed_dihedral(np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]))
    with pytest.raises(ValueError, match="finite"):
        signed_dihedral(np.full((4, 3), np.nan))


def test_reference_dihedral_follows_handedness() -> None:
    prim = seeded_primitive(MENTHOL)
    reference = capture_stereo(prim)
    assert all(0.0 <= center.phi0 < 2 * np.pi for center in reference.centers)

    mirrored = capture_stereo_from_arrays(
        reference.elements, atomic_bond_graph(prim)[1], positions_of(prim) * MIRROR
    )
    for center, mirror_center in zip(reference.centers, mirrored.centers):
        assert mirror_center.sign == -center.sign
        assert np.isclose(np.sin(mirror_center.phi0), -np.sin(center.phi0))


def test_reference_dihedral_invariant_under_rigid_motion() -> None:
    prim = seeded_primitive(MENTHOL)
    reference = capture_stereo(prim)
    prim.rigidly_transform(
        RigidTransform.from_components(
            rotation=Rotation.from_euler("zyx", [0.3, 2.0, -1.0]),
            translation=[3.0, 4.0, -5.0],
        )
    )
    moved = capture_stereo(prim)
    assert np.allclose(
        [center.phi0 for center in moved.centers],
        [center.phi0 for center in reference.centers],
    )


def test_explicit_atom_ordering_permutes_centers() -> None:
    prim = seeded_primitive("C[C@H](O)CC")
    atoms, _ = atomic_bond_graph(prim)
    order = list(range(len(atoms)))[::-1]
    reordered_atoms = [atoms[idx] for idx in order]
    reference = capture_stereo(prim)
    reordered = capture_stereo(prim, atoms=reordered_atoms)

    new_index = {old: new for new, old in enumerate(order)}
    assert sorted(center.atom for center in reordered.centers) == sorted(
        new_index[center.atom] for center in reference.centers
    )
    assert reordered.elements == tuple(atom.element.symbol for atom in reordered_atoms)
    assert check_stereo(prim, reordered, atoms=reordered_atoms).passed

    mirrored = positions_of(prim)[order] * MIRROR
    assert len(check_stereo(mirrored, reordered).inverted) == len(reordered.centers)


def test_explicit_atom_ordering_must_cover_every_atom() -> None:
    prim = seeded_primitive("C[C@H](O)CC")
    atoms, _ = atomic_bond_graph(prim)
    with pytest.raises(ValueError, match="each atom"):
        capture_stereo(prim, atoms=atoms[:-1])


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (MENTHOL, [3, 6, 9]),
        (MESO_TARTARIC_ACID, [1, 5]),
        ("C[C@H](O)CC", [1]),
        ("CC(O)CC", [1]),  # unspecified, but still a potential stereocenter
        ("CCO", []),
    ],
)
def test_potential_stereocenters(smiles: str, expected: list[int]) -> None:
    with suppress_rdkit_logs():
        mol = sanitized_mol(MolFromSmiles(smiles), add_Hs=True)
    assert potential_stereocenters(mol) == expected


def test_potential_stereocenters_of_polymer_are_backbone_methines() -> None:
    from mupt.interfaces.rdkit import primitive_to_rdkit_mols

    root, resname_map = polypropylene_universe(["head", "mid", "mid", "mid", "tail"])
    (mol,) = primitive_to_rdkit_mols(
        root, resname_map=resname_map, default_atom_position=np.zeros(3)
    )
    assert potential_stereocenters(mol) == backbone_methines(root)


def test_report_summary_is_serializable() -> None:
    import json

    prim = seeded_primitive(MENTHOL)
    reference = capture_stereo(prim)
    summary = check_stereo(positions_of(prim) * MIRROR, reference).summary()
    assert json.loads(json.dumps(summary)) == summary
    assert not summary["passed"]
    assert summary["n_centers"] == len(reference.centers)
    assert summary["inverted"] == [center.atom for center in reference.centers]
