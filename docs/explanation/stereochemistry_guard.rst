Keeping stereocenters through coordinate building
=================================================

A MuPT system usually starts from coordinates that already have the right
chemistry. :func:`~mupt.interfaces.smiles.primitive_from_smiles` embeds each
residue from a SMILES string, and every ``@`` or ``@@`` in that string becomes
a real handedness in the embedded geometry. Coordinate builders then move the
atoms. Rigid transformations don't change stereochemistry. Physics-based
builders may move individual atoms, potentially inverting stereocenters. In
all-atom DPD
(:class:`~mupt.builders.all_atom_dpd.AllAtomDPDBuilder`) the pair repulsion is
soft, so an atom can pass through a stereocenter and invert it. A later
classical force field cannot invert it back, and the simulation silently
describes a different molecule (`#116
<https://github.com/MuPT-hub/mupt/issues/116>`_).

The stereochemistry guard in :mod:`mupt.chemistry.stereo` detects this. It
reads the handedness of every stereocenter in the starting coordinates, and
reports any center whose handedness differs in a later set of coordinates.
It is a diagnostic on coordinates only. Primitives do not store
stereocenters or labels. The all-atom DPD builder can also restrain
stereocenters while it runs; see `Preserving stereocenters in all-atom DPD`_.


Using the guard
---------------

Capture a reference before building, and check the result afterwards:

.. code-block:: python

    from mupt.chemistry.stereo import capture_stereo, assert_stereo_preserved

    reference = capture_stereo(root)        # starting coordinates
    result = builder.build(root)            # any builder; AA-DPD mutates root
    assert_stereo_preserved(root, reference, box_lengths=result.box_lengths_a)

:func:`~mupt.chemistry.stereo.check_stereo` returns a
:class:`~mupt.chemistry.stereo.StereoReport` instead of raising. Every
recorded center gets one status: ``preserved``, ``inverted``,
``near_planar`` (flattened, so its handedness is ambiguous) or ``nonfinite``.
Both functions also accept a plain ``(N, 3)`` array of positions in place of
a Primitive, for checking coordinates that have not been written back yet.
Pass ``box_lengths`` when the coordinates are wrapped into a periodic box.


What counts as a stereocenter
-----------------------------

The guard operates on a flat slice of the hierarchy: every atom and bond below
the Primitive it is given, regardless of residues or other grouping. Every
atom with four bonded neighbors is checked, with no exceptions. That includes
CH2 and CH3 carbons, whose inversion would give back the same molecule, so the
contract is simply that no tetrahedral center changes handedness. Hydrogens
must be explicit. To check only some atoms, pass their indices as ``centers``.
:func:`~mupt.chemistry.stereo.potential_stereocenters` gives the indices of the
atoms RDKit perceives as potential tetrahedral stereocenters (specified or not)
in an RDKit molecule, for checking only the centers that matter chemically.


Handedness at residue junctions
-------------------------------

A stereocenter at a residue junction has one neighbor in the adjacent residue.
Before residues are placed, that neighbor can be anywhere, so
:func:`~mupt.chemistry.stereo.capture_stereo` uses the linker of the Connector
that forms the bond instead. The linker sits where the residue SMILES put its
``*``, so the reference is the handedness the residue template defines. Inside
a residue, the linker and the neighbor atom are in the same place.

The check always uses atom positions. Some builders, including AA-DPD, move
atoms but not Connectors, so capture the reference before building. To capture
from a builder's output, pass ``use_connector_positions=False``.


How handedness is measured
--------------------------

For a center with neighbors a, b, c and d (in atom index order), the guard
computes the signed volume ``det(a - d, b - d, c - d)`` and divides it by the
three edge lengths, giving a value between -1 and 1 that does not depend on
bond lengths. A reflection flips its sign. Rotations and translations do not
change it. The sign is compared between two geometries of the same atoms, so
we don't need to assign R/S.

Each recorded center also stores ``phi0``, the signed dihedral of its
neighbors a-b-c-d (see :func:`~mupt.chemistry.stereo.signed_dihedral`). Like
the volume, its sign flips under reflection, and builders can use it to
restrain a center's handedness.


Preserving stereocenters in all-atom DPD
----------------------------------------

Detecting an inversion after the fact doesn't fix it, so
:class:`~mupt.builders.all_atom_dpd.AllAtomDPDBuilder` can also prevent it.
Set ``preserve_stereo=True`` in
:class:`~mupt.builders.all_atom_dpd.AllAtomDPDSettings`:

.. code-block:: python

    from mupt.builders.all_atom_dpd import AllAtomDPDBuilder, AllAtomDPDSettings

    settings = AllAtomDPDSettings(preserve_stereo=True, resname_map=resname_map)
    result = AllAtomDPDBuilder(settings=settings).build(root)
    result.diagnostics["stereo"]  # {"passed": True, "n_centers": ..., ...}

With this setting, the builder:

1. Captures a reference from the input coordinates before anything moves,
   using the residue templates' linkers for junction centers as described above.
   ``stereo_centers="stereocenters"`` (the default) selects RDKit's potential
   tetrahedral stereocenters; ``"all_tetrahedral"`` selects every atom with
   four bonded neighbors, as the guard does by default.
2. Restrains every selected center with a periodic dihedral on its four
   substituents, ``U = k/2 [1 - cos(phi - phi0)]``. A harmonic improper can't
   tell a center from its mirror image, but this term can. The force constant is
   ``stereo_restraint_k``.
3. Records the number of inverted, near-planar and nonfinite centers at every
   convergence check (``stereo_inverted`` and so on in the diagnostics).
4. Audits the final coordinates. With ``stereo_on_failure="raise"`` (the
   default), a failed audit raises
   :class:`~mupt.chemistry.stereo.StereoInversionError` without writing any
   coordinates to the hierarchy; the error carries the ``report`` and the
   partial build ``result``. With ``"warn"``, it logs a warning and writes the
   coordinates anyway.

The restraint has to stop inversions rather than undo them. Once a center has
passed through planarity, the stiff angle terms of the DPD run keep it from
flipping back, so a weak restraint barely helps, and running longer doesn't
either. On polypropylene chains, the default ``stereo_restraint_k = 5000``
kept every backbone stereocenter over 20 seeds without slowing convergence,
while much larger values stall it. A CH2 or CH3 flip gives back the same
molecule, which is why only potential stereocenters are restrained by default.

The restraint is a finite bias, not a constraint, so the final audit is always
run. Restraint terms share the builder's periodic dihedral force, so
``dihedral_energy`` in the diagnostics includes their energy, while
``counts["dihedral"]`` counts only force-field dihedrals and
``counts["stereo_restraint"]`` counts the restraints.
