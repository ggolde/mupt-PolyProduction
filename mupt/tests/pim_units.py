"""PIM-1 repeat units for testing Primitives with multiple bonds between children"""

from functools import lru_cache

from rdkit import Chem

from mupt.chemistry.rdkit.rdloggers import suppress_rdkit_logs
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.mupr.connection import ConnectorHandle
from mupt.mupr.primitives import Primitive, PrimitiveHandle


# port numbers are the atom map numbers on wild-type atoms,
# with each pair of ports sitting adjacent to each other on the same ring
TTSBI_SMILES = "C12(CC(C)(C)c3cc(O[*:3])c(O[*:4])cc31)CC(C)(C)c1cc(O[*:1])c(O[*:2])cc12"
TFTPN_SMILES = "N#Cc1c([*:5])c([*:6])c(C#N)c([*:7])c1[*:8]"

# (TTSBI port, TFTPN port) bonded pairs for each dibenzodioxin linkage pattern;
# the TTSBI->TFTPN junction uses ports 3/4, the TFTPN->TTSBI junction ports 1/2
LINKAGE_PATTERNS: dict[
    str, tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
] = {
    "1-3/2-4": (((3, 5), (4, 6)), ((7, 1), (8, 2))),
    "1-4/2-3": (((3, 6), (4, 5)), ((7, 2), (8, 1))),
}


def unit_with_ports(
    smiles: str,
    embed_positions: bool = False,
) -> tuple[Primitive, dict[int, ConnectorHandle]]:
    """
    Build a Primitive from SMILES, along with a map from wild-type atom
    map numbers to the handles of the corresponding Connectors

    Makes all Connectors mutually bondable by designating
    the wild-type symbol as attachable to each anchor

    Each SMILES is only embedded once per session (successive RDKit embeddings
    differ slightly), and a fresh copy of that same Primitive is returned every call
    """
    prim, ports = _cached_unit_with_ports(smiles, embed_positions)
    return prim.copy(), dict(ports)


@lru_cache(maxsize=None)
def _cached_unit_with_ports(
    smiles: str,
    embed_positions: bool,
) -> tuple[Primitive, dict[int, ConnectorHandle]]:
    mol = Chem.MolFromSmiles(smiles)
    idx_to_map_num = {
        atom.GetIdx(): atom.GetAtomMapNum()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    }
    with suppress_rdkit_logs():
        prim = primitive_from_smiles(smiles, embed_positions=embed_positions)
    for conn in prim.connectors.values():
        conn.anchor.attachables.add("*")

    return prim, {
        idx_to_map_num[conn.linker.attachment]: conn_handle
        for conn_handle, conn in prim.connectors.items()
    }


def pim_ladder_chain(
    num_units: int,
    linkage: str = "1-3/2-4",
    embed_positions: bool = True,
    label: str = "pim",
) -> tuple[Primitive, list[PrimitiveHandle]]:
    """
    Alternating TTSBI-TFTPN-TTSBI-... ladder chain with
    two bonds across every junction, using the given linkage pattern
    """
    ttsbi, ports_ttsbi = unit_with_ports(TTSBI_SMILES, embed_positions=embed_positions)
    tftpn, ports_tftpn = unit_with_ports(TFTPN_SMILES, embed_positions=embed_positions)
    ttsbi_to_tftpn, tftpn_to_ttsbi = LINKAGE_PATTERNS[linkage]

    chain = Primitive(label=label)
    handles = [
        chain.attach_child((ttsbi if (i % 2 == 0) else tftpn).copy())
        for i in range(num_units)
    ]
    for i, (handle_1, handle_2) in enumerate(zip(handles[:-1], handles[1:])):
        if i % 2 == 0:  # TTSBI -> TFTPN
            for port_ttsbi, port_tftpn in ttsbi_to_tftpn:
                chain.connect_children(
                    handle_1, ports_ttsbi[port_ttsbi], handle_2, ports_tftpn[port_tftpn]
                )
        else:  # TFTPN -> TTSBI
            for port_tftpn, port_ttsbi in tftpn_to_ttsbi:
                chain.connect_children(
                    handle_1, ports_tftpn[port_tftpn], handle_2, ports_ttsbi[port_ttsbi]
                )
    chain.check_self_consistent()

    return chain, handles
