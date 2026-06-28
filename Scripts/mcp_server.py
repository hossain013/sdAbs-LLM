import os
import asyncio
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from mcp.types import *

import openprotein
from openprotein.molecules import Complex, Protein, Ligand


# ---------------------------------------------------
# Load environment & authenticate
# ---------------------------------------------------
load_dotenv()

username = os.getenv("OP_USERNAME", None)
password = os.getenv("OP_PASSWORD")

if not username:
    raise RuntimeError("OPENPROTEIN_API_KEY not set")

session = openprotein.connect(username=username, password=password)


# ---------------------------------------------------
# Create FastMCP Server
# ---------------------------------------------------
server = FastMCP( 
    "openprotein_mcp", 
    dependencies=[ 
        "httpx", 
        "python-dotenv",
        "openprotein-python", 
        "pandas", 
        "biopython", 
        "modlamp", 
        "pfeature", 
        "mhcflurry", 
        # "epitopepredict", 
        # "mhcnuggets", 
        ] 
        )

# ===================================================
#   ESMFold
# ===================================================
@server.tool("predict_esmfold", "Predict structure using ESMFold")
async def predict_esmfold(args):
    seq = args["sequence"]

    model = session.fold.get_model("esmfold")
    job = model.fold([seq.encode()], num_recycles=1)
    job.wait_until_done()

    pdb_bytes = job.get()[0][1]
    pdb = pdb_bytes.decode()

    return {"pdb": pdb}


# ===================================================
#   AlphaFold2
# ===================================================
@server.tool("predict_alphafold2", "Predict structure using AlphaFold2")
async def predict_alphafold2(args):
    seq = args["sequence"]

    msa = session.align.create_msa(seq.encode())
    msa.wait_until_done()

    model = session.fold.get_model("alphafold2")
    job = model.fold(msa, num_models=1)
    job.wait_until_done()

    cif = job.wait().decode()
    return {"cif": cif}


# ===================================================
#   Boltz-2 (Protein 3D structure prediction + Affinity)
# ===================================================
@server.tool("predict_boltz2", "Predict complex using Boltz-2")
async def predict_boltz2(args):

    sequences = args["protein_sequence"]
    sequences = [sequences] if isinstance(sequences, str) else sequences
    ligand_ccd = args["ligand"]

    if not sequences:
        raise ValueError("Protein sequence is empty.")

    # -----------------------------
    # Create Proteins
    # -----------------------------
    proteins = []
    chain_ids = ["A", "B", "C", "D"]

    for i, seq in enumerate(sequences):
        if isinstance(seq, bytes):
            seq = seq.decode()

        p = Protein(sequence=seq)
        p.chain_id = chain_ids[i] if i < len(chain_ids) else chr(65 + i)
        proteins.append(p)

    # -----------------------------
    # Create Ligand
    # -----------------------------
    ligand = Ligand(ccd=ligand_ccd)

    # -----------------------------
    # Assemble Complex
    # -----------------------------
    complex_dict = {p.chain_id: p for p in proteins}
    complex_dict["L"] = ligand
    complex_obj = Complex(complex_dict)

    # -----------------------------
    # Create MSA
    # -----------------------------
    msa_query = [p.sequence for p in complex_obj.get_proteins().values()]
    msa_seed = b":".join(
        [s if isinstance(s, bytes) else s.encode() for s in msa_query]
    )

    msa = session.align.create_msa(seed=msa_seed)

    for p in complex_obj.get_proteins().values():
        p.msa = msa

    # -----------------------------
    # Run Boltz-2
    # -----------------------------
    fold_job = session.fold.boltz2.fold(
        sequences=[complex_obj],
        properties=[{"affinity": {"binder": "L"}}]
    )

    fold_job.wait_until_done(verbose=True, timeout=900)

    # -----------------------------
    # Retrieve Results
    # -----------------------------
    result = fold_job.get()
    predicted_complex = result[0][0]

    # -----------------------------
    # Extract CIF
    # -----------------------------
    mmcif = predicted_complex.to_string(format="cif")

    # -----------------------------
    # Extract pLDDT
    # -----------------------------
    plddt_scores = fold_job.get_plddt()[0]  # shape (1, N)
    plddt_scores = plddt_scores.tolist()    # convert to JSON serializable

    # -----------------------------
    # Extract PAE
    # -----------------------------
    pae_matrix = fold_job.get_pae()[0]
    pae_matrix = pae_matrix.tolist()

    # -----------------------------
    # Extract PDE
    # -----------------------------
    pde_matrix = fold_job.get_pde()[0]
    pde_matrix = pde_matrix.tolist()

    # -----------------------------
    # Extract Confidence
    # -----------------------------
    confidence_obj = fold_job.get_confidence()[0][0]
    confidence_scores = confidence_obj.model_dump()

    # -----------------------------
    # Extract Affinity
    # -----------------------------
    affinity_obj = fold_job.get_affinity()[0]

    # -----------------------------
    # Final Clean Return
    # -----------------------------
    # Instead of returning the full mmcif, return summary metrics
    return {
        "status": "success",
        "affinity": {
            "overall": affinity_obj.affinity_pred_value,
            "probability": affinity_obj.affinity_probability_binary,
            "per_chain": {
                "A": {
                    "predicted": affinity_obj.affinity_pred_value1,
                    "probability": affinity_obj.affinity_probability_binary1
                }
            }
        },
        "structure_info": {
            "format": "CIF",
            "size_bytes": len(mmcif),
            "num_atoms": mmcif.count("ATOM"),
        },
        "confidence_metrics": {
            "mean_plddt": "extracted_from_cif",  # You'll need to parse this
            "note": "Full CIF file too large to return"
        },
    }
# @server.tool("predict_boltz2", "Predict complex using Boltz-2")
# async def predict_boltz2(args):
#     sequences = args["protein_sequence"]
#     sequences = [sequences] if isinstance(sequences,str) else sequences
#     ligand_ccd = args["ligand"]

#     # Create protein objects
#     proteins = []
#     for seq in sequences:
#         p = Protein(sequence=seq)
#         p.chain_id = ["A"]   # assign chain A for each protein
#         proteins.append(p)

#     # Ligand
#     ligands = [Ligand(ccd=ligand_ccd)]

#     # Create MSA
#     msa_seed = ":".join([p.sequence.decode() for p in proteins])
#     msa = session.align.create_msa(seed=msa_seed)

#     for p in proteins:
#         p.msa = msa

#     # Run Boltz-2 folding
#     job = session.fold.boltz2.fold(
#         proteins=proteins,
#         ligands=ligands,
#         properties=[{"affinity": {"binder": "L"}}]
#     )
#     job.wait_until_done()

#     mmcif = job.get().decode()
#     affinity_value = job.affinity.affinity_pred_value

#     return {
#         "cif": mmcif,
#         "affinity": affinity_value
#     }
# ===================================================
#   Boltz-2 (Protein 3D structure prediction)
# ===================================================
# @server.tool("predict_boltz2", "Predict 3D structure with metrics using Boltz-2 (single chain, ATP ligand)")
# async def predict_boltz2(args):
#     sequence = args["protein_sequence"]  # single chain
#     ligand_ccd = "ATP"  # fixed ligand

#     # -----------------------------
#     # Create protein object
#     # -----------------------------
#     protein = Protein(sequence=sequence)
#     protein.chain_id = ["A"]  # single chain

#     # -----------------------------
#     # Ligand object
#     # -----------------------------
#     ligand = Ligand(ccd=ligand_ccd, chain_id="L")

#     # -----------------------------
#     # Create MSA (single sequence mode)
#     # -----------------------------
#     # For single chain validation, can optionally skip MSA
#     protein.msa = Protein.single_sequence_mode

#     # -----------------------------
#     # Run Boltz-2 folding
#     # -----------------------------
#     fold_job = session.fold.boltz2.fold(
#         proteins=[protein],
#         ligands=[ligand],
#         properties=[{"affinity": {"binder": "L"}}]  # request binding affinity
#     )

#     fold_job.wait_until_done(verbose=True)

#     # -----------------------------
#     # Retrieve 3D structure (mmCIF)
#     # -----------------------------
#     structure = fold_job.get()[0]
#     mmcif = structure.to_string(format="cif")

#     # -----------------------------
#     # Retrieve affinity
#     # -----------------------------
#     affinity_data = fold_job.get_affinity()[0]
#     affinity = {
#         "overall": affinity_data.affinity_pred_value,
#         "probability": affinity_data.affinity_probability_binary,
#         "per_chain": {
#             "A": {
#                 "predicted": affinity_data.affinity_pred_value1,
#                 "probability": affinity_data.affinity_probability_binary1
#             }
#         }
#     }

#     # -----------------------------
#     # Retrieve metrics: pLDDT, PAE, PDE, confidence
#     # -----------------------------
#     metrics = {
#         "plddt": fold_job.get_plddt()[0].tolist(),
#         "pae": fold_job.get_pae()[0].tolist(),
#         "pde": fold_job.get_pde()[0].tolist(),
#         "confidence": fold_job.get_confidence()[0].model_dump()  # dict
#     }

#     return {
#         "cif": mmcif,
#         "affinity": affinity,
#         "metrics": metrics
#     }

# ===================================================
#   RosettaFold-3
# ===================================================
@server.tool("predict_rosettafold3", "Predict complex using RosettaFold-3")
async def predict_rosettafold3(args):
    sequences = args["protein_sequence"]
    ligand_ccd = args["ligand"]

    proteins = []
    for seq in sequences:
        p = Protein(sequence=seq)
        p.chain_id = ["A"]
        proteins.append(p)

    ligands = [Ligand(ccd=ligand_ccd, chain_id="L")]

    msa_seed = ":".join([p.sequence.decode() for p in proteins])
    msa = session.align.create_msa(seed=msa_seed)

    for p in proteins:
        p.msa = msa

    job = session.fold.rosettafold_3.fold(
        proteins=proteins,
        ligands=ligands
    )
    job.wait_until_done()

    cif = job.get().decode()

    return {"cif": cif}


# ===================================================
#   MCP Entry Point
# ===================================================
if __name__ == "__main__":
    asyncio.run(server.run())