# evaluate_on_s669.py
# Automate evaluation of protein language models on S669 stability dataset

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
#from Bio.PDB import PDBList, PDBParser
#from Bio.PDB.Polypeptide import aa1, is_aa
from Bio.PDB import PDBList, PDBParser
from Bio.PDB.Polypeptide import aa1, is_aa
import os
import requests
from tqdm import tqdm
import json

# === 1. Configuration ===
S669_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5528971/bin/NIHMS876490-supplement-Supp_Table_S1.csv"
DATA_DIR = "data"
PDB_DIR = os.path.join(DATA_DIR, "pdb")
MODEL_PATH = "saved_real_enzyme_model"  # Path to your saved GNN+LLM
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PDB_DIR, exist_ok=True)

# === 2. Amino Acid Vocabulary & Hydrophobicity (for tokenization) ===
AA_VOCAB = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    '<PAD>', '<UNK>', '<CLS>', '<MASK>'
]
aa_to_idx = {aa: idx for idx, aa in enumerate(AA_VOCAB)}
idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}

# === 3. Download and Load S669 Dataset ===
def download_s669():
    """Download S669 dataset from NCBI."""
    path = os.path.join(DATA_DIR, "s669.csv")
    if os.path.exists(path):
        print("✅ S669 already downloaded.")
        return path

    print("📥 Downloading S669 dataset...")
    try:
        response = requests.get(S669_URL)
        response.raise_for_status()
        with open(path, 'w') as f:
            f.write(response.text)
        print(f"✅ Saved to {path}")
        return path
    except Exception as e:
        raise ConnectionError(f"Failed to download S669: {e}")

def load_s669():
    """Load and clean S669 mutations."""
    path = download_s669()
    df = pd.read_csv(path)

    # Rename and filter columns
    df = df.rename(columns={
        'PDB': 'pdb_id',
        'Chain': 'chain',
        'WT': 'wildtype',
        'Mut': 'mutant',
        'Position': 'position',
        'ddg': 'ddg'
    })[['pdb_id', 'chain', 'wildtype', 'position', 'mutant', 'ddg']]

    # Filter valid entries
    df = df.dropna()
    df = df[df['wildtype'].isin(aa_to_idx.keys())]
    df = df[df['mutant'].isin(aa_to_idx.keys())]
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df = df.dropna()

    print(f"✅ Loaded {len(df)} mutations from S669")
    return df

# === 4. Fetch PDB Files ===
def download_pdb(pdb_id):
    """Download PDB file."""
    pdb_file = os.path.join(PDB_DIR, f"{pdb_id.upper()}.pdb")
    if os.path.exists(pdb_file):
        return pdb_file

    pdbl = PDBList()
    try:
        pdbl.retrieve_pdb_file(pdb_id, pdir=PDB_DIR, file_format="pdb")
        old_name = os.path.join(PDB_DIR, f"pdb{pdb_id.lower()}.ent")
        if os.path.exists(old_name):
            os.rename(old_name, pdb_file)
        return pdb_file
    except Exception as e:
        print(f"❌ Failed to download {pdb_id}: {e}")
        return None

# === 5. Parse PDB to Get Wild-Type Sequence ===
def parse_pdb_sequence(pdb_path, chain_id, start_pos=None, end_pos=None):
    """Extract sequence from PDB."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", pdb_path)
        seq = ""
        positions = []
        for model in structure:
            for chain in model:
                if chain.id != chain_id:
                    continue
                for residue in chain:
                    if not is_aa(residue.get_resname(), standard=True):
                        continue
                    try:
                        pos = residue.get_id()[1]  # Residue number
                        aa = aa1(residue.get_resname())
                        seq += aa
                        positions.append(pos)
                    except KeyError:
                        continue
        return seq, positions
    except Exception as e:
        print(f"PDB parsing error: {e}")
        return None, None

# === 6. Model Loading (Update based on your model class) ===
def load_model(model_path):
    """Load your trained GNN+LLM model."""
    try:
        # You may need to define GNNPrior and GNNEnhancedProteinLM here or import
        from gnn_aware_protein_llm import load_full_model
        llm, gnn_prior = load_full_model(model_path, device=DEVICE)
        return llm, gnn_prior
    except ImportError:
        raise ImportError("Make sure gnn_aware_protein_llm.py is in PYTHONPATH")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# === 7. Compute Mutation Score (ΔLogP) ===
@torch.no_grad()
def compute_mutation_score(llm, gnn_prior, wt_seq, position, mutant_aa, chain_coords=None):
    """
    Compute log-likelihood difference: ΔLogP = logP(mutant) - logP(wildtype)
    This serves as a proxy for predicted ΔΔG.
    """
    def tokenize(seq):
        ids = [aa_to_idx['<CLS>']] + [
            aa_to_idx.get(aa, aa_to_idx['<UNK>']) for aa in seq
        ]
        pad_len = MAX_LEN - len(seq)
        ids += [aa_to_idx['<PAD>']] * pad_len
        return torch.tensor([ids]).to(DEVICE)

    wt_ids = tokenize(wt_seq)
    pos_in_seq = position - 1  # zero-indexed

    if pos_in_seq < 0 or pos_in_seq >= len(wt_seq):
        return None

    # Create mutant sequence
    mut_seq = list(wt_seq)
    mut_seq[pos_in_seq] = mutant_aa
    mut_seq = ''.join(mut_seq)
    mut_ids = tokenize(mut_seq)

    # Forward pass
    try:
        # If using GNN, build graph
        if gnn_prior is not None and chain_coords is not None:
            from gnn_aware_protein_llm import structure_to_graph
            graph = structure_to_graph(chain_coords, wt_seq)
            if graph is None:
                return None
            batched_graph = graph.clone().to(DEVICE)
            gnn_embeddings = gnn_prior(batched_graph)
        else:
            gnn_embeddings = None

        # Get logits
        _, wt_logits = llm(wt_ids, labels=wt_ids)  # Only need logits shape
        _, mut_logits = llm(mut_ids, labels=mut_ids)

        # Extract log-prob at mutation site
        wt_logprobs = torch.log_softmax(wt_logits[0], dim=-1)
        mut_logprobs = torch.log_softmax(mut_logits[0], dim=-1)

        wt_idx = aa_to_idx[wt_seq[pos_in_seq]]
        mut_idx = aa_to_idx[mutant_aa]

        wt_logp = wt_logprobs[pos_in_seq + 1, wt_idx].item()  # +1 for <CLS>
        mut_logp = mut_logprobs[pos_in_seq + 1, mut_idx].item()

        return mut_logp - wt_logp  # ΔLogP (higher = more favorable)

    except Exception as e:
        print(f"Error computing score: {e}")
        return None

# === 8. Main Evaluation Loop ===
def evaluate_on_s669(model_path=MODEL_PATH):
    print("🔬 Evaluating model on S669 dataset...\n")

    # Load data
    df = load_s669()
    llm, gnn_prior = load_model(model_path)

    predictions = []
    targets = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        pdb_id = row['pdb_id']
        chain = row['chain']
        wt_aa = row['wildtype']
        pos = int(row['position'])
        mut_aa = row['mutant']
        true_ddg = row['ddg']

        # Download PDB
        pdb_file = download_pdb(pdb_id)
        if not pdb_file:
            continue

        # Extract sequence and coordinates
        wt_seq, pdb_positions = parse_pdb_sequence(pdb_file, chain)
        if not wt_seq or pos not in pdb_positions:
            continue

        # Map PDB position to sequence index
        try:
            seq_idx = pdb_positions.index(pos)
            if wt_seq[seq_idx] != wt_aa:
                print(f"⚠️ Mismatch at {pdb_id}:{pos}: expected {wt_aa}, got {wt_seq[seq_idx]}")
                continue
        except ValueError:
            continue

        # Optional: extract Cα coords for GNN
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("x", pdb_file)
        coords = []
        for model in structure:
            for ch in model:
                if ch.id == chain:
                    for residue in ch:
                        if residue.get_id()[1] == pos and is_aa(residue.get_resname(), standard=True):
                            try:
                                coords.append(residue['CA'].coord)
                            except KeyError:
                                pass
        chain_coords = np.array(coords) if len(coords) > 0 else None

        # Compute prediction
        pred_score = compute_mutation_score(llm, gnn_prior, wt_seq, pos, mut_aa, chain_coords)
        if pred_score is not None:
            predictions.append(pred_score)
            targets.append(true_ddg)

    # Convert to numpy
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Reverse sign: lower ΔΔG = more stable → higher likelihood
    # But our ΔLogP: higher = better → so correlate negative ΔΔG with high ΔLogP
    # Thus: predicted stabilizing effect = -ΔLogP
    reversed_preds = -predictions

    # Compute metrics
    r, _ = pearsonr(reversed_preds, targets)
    rho, _ = spearmanr(reversed_preds, targets)

    # Binary classification: stabilizing (ΔΔG < 0) vs destabilizing
    y_true_class = (targets < 0).astype(int)
    y_pred_prob = -reversed_preds  # higher = more likely stabilizing
    auc = roc_auc_score(y_true_class, y_pred_prob)

    print("\n✅ Evaluation Complete!")
    print(f"Number of mutations evaluated: {len(predictions)}")
    print(f"Pearson r: {r:.3f}")
    print(f"Spearman ρ: {rho:.3f}")
    print(f"AUC (stabilizing): {auc:.3f}")

    # Save results
    results = {
        "pearson_r": float(r),
        "spearman_rho": float(rho),
        "auc": float(auc),
        "count_evaluated": len(predictions),
        "predictions": predictions.tolist(),
        "targets": targets.tolist()
    }
    with open("s669_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"📊 Results saved to s669_evaluation_results.json")

    return results

# === Run ===
if __name__ == "__main__":
    evaluate_on_s669(MODEL_PATH)