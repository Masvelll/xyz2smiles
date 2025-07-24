#!/usr/bin/env python3
"""generate_embeddings_lmdb_optimized.py

Batch‑wise generation of Uni‑Mol embeddings with memory‑safe LMDB writing.
Changes compared to the original version
----------------------------------------
* Writes in **chunks** (`commit_every`) instead of one transaction per molecule.
* Stores tensors in **float16** and compresses with **LZ4** (≈×6 smaller).
* Saves only the **upper triangular** part of pair embeddings.
* Frees GPU/CPU memory every batch (`del`, `gc.collect`, `torch.cuda.empty_cache`).
* Optional progress log with current RSS.
* Accepts the same CLI flags; drop‑in replacement.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
from pathlib import Path

import lmdb
import lz4.frame as lz4
import numpy as np
import psutil
import torch
from tqdm import tqdm

# -- Приоритет: сначала добавить пути к Uni‑Mol / Uni‑Core ------------------
UNIMOL_CODE_PATH = Path("../Uni-Mol/unimol").resolve()
UNICORE_CODE_PATH = Path("../Uni-Core").resolve()
for p in (str(UNIMOL_CODE_PATH), str(UNICORE_CODE_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from unimol.models.unimol import UniMolModel, unimol_base_architecture
except ImportError as e:  # pragma: no cover
    sys.exit(f"[FATAL] Cannot import Uni‑Mol/Uni‑Core: {e}\nCheck paths: {UNIMOL_CODE_PATH}, {UNICORE_CODE_PATH}")

# ---------------------------------------------------------------------------


class CorrectDictionary:
    symbols = [
        "[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "N", "O", "S", "H", "Cl", "F", "Br", "I",
        "Si", "P", "B", "Na", "K", "Al", "Ca", "Sn", "As", "Hg", "Fe", "Zn", "Cr", "Se", "Gd", "Au", "Li", "DUMMY",
    ]

    def __init__(self):
        assert len(self.symbols) == 31, "Dictionary size must be 31!"
        self.indices = {s: i for i, s in enumerate(self.symbols)}
        self.atomic_num_to_idx = {1: 3, 6: 4, 7: 5, 8: 6, 9: 7, 14: 8, 15: 9, 16: 10, 17: 11}

    # ----- helpers -----
    def __len__(self):
        return len(self.symbols)

    def pad(self):
        return self.indices["[PAD]"]

    def bos(self):
        return self.indices["[CLS]"]

    def index(self, atomic_num: int):
        return self.atomic_num_to_idx.get(atomic_num, self.atomic_num_to_idx[8])


DATASET_ATOM_MAP = [6, 8, 7, 9, 16, 17, 35, 53, 15]


def collate_fn(batch, dictionary, *, max_len=512):
    """Prepares a mini‑batch for Uni‑Mol inference.
    Filters out too‑long molecules and returns `None` if batch became empty."""
    batch = [mol for mol in batch if mol and "one_hot" in mol and len(mol["one_hot"]) + 1 <= max_len]
    if not batch:
        return None

    max_batch_len = max(len(mol["one_hot"]) for mol in batch) + 1  # + CLS
    n, m = len(batch), max_batch_len
    device = torch.device("cpu")  # create on CPU first

    src_tokens = torch.full((n, m), dictionary.pad(), dtype=torch.long)
    src_coord = torch.zeros(n, m, 3)
    src_distance = torch.zeros(n, m, m)
    src_edge_type = torch.zeros(n, m, m, dtype=torch.long)

    for i, mol in enumerate(batch):
        one_hot: torch.Tensor = mol["one_hot"]
        coords: torch.Tensor = mol["positions"]
        n_atoms = len(one_hot)

        atom_indices = one_hot.argmax(dim=1)
        atomic_nums = [DATASET_ATOM_MAP[idx] for idx in atom_indices]
        atom_tokens = torch.tensor([dictionary.index(num) for num in atomic_nums], dtype=torch.long)

        mol_tokens = torch.cat([torch.tensor([dictionary.bos()]), atom_tokens])
        mol_coords = torch.cat([torch.zeros(1, 3), coords])

        src_tokens[i, : n_atoms + 1] = mol_tokens
        src_coord[i, : n_atoms + 1] = mol_coords
        dists = torch.cdist(mol_coords, mol_coords)
        src_distance[i, : n_atoms + 1, : n_atoms + 1] = dists

        etypes = mol_tokens.view(-1, 1) * len(dictionary) + mol_tokens.view(1, -1)
        src_edge_type[i, : n_atoms + 1, : n_atoms + 1] = etypes

    return {
        "src_tokens": src_tokens,
        "src_distance": src_distance,
        "src_coord": src_coord,
        "src_edge_type": src_edge_type,
    }


# ------------------------- main pipeline -----------------------------------

def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Generate Uni‑Mol embeddings and store them in LMDB.")
    parser.add_argument("--data_path", default="../data/geom_train.pt", help="Path to PyTorch dataset file")
    parser.add_argument("--lmdb_path", default="../data/molecule_embeddings.lmdb", help="Destination LMDB")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_h", action="store_true", help="Use weights trained without hydrogens")
    parser.add_argument("--commit_every", type=int, default=1000, help="LMDB commit frequency (molecules)")
    parser.add_argument("--log_every", type=int, default=2000, help="How often to print RSS (molecules)")
    args = parser.parse_args()

    # ---- model -----------------------------------------------------------
    print("[INFO] Building Uni‑Mol model …")
    dictionary = CorrectDictionary()
    model_args = argparse.Namespace()
    unimol_base_architecture(model_args)
    model_args.mode = "infer"
    model = UniMolModel(model_args, dictionary)

    weights = "mol_pre_no_h_220816.pt" if args.no_h else "mol_pre_all_h_220816.pt"
    weights_path = Path("../Uni-Mol/weights") / weights
    if not weights_path.exists():
        sys.exit(f"[FATAL] Weights file not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")["model"]
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"[INFO] Model ready on {device}.")

    # ---- data ------------------------------------------------------------
    data_path = Path(args.data_path)
    if not data_path.exists():
        sys.exit(f"[FATAL] Dataset file not found: {data_path}")
    print(f"[INFO] Loading dataset ({data_path.stat().st_size/1e9:.1f} GB) …")
    all_mols = torch.load(data_path, map_location="cpu")
    print(f"[INFO] Loaded {len(all_mols):,} molecules.")

    # ---- LMDB ------------------------------------------------------------
    env = lmdb.open(
        args.lmdb_path,
        map_size=1024**3 * 200,  # 200 GiB max
        subdir=False,
        meminit=False,
        writemap=True,
        readahead=False,
    )
    txn = env.begin(write=True)

    processed = failed = 0
    rss_prev = psutil.Process(os.getpid()).memory_info().rss

    with torch.no_grad():
        for start in tqdm(range(0, len(all_mols), args.batch_size), unit="mols"):
            batch_raw = all_mols[start : start + args.batch_size]
            batch = collate_fn(batch_raw, dictionary)
            if batch is None:
                continue

            # move tensors to GPU
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            try:
                enc_atom, enc_pair = model(**batch)
            finally:
                # 'src_distance' no longer needed after forward pass
                del batch["src_distance"]

            padding_mask = (batch["src_tokens"] != dictionary.pad()).cpu()

            for j in range(enc_atom.size(0)):
                true_len = int(padding_mask[j].sum())
                if not true_len:
                    continue

                atom_emb = enc_atom[j, :true_len].half().cpu()
                pair_emb = torch.triu(enc_pair[j, :true_len, :true_len]).half().cpu()

                blob = pickle.dumps({"atom": atom_emb, "pair": pair_emb}, protocol=pickle.HIGHEST_PROTOCOL)
                blob = lz4.compress(blob, compression_level=3)
                mol_index = start + j
                txn.put(str(mol_index).encode(), blob)
                processed += 1

                if processed % args.commit_every == 0:
                    txn.commit(); txn = env.begin(write=True)

                if processed % args.log_every == 0:
                    rss_now = psutil.Process(os.getpid()).memory_info().rss
                    delta = (rss_now - rss_prev) / 1e9
                    print(f"\n[RSS] {rss_now/1e9:6.1f} GB (Δ {delta:+.2f} GB) after {processed:,} molecules")
                    rss_prev = rss_now

            # ---- free GPU/CPU RAM -------------------------------------
            del enc_atom, enc_pair, batch
            torch.cuda.empty_cache()
            gc.collect()

    # final commit & close -----------------------------------------------
    txn.commit(); env.sync(); env.close()

    print("\n==============================================")
    print("       EMBEDDING GENERATION FINISHED")
    print("==============================================")
    print(f"Processed successfully : {processed:,}")
    print(f"Failed (skipped)       : {failed:,}")
    print(f"LMDB saved to          : {args.lmdb_path}")


if __name__ == "__main__":
    main()
