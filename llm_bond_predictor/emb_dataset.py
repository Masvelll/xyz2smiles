import lmdb
import pickle
import json
import os
import torch

from torch.utils.data import Dataset
from typing import List

class PairedLMDBJsonlDataset(Dataset):
    def __init__(self, lmdb_dir, jsonl_path):
        # Открываем LMDB один раз
        if not os.path.exists(lmdb_dir):
            raise FileNotFoundError(f"LMDB directory not found: {lmdb_dir}")
        self.lmdb_env = lmdb.open(
            lmdb_dir,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=1
        )
        print('Loading embeddings...')
        with self.lmdb_env.begin() as txn:
            # self.lmdb_keys = list(txn.cursor().iternext(keys=True, values=False))
            cursor = txn.cursor()
            self.lmdb_keys = sorted(
                cursor.iternext(keys=True, values=False),
                key=lambda k: int(k.decode())
            )
        # print(self.lmdb_keys[0:12])

        # Читаем .jsonl в список один раз (быстро)
        print('Loading json prompts...')
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.prompts = [json.loads(line) for line in f]

        if len(self.prompts) != len(self.lmdb_keys):
            raise ValueError(f"❌ Несовпадение длины LMDB {len(self.lmdb_keys)} и JSONL {len(self.prompts)}.")

    def __len__(self):
        return len(self.lmdb_keys)
    
    def __getitem__(self, idx):
        key = self.lmdb_keys[idx]

        # Загружаем эмбеддинги
        with self.lmdb_env.begin(write=False) as txn:
            buf = txn.get(key)
            embedding = pickle.loads(buf)

        # Загружаем prompt и метки
        raw = self.prompts[idx]
        input_text = raw["input"]        # string
        output_text = raw["output"]      # string

        # Парсим пары из input (до строки 'Atoms:')
        lines = input_text.strip().splitlines()
        atom_start_idx = [i for i, line in enumerate(lines) if line.startswith("Atoms:")]
        if not atom_start_idx:
            raise RuntimeError(f"Не найдена строка 'Atoms:' в input на idx={idx}")
        bond_lines = lines[1:atom_start_idx[0]-1]  # пропускаем заголовок
        

        try:
            pairs = [tuple(map(int, line.strip().split())) for line in bond_lines]
            labels = list(map(int, output_text.strip().split()))
            if len(pairs) != len(labels):
                raise ValueError(f"🔴 Несовпадение пар и меток на idx={idx}: {len(pairs)} vs {len(labels)}")
            labeled_pairs = [(i, j, l) for (i, j), l in zip(pairs, labels)]
        except Exception as e:
            raise RuntimeError(f"Ошибка при парсинге input/output на idx={idx}: {e}")

        return {
            "atom_embeddings": embedding["atom_embeddings"],     # numpy [N, D_atom]
            "pair_embeddings": embedding["pair_embeddings"],     # numpy [N, N, D_edge]
            "labels": labeled_pairs,                             # список (i, j, label)
            "prompt": input_text                                 # str (простой текст)
        }


def collate_graph_batch(batch: List[dict]):
    batch_size = len(batch)
    D_atom = batch[0]["atom_embeddings"].shape[1]
    D_edge = batch[0]["pair_embeddings"].shape[2]

    N_max = max([item["atom_embeddings"].shape[0] for item in batch])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    atom_tensor = torch.zeros((batch_size, N_max, D_atom), dtype=torch.float32).to(device)
    pair_tensor = torch.zeros((batch_size, N_max, N_max, D_edge), dtype=torch.float32).to(device)
    
    # Add attention masks for padded positions
    atom_mask = torch.zeros((batch_size, N_max), dtype=torch.bool).to(device)
    pair_mask = torch.zeros((batch_size, N_max, N_max), dtype=torch.bool).to(device)

    labels = []
    pair_indices = []
    prompts = []

    for b, item in enumerate(batch):
        N = item["atom_embeddings"].shape[0]
        
        atom_tensor[b, :N] = torch.as_tensor(item["atom_embeddings"], dtype=torch.float32).to(device)
        pair_tensor[b, :N, :N] = torch.as_tensor(item["pair_embeddings"], dtype=torch.float32).to(device)
        
        # Mark valid positions (True = valid, False = padded)
        atom_mask[b, :N] = True
        pair_mask[b, :N, :N] = True

        for i, j, lbl in item["labels"]:
            pair_indices.append((b, i, j))
            labels.append(lbl)

        prompts.append(item["prompt"])  # уже str
        
    ret_dict = {
        "atom_embeddings": atom_tensor,
        "pair_embeddings": pair_tensor,
        "atom_mask": atom_mask,
        "pair_mask": pair_mask,
        "pair_indices": torch.tensor(pair_indices, dtype=torch.long).to(device),
        "labels": torch.tensor(labels, dtype=torch.long).to(device),
        "prompts": prompts
    }
    # for key, item in ret_dict.items():
    #     if torch.is_tensor(item):
    #         print(key, item.shape)
    #     else:
    #         print(key, len(item))
    return ret_dict
