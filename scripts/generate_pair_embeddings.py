import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import traceback

# --- НАЧАЛО: Настройка путей и импортов (не меняется) ---
# Укажите правильный путь к вашему репозиторию Uni-Mol
UNIMOL_CODE_PATH = '/home/user12/burov/unimol_embeddings_project/unimol_repo/unimol' 
if UNIMOL_CODE_PATH not in sys.path:
    sys.path.insert(0, UNIMOL_CODE_PATH)

from unimol.models.unimol import UniMolModel, unimol_base_architecture
print(">>> Код Uni-Mol импортирован.")
# --- КОНЕЦ ---


# --- НАЧАЛО: Словарь и collate_fn (не меняется) ---
class CorrectDictionary:
    def __init__(self):
        self.symbols = [
            "<pad>", "<bos>", "<eos>", 'H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl',
            '<dummy_13>', '<dummy_14>', '<dummy_15>', '<dummy_16>', '<dummy_17>',
            '<dummy_18>', '<dummy_19>', '<dummy_20>', '<dummy_21>', '<dummy_22>',
            '<dummy_23>', '<dummy_24>', '<dummy_25>', '<dummy_26>', '<dummy_27>',
            '<dummy_28>', '<dummy_29>', '<dummy_30>', '<dummy_31>'
        ]
        assert len(self.symbols) == 31, "Размер словаря модели должен быть 31!"
        self.indices = {s: i for i, s in enumerate(self.symbols)}
        self.atomic_num_to_idx = {
            1: 3, 6: 4, 7: 5, 8: 6, 9: 7, 14: 8, 15: 9, 16: 10, 17: 11
        }
    def __len__(self): return len(self.symbols)
    def pad(self): return self.indices["<pad>"]
    def bos(self): return self.indices["<bos>"]
    def index(self, atomic_num):
        return self.atomic_num_to_idx.get(atomic_num, self.atomic_num_to_idx[8])

DATASET_ATOM_MAP = [6, 8, 7, 9, 16, 17, 35, 53, 15]

def collate_fn(batch, dictionary, max_len=512):
    batch = [mol for mol in batch if mol and 'one_hot' in mol and len(mol['one_hot']) + 1 <= max_len]
    if not batch: return None
    max_batch_len = max(len(mol['one_hot']) for mol in batch) + 1
    src_tokens = torch.full((len(batch), max_batch_len), dictionary.pad(), dtype=torch.long)
    src_coord = torch.zeros(len(batch), max_batch_len, 3, dtype=torch.float)
    src_distance = torch.zeros(len(batch), max_batch_len, max_batch_len, dtype=torch.float)
    src_edge_type = torch.zeros(len(batch), max_batch_len, max_batch_len, dtype=torch.long)
    for i, mol in enumerate(batch):
        one_hot, coords = mol['one_hot'], mol['positions']
        n_atoms = len(one_hot)
        atom_indices = torch.argmax(one_hot, dim=1)
        atomic_nums = [DATASET_ATOM_MAP[idx] for idx in atom_indices]
        atom_tokens = torch.tensor([dictionary.index(num) for num in atomic_nums], dtype=torch.long)
        mol_tokens = torch.cat([torch.tensor([dictionary.bos()], dtype=torch.long), atom_tokens])
        mol_coords = torch.cat([torch.zeros(1, 3), coords], dim=0)
        src_tokens[i, :n_atoms+1] = mol_tokens
        src_coord[i, :n_atoms+1] = mol_coords
        dist_matrix = torch.cdist(mol_coords, mol_coords)
        src_distance[i, :n_atoms+1, :n_atoms+1] = dist_matrix
        edge_types = mol_tokens.view(-1, 1) * len(dictionary) + mol_tokens.view(1, -1)
        src_edge_type[i, :n_atoms+1, :n_atoms+1] = edge_types
    return {
        'src_tokens': src_tokens,
        'src_distance': src_distance,
        'src_coord': src_coord,
        'src_edge_type': src_edge_type
    }
# --- КОНЕЦ ---


# --- НАЧАЛО: Создание модели и загрузка весов (не меняется) ---
print(">>> Создаем модель и загружаем веса...")
args = argparse.Namespace()
dictionary = CorrectDictionary()
unimol_base_architecture(args)
# ВАЖНО: Модель должна возвращать и парные представления.
# Они возвращаются как в 'train', так и в 'infer' режиме, поэтому это менять не нужно.
args.mode = 'infer'
model = UniMolModel(args, dictionary)
print("Архитектура модели создана.")
WEIGHTS_PATH = '/home/user12/.conda/envs/xyz_smiles/lib/python3.10/site-packages/unimol_tools/weights/mol_pre_all_h_220816.pt'
print(f"Загружаем веса из локального файла: {WEIGHTS_PATH}")
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")["model"]
model.load_state_dict(state_dict, strict=False) 
print(">>> УСПЕХ! Веса загружены, модель готова.")
# --- КОНЕЦ ---


# --- Основная часть скрипта ---
data_path = 'geom_train.pt'
### ИЗМЕНЕНО: Новые пути для сохранения результатов ###
output_path_pairs = 'geom_train_pair_embeddings.pt'
output_path_masks = 'geom_train_pair_padding_masks.pt'

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")
print("Загружаем данные...")
all_molecules_data = torch.load(data_path, map_location='cpu')
print(f"Загружено {len(all_molecules_data)} молекул.")
model.to(device)
model.eval()

### ИЗМЕНЕНО: Создаем списки для парных эмбеддингов и масок ###
all_pair_embeddings = []
all_padding_masks = []

print("Начинаем генерацию ПАРНЫХ эмбеддингов...")
with torch.no_grad():
    for i in tqdm(range(0, len(all_molecules_data), batch_size)):
        batch_data = all_molecules_data[i:i+batch_size]
        batch = collate_fn(batch_data, dictionary)
        if batch is None: continue
        for key in batch: batch[key] = batch[key].to(device)
            
        try:
            # Получаем внутренние представления от энкодера
            ### ИЗМЕНЕНО: Нам теперь нужен и второй элемент - encoder_pair_rep ###
            encoder_rep, encoder_pair_rep = model(**batch)

            # `encoder_pair_rep` имеет размер (batch_size, max_len, max_len, num_heads)
            # Мы хотим сохранить его на CPU
            pair_reps_batch = encoder_pair_rep.cpu()

            # Также нам нужны маски, чтобы знать, где реальные атомы, а где padding
            # `src_tokens` имеет размер (batch_size, max_len)
            padding_masks_batch = (batch['src_tokens'] != dictionary.pad()).cpu()

            # Так как молекулы в батче разной длины, мы не можем просто объединить их в один тензор.
            # Мы "разбираем" батч обратно на отдельные молекулы и сохраняем их.
            for j in range(pair_reps_batch.size(0)):
                # Находим реальную длину молекулы (включая BOS токен)
                true_len = padding_masks_batch[j].sum()
                
                # Обрезаем матрицу парных представлений до реального размера
                pair_rep_single = pair_reps_batch[j, :true_len, :true_len, :]
                all_pair_embeddings.append(pair_rep_single)

                # Также сохраняем маску для этой молекулы
                padding_mask_single = padding_masks_batch[j, :true_len]
                all_padding_masks.append(padding_mask_single)

        except Exception as e:
            print(f"Ошибка при обработке батча {i//batch_size}. Пропускаем. Ошибка: {e}")
            continue

### ИЗМЕНЕНО: Логика сохранения ###
if all_pair_embeddings:
    print(f"\n==============================================")
    print(f"     ГЕНЕРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!     ")
    print(f"==============================================")
    print(f"Успешно обработано: {len(all_pair_embeddings)} из {len(all_molecules_data)} молекул.")
    
    # Сохраняем список тензоров как есть.
    # Это самый гибкий формат для данных переменной длины.
    torch.save(all_pair_embeddings, output_path_pairs)
    print(f"Парные эмбеддинги сохранены в файл: {output_path_pairs}")

    torch.save(all_padding_masks, output_path_masks)
    print(f"Маски сохранены в файл: {output_path_masks}")
else:
    print("\nНе удалось сгенерировать парные эмбеддинги.")