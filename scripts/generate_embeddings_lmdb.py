import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import traceback
import lmdb
import pickle

# --- НАЧАЛО: Настройка путей и импортов (ФИНАЛЬНАЯ ВЕРСИЯ) ---
# 1. Путь к клонированному репозиторию Uni-Mol
UNIMOL_CODE_PATH = '/home/user12/burov/unimol_embeddings_project/unimol_repo/unimol' 
if UNIMOL_CODE_PATH not in sys.path:
    sys.path.insert(0, UNIMOL_CODE_PATH)

# 2. Путь к клонированному репозиторию Uni-Core (РЕШЕНИЕ ПРОБЛЕМЫ)
UNICORE_CODE_PATH = '/home/user12/burov/unimol_embeddings_project/Uni-Core' 
if UNICORE_CODE_PATH not in sys.path:
    sys.path.insert(0, UNICORE_CODE_PATH)

try:
    # Теперь этот импорт должен сработать, так как Python найдет и unimol, и unicore
    from unimol.models.unimol import UniMolModel, unimol_base_architecture
    print(">>> Код Uni-Mol и его зависимости успешно импортированы.")
except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать Uni-Mol или Uni-Core. {e}")
    print("Проверьте, что пути UNIMOL_CODE_PATH и UNICORE_CODE_PATH указаны верно и репозитории существуют.")
    sys.exit(1)
# --- КОНЕЦ ---


# --- Остальная часть скрипта остается АБСОЛЮТНО БЕЗ ИЗМЕНЕНИЙ ---
# (Словарь CorrectDictionary, collate_fn, main() и т.д.)

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
        src_distance[i, :n_atoms+1, :n_atoms+1] = torch.cdist(mol_coords, mol_coords)
        edge_types = mol_tokens.view(-1, 1) * len(dictionary) + mol_tokens.view(1, -1)
        src_edge_type[i, :n_atoms+1, :n_atoms+1] = edge_types
    return {'src_tokens': src_tokens, 'src_distance': src_distance, 'src_coord': src_coord, 'src_edge_type': src_edge_type}

def main():
    parser = argparse.ArgumentParser(description='Генерация эмбеддингов Uni-Mol и сохранение в LMDB.')
    parser.add_argument('--data_path', type=str, default='/home/user12/maslov/mydatasets/geom_test.pt', help='Путь к файлу с данными молекул')
    parser.add_argument('--lmdb_path', type=str, default='molecule_embeddings_test.lmdb', help='Путь для сохранения LMDB базы данных')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--no_h', action='store_true', help='Использовать веса модели, обученной без атомов водорода')
    args_cmd = parser.parse_args()
    
    print(">>> Создаем модель и загружаем веса...")
    args = argparse.Namespace()
    dictionary = CorrectDictionary()
    unimol_base_architecture(args)
    args.mode = 'infer'
    model = UniMolModel(args, dictionary)
    print("Архитектура модели создана.")
    
    weights_filename = 'mol_pre_no_h_220816.pt' if args_cmd.no_h else 'mol_pre_all_h_220816.pt'
    print(f"INFO: Загружаются веса {'БЕЗ водородов' if args_cmd.no_h else 'со всеми атомами'}.")
    
    WEIGHTS_PATH = f'./unimol_repo/weights/mol_pre_no_h_220816.pt'
    if not os.path.exists(WEIGHTS_PATH):
        print(f"ОШИБКА: Файл с весами не найден: {WEIGHTS_PATH}")
        return
        
    print(f"Загружаем веса из: {WEIGHTS_PATH}")
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")["model"]
    model.load_state_dict(state_dict, strict=False) 
    print(">>> УСПЕХ! Веса загружены, модель готова.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    
    if not os.path.exists(args_cmd.data_path):
        print(f"ОШИБКА: Файл с данными не найден: {args_cmd.data_path}")
        return
    
    print(f"Загружаем данные из: {args_cmd.data_path}")
    all_molecules_data = torch.load(args_cmd.data_path, map_location='cpu')
    print(f"Загружено {len(all_molecules_data)} молекул.")
    model.to(device)
    model.eval()

    env = lmdb.open(args_cmd.lmdb_path, map_size=1024**3 * 200)

    print(f"Начинаем генерацию эмбеддингов и сохранение в LMDB: {args_cmd.lmdb_path}")
    processed_count = 0
    failed_count = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(all_molecules_data), args_cmd.batch_size)):
            batch_data = all_molecules_data[i:i+args_cmd.batch_size]
            batch = collate_fn(batch_data, dictionary)
            if batch is None: continue
            for key in batch: batch[key] = batch[key].to(device)
            try:
                encoder_rep, encoder_pair_rep = model(**batch)
                padding_masks_batch = (batch['src_tokens'] != dictionary.pad()).cpu()
                for j in range(encoder_rep.size(0)):
                    try:
                        true_len = padding_masks_batch[j].sum().item()
                        if true_len == 0: continue
                        atom_embeddings = encoder_rep[j, :true_len, :].cpu()
                        pair_embeddings = encoder_pair_rep[j, :true_len, :true_len, :].cpu()
                        molecule_index = i + j
                        molecule_data = {'atom_embeddings': atom_embeddings, 'pair_embeddings': pair_embeddings}
                        with env.begin(write=True) as txn:
                            txn.put(str(molecule_index).encode(), pickle.dumps(molecule_data))
                        processed_count += 1
                    except Exception as e:
                        failed_count += 1
                        continue
            except Exception as e:
                print(f"Критическая ошибка при обработке батча {i//args_cmd.batch_size}. Пропускаем. Ошибка: {e}")
                failed_count += len(batch_data)
                continue
    env.close()

    print(f"\n==============================================")
    print(f"     ГЕНЕРАЦИЯ ЗАВЕРШЕНА!     ")
    print(f"==============================================")
    print(f"Успешно обработано: {processed_count} молекул")
    print(f"Не удалось обработать: {failed_count} молекул")
    print(f"LMDB база данных сохранена в: {args_cmd.lmdb_path}")

if __name__ == "__main__":
    main()