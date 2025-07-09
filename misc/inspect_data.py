import torch
import lmdb
import pickle
import numpy as np

def inspect_data():
    """
    Загружает первую молекулу из original_data и первую запись из LMDB
    и выводит всю информацию о них для сравнения.
    """
    
    LMDB_PATH = './molecule_embeddings.lmdb'
    ORIGINAL_DATA_PATH = 'geom_train.pt'
    
    print("="*60)
    print("           ИНСПЕКЦИЯ ДАННЫХ: MOLECULE #0")
    print("="*60)

    # --- 1. Загружаем данные из geom_train.pt ---
    try:
        original_data = torch.load(ORIGINAL_DATA_PATH, map_location='cpu')
        mol_data_original = original_data[0]
        print("\n--- ДАННЫЕ ИЗ `geom_train.pt` (Молекула #0) ---")
        print(f"  Ключи: {list(mol_data_original.keys())}")
        
        one_hot = mol_data_original.get('one_hot')
        positions = mol_data_original.get('positions')
        edge_index = mol_data_original.get('edge_index')
        bond_orders = mol_data_original.get('bond_orders')
        
        if one_hot is not None:
            print(f"  Количество атомов (`one_hot`): {one_hot.shape[0]}")
        else:
            print("  Ключ 'one_hot' отсутствует!")
            
        if edge_index is not None:
            print(f"  Количество связей (`edge_index`): {edge_index.shape[1]}")
        else:
            print("  Ключ 'edge_index' отсутствует!")
            
        if bond_orders is not None:
            print(f"  Уникальные типы связей (`bond_orders`): {torch.unique(bond_orders).tolist()}")
            # Выведем первые 5 типов связей для примера
            print(f"  Примеры `bond_orders`: {bond_orders[:5].tolist()}")
        else:
            print("  Ключ 'bond_orders' отсутствует!")
            
    except Exception as e:
        print(f"!!! Ошибка при чтении geom_train.pt: {e}")
        return

    # --- 2. Загружаем данные из molecule_embeddings.lmdb ---
    try:
        env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            # Получаем значение для ключа '0'
            value = txn.get(b'0')
            if value is None:
                print("\n!!! ОШИБКА: В LMDB нет записи для молекулы с ключом '0'.")
                env.close()
                return

        deserialized_data = pickle.loads(value)
        
        print("\n--- ДАННЫЕ ИЗ `molecule_embeddings.lmdb` (Молекула #0) ---")
        print(f"  Тип данных после распаковки: {type(deserialized_data)}")
        
        if isinstance(deserialized_data, dict):
            print(f"  Ключи в словаре: {list(deserialized_data.keys())}")
            pair_embeddings = deserialized_data.get('pair_embeddings')
            if pair_embeddings is not None:
                # Конвертируем в numpy для вывода
                if isinstance(pair_embeddings, torch.Tensor):
                    pair_embeddings = pair_embeddings.numpy()
                print(f"  Shape парных эмбеддингов: {pair_embeddings.shape}")
            else:
                print("  Ключ 'pair_embeddings' отсутствует!")
        else:
            print(f"  Данные в LMDB - это не словарь, а объект типа {type(deserialized_data)}")

    except Exception as e:
        print(f"!!! Ошибка при чтении molecule_embeddings.lmdb: {e}")
        return
        
    finally:
        if 'env' in locals() and env:
            env.close()
            
    # --- 3. Финальный анализ ---
    print("\n" + "="*60)
    print("                     АНАЛИЗ")
    print("="*60)
    
    num_atoms = one_hot.shape[0] if one_hot is not None else -1
    embedding_dim = pair_embeddings.shape[0] if 'pair_embeddings' in locals() and pair_embeddings is not None else -1
    
    print(f"Кол-во атомов в `geom_train.pt`: {num_atoms}")
    print(f"Размерность эмбеддинга в `lmdb`:  {embedding_dim} (ожидается {num_atoms + 1})")
    
    if embedding_dim == num_atoms + 1:
        print("\n[OK] Размерности совпадают (с учетом BOS токена).")
    else:
        print("\n[!!!] ВНИМАНИЕ: Размерности НЕ СОВПАДАЮТ! Это может быть причиной проблемы.")
        
    print("\n[ПРОВЕРКА] Типы связей, которые мы ожидаем: {1.0, 1.5, 2.0, 3.0}")
    print(f"[ФАКТ]   Уникальные типы связей в ваших данных: {torch.unique(bond_orders).tolist() if bond_orders is not None else 'N/A'}")
    print("\nСравните эти две строки. Если они не совпадают, мы нашли проблему.")


if __name__ == "__main__":
    inspect_data()