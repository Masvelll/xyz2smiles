import torch
import numpy as np
import os
import lmdb
import pickle
from collections import Counter
import traceback

def debug_main_v3():
    print("="*60)
    print("      ЗАПУСК В РЕЖИМЕ ОТЛАДКИ V3")
    print("="*60)
    
    # --- Шаг 1: Загрузка данных ---
    LMDB_PATH = './molecule_embeddings.lmdb'
    ORIGINAL_DATA_PATH = 'geom_train.pt'
    
    try:
        original_data = torch.load(ORIGINAL_DATA_PATH, map_location='cpu')
        env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            db_size = txn.stat()['entries']
        print(f"[OK] Все данные и LMDB ({db_size} записей) успешно загружены.")
    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при загрузке: {e}")
        return

    # --- Шаг 2: Анализ ---
    print("\n>>> Шаг 2: Анализ ОДНОЙ записи из LMDB...")
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        
        # --- ИСПРАВЛЕНИЕ: Правильный способ получить первую запись ---
        if not cursor.first():
            print("!!! ОШИБКА: База данных LMDB пуста!")
            env.close()
            return
            
        key, value = cursor.item()
        # --------------------------------------------------------
        
        print("\n--- Анализ первой записи из LMDB ---")
        try:
            molecule_index = int(key.decode())
            print(f"Индекс молекулы: {molecule_index}")

            deserialized_data = pickle.loads(value)
            print(f"Тип данных после pickle.loads(): {type(deserialized_data)}")
            
            if isinstance(deserialized_data, dict):
                print(f"Ключи в словаре: {list(deserialized_data.keys())}")
                if 'pair_embeddings' not in deserialized_data:
                    raise KeyError("Ключ 'pair_embeddings' отсутствует в словаре!")
                mol_embeddings = deserialized_data['pair_embeddings']
            else:
                raise TypeError(f"Ожидался словарь, но получен {type(deserialized_data)}")

            print(f"Тип извлеченных эмбеддингов: {type(mol_embeddings)}")
            print(f"Shape эмбеддингов: {mol_embeddings.shape}")

            mol_data = original_data[molecule_index]
            print("[OK] Соответствующая молекула из original_data найдена.")

            if not all(k in mol_data for k in ['edge_index', 'bond_orders']):
                 raise ValueError("Отсутствуют ключи edge_index/bond_orders в original_data")
            
            edge_index, bond_orders = mol_data['edge_index'], mol_data['bond_orders']
            
            j = 0 # Проверяем первую связь
            atom1_idx = edge_index[0, j].item() + 1
            atom2_idx = edge_index[1, j].item() + 1
            print(f"Проверяем связь между атомами с индексами {atom1_idx} и {atom2_idx}")
            
            if (atom1_idx >= mol_embeddings.shape[0] or atom2_idx >= mol_embeddings.shape[1]):
                raise IndexError(f"Индекс связи ({atom1_idx}, {atom2_idx}) выходит за пределы матрицы эмбеддингов ({mol_embeddings.shape})!")
            
            print("[OK] Индексы в пределах нормы.")
            print("\nВЫВОД: Проблема, скорее всего, была в способе итерации по курсору.")
            print("Если этот скрипт отработал, значит основная логика верна.")

        except Exception as e:
            print(f"\n!!! ПОЙМАНА ОШИБКА: {e}")
            traceback.print_exc()
            
    env.close()

if __name__ == "__main__":
    debug_main_v3()