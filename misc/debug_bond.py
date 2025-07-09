import torch
import numpy as np
from tqdm import tqdm
import os
import lmdb
import pickle
from collections import Counter

def debug_main_v5():
    print("="*60)
    print("      ЗАПУСК В РЕЖИМЕ ОТЛАДКИ V5 (Анализ типов связей)")
    print("="*60)
    
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

    print("\n>>> Шаг 2: Анализ содержимого `bond_orders`...")
    
    # --- ОТЛАДКА: Считаем все уникальные типы связей ---
    bond_types_found = Counter()
    processed_molecules_with_bonds = 0
    total_bonds_checked = 0

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        
        # Проверяем первые 10000 записей, чтобы собрать достаточно статистики
        for i, (key, value) in enumerate(tqdm(cursor.iternext(keys=True, values=True), total=db_size, desc="Анализ молекул")):
            if i >= 10000:
                break
            
            try:
                molecule_index = int(key.decode())
                if molecule_index >= len(original_data): continue
                
                mol_data = original_data[molecule_index]
                if not all(k in mol_data for k in ['edge_index', 'bond_orders']): continue
                
                bond_orders = mol_data['bond_orders']
                
                # Итерируемся по всем связям и просто считаем их типы
                for j in range(bond_orders.shape[0]):
                    bond_type = bond_orders[j].item()
                    bond_types_found[bond_type] += 1
                
                total_bonds_checked += bond_orders.shape[0]
                processed_molecules_with_bonds += 1

            except Exception:
                continue
            
    env.close()

    print("\n" + "="*60)
    print("           РЕЗУЛЬТАТЫ АНАЛИЗА ТИПОВ СВЯЗЕЙ")
    print("="*60)
    print(f"Проверено молекул: {processed_molecules_with_bonds}")
    print(f"Всего связей проанализировано: {total_bonds_checked}")
    print("\nНайдены следующие уникальные типы связей (тип: количество):")
    
    if not bond_types_found:
        print("    !!! Связей не найдено вообще. Возможно, `bond_orders` всегда пустой.")
    else:
        for bond_type, count in sorted(bond_types_found.items()):
            print(f"    - Тип {bond_type}: {count} раз")
    
    print("="*60)
    
    bond_to_label_map = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3}
    print("\nНаш текущий словарь для сравнения:")
    print(bond_to_label_map)
    
    print("\nВЫВОД: Сравните 'Найденные типы' с 'Нашим словарем'.")
    print("Если типы не совпадают (например, 1 вместо 1.0), это и есть причина проблемы.")


if __name__ == "__main__":
    debug_main_v5()