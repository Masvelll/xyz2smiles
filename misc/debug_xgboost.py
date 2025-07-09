import torch
import numpy as np
import os
import lmdb
import pickle
from collections import Counter

### --- Основной код отладчика --- ###

def debug_main():
    print("="*60)
    print("      ЗАПУСК В РЕЖИМЕ ДЕТАЛЬНОЙ ОТЛАДКИ")
    print("="*60)
    
    # --- Шаг 1: Проверка путей и загрузка данных ---
    print("\n>>> Шаг 1: Проверка путей и загрузка данных...")
    
    LMDB_PATH = './molecule_embeddings.lmdb'
    ORIGINAL_DATA_PATH = 'geom_train.pt'
    
    if not os.path.exists(ORIGINAL_DATA_PATH):
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Файл с исходными данными не найден: {ORIGINAL_DATA_PATH}")
        return
        
    try:
        original_data = torch.load(ORIGINAL_DATA_PATH, map_location='cpu')
        num_molecules_total = len(original_data)
        print(f"    [OK] Загружено {num_molecules_total} исходных молекул.")
    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при загрузке исходных данных: {str(e)}")
        return

    if not os.path.exists(LMDB_PATH):
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: База данных LMDB не найдена: {LMDB_PATH}")
        return

    try:
        env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            db_size = txn.stat()['entries']
        print(f"    [OK] База данных LMDB '{LMDB_PATH}' успешно открыта. Записей: {db_size}")
    except lmdb.Error as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось открыть LMDB базу данных '{LMDB_PATH}'. {e}")
        return

    # --- Шаг 2: Итерация по LMDB и анализ данных ---
    print("\n>>> Шаг 2: Анализ данных из LMDB (проверяем первые 1000 записей)...")
    
    # Переменные для отладочной статистики
    X_features = []
    y_labels = []
    processed_count = 0
    errors = Counter()
    bond_types_found = Counter()
    
    bond_to_label_map = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3}
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        
        # Используем `enumerate` и ограничиваем проверку для скорости
        for i, (key, value) in enumerate(cursor):
            if i >= 1000: # Проверяем только первые 1000 записей
                break
            
            try:
                molecule_index = int(key.decode())
                
                if molecule_index >= num_molecules_total:
                    errors['index_out_of_bounds'] += 1
                    continue

                mol_data = original_data[molecule_index]
                
                if not all(k in mol_data for k in ['edge_index', 'bond_orders']):
                    errors['missing_keys_in_original_data'] += 1
                    continue
                
                molecule_embeddings_data = pickle.loads(value)
                
                if 'pair_embeddings' not in molecule_embeddings_data:
                    errors['missing_pair_embeddings_in_lmdb'] += 1
                    continue

                mol_embeddings = molecule_embeddings_data['pair_embeddings']
                
                edge_index, bond_orders = mol_data['edge_index'], mol_data['bond_orders']
                
                found_bonds_in_mol = 0
                for j in range(edge_index.shape[1]):
                    atom1_idx, atom2_idx = edge_index[0, j].item() + 1, edge_index[1, j].item() + 1
                    
                    if (atom1_idx >= mol_embeddings.shape[0] or atom2_idx >= mol_embeddings.shape[1]):
                        errors['embedding_index_out_of_bounds'] += 1
                        continue
                        
                    bond_type = bond_orders[j].item()
                    bond_types_found[bond_type] += 1 # Считаем все типы связей, которые видим
                    
                    if bond_type not in bond_to_label_map:
                        errors['unknown_bond_type'] += 1
                        continue
                    
                    label = bond_to_label_map[bond_type]
                    feature_vec = (mol_embeddings[atom1_idx, atom2_idx] + mol_embeddings[atom2_idx, atom1_idx]) / 2.0
                    
                    X_features.append(feature_vec)
                    y_labels.append(label)
                    found_bonds_in_mol += 1
                
                if found_bonds_in_mol > 0:
                    processed_count += 1

            except Exception as e:
                errors[f'general_error_{type(e).__name__}'] += 1
                continue
                
    env.close()

    # --- Шаг 3: Вывод детальной статистики ---
    print("\n" + "="*60)
    print("           РЕЗУЛЬТАТЫ ОТЛАДКИ")
    print("="*60)
    print(f"Проверено записей в LMDB:      {min(1000, db_size)}")
    print(f"Из них успешно обработано молекул (найдены связи): {processed_count}")
    print(f"Всего найдено связей (добавлено в датасет): {len(X_features)}")
    print("-" * 60)
    print("Статистика по ошибкам и пропускам:")
    if not errors:
        print("    [OK] Критических ошибок при обработке не найдено.")
    for error_type, count in errors.items():
        print(f"    - {error_type}: {count} раз")
    print("-" * 60)
    print("Статистика по найденным типам связей:")
    if not bond_types_found:
        print("    Связей не найдено вообще.")
    for bond_type, count in bond_types_found.items():
        is_known = " (известный)" if bond_type in bond_to_label_map else " (НЕИЗВЕСТНЫЙ)"
        print(f"    - Тип связи {bond_type}: {count} раз{is_known}")
    print("="*60)
    
    if not X_features:
        print("\nВЫВОД: Датасет остался пустым. Проанализируйте статистику выше.")
        print("Наиболее вероятные причины:")
        print("  1. В `original_data` нет ключей `edge_index` или `bond_orders`.")
        print("  2. Все типы связей в данных не входят в `bond_to_label_map` (1.0, 1.5, 2.0, 3.0).")
        print("  3. Ошибка в индексации (проверьте 'embedding_index_out_of_bounds').")
    else:
        print("\nВЫВОД: Датасет успешно сформирован! Проблема, скорее всего, была в другом месте.")
        print("Теперь можно интегрировать эту логику в основной скрипт `run_classification.py`.")

if __name__ == "__main__":
    debug_main()