import torch
import numpy as np

# Этот скрипт не требует никаких других библиотек

def final_inspection():
    print("="*60)
    print("      ФИНАЛЬНАЯ ИНСПЕКЦИЯ: СМОТРИМ НА СЫРЫЕ ДАННЫЕ")
    print("="*60)
    
    ORIGINAL_DATA_PATH = 'geom_train.pt'
    
    try:
        original_data = torch.load(ORIGINAL_DATA_PATH, map_location='cpu')
        print(f"[OK] Файл '{ORIGINAL_DATA_PATH}' загружен.")
    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при загрузке: {e}")
        return

    if not original_data:
        print("Файл пуст!")
        return
        
    # Берем самую первую молекулу
    mol_data = original_data[0]
    
    print("\n--- Анализ молекулы #0 ---")
    print(f"Ключи: {list(mol_data.keys())}")
    
    # --- Анализ edge_index ---
    if 'edge_index' in mol_data:
        edge_index = mol_data['edge_index']
        print("\n--- ДЕТАЛИ `edge_index` ---")
        print(f"  Тип данных: {edge_index.dtype}")
        print(f"  Размер (shape): {edge_index.shape}")
        # Печатаем первые 5 пар индексов
        print(f"  Первые 5 связей (колонки): \n{edge_index[:, :5]}")
    else:
        print("\n!!! КЛЮЧ `edge_index` ОТСУТСТВУЕТ !!!")
        
        
    # --- Анализ one_hot ---
    if 'one_hot' in mol_data:
        one_hot = mol_data['one_hot']
        print("\n--- ДЕТАЛИ `one_hot` ---")
        print(f"  Тип данных: {one_hot.dtype}")
        print(f"  Размер (shape): {one_hot.shape}")
        print(f"  Количество атомов: {one_hot.shape[0]}")
    else:
        print("\n!!! КЛЮЧ `one_hot` ОТСУТСТВУЕТ !!!")


    # --- Анализ bond_orders ---
    if 'bond_orders' in mol_data:
        bond_orders = mol_data['bond_orders']
        print("\n--- ДЕТАЛИ `bond_orders` ---")
        print(f"  Тип данных: {bond_orders.dtype}")
        print(f"  Размер (shape): {bond_orders.shape}")
        
        # Проверяем, совпадает ли количество связей
        if 'edge_index' in mol_data and edge_index.shape[1] == bond_orders.shape[0]:
            print("  [OK] Количество связей в `edge_index` и `bond_orders` совпадает.")
        elif 'edge_index' in mol_data:
            print(f"  [!!!] ВНИМАНИЕ: Несовпадение! Связей в `edge_index`: {edge_index.shape[1]}, а в `bond_orders`: {bond_orders.shape[0]}")
        
        # Печатаем первые 5 векторов типов связей
        print(f"  Первые 5 векторов `bond_orders`: \n{bond_orders[:5]}")
        
        # Декодируем и считаем типы
        if bond_orders.ndim == 2: # Если это матрица one-hot
            bond_type_indices = torch.argmax(bond_orders, dim=1)
            print(f"\n  Декодированные индексы типов связей (первые 15): {bond_type_indices[:15].tolist()}")
            
            from collections import Counter
            counts = Counter(bond_type_indices.tolist())
            print("  Статистика по декодированным индексам:")
            for bond_idx, count in sorted(counts.items()):
                print(f"    - Индекс {bond_idx}: {count} раз")
        else:
            print("\n  Это не one-hot матрица, а простой вектор. Уникальные значения:")
            print(f"    {torch.unique(bond_orders)}")

    else:
        print("\n!!! КЛЮЧ `bond_orders` ОТСУТСТВУЕТ !!!")
        
    print("\n" + "="*60)


if __name__ == "__main__":
    final_inspection()