import torch
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import lmdb
import pickle
import pandas as pd
import gc

def calculate_exact_match(y_true, y_pred, molecule_ids):
    """Вычисляет Exact Match Accuracy по молекулам"""
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'mol_id': molecule_ids})
    mol_groups = df.groupby('mol_id')
    exact_matches = sum(group['true'].equals(group['pred']) for _, group in mol_groups)
    return exact_matches / len(mol_groups)

def main():
    # Оптимизация памяти
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'

    print(">>> Шаг 1: Инициализация данных...")
    LMDB_PATH = './molecule_embeddings.lmdb'
    ORIGINAL_DATA_PATH = './geom_train.pt'
    
    try:
        # Загрузка с оптимизацией памяти
        original_data = torch.load(ORIGINAL_DATA_PATH, map_location='cpu')
        env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
        with env.begin() as txn:
            db_size = txn.stat()['entries']
        print(f"[OK] Данные загружены. Молекул: {len(original_data)}, Записей в LMDB: {db_size}")
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return

    print("\n>>> Шаг 2: Формирование датасета...")
    X, y, mol_ids = [], [], []
    
    with env.begin(write=False) as txn:
        for key, value in tqdm(txn.cursor(), total=db_size, desc="Чтение LMDB"):
            try:
                mol_idx = int(key.decode())
                if mol_idx >= len(original_data): continue
                
                mol_data = original_data[mol_idx]
                if not all(k in mol_data for k in ['edge_index', 'bond_orders']): continue
                
                emb_data = pickle.loads(value)
                if 'pair_embeddings' not in emb_data: continue
                
                embeddings = emb_data['pair_embeddings'].numpy() if torch.is_tensor(emb_data['pair_embeddings']) else emb_data['pair_embeddings']
                edge_index = mol_data['edge_index']
                bond_types = torch.argmax(mol_data['bond_orders'], dim=1)
                
                for j in range(edge_index.shape[0]):
                    a1, a2 = edge_index[j, 0].item()+1, edge_index[j, 1].item()+1
                    if a1 >= embeddings.shape[0] or a2 >= embeddings.shape[1]: continue
                    
                    X.append((embeddings[a1, a2] + embeddings[a2, a1]) / 2)
                    y.append(bond_types[j].item())
                    mol_ids.append(mol_idx)
            except Exception:
                continue
    
    env.close()
    del original_data
    gc.collect()

    if not X:
        print("ОШИБКА: Не удалось сформировать датасет")
        return

    print(f"\nДатасет сформирован. Связей: {len(y)}, Уникальных молекул: {len(set(mol_ids))}")

    # Преобразование и разделение данных
    X = np.array(X, dtype=np.float32)
    y = LabelEncoder().fit_transform(y)
    
    # Разделение с сохранением молекулярной структуры
    X_train, X_test, y_train, y_test, mol_train, mol_test = train_test_split(
        X, y, mol_ids, test_size=0.2, random_state=42, stratify=y
    )

    print("\n>>> Шаг 3: Обучение RandomForest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    print("\n>>> Шаг 4: Оценка модели...")
    # Общая accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nОбщая Accuracy: {acc:.4f}")

    # Exact Match по молекулам
    exact_match = calculate_exact_match(y_test, y_pred, mol_test)
    print(f"Exact Match Accuracy: {exact_match:.4f}")

    # Сохранение результатов
    pd.DataFrame({
        'true': y_test,
        'pred': y_pred,
        'mol_id': mol_test
    }).to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()