import torch
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import os
import lmdb
import pickle
import pandas as pd # <-- Импортируем pandas для сохранения CSV

### --- Функция для красивого вывода (без изменений) --- ###
def print_custom_report(y_true, y_pred, le, label_map, title="ОЦЕНКА МОДЕЛИ"):
    """
    Печатает отчет по метрикам в заданном табличном формате.
    """
    # Фильтруем метки, чтобы остались только те, что есть в label_map
    labels_to_report_encoded = le.transform([k for k in label_map.keys() if k in le.classes_])
    
    if len(labels_to_report_encoded) == 0:
        print("В данных для отчета нет ни одного из целевых классов.")
        return

    # Рассчитываем метрики для каждого класса
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels_to_report_encoded, zero_division=0)
    
    # Рассчитываем общие метрики
    acc_total = accuracy_score(y_true, y_pred)
    p_total, r_total, f1_total, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # --- Печать таблицы ---
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)
    print(f"{'':<15} | {'Acc':^10} | {'F1':^10} | {'Precision':^12} | {'Recall':^10}")
    print("-"*60)
    
    print(f"{'total':<15} | {acc_total:^10.3f} | {f1_total:^10.3f} | {p_total:^12.3f} | {r_total:^10.3f}")
    
    for i, class_name in label_map.items():
        if i in le.classes_:
            encoded_label_val = le.transform([i])[0]
            # Находим позицию нашей метки в отсортированном списке `labels_to_report_encoded`
            report_idx = np.where(labels_to_report_encoded == encoded_label_val)[0]
            if len(report_idx) > 0:
                idx = report_idx[0]
                mask = y_true == encoded_label_val
                if np.sum(mask) > 0:
                    class_acc = accuracy_score(y_true[mask], y_pred[mask])
                else:
                    class_acc = 0.0 # Если класса нет в y_true, точность 0
                print(f"{class_name.lower():<15} | {class_acc:^10.3f} | {f1[idx]:^10.3f} | {p[idx]:^12.3f} | {r[idx]:^10.3f}")

    print("="*60)

def main():
    print(">>> Шаг 1: Инициализация данных...")
    LMDB_PATH = './molecule_embeddings.lmdb'
    ORIGINAL_DATA_PATH = './geom_train.pt'
    
    try:
        original_data = torch.load(ORIGINAL_DATA_PATH, map_location='cpu')
        env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            db_size = txn.stat()['entries']
        print(f"[OK] Все данные и LMDB ({db_size} записей) успешно загружены.")
    except Exception as e:
        print(f"ОШИБКА при загрузке: {e}")
        return

    print("\n>>> Шаг 2: Формирование датасета (X, y)...")
    
    X_features, y_labels = [], []
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        iterator = tqdm(cursor.iternext(keys=True, values=True), total=db_size, desc="Обработка молекул")
        
        for key, value in iterator:
            try:
                molecule_index = int(key.decode())
                if molecule_index >= len(original_data): continue
                molecule_embeddings_data = pickle.loads(value)
                if 'pair_embeddings' not in molecule_embeddings_data: continue
                
                mol_embeddings = molecule_embeddings_data['pair_embeddings']
                if isinstance(mol_embeddings, torch.Tensor):
                    mol_embeddings = mol_embeddings.numpy()

                mol_data = original_data[molecule_index]
                if not all(k in mol_data for k in ['edge_index', 'bond_orders']): continue
                
                edge_index = mol_data['edge_index']
                if edge_index.shape[0] == 0: continue
                
                bond_orders_matrix = mol_data['bond_orders']
                if edge_index.shape[0] != bond_orders_matrix.shape[0]: continue
                
                bond_type_indices = torch.argmax(bond_orders_matrix, dim=1)
                
                for j in range(edge_index.shape[0]):
                    atom1_idx, atom2_idx = edge_index[j, 0].item() + 1, edge_index[j, 1].item() + 1
                    if (atom1_idx >= mol_embeddings.shape[0] or atom2_idx >= mol_embeddings.shape[1]): continue
                    label = bond_type_indices[j].item()
                    
                    feature_vec = (mol_embeddings[atom1_idx, atom2_idx] + mol_embeddings[atom2_idx, atom1_idx]) / 2.0
                    X_features.append(feature_vec)
                    y_labels.append(label)
            except Exception:
                continue
            
    env.close()

    if not X_features:
        print("\nОШИБКА: Не удалось сформировать датасет.")
        return
    
    print(f"\nДатасет сформирован. Всего связей для анализа: {len(y_labels)}")
    
    X = np.array(X_features, dtype=np.float32)
    y_raw_labels = np.array(y_labels, dtype=np.int32)
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw_labels)
    
    report_label_map = {
        0: 'single bond', 1: 'double bond', 2: 'triple bond', 3: 'Aromatic'
    }
    
    num_classes = len(np.unique(y))
    print(f"Обнаружено {num_classes} уникальных типов связей в датасете.")

    print("\n>>> Шаг 3: Обучение XGBoost...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=50, max_depth=6, n_jobs=4, learning_rate=0.01
    )
    model.fit(X_train, y_train, verbose=False)
    print("Модель XGBoost обучена.")
    
    ### --- НАЧАЛО: Оценка на ОБУЧАЮЩЕЙ выборке --- ###
    print("\n>>> Шаг 4: Оценка модели...")
    y_pred_train = model.predict(X_train)
    print_custom_report(y_train, y_pred_train, le, report_label_map, title="Табл. № (результаты на TRAIN выборке)")
    ### --- КОНЕЦ: Оценка на ОБУЧАЮЩЕЙ выборке --- ###

    ### --- НАЧАЛО: Оценка и СОХРАНЕНИЕ для ТЕСТОВОЙ выборки --- ###
    y_pred_test = model.predict(X_test)
    
    # Выводим отчет в кастомном формате для тестовой выборки
    print_custom_report(y_test, y_pred_test, le, report_label_map, title="Табл. № (результаты на TEST выборке)")
    
    # --- Сохраняем предсказания в CSV ---
    print("\nСохранение предсказаний на тестовой выборке...")
    
    # Обратно преобразуем закодированные метки в оригинальные (0, 1, 2, 3...)
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred_test)
    
    # Создаем DataFrame
    predictions_df = pd.DataFrame({
        'true_label': y_test_original,
        'predicted_label': y_pred_original
    })
    
    # Сохраняем в файл
    output_csv_path = 'test_predictions.csv'
    predictions_df.to_csv(output_csv_path, index=False)
    print(f"Предсказания успешно сохранены в файл: {output_csv_path}")
    ### --- КОНЕЦ: Оценка и СОХРАНЕНИЕ для ТЕСТОВОЙ выборки --- ###

if __name__ == "__main__":
    main()