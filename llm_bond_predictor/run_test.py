import torch
from torch.utils.data import DataLoader
import sys
import json
sys.path.insert(1, '/home/user12/spavlenko')
import emb_dataset as ds     # ваш модуль
from llm_bond import BondPredictorLLM
from llm_bond import BondPredictorGRPO
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm

LMDB_DIR   = "/home/user12/burov/unimol_embeddings_project/molecule_embeddings_test.lmdb"           # где лежат эмбеддинги
JSONL_PATH = "/home/user12/prompts2/prompts_test.jsonl"           # и prompts.jsonl
BATCH_SIZE = 16

dataset = ds.PairedLMDBJsonlDataset(LMDB_DIR, JSONL_PATH)
loader  = DataLoader(dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     collate_fn=ds.collate_graph_batch,
                     num_workers=0)


# CKPT_PATH = "lightning_logs/version_0/checkpoints/epoch=2-step=294618.ckpt"
CKPT_PATH = "/home/user12/maslov/ckpts/best-val/acc=0.9981.ckpt"
print('Loading model...')
model = BondPredictorGRPO.load_from_checkpoint(CKPT_PATH).to("cuda").eval()
model.eval()
y_true, y_pred = [], []
print('Making predictions...')
with torch.no_grad():
    for batch in tqdm(loader, total=len(loader), desc="Inference", unit="batch"):
        # перенос только тензоров
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to("cuda", non_blocking=True)   # чуть быстрее

        logits = model.forward(batch)                       # [K, num_classes]
        y_pred.extend(logits.argmax(-1).cpu().tolist())
        y_true.extend(batch["labels"].cpu().tolist())



print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred, average='weighted'))
print("Precision:", precision_score(y_true, y_pred, average='weighted'))
print("Recall", recall_score(y_true, y_pred, average='weighted'))

target_names = ["no-bond", "single", "double", "triple", "aromatic"]
print(classification_report(y_true, y_pred,
                            target_names=target_names,
                            digits=4))

# predictions = []

# with torch.no_grad():
#     for batch in tqdm(loader, total=len(loader), desc="Inference", unit="batch"):
#         # На GPU
#         for k, v in batch.items():
#             if torch.is_tensor(v):
#                 batch[k] = v.to("cuda")

#         logits = model(batch)  # [K, num_classes]
#         preds = logits.argmax(-1).cpu().tolist()
#         labels = batch["labels"].cpu().tolist()
#         pairs = batch["pair_indices"].cpu().tolist()
#         prompts = batch["prompts"]  # List[str]

#         # Сгруппируем по молекулам
#         mol_preds = dict()
#         for (mol_idx, i, j), pred, label in zip(pairs, preds, labels):
#             if mol_idx not in mol_preds:
#                 mol_preds[mol_idx] = {"pairs": [], "labels": []}
#             mol_preds[mol_idx]["pairs"].append([i, j, pred])
#             mol_preds[mol_idx]["labels"].append([i, j, label])

#         for mol_idx, entry in mol_preds.items():
#             predictions.append({
#                 "prompt": prompts[mol_idx],
#                 "predictions": entry["pairs"],
#                 "labels": entry["labels"]  # можно опустить
#             })

# N = len(predictions)
# acc = sum([pred['predictions'] == pred['labels'] for pred in predictions]) / N
# print(f'Exact match: {acc:.4f}')
# # Сохраним в JSONL
# with open("bond_predictions.jsonl", "w") as f:
#     for item in tqdm(predictions, desc='Json writing'):
#         f.write(json.dumps(item) + "\n")

