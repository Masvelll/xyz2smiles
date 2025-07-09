import sys 
import torch
import numpy as np
import json

from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer, AutoModelForCausalLM


pairs = torch.load('../burov/unimol_embeddings_project/geom_train_pair_embeddings.pt', weights_only=True)
geom = torch.load('./datasets/geom_train.pt', weights_only=True)


k         = 128
batchsize = 10_000           # сколько 64-векторов обрабатываем за раз
max_pairs = 10_000_000          # всего примеров, на которых «доведём» центры

kmeans = MiniBatchKMeans(n_clusters=k,
                         batch_size=batchsize,
                         init_size=k*3,        # можно побольше для устойчивости
                         verbose=0,
                         random_state=42)

seen = 0
for mol in tqdm(pairs):                   # pair_list: list[T(N,N,64)]
    vecs = mol.reshape(-1, 64)                # (N²,64)   – в gpu/cpu памяти молекулы
    # --- случайно берём не больше batchsize векторов ---
    if vecs.size(0) > batchsize:
        idx = torch.randperm(vecs.size(0))[:batchsize]
        vecs = vecs[idx]

    kmeans.partial_fit(vecs.cpu().numpy())    # учим на CPU, по кусочкам
    seen += vecs.size(0)
    if seen >= max_pairs:                     # хватит примеров – выходим
        break

centers = torch.tensor(kmeans.cluster_centers_)   # (128,64)  • готово
torch.save(centers, "centers_pair128.pt")


def vec2tok(v: torch.Tensor) -> str | list[str]:
    """
    v : Tensor (64,)       → '<p042>'
        Tensor (M,64)      → ['<p042>', '<p118>', ...] длиной M
    """
    v = v.to(centers)                       # убедимся, что на том же девайсе

    if v.ndim == 1:                         # одиночный вектор
        idx = torch.cdist(v[None], centers).argmin().item()
        return f"<p{idx:03d}>"

    # батч M×64  → M индексов
    idx = torch.cdist(v, centers).argmin(dim=1).tolist()   # List[int] длиной M
    return [f"<p{i:03d}>" for i in idx]                    # List[str]


model_name = "Qwen/Qwen1.5-1.8B-Chat"   # или другая
tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model      = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 128 новых «pair-токенов»
new_tokens = [f"<p{i:03d}>" for i in range(128)]

num_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
print("Добавили:", num_added)      # должно быть 128
model.resize_token_embeddings(len(tokenizer))   # 🔑 расширяем эмбеддинг



SYSTEM = ("You are a chemist. For each atom pair within 2 Å classify the "
          "bond type. Labels: 0 no-bond, 1 single, 2 double, 3 triple, 4 aromatic. "
          "Return ONLY JSON list [{\"pair\":[i,j],\"label\":n}].")

ORDER2LABEL = {0:1, 1:2, 2:3, 3:4, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}
ALLOWED = ['C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P']

dev = 'cuda'
centers_gpu = centers.to(dev)
with open("bond_prompts.jsonl", "w") as fout:
    
    for mol, pair in tqdm(zip(geom, pairs), total=len(geom)):
        xyz   = mol["positions"].to(dev, non_blocking=True)      # (N,3)
        types = mol["one_hot"].argmax(-1)                        # остаётся на CPU
        pair  = pair.to(dev, non_blocking=True)                  # (N,N,64)

        dmat  = torch.cdist(xyz, xyz)                            # GPU, fp32
        i_idx, j_idx = (dmat<=2.0).nonzero(as_tuple=True)        # тоже GPU
        if not len(i_idx):  continue

        vecs   = pair[i_idx, j_idx]                              # (M,64)_gpu
        idx    = torch.cdist(vecs, centers_gpu).argmin(dim=1)        # (M,)_gpu
        toks   = [f"<p{i:03d}>" for i in idx.tolist()]                            # List[str]  (исправленный!)

        lines = [f"[{i},{j}]={tok}"
                for (i,j),tok in zip(zip(i_idx.tolist(), j_idx.tolist()), toks)]

        # ---------- метки ------------------------------------------------------
        edge2order = {(int(u),int(v)): bo.nonzero(as_tuple=True)[0].item()
                    for (u,v), bo in zip(mol["edge_index"], mol["bond_orders"])}
        edge2order |= {(v,u):o for (u,v),o in edge2order.items()}

        labels = [0 if (i,j) not in edge2order else ORDER2LABEL[edge2order[(i,j)]]
                for i,j in zip(i_idx.tolist(), j_idx.tolist())]

        assistant = json.dumps([{"pair":[i,j],"label":l}
                                for (i,j),l in zip(zip(i_idx.tolist(), j_idx.tolist()),
                                                labels)],
                            separators=(",",":"))

        atom_line = "Atoms (index→type): " + \
                    ", ".join(f"{idx}:{ALLOWED[types[idx]]}"
                            for idx in range(len(types)))

        prompt = {"messages":[
            {"role":"system","content":SYSTEM},
            {"role":"user",
            "content":"Pairs within 2 Å:\n" + "\n".join(lines) + "\n\n" + atom_line},
            {"role":"assistant","content":assistant}
        ]}
        fout.write(json.dumps(prompt)+"\n")

