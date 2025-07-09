from tqdm import tqdm
import torch, json

geom = torch.load('./datasets/geom_train.pt')

max_dist = 2.0
out = open("bond_prompts.jsonl", "w")

for mol in tqdm(geom):                           # geom — это list(dict)
    xyz   = mol["coords"]        # (N, 3)
    types = mol["one_hot"].argmax(-1)           # (N,)

    # 1. полная матрица расстояний
    dmat = torch.cdist(xyz, xyz, p=2)           # (N, N)

    # 2. выбираем пары ≤ 2 Å (и не i==j)
    idx   = (dmat > 0) & (dmat <= max_dist)
    pairs = idx.nonzero(as_tuple=False)         # (M, 2)

    if len(pairs) == 0:
        continue

    # ────────────────── постройте dict → (i,j) → one-hot в bond_orders ──────────
    edge2order = {}
    ei, bo = mol["edge_index"], mol["bond_orders"]          # (E,2), (E,10)
    for k in range(ei.shape[0]):
        u, v = map(int, ei[k])
        edge2order[(u, v)] = bo[k]
    # undirected: сразу добавим обратное направление
        edge2order[(v, u)] = bo[k]

    # ────────────────── определяем label для каждой пары ≤ 2 Å ───────────────────
    labels = []
    for i, j in pairs:                                      # pairs — Tensor (M,2)
        vec = edge2order.get((int(i), int(j)))              # None, если нет связи
        if vec is None:
            bond = 0                                        # «no-bond»
        else:
            idx = vec.nonzero(as_tuple=True)[0].item()      # 0…9
            bond = {0:0, 1:1, 2:2, 3:3, 4:1}.get(idx, 0)    # сводим к 0-3
        labels.append(bond)


    # 4. строим prompt
    atom_map = ", ".join(f"{i}:{'CONHXBOSPFI'[types[i]]}"   # грубое сокращение; лучше RDKit
                         for i in range(len(types)))
    dist_map = "\n".join(f"[{int(i)},{int(j)}]={dmat[i,j]:.3f}"
                         for i, j in pairs)
    system = ("You are a chemist. Task: classify bond type (0 no-bond, 1 single, "
              "2 double, 3 triple). Return ONLY JSON list "
              "`[{\"pair\":[i,j],\"label\":n}, ...]` in the given order.")
    user   = f"Atoms (index→element): {atom_map}\nDistances (Å):\n{dist_map}"
    assistant = json.dumps(
        [{"pair":[int(i), int(j)], "label": int(l)}
         for (i, j), l in zip(pairs, labels)],
        separators=(",", ":"))

    json.dump({"messages":[
        {"role":"system", "content":system},
        {"role":"user",    "content":user},
        {"role":"assistant","content":assistant}
    ]}, out); out.write("\n")
