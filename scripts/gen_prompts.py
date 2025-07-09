import json
import torch
from tqdm import tqdm

ORDER2LABEL = {0:1, 1:2, 2:3, 3:4, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}
ALLOWED = ['C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P']

geom = torch.load('./mydatasets/geom_test.pt', weights_only=True)
dev = 'cuda'

with open("bond_prompts_test.jsonl", "w") as fout:
    for mol in tqdm(geom, total=len(geom)):
        xyz   = mol["positions"].to(dev)                  # (N, 3)
        types = mol["one_hot"].argmax(dim=-1).cpu()       # (N,)
        dmat  = torch.cdist(xyz, xyz)                     # (N, N)

        i_idx, j_idx = (dmat <= 2.0).nonzero(as_tuple=True)
        if not len(i_idx): continue

        # Определение порядка связи
        edge2order = {
            (int(u), int(v)): bo.nonzero(as_tuple=True)[0].item()
            for (u, v), bo in zip(mol["edge_index"], mol["bond_orders"])
        }
        edge2order |= {(v, u): o for (u, v), o in edge2order.items()}

        # Формируем пары и метки
        ij_pairs = list(zip(i_idx.tolist(), j_idx.tolist()))
        labels = [
            0 if (i, j) not in edge2order else ORDER2LABEL[edge2order[(i, j)]]
            for i, j in ij_pairs
        ]

        # Строим строки
        input_lines = "\n".join(f"{i} {j}" for i, j in ij_pairs)
        output_lines = "\n".join(str(label) for label in labels)
        atom_types = " ".join(f"{i}:{ALLOWED[types[i]]}" for i in range(len(types)))

        # Постановка задачи
        prompt_text = (
            "You are a chemist. For each atom pair within 2 Å classify the bond type:\n"
            + input_lines + "\n\nAtoms: " + atom_types
        )

        fout.write(json.dumps({
            "input": prompt_text,
            "output": output_lines
        }) + "\n")
