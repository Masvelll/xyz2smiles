import torch
import torch.nn as nn

class EmbeddingProjector(nn.Module):
    def __init__(self, d_atom: int, d_edge: int, d_llm: int, device="cuda"):
        super().__init__()
        self.atom_proj = nn.Linear(d_atom, d_llm)
        self.edge_proj = nn.Linear(d_edge, d_llm)
        self.device = device

    def forward(self, embedding: dict, label_pairs: list):
        # Переводим в тензоры сразу и на нужное устройство
        A = torch.tensor(embedding["atom_embeddings"], device=self.device)     # [N, D_atom]
        E = torch.tensor(embedding["pair_embeddings"], device=self.device)     # [N, N, D_edge]

        label_pairs = torch.tensor(label_pairs, device=self.device)            # [num_pairs, 3]
        idx_i = label_pairs[:, 0]  # [num_pairs]
        idx_j = label_pairs[:, 1]
        labels = label_pairs[:, 2] # [num_pairs]

        A_proj = self.atom_proj(A)    # [N, D_llm]
        E_proj = self.edge_proj(E)    # [N, N, D_llm]

        # Получаем a_i, a_j, e_ij векторизованно
        a_i = A_proj[idx_i]                          # [num_pairs, D_llm]
        a_j = A_proj[idx_j]
        e_ij = E_proj[idx_i, idx_j]                  # [num_pairs, D_llm]

        tokens = torch.cat([a_i, a_j, e_ij], dim=-1)  # [num_pairs, 3 * D_llm]

        return tokens, labels
