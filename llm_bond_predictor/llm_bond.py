import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from torch.distributions import Categorical
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,)
from torchmetrics.functional import precision, recall, f1_score


class BondPredictorLLM(pl.LightningModule):
    def __init__(
        self,
        llm_id: str,
        d_atom: int,
        d_edge: int,
        d_llm: int,
        num_classes: int = 4,
        lr: float = 2e-5,
        concat_order: str = "graph_first",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tok = AutoTokenizer.from_pretrained(llm_id)
        # self.llm = AutoModel.from_pretrained(llm_id)
        self.llm = T5EncoderModel.from_pretrained(llm_id)
        self.lr = lr
        self.concat_order = concat_order

        # Проекции эмбеддингов атомов и связей
        self.atom_proj = nn.Linear(d_atom, d_llm)
        self.edge_proj = nn.Linear(d_edge, d_llm)

        # Итоговая проекция токена (A_i, A_j, E_ij) -> d_llm
        self.token_proj = nn.Linear(3 * d_llm, d_llm)

        # Классификатор
        self.classifier = nn.Linear(d_llm, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        
        atom_emb = batch["atom_embeddings"]  # [B, N, D_atom]
        pair_emb = batch["pair_embeddings"]  # [B, N, N, D_edge]
        pair_idx = batch["pair_indices"]     # [K, 3] -> (mol_idx, i, j)
        prompts  = batch["prompts"]          # List[str]

        B, N, _ = atom_emb.shape
        K = pair_idx.shape[0]

        atom_emb_llm = self.atom_proj(atom_emb)       # [B, N, d_llm]
        edge_emb_llm = self.edge_proj(pair_emb)       # [B, N, N, d_llm]

        # собираем input_tokens [K, 3*d_llm] -> [K, d_llm]
        mol_idx, i_idx, j_idx = pair_idx[:, 0], pair_idx[:, 1], pair_idx[:, 2]
        # assert torch.all(mol_idx < atom_emb_llm.size(0))
        # print(i_idx, atom_emb_llm.size(1))
        # print(atom_emb_llm.shape, i_idx)
        assert torch.all(i_idx  < atom_emb_llm.size(1))
        # assert torch.all(j_idx  < atom_emb_llm.size(1))
        A_i = atom_emb_llm[mol_idx, i_idx]
        A_j = atom_emb_llm[mol_idx, j_idx]
        E_ij = edge_emb_llm[mol_idx, i_idx, j_idx]
        token_input = torch.cat([A_i, A_j, E_ij], dim=-1)
        input_tokens = self.token_proj(token_input)  # [K, d_llm]

        # Process each prompt separately and collect embeddings for each pair
        logits_dict = {}
        
        # Group pairs by molecule
        pairs_by_mol = {}
        for k, (mol_idx, i_idx, j_idx) in enumerate(pair_idx):
            mol_idx = mol_idx.item()
            if mol_idx not in pairs_by_mol:
                pairs_by_mol[mol_idx] = []
            pairs_by_mol[mol_idx].append(k)
        
        # Process each molecule separately
        for mol_idx, pair_indices_for_mol in pairs_by_mol.items():
            # Get tokens for this molecule
            mol_tokens = input_tokens[pair_indices_for_mol]  # [num_pairs_in_mol, d_llm]
            
            # Tokenize the prompt for this molecule
            tokenized = self.tok(prompts[mol_idx], return_tensors="pt", max_length=1024).to(self.device)
            prompt_embeds = self.llm.get_input_embeddings()(tokenized.input_ids).squeeze(0)  # [T, d_llm]

            # Concatenate tokens and prompt
            if self.concat_order == "graph_first":
                full_input = torch.cat([mol_tokens, prompt_embeds], dim=0)
                offset = 0
            else:
                full_input = torch.cat([prompt_embeds, mol_tokens], dim=0)
                offset = prompt_embeds.shape[0]

            # Process through LLM
            out = self.llm(inputs_embeds=full_input.unsqueeze(0), return_dict=True)
            hidden = out.last_hidden_state.squeeze(0)  # [T+num_pairs, d_llm]

            # Extract graph outputs
            graph_out = hidden[offset:offset + len(pair_indices_for_mol)]  # [num_pairs, d_llm]
            mol_logits = self.classifier(graph_out)  # [num_pairs, num_classes]
            
            # Store logits with their original indices
            for i, original_idx in enumerate(pair_indices_for_mol):
                logits_dict[original_idx] = mol_logits[i]
        
        # Reconstruct logits in original order
        logits = torch.stack([logits_dict[k] for k in range(K)], dim=0)  # [K, num_classes]
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch["labels"])
        acc = (logits.argmax(-1) == batch["labels"]).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch["labels"])
        preds = logits.argmax(-1)
        gold = batch["labels"]
        acc = (preds == gold).float().mean()

        prec = precision(preds, 
                         gold,
                         num_classes=self.hparams.num_classes,
                         task="multiclass",                    
                         average="macro")       # macro-avg за батч
        rec  = recall(preds, gold,
                    num_classes=self.hparams.num_classes,
                    task="multiclass",                    
                    average="macro")
        f1   = f1_score(preds, gold,
                        num_classes=self.hparams.num_classes,
                        task="multiclass",                    
                        average="macro")
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/precision", prec, prog_bar=True)
        self.log("val/recall", rec, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    

class BondPredictorGRPO(BondPredictorLLM):
    def __init__(self, *args,
                 group_size: int = 8,
                 kl_beta: float = 0.1,
                 ce_alpha: float = 0.3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size
        self.kl_beta   = kl_beta
        self.ce_alpha    = ce_alpha


        # # --- NEW: метрики для train/val --------------------------
        # self.train_precision = MulticlassPrecision(
        #     num_classes=self.hparams.num_classes, average="macro"
        # )
        # self.train_recall    = MulticlassRecall(
        #     num_classes=self.hparams.num_classes, average="macro"
        # )
        # self.train_f1        = MulticlassF1Score(
        #     num_classes=self.hparams.num_classes, average="macro"
        # )
        # # клон без состояния для валидации
        # self.val_precision = self.train_precision.clone()
        # self.val_recall    = self.train_recall.clone()
        # self.val_f1        = self.train_f1.clone()

        # замораживаем копию SFT-политики (π₀)
        self.ref_model = copy.deepcopy(self).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    # ---------- RL-utils ----------
    @staticmethod
    def _sample_actions(logits, g):
        """возвращает actions [K,g] и logπ(a|s) той же формы"""
        log_probs = F.log_softmax(logits, -1)              # [K,C]
        probs = log_probs.exp()
        K, C = probs.shape
        actions = torch.multinomial(probs, g, replacement=True)  # [K,g]
        lp = log_probs.gather(1, actions)                       # [K,g]
        return actions, lp

    @staticmethod
    def _kl_div(p_logits, q_logits):
        """KL(π||π₀) усреднённый по батчу"""
        p_log = F.log_softmax(p_logits, -1)
        q     = F.softmax(q_logits,   -1)
        return F.kl_div(p_log, q, reduction="batchmean")

    # ---------- Lightning ----------
    def training_step(self, batch, batch_idx):
        # текущая политика
        logits = self.forward(batch)                       # [K,C]
        actions, logp = self._sample_actions(logits, self.group_size)

        # вознаграждения 1/0
        gold = batch["labels"]                # [K]
        rewards = (actions == gold.unsqueeze(1)).float()                # [K,g]
        mean_r  = rewards.mean(dim=1, keepdim=True)        # baseline
        adv     = rewards - mean_r                         # A_i

        # policy-gradient часть
        pg_loss = -(adv.detach() * logp).mean()
        
        # CE-добавка
        ce_loss  = F.cross_entropy(logits, batch["labels"])
        
        # KL к SFT-политике
        with torch.no_grad():
            ref_logits = self.ref_model.forward(batch)     # [K,C]
        kl_loss = self._kl_div(logits, ref_logits)

        loss = pg_loss + self.kl_beta * kl_loss + self.ce_alpha * ce_loss
                # --- NEW: обновляем и логируем метрики -------------------
        preds = logits.argmax(-1)
        
        # self.train_precision.update(preds, batch["labels"])
        # self.train_recall.update(preds, batch["labels"])
        # self.train_f1.update(preds, batch["labels"])
        
        prec = precision(preds, gold,
                        num_classes=self.hparams.num_classes,
                        task="multiclass",                    
                        average="macro")       # macro-avg за батч
        rec  = recall(preds, gold,
                    num_classes=self.hparams.num_classes,
                    task="multiclass",                    
                    average="macro")
        f1   = f1_score(preds, gold,
                        num_classes=self.hparams.num_classes,
                        task="multiclass",                    
                        average="macro")
        self.log_dict({
            "train/loss": loss,
            "train/PG": pg_loss,
            "train/KL": kl_loss,
            "train/ce_loss": ce_loss,
            "train/acc": (logits.argmax(-1) == gold.squeeze()).float().mean(),
            "train/precision": prec,
            "train/recall":    rec,
            "train/f1":        f1,
        }, prog_bar=True,
                      on_step=True,
                      on_epoch=True,)
        return loss

