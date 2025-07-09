import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
import sys 
sys.path.insert(1, '/home/user12/spavlenko')
import emb_dataset                     as ds
from llm_bond     import BondPredictorLLM, BondPredictorRPO 
from pytorch_lightning.loggers import WandbLogger            
import wandb 
from pytorch_lightning.callbacks import ModelCheckpoint


# ---- Параметры ----------------------------------------------------
LMDB_DIR   = "/home/user12/burov/unimol_embeddings_project/molecule_embeddings.lmdb"                 # каталог с .mdb / data.mdb
JSONL_PATH = "/home/user12/prompts2/prompts.jsonl"   # твой .jsonl c промптами
LLM_ID     = "laituan245/molt5-base"       
BATCH_SIZE = 4                       # ↑ при LoRA можно 8-16
EPOCHS     = 3
D_LLM      = 768                    # hidden_size вашей модели
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------------------------

# 1) Датасет + train/val разбиение

full_ds   = ds.PairedLMDBJsonlDataset(LMDB_DIR, JSONL_PATH)
train_ds, val_ds = random_split(full_ds, [0.9, 0.1])
print('Loading train data...')
loader_train = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=ds.collate_graph_batch)
print('Loading val data...')
loader_val   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=ds.collate_graph_batch)

# 2) Из выборки считываем размеры эмбеддингов
probe = full_ds[0]
D_ATOM = probe["atom_embeddings"].shape[1]
D_EDGE = probe["pair_embeddings"].shape[2]

# 3) Модель
model = BondPredictorRPO(
    llm_id   = LLM_ID,
    d_atom   = D_ATOM,
    d_edge   = D_EDGE,
    d_llm    = D_LLM,        # должен совпадать с hidden_size!
    num_classes = 5,         # 0..4
    lr = 2e-5,
    concat_order = "graph_first"  # можно "prompt_first"
)

# 4) Wandb-логгер
wandb_logger = WandbLogger(
    entity='lokofanko22-m-v-lomonosovmoscow-state-university', 
    project = "llm grpo",
    name    = "molt5-grpo-run",
    log_model = "all",          # загрузить веса в W&B Artifacts
    save_dir  = "./wandb"       # локальная копия
)
# 5) чек-пойнт коллбэк (лёгкий файл только с весами)
ckpt_cb = ModelCheckpoint(
    dirpath      = "ckpts",
    filename     = "best-{val/acc:.4f}",
    monitor      = "val/acc",
    mode         = "max",
    save_top_k   = 1,
    save_last    = False,
    save_weights_only = True)
# Дополнительно писать градиенты/параметры раз в N шагов
# wandb_logger.watch(model, log="gradients", log_freq=100)

# 4) Тренер
trainer = Trainer(
    devices=[2], 
    callbacks=[ckpt_cb],
    accelerator="gpu" if DEVICE=="cuda" else "cpu",
    precision="bf16-mixed" if DEVICE=="cuda" else 32,
    max_epochs=EPOCHS,
    gradient_clip_val=1.0,
    log_every_n_steps=10,
    logger=wandb_logger
)

trainer.fit(model, loader_train, loader_val)
