# xyz2smiles

**xyz2smiles** is a research project focused on reconstructing molecular SMILES strings directly from 3D atomic coordinates. The main challenge is predicting the bond types between atoms using geometric information alone, without relying on heuristic rules or templates.

This repository explores multiple approaches — from classical machine learning to large language models — to achieve end-to-end SMILES reconstruction from XYZ input.

---

## 🗂️ Repository Structure


├── baselines/ # Reference implementations (e.g. Yuel-Bond) \
├── data/ # Dataset storage and preprocessing tools \
├── llm_bond_predictor/ # Our core pipeline using MolT5/GRPO + Uni-Mol embeddings \
├── notebooks/ # Interactive demos and model evaluation (Jupyter) \
├── scripts  # Training/inference utilities for all models \


---

## Methods


### 🔹 Baselines

- **RDKit (heuristic)**

    - Traditional rule-based method for bond perception based on interatomic distances and valence checks.

    - We used RDKit’s rdDetermineBonds() as a baseline to extract bond graphs from 3D coordinates.

    - Evaluated primarily on the QM9 dataset, where it performs well due to regular bonding patterns.


- **Yuel-Bond**: Graph Neural Network baseline for bond prediction.
  - Original repo: [https://bitbucket.org/dokhlab/yuel_bond](https://bitbucket.org/dokhlab/yuel_bond)
  - Used for comparison only; not actively modified in this repo.
  - To use as baseline:
    ```bash
    git clone https://bitbucket.org/dokhlab/yuel_bond baselines/yuel_bond
    ```


### 🔹 Classical
- **Random Forest** classifier trained on pairwise Uni-Mol embeddings.

```bash
python scripts/rf.py
```

- **XGBoost** on pairwise Uni-Mol embeddings.

```bash
python scripts/xgboost.py
```

### 🔹 LLM-based
- **MolT5**: Large language model that processes concatenated pair embeddings vectors and encoded prompt.
- **MolT5 + GRPO**: Finetuned version using gradient-based rewiring for enhanced accuracy.

Located under `llm_bond_predictor/`.

---

## Results

Bond type classification was evaluated on the [GEOM dataset](https://github.com/learningmatter-mit/geom). We report both:
- **Per-bond metrics** (F1, precision, recall)
- **Exact match** for whole-molecule reconstruction

See our `notebooks/` for visualization and summary tables.

---

## Training & Inference

You can train or evaluate models via:

```bash
# Train MolT5 on bond prediction
python llm_bond_predictor/train_bond.py 

python llm_bond_predictor/run_test.py
```


## Installation

We recommend using a fresh Python 3.10+ environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt