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

| Model         | Total Accuracy | Exact Match |
|---------------|----------------|-------------|
| Yuel-Bond     | 0.978          | 0.934       |
| Random Forest | 0.983          |  -       |
| MolT5         | 0.999          | 0.978       |
| MolT5+GRPO    | 0.998          | 0.978       |


See notebooks/ for more info



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
```

In order to download QM9 dataset for RDKit baseline:

```bash

wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
```

Geom dataset download:

```bash

wget https://zenodo.org/records/15353365/files/geom_train.pt -O datasets/geom_train.pt

wget https://zenodo.org/records/15353365/files/geom_train.pt -O datasets/geom_test.pt
```

## Citation

If you use this project in your research or build upon it, please cite the relevant foundational works:
```
@article{zhou2023unimol,
  title={Uni-Mol: A Universal 3D Molecular Representation Learning Framework},
  author={Zhou, Kun and Gong, Ming and Shen, Yang and others},
  journal={arXiv preprint arXiv:2210.01776},
  year={2023}
}

@article{raffel2020exploring,
  title={Exploring the limits of transfer learning with a unified text-to-text transformer},
  author={Raffel, Colin and others},
  journal={JMLR},
  year={2020}
}

@article{wang2025multimodal,
  title={Multimodal Bond Perception for 3D Molecular Graph Generation},
  author={Wang, et al.},
  journal={To appear, NeurIPS 2025},
  year={2025}
}
```

If you cite this project directly, use:

```
@misc{xyz2smiles2025,
  author       = {Pavel Maslov, Stepan Pavlenko, Ivan Burov},
  title        = {xyz2smiles: End-to-end recovery of SMILES strings from 3D atomic coordinates},
  year         = 2025,
  note         = {AIRI Summer School Project},
  url          = {https://github.com/Masvelll/xyz2smiles}
}

**Project Authors:** [Pavel Maslov](https://github.com/Masvelll) | [Stepan Pavlenko](https://github.com/PavlenkoSS) | [Ivan Burov](https://github.com/lokofanko)
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.