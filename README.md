# molecular_dissolution_prediction

This repository contains the source code, trained models, and example data for the study:

**"Integrating artificial intelligence and physiologically based pharmacokinetic modelling to predict in vitro and in vivo fate of amorphous solid dispersions."**
 
📄 Publication: *[To be added upon acceptance]*

---

## 🌟 Highlights

- Predicts **molecular dissolution profiles** of ASD formulations using machine learning
- Integrates with **PBPK models** to simulate in vivo systemic absorption
- Uses **TabPFN**, a transformer-based foundation model for tabular data
- Includes model interpretability using **SHAP analysis**

---

## 📦 Contents

| Folder/File       | Description                                         |
|-------------------|-----------------------------------------------------|
| `models/`         | Pre-trained `.pkl` models (e.g., TabPFN, RF)        |
| `data/`           | Sample input CSV for prediction                     |
| `src/`            | Scripts for preprocessing, prediction, SHAP         |
| `requirements.txt`| Python dependencies                                 |

---

## 🛠️ Installation

We recommend using a virtual environment.

```bash
git clone https://github.com/xappiny/molecular_dissolution_prediction.git
cd molecular_dissolution_prediction
pip install -r requirements.txt
