# 🔍 Fraud Detection Pipeline
### DecodeLabs | Project 2

---

## 📌 Project Overview
A supervised machine learning pipeline to detect fraudulent credit card transactions in a highly imbalanced dataset. The pipeline uses **SMOTE** for class balancing and evaluates models using **Precision, Recall, and ROC-AUC** — completely discarding misleading Accuracy metrics.

---

## 📊 Dataset
- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807
- **Legitimate:** 284,315 (99.83%)
- **Fraudulent:** 492 (0.17%)
- **Features:** 30 (V1–V28 PCA components + Time + Amount)

> ⚠️ Dataset not included in repo (150MB+). Download from Kaggle and place `creditcard.csv` in the project folder.

---

## 🛠️ Tech Stack
- Python 3.x
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

---

## ⚙️ Pipeline Steps

```
1. Load Dataset
2. EDA — Class Distribution
3. Stratified Train/Test Split (80/20)
4. SMOTE — Applied only on training data (no leakage)
5. StandardScaler — For Logistic Regression
6. Train Models — Logistic Regression + Random Forest
7. Evaluate — Precision, Recall, F1, ROC-AUC
8. Visualize — Confusion Matrix, ROC Curve, Feature Importance
```

---

## 📈 Results

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.1234 | 0.8878 | 0.2167 | 0.9778 |
| **Random Forest** | **0.4195** | **0.8776** | **0.5677** | **0.9820** ⭐ |

**Winner: Random Forest** — Higher Precision, F1, and ROC-AUC

---

## 📁 Output Files

| File | Description |
|---|---|
| `class_distribution.png` | Fraud vs Legitimate bar chart |
| `confusion_matrices.png` | Both models side by side |
| `roc_curves.png` | ROC-AUC comparison |
| `feature_importance.png` | Top 15 features (Random Forest) |
| `model_comparison.csv` | Final metrics table |

---

## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

**2. Place dataset in project folder:**
```
Project_2/
├── creditcard.csv
└── fraud_detection_pipeline.py
```

**3. Run:**
```bash
python fraud_detection_pipeline.py
```

---

## 🧠 Key Concepts Used

- **SMOTE** — Synthetic Minority Over-sampling to handle class imbalance
- **Stratified Split** — Preserves fraud ratio in train/test sets
- **No Accuracy** — Used Precision, Recall, F1, ROC-AUC instead
- **Data Leakage Prevention** — SMOTE & Scaler applied only on training data

---

## 👨‍💻 Author
**Tayyaba Nizamani** | DecodeLabs Intern | Batch 2026
