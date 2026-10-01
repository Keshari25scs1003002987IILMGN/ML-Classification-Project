# Task 1: ML Classification Project

**Alfido Tech Internship Program - Artificial Intelligence Track**

Binary classification of breast tumours (malignant vs benign) on the Breast Cancer Wisconsin Diagnostic dataset (569 samples, 30 numeric features), which ships with scikit-learn, so no download is needed.

## Results (held-out test set, 114 samples)

Selected model: **SVM (RBF kernel, C=10)**, chosen by best cross-validated F1 on the training data.

| Metric | Score |
|---|---|
| Accuracy | 0.974 |
| Precision | 1.000 |
| Recall | 0.929 |
| F1 | 0.963 |
| ROC-AUC | 0.993 |

Confusion matrix: 72 TN, 0 FP, 3 FN, 39 TP. Positive class = malignant.

Four algorithms were compared (Logistic Regression, Random Forest, SVM, Gradient Boosting) with 5-fold stratified CV and GridSearchCV. Their scores are within about 1 point of each other, so the ranking is not statistically decisive with only 114 test samples. Details are in `report/Task1_Report.pdf`.

## Repository layout

```
.
├── README.md
├── requirements.txt            # exact package versions used
├── requirements-report.txt     # only needed to rebuild the PDF
├── notebooks/
│   └── task1_ml_classification.ipynb   # main deliverable (code, plots, metrics)
├── src/
│   └── train_classification.py         # same pipeline as a plain script
├── data/breast_cancer.csv              # copy of the dataset
├── images/                             # all plots (PNG)
├── results/                            # CV, tuning, test metrics (CSV/JSON)
├── models/best_model.joblib            # saved final model
└── report/
    ├── Task1_Report.pdf                # submission document
    └── build_report.py                 # regenerates the PDF
```

## Environment setup

Tested with Python 3.12. Python 3.10+ should work.

**Windows (PowerShell / CMD)**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Option A: notebook
```bash
jupyter notebook notebooks/task1_ml_classification.ipynb
```
Then choose **Kernel > Restart & Run All**.

Option B: script (from the repository root)
```bash
python src/train_classification.py
```
Both write plots to `images/`, metrics to `results/` and the model to `models/best_model.joblib`. A fixed random seed (42) makes results reproducible.

Option C: Google Colab. Upload the notebook, then run the first cell after adding `!pip install -r requirements.txt` if needed. The notebook uses only standard libraries.

## Run inference with the saved model

```python
import joblib, pandas as pd
model = joblib.load("models/best_model.joblib")
X = pd.read_csv("data/breast_cancer.csv").drop(columns="target")
print(model.predict_proba(X.iloc[[0]])[0, 1])   # probability of malignant
```

## Rebuild the PDF report (optional)

```bash
pip install -r requirements-report.txt
python report/build_report.py https://github.com/<your-username>/<your-repo>
```
The URL argument is printed on the title page of the PDF.

## Notes and limitations

* Scaling is inside each pipeline, so it is fitted on training folds only (no data leakage).
* The test set is used once, for the final evaluation. Model selection uses cross-validation only.
* The dataset is small and clean. This is a learning project, not a clinical tool.
