"""Build report/Task1_Report.pdf from the saved results and plots.
Usage (from repo root):  python report/build_report.py [GITHUB_REPO_URL]
"""
import sys, json
from pathlib import Path
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, KeepTogether, Preformatted)

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/<your-username>/alfido-task1-ml-classification"
OUT = ROOT / "report" / "Task1_Report.pdf"

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=14.5, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#8B1A1A"), spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor("#222222"), spaceBefore=8, spaceAfter=4)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"), alignment=1)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
CODE = ParagraphStyle("code", parent=ss["Code"], fontSize=8.5, leading=11, backColor=colors.HexColor("#F3F3F3"),
                      borderPadding=5, spaceAfter=8)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)

def P(t, s=BODY): return Paragraph(t, s)
def bullets(items): return [Paragraph(i, BUL, bulletText="\u2022") for i in items]

def img(name, width_cm=16, caption=None):
    path = ROOT / "images" / name
    w, h = PILImage.open(path).size
    width = width_cm * cm
    parts = [Image(str(path), width=width, height=width * h / w)]
    if caption: parts += [Spacer(1, 3), P(caption, CAP)]
    return KeepTogether(parts + [Spacer(1, 8)])

def table(df, col_widths=None, header_bg="#8B1A1A", highlight_row=None):
    HS = ParagraphStyle("hs", parent=CELL, textColor=colors.white)
    data = [[P(f"<b>{c}</b>", HS) for c in [df.index.name or ""] + list(df.columns)]]
    for idx, row in df.iterrows():
        data.append([P(str(idx), CELL)] + [P(str(v), CELL) for v in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F2F2")]),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    t.setStyle(TableStyle(st))
    return t

# header text color fix for white-on-red header
def hdr(df, **kw):
    t = table(df, **kw)
    return t

cv = pd.read_csv(ROOT / "results" / "cv_comparison.csv", index_col=0)
cv.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]; cv.index.name = "Model"
tune = pd.read_csv(ROOT / "results" / "tuning_results.csv", index_col=0)
test = pd.read_csv(ROOT / "results" / "test_metrics.csv", index_col=0); test.index.name = "Model"
final = json.load(open(ROOT / "results" / "final_metrics.json"))
m, cm_ = final["metrics"], final["confusion_matrix"]

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.2 * cm, "Alfido Tech Internship - AI Track - Task 1: ML Classification Project")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
                        title="Task 1 - ML Classification Project", author="Alfido Tech Intern")
s = []

# ---- Title block
s += [Spacer(1, 1.2*cm),
      P("Task 1: ML Classification Project", ParagraphStyle("t", parent=ss["Title"], fontSize=24, textColor=colors.HexColor("#8B1A1A"))),
      P("Alfido Tech Internship Program - Artificial Intelligence Track",
        ParagraphStyle("st", parent=BODY, alignment=1, fontSize=12)),
      Spacer(1, 0.6*cm)]

info = [["Intern name", "[Your name]"], ["Task", "Task 1 - ML Classification Project"],
        ["Dataset", "Breast Cancer Wisconsin Diagnostic (scikit-learn built-in, 569 samples, 30 features)"],
        ["Final model", f"{final['model']} (RBF kernel)"],
        ["GitHub repository", REPO_URL],
        ["Main notebook", REPO_URL + "/blob/main/notebooks/task1_ml_classification.ipynb"],
        ["Environment", "Python 3.12, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, seed = 42"]]
t = Table([[P(f"<b>{a}</b>", CELL), P(b, CELL)] for a, b in info], colWidths=[3.8*cm, 13.2*cm])
t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#BBBBBB")),
                       ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F7F2F2")),
                       ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
s += [t, Spacer(1, 0.5*cm)]

s += [P("Summary", H1),
      P(f"A supervised classifier was built to predict whether a breast tumour is malignant or benign. Four algorithms "
        f"(Logistic Regression, Random Forest, SVM and Gradient Boosting) were compared with 5-fold stratified cross-validation "
        f"and tuned with grid search. The final model, <b>{final['model']}</b>, reached <b>accuracy {m['Accuracy']:.3f}, precision {m['Precision']:.3f}, "
        f"recall {m['Recall']:.3f}, F1 {m['F1']:.3f} and ROC-AUC {m['ROC-AUC']:.3f}</b> on a held-out test set of 114 samples "
        f"({cm_['TN']} TN, {cm_['FP']} FP, {cm_['FN']} FN, {cm_['TP']} TP; positive class = malignant)."),
      P("All four models scored within about one percentage point of each other, so the choice between the top models is not "
        "statistically decisive with a test set this small.")]

# ---- 1 Problem
s += [P("1. Problem and dataset", H1),
      P("<b>Goal:</b> build and evaluate a supervised classification model. <b>Task:</b> predict tumour malignancy from 30 numeric "
        "features (radius, texture, perimeter, area, smoothness, concavity and so on) computed from digitised images of a fine-needle aspirate. "
        "Each feature appears as a mean, a standard error and a worst value."),
      P("The dataset has 569 samples: 357 benign (62.7%) and 212 malignant (37.3%). The imbalance is mild, so stratified splitting is used and "
        "F1 and recall are reported in addition to accuracy. The positive class is defined as <b>malignant</b>, so recall means the share of real cancers that were caught.")]
s += [img("01_class_and_features.png", 15.5, "Figure 1. Class distribution and selected features by class (log scale).")]

# ---- 2 Preprocessing
s += [P("2. Data preprocessing", H1)] + bullets([
    "<b>Quality checks:</b> 0 missing values, 0 duplicate rows, all features numeric. Cleaning code (drop duplicates / missing) is included so the pipeline works on messier data.",
    "<b>Target encoding:</b> scikit-learn codes malignant as 0; it was flipped so malignant = 1 (positive class).",
    "<b>Scaling:</b> StandardScaler is placed <i>inside</i> the pipelines of Logistic Regression and SVM, so it is fitted on training folds only. This prevents data leakage. Tree models need no scaling.",
    "<b>Split:</b> 80/20 stratified train/test split (455 train, 114 test), random_state = 42. The test set is used once at the end."])
s += [img("02_histograms.png", 17, "Figure 2. Feature distributions by class. Malignant tumours tend to have larger radius and area and more concave points."),
      img("03_correlation_heatmap.png", 12.5, "Figure 3. Correlation heatmap of the mean features and the target. Radius, perimeter and area are almost perfectly correlated.")]

# ---- 3 Method
s += [P("3. Methodology", H1),
      P("<b>Cross-validation.</b> Each algorithm was evaluated with 5-fold stratified cross-validation on the training set using accuracy, precision, recall, F1 and ROC-AUC."),
      P("<b>Algorithms compared.</b> Logistic Regression (linear baseline, interpretable), Random Forest (bagged trees), SVM with RBF kernel (non-linear margin classifier) and Gradient Boosting (boosted trees)."),
      P("<b>Tuning.</b> GridSearchCV with 5-fold CV, optimising F1, on the training set only. Grids: Logistic Regression C; Random Forest n_estimators, max_depth, min_samples_leaf; SVM C and gamma; Gradient Boosting n_estimators, learning_rate, max_depth."),
      P("<b>Selection rule.</b> The final model is the one with the best cross-validated F1 on the training data, so the test set remains an unbiased estimate.")]

# ---- 4 Results
s += [P("4. Results", H1), P("4.1 Cross-validation (training set, before tuning), mean +/- std over 5 folds", H2),
      table(cv, col_widths=[3.4*cm] + [2.7*cm]*5), Spacer(1, 8),
      img("04_cv_comparison.png", 16, "Figure 4. Distribution of per-fold F1 and ROC-AUC for the four algorithms.")]

tdf = tune.copy(); tdf.index.name = "Model"
s += [P("4.2 Hyper-parameter tuning", H2), table(tdf, col_widths=[3.4*cm, 2.6*cm, 11*cm]), Spacer(1, 8)]

s += [P("4.3 Held-out test set (114 samples, tuned models)", H2), table(test.map(lambda v: f"{v:.3f}"), col_widths=[3.4*cm] + [2.7*cm]*5), Spacer(1, 8),
      img("05_confusion_matrices.png", 17, "Figure 5. Test-set confusion matrices (positive class = malignant)."),
      img("06_roc_curves.png", 10.5, "Figure 6. ROC curves on the test set. All models have AUC above 0.99.")]

# ---- 5 Model selection
s += [P("5. Model selection and discussion", H1),
      P(f"<b>Selected: {final['model']}</b> (C = 10, gamma = scale). It had the best cross-validated F1 after tuning (0.967) and on the test set achieved:"),
      table(pd.DataFrame({"Score": {k: f"{v:.3f}" for k, v in m.items()}}), col_widths=[4*cm, 3*cm]), Spacer(1, 8)]
s += bullets([
    "<b>Precision 1.000:</b> every tumour flagged as malignant was truly malignant (0 false positives).",
    "<b>Recall 0.929:</b> 3 of 42 malignant tumours were missed. In a medical setting a missed cancer is the costlier error, so recall is the metric to improve, for example by lowering the decision threshold at the cost of some false positives.",
    "<b>Differences between models are small.</b> Cross-validated F1 spans only about 0.95 to 0.97 after tuning, and with 114 test samples a single extra mistake changes accuracy by 0.9 points. Logistic Regression is almost as good and far easier to explain, so it is a reasonable choice if interpretability matters more than the last fraction of a point.",
    "<b>No sign of overfitting:</b> cross-validation scores (F1 around 0.96) and test scores (F1 0.963) agree closely."])
s += [img("07_feature_importance.png", 13, "Figure 7. Permutation importance of the SVM on the test set. Texture and concavity features matter most. "
          "Because many features are strongly correlated, importance is shared among them and the error bars are wide.")]

# ---- 6 Limitations
s += [P("6. Limitations and next steps", H1)] + bullets([
    "Small, clean research dataset (569 samples); the results will not transfer directly to real clinical data.",
    "The 114-sample test set gives wide confidence intervals; repeated cross-validation or a larger dataset would give a firmer ranking.",
    "Next steps: threshold tuning to raise recall, probability calibration, and SHAP explanations for individual predictions."])

# ---- 7 Reproduce
s += [P("7. How to reproduce (exact commands)", H1),
      P("Repository: " + REPO_URL), P("<b>Setup (Windows)</b>", H2),
      Preformatted("python -m venv .venv\n.venv\\Scripts\\activate\npip install -r requirements.txt", CODE),
      P("<b>Setup (macOS / Linux)</b>", H2),
      Preformatted("python3 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt", CODE),
      P("<b>Run</b>", H2),
      Preformatted("jupyter notebook notebooks/task1_ml_classification.ipynb   # then Kernel > Restart & Run All\n"
                   "# or, headless, from the repo root:\npython src/train_classification.py", CODE),
      P("Outputs: plots in <font face='Courier'>images/</font>, metrics in <font face='Courier'>results/</font>, saved model in "
        "<font face='Courier'>models/best_model.joblib</font>. Deliverables: (1) notebook with code, plots and metrics, (2) this short report on model selection and results."),
      P("<b>Deliverable checklist</b>", H2)] + bullets([
    "Data preprocessing, train/test split, cross-validation: done (sections 2 and 3)",
    "At least two algorithms compared: done (four algorithms)",
    "Accuracy, precision, recall, F1, ROC-AUC reported: done (section 4)",
    "Notebook with code, plots and metrics: <font face='Courier'>notebooks/task1_ml_classification.ipynb</font>",
    "README with environment setup and run commands: <font face='Courier'>README.md</font>"])

doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
