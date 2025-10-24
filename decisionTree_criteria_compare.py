# decisionTree_criteria_compare.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ------------------ Config ------------------ #
SPLIT_DIR = Path("prepared_final")
OUT_DIR   = SPLIT_DIR / "criteria_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CRITERIA = ["gini", "entropy"]
DEPTH_RANGE = range(1, 31)  # 1..30
CV_FOLDS = 5
RANDOM_STATE = 42

# ------------------ Data loading ------------------ #
def load_splits(split_dir: Path):
    train = pd.read_csv(split_dir / "train.csv")
    test  = pd.read_csv(split_dir / "test.csv")
    assert "label" in train.columns and "label" in test.columns
    X_train = train.drop(columns=["label"])
    y_train = train["label"].astype(int)
    X_test  = test.drop(columns=["label"])
    y_test  = test["label"].astype(int)
    return X_train, y_train, X_test, y_test

# ------------------ Utilities ------------------ #
def safe_auc(y_true, proba):
    try:
        return roc_auc_score(y_true, proba)
    except Exception:
        return np.nan

def cv_mean_std(model, X, y, scoring, cv):
    try:
        scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=None)
        return float(scores.mean()), float(scores.std(ddof=1))
    except Exception:
        return np.nan, np.nan

def tune_depth_one_std(X_train, y_train, criterion: str, depth_range, cv_folds=5):
    """Return chosen depth using One-Std rule on CV accuracy + full per-depth table."""
    rows = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    for d in tqdm(depth_range, desc=f"Tuning ({criterion})", ncols=100):
        base = DecisionTreeClassifier(max_depth=d, criterion=criterion, random_state=RANDOM_STATE)
        cv_mean, cv_std = cv_mean_std(base, X_train, y_train, "accuracy", skf)
        rows.append({"max_depth": d, "cv_acc_mean": cv_mean, "cv_acc_std": cv_std})

    df = pd.DataFrame(rows)
    # One-Std rule
    best_idx = int(df["cv_acc_mean"].idxmax())
    best_mean = float(df.loc[best_idx, "cv_acc_mean"])
    best_std  = float(df.loc[best_idx, "cv_acc_std"])
    threshold = best_mean - best_std
    candidate = df[df["cv_acc_mean"] >= threshold]
    chosen_depth = int(candidate["max_depth"].min())
    return chosen_depth, df, threshold, best_mean, best_std

def train_eval_tree(X_train, y_train, X_test, y_test, criterion: str, depth: int):
    """Train final model and evaluate on train/test. Return metrics dict and model."""
    clf = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # Train metrics
    ytr_pred = clf.predict(X_train)
    ytr_prob = clf.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, ytr_pred)
    train_f1  = f1_score(y_train, ytr_pred)
    train_bal = balanced_accuracy_score(y_train, ytr_pred)
    train_auc = safe_auc(y_train, ytr_prob)

    # Test metrics
    yte_pred = clf.predict(X_test)
    yte_prob = clf.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, yte_pred)
    test_f1  = f1_score(y_test, yte_pred)
    test_bal = balanced_accuracy_score(y_test, yte_pred)
    test_auc = safe_auc(y_test, yte_prob)
    cm = confusion_matrix(y_test, yte_pred)

    info = {
        "criterion": criterion,
        "chosen_max_depth": depth,
        "node_count": int(clf.tree_.node_count),
        "tree_depth": int(clf.tree_.max_depth),
        # train
        "train_accuracy": float(train_acc),
        "train_f1": float(train_f1),
        "train_balanced_accuracy": float(train_bal),
        "train_auc": float(train_auc) if not pd.isna(train_auc) else np.nan,
        # test
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "test_balanced_accuracy": float(test_bal),
        "test_auc": float(test_auc) if not pd.isna(test_auc) else np.nan,
        "confusion_matrix": cm.tolist(),
    }
    return clf, info, yte_prob

def plot_cv_curves(curves_dict, out_path: Path, title="CV Accuracy vs max_depth (Gini vs Entropy)"):
    plt.figure(figsize=(10, 6))
    for name, df in curves_dict.items():
        plt.plot(df["max_depth"], df["cv_acc_mean"], marker='o', label=f"{name} (CV mean)")
        # shade mean-std band
        plt.fill_between(
            df["max_depth"],
            df["cv_acc_mean"] - df["cv_acc_std"],
            df["cv_acc_mean"] + df["cv_acc_std"],
            alpha=0.15
        )
    plt.xlabel("max_depth"); plt.ylabel("CV Accuracy")
    plt.title(title); plt.grid(True); plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_test_accuracy_curves(test_acc_tables, out_path: Path):
    plt.figure(figsize=(10, 6))
    for name, df in test_acc_tables.items():
        plt.plot(df["max_depth"], df["test_accuracy"], marker='o', label=f"{name} (Test acc)")
    plt.xlabel("max_depth"); plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs max_depth (Gini vs Entropy)")
    plt.grid(True); plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def build_test_accuracy_table(X_train, y_train, X_test, y_test, criterion, depth_range):
    rows = []
    for d in depth_range:
        clf = DecisionTreeClassifier(max_depth=d, criterion=criterion, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        yte_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, yte_pred)
        rows.append({"max_depth": d, "test_accuracy": float(acc)})
    return pd.DataFrame(rows)

def plot_roc_curves(y_test, probas_dict, out_path: Path):
    plt.figure(figsize=(10,6))
    for name, prob in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves (Gini vs Entropy)")
    plt.grid(True); plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_top_features(model, feature_names, out_path: Path, top_n=15, title_prefix=""):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    names = np.array(feature_names)[idx]
    vals  = importances[idx]

    plt.figure(figsize=(10, max(5, top_n*0.35)))
    plt.barh(range(len(vals)), vals[::-1])
    plt.yticks(range(len(vals)), names[::-1])
    plt.xlabel("Importance")
    plt.title(f"{title_prefix} Top-{top_n} Feature Importances")
    plt.grid(axis='x', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------ Main ------------------ #
def main():
    X_train, y_train, X_test, y_test = load_splits(SPLIT_DIR)
    feature_names = list(X_train.columns)

    summaries = []
    curves = {}
    test_curve_tables = {}
    final_models = {}
    final_probs = {}

    # For each criterion: tune depth (one-std), train final, evaluate
    for crit in CRITERIA:
        chosen_depth, cv_table, threshold, best_mean, best_std = tune_depth_one_std(
            X_train, y_train, criterion=crit, depth_range=DEPTH_RANGE, cv_folds=CV_FOLDS
        )
        # Save CV table
        cv_path = OUT_DIR / f"cv_table_{crit}.csv"
        cv_table.to_csv(cv_path, index=False)

        # Build test accuracy curve (optional but useful plot)
        test_acc_tbl = build_test_accuracy_table(X_train, y_train, X_test, y_test, crit, DEPTH_RANGE)
        test_acc_tbl.to_csv(OUT_DIR / f"test_accuracy_per_depth_{crit}.csv", index=False)

        curves[crit] = cv_table
        test_curve_tables[crit] = test_acc_tbl

        # Train & evaluate final
        model, info, yte_prob = train_eval_tree(X_train, y_train, X_test, y_test, crit, chosen_depth)
        info.update({
            "cv_best_mean": best_mean,
            "cv_best_std": best_std,
            "cv_threshold_one_std": threshold
        })
        summaries.append(info)
        final_models[crit] = model
        final_probs[crit] = yte_prob

        # Save top features
        plot_top_features(model, feature_names, OUT_DIR / f"top_features_{crit}.png",
                          top_n=15, title_prefix=f"{crit.capitalize()}")

    # Save comparison summary
    summary_df = pd.DataFrame(summaries)
    summary_path_csv = OUT_DIR / "criteria_summary.csv"
    summary_path_json = OUT_DIR / "criteria_summary.json"
    summary_df.to_csv(summary_path_csv, index=False)
    summary_path_json.write_text(json.dumps(summary_df.to_dict(orient="records"), indent=2))
    print(f"[INFO] Saved criteria summary -> {summary_path_csv}")

    # Plots: CV curves, Test accuracy curves, ROC curves
    plot_cv_curves(curves, OUT_DIR / "cv_curves.png")
    plot_test_accuracy_curves(test_curve_tables, OUT_DIR / "test_accuracy_curves.png")
    plot_roc_curves(y_test, final_probs, OUT_DIR / "roc_curves.png")

    # Quick console summary
    print("\n=== Final Comparison (One-Std tuned) ===")
    print(summary_df[[
        "criterion","chosen_max_depth","tree_depth","node_count",
        "test_accuracy","test_f1","test_balanced_accuracy","test_auc"
    ]].to_string(index=False))

if __name__ == "__main__":
    main()
