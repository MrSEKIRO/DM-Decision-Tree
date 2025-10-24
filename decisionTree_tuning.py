# decisionTree_tuning.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm   # <-- NEW

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ------------------ Config ------------------ #
SPLIT_DIR = Path("prepared_final")
OUT_DIR   = SPLIT_DIR / "tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DEPTH_RANGE = range(1, 31)      # 1..30
GAP_ALPHA = 0.5                     # (B) your choice
ACC_THRESHOLD = 0.93                # (B) your choice
CV_FOLDS = 5                        # include CV + Test (B)

RANDOM_STATE = 42


# ------------------ Helpers ------------------ #
def _load_splits(split_dir: Path):
    train = pd.read_csv(split_dir / "train.csv")
    test  = pd.read_csv(split_dir / "test.csv")
    assert "label" in train.columns and "label" in test.columns
    X_train = train.drop(columns=["label"])
    y_train = train["label"].astype(int)
    X_test  = test.drop(columns=["label"])
    y_test  = test["label"].astype(int)
    return X_train, y_train, X_test, y_test


def _safe_auc(y_true, proba):
    try:
        return roc_auc_score(y_true, proba)
    except Exception:
        return np.nan


def _cv_metric(model, X, y, scoring, skf):
    try:
        scores = cross_val_score(model, X, y, scoring=scoring, cv=skf, n_jobs=None)
        return scores.mean(), scores.std(ddof=1)
    except Exception:
        return np.nan, np.nan


# ------------------ Main routine ------------------ #
def main():
    X_train, y_train, X_test, y_test = _load_splits(SPLIT_DIR)
    feature_names = list(X_train.columns)

    rows = []
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --------- PROGRESS BAR HERE --------- #
    for depth in tqdm(MAX_DEPTH_RANGE, desc="Tuning max_depth", ncols=100):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

        # Train metrics
        ytr_pred = clf.predict(X_train)
        ytr_proba = clf.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, ytr_pred)
        train_f1  = f1_score(y_train, ytr_pred)
        train_bal = balanced_accuracy_score(y_train, ytr_pred)
        train_auc = _safe_auc(y_train, ytr_proba)

        # Test metrics
        yte_pred = clf.predict(X_test)
        yte_proba = clf.predict_proba(X_test)[:, 1]
        test_acc = accuracy_score(y_test, yte_pred)
        test_f1  = f1_score(y_test, yte_pred)
        test_bal = balanced_accuracy_score(y_test, yte_pred)
        test_auc = _safe_auc(y_test, yte_proba)

        gap = max(0.0, train_acc - test_acc)
        gap_penalized = test_acc - GAP_ALPHA * gap

        # CV metrics
        cv_acc_mean, cv_acc_std = _cv_metric(DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE), X_train, y_train, "accuracy", skf)
        cv_f1_mean,  cv_f1_std  = _cv_metric(DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE), X_train, y_train, "f1", skf)
        cv_bal_mean, cv_bal_std = _cv_metric(DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE), X_train, y_train, "balanced_accuracy", skf)
        cv_auc_mean, cv_auc_std = _cv_metric(DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE), X_train, y_train, "roc_auc", skf)

        node_count = clf.tree_.node_count

        rows.append({
            "max_depth": depth,
            # TEST metrics
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_balanced_accuracy": test_bal,
            "test_auc": test_auc,
            # TRAIN metrics
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "train_balanced_accuracy": train_bal,
            "train_auc": train_auc,
            "generalization_gap": gap,
            "gap_penalized": gap_penalized,
            # CV
            "cv_acc_mean": cv_acc_mean, "cv_acc_std": cv_acc_std,
            "cv_f1_mean": cv_f1_mean,   "cv_f1_std": cv_f1_std,
            "cv_bal_mean": cv_bal_mean, "cv_bal_std": cv_bal_std,
            "cv_auc_mean": cv_auc_mean, "cv_auc_std": cv_auc_std,
            # model size
            "node_count": node_count,
        })

    scores_df = pd.DataFrame(rows)
    scores_path = OUT_DIR / "tuning_results.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"[INFO] Saved per-depth metrics -> {scores_path}")

    # ---------- Selection rules ---------- #
    bests = []

    # 1) Baseline: Highest test accuracy
    idx = int(scores_df["test_accuracy"].idxmax())
    bests.append(("Best_Test_Accuracy", int(scores_df.loc[idx, "max_depth"]),
                  float(scores_df.loc[idx, "test_accuracy"])))

    # 2) One-Std rule on CV accuracy: pick smallest depth with CV >= best_mean - best_std
    best_cv_idx = int(scores_df["cv_acc_mean"].idxmax())
    best_cv_mean = float(scores_df.loc[best_cv_idx, "cv_acc_mean"])
    best_cv_std = float(scores_df.loc[best_cv_idx, "cv_acc_std"])
    threshold = best_cv_mean - best_cv_std
    candidate = scores_df[scores_df["cv_acc_mean"] >= threshold]
    chosen_depth_one_std = int(candidate["max_depth"].min())
    chosen_acc_one_std = float(candidate.loc[candidate["max_depth"] == chosen_depth_one_std, "cv_acc_mean"].iloc[0])
    bests.append(("One_Std_Rule", chosen_depth_one_std, chosen_acc_one_std))

    # 3) Gap rule: maximize (test_acc - alpha*(train-test))
    idx = int(scores_df["gap_penalized"].idxmax())
    bests.append(("Gap_Rule_a=0.5", int(scores_df.loc[idx, "max_depth"]),
                  float(scores_df.loc[idx, "gap_penalized"])))

    # 4) Max F1-score (test)
    idx = int(scores_df["test_f1"].idxmax())
    bests.append(("Max_Test_F1", int(scores_df.loc[idx, "max_depth"]),
                  float(scores_df.loc[idx, "test_f1"])))

    # 5) Balanced Accuracy (test)
    idx = int(scores_df["test_balanced_accuracy"].idxmax())
    bests.append(("Max_Test_BalancedAcc", int(scores_df.loc[idx, "max_depth"]),
                  float(scores_df.loc[idx, "test_balanced_accuracy"])))

    # 6) AUC-ROC (test)
    idx = int(scores_df["test_auc"].idxmax())
    bests.append(("Max_Test_AUC", int(scores_df.loc[idx, "max_depth"]),
                  float(scores_df.loc[idx, "test_auc"])))

    # 7) Minimum tree size with acceptable performance (smallest depth s.t. test_acc >= threshold)
    ok = scores_df[scores_df["test_accuracy"] >= ACC_THRESHOLD]
    if not ok.empty:
        idx = int(ok["max_depth"].idxmin())
        bests.append(("Min_Size_Acc>=%.2f" % ACC_THRESHOLD,
                      int(scores_df.loc[idx, "max_depth"]),
                      float(scores_df.loc[idx, "test_accuracy"])))
    else:
        bests.append(("Min_Size_Acc>=%.2f" % ACC_THRESHOLD, None, None))

    bests_df = pd.DataFrame(bests, columns=["criterion", "chosen_max_depth", "score"])
    bests_path = OUT_DIR / "best_depths_comparison.csv"
    bests_df.to_csv(bests_path, index=False)
    print(f"[INFO] Saved best depth per criterion -> {bests_path}")

    # Also dump JSON for convenience
    (OUT_DIR / "best_depths_comparison.json").write_text(json.dumps(bests_df.to_dict(orient="records"), indent=2))

    # ---------- Plots ---------- #
    # A) Curves of metrics vs depth
    plt.figure(figsize=(11, 7))
    plt.plot(scores_df["max_depth"], scores_df["test_accuracy"], marker="o", label="Test Accuracy")
    plt.plot(scores_df["max_depth"], scores_df["cv_acc_mean"], linestyle="--", marker="s", label="CV Acc (mean)")
    plt.plot(scores_df["max_depth"], scores_df["test_f1"], marker="o", label="Test F1")
    plt.plot(scores_df["max_depth"], scores_df["test_balanced_accuracy"], marker="o", label="Test Balanced Acc")
    plt.plot(scores_df["max_depth"], scores_df["test_auc"], marker="o", label="Test AUC")
    plt.xlabel("max_depth")
    plt.ylabel("Score")
    plt.title("Decision Tree metrics vs max_depth")
    plt.grid(True)
    plt.legend()
    plot1 = OUT_DIR / "metric_curves.png"
    plt.savefig(plot1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved curves -> {plot1}")

    # B) Bar chart: chosen depths by criterion
    plt.figure(figsize=(10, 6))
    y = np.arange(len(bests_df))
    plt.barh(y, bests_df["chosen_max_depth"].fillna(-1).astype(int))
    plt.yticks(y, bests_df["criterion"])
    plt.xlabel("Chosen max_depth")
    plt.title("Chosen depth per selection criterion")
    plt.grid(axis="x", linestyle=":", linewidth=0.5)
    plot2 = OUT_DIR / "best_depths_comparison.png"
    plt.savefig(plot2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved comparison chart -> {plot2}")

    print("\n[SUMMARY] Best depths by criterion:")
    print(bests_df.to_string(index=False))


if __name__ == "__main__":
    main()