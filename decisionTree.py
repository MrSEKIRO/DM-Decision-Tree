import math
import re
import json
from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from data_loader import read_csv_robust, EXPECTED_COLUMNS

# Step 1: Balance Dataset
def balance_dataset(df: pd.DataFrame, target_col="label", method="undersample") -> pd.DataFrame:
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").round().astype("Int64")
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    counts = df[target_col].value_counts()
    if len(counts) != 2:
        raise ValueError(f"Expected binary classes; got {counts.to_dict()}")

    cls_min, cls_max = counts.idxmin(), counts.idxmax()
    n_min, n_max = counts.min(), counts.max()

    if method == "undersample":
        df_min = df[df[target_col] == cls_min]
        df_max = df[df[target_col] == cls_max].sample(n=n_min, random_state=42)
        df_bal = pd.concat([df_min, df_max], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    elif method == "oversample":
        df_min = df[df[target_col] == cls_min].sample(n=n_max, replace=True, random_state=42)
        df_max = df[df[target_col] == cls_max]
        df_bal = pd.concat([df_min, df_max], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        raise ValueError("method must be 'undersample' or 'oversample'")
    return df_bal

# =========================================
# Step 4: Feature Engineering from `domain`
# =========================================
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_SHORTENERS = {"bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "adf.ly"}

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    cnt = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in cnt.values())

def _safe_urlsplit(u: str):
    if not isinstance(u, str):
        u = ""
    us = u.strip()
    if not us:
        return urlsplit("http://")
    if not us.startswith(("http://", "https://")):
        us = "http://" + us
    try:
        return urlsplit(us)
    except Exception:
        return urlsplit("http://")

def _extract_host(host: str) -> str:
    return (host or "").strip().strip("[]")

def _tld_of_host(host: str) -> str:
    parts = host.split(".")
    return parts[-1].lower() if len(parts) >= 2 else ""

def _is_ip(host: str) -> bool:
    return bool(_IPV4_RE.match(host))

def add_domain_features(df: pd.DataFrame, domain_col: str = "domain") -> pd.DataFrame:
    if domain_col not in df.columns:
        return df.copy()

    df = df.copy()
    tlds_to_onehot = ["com", "net", "org", "ir", "ru", "info"]
    keywords = ["paypal", "login", "secure", "verify", "update", "bank"]

    def fe(url_text: str):
        raw = url_text if isinstance(url_text, str) else ""
        raw_l = raw.strip().lower()
        parts = _safe_urlsplit(raw)
        host = _extract_host(parts.hostname or "")
        path = parts.path or ""
        query = parts.query or ""

        url_length = len(raw)
        host_length = len(host)
        path_length = len(path)
        query_length = len(query)

        num_slashes = raw.count("/")
        num_dots = raw.count(".")
        num_digits = sum(c.isdigit() for c in raw)
        digit_ratio = (num_digits / url_length) if url_length > 0 else 0.0

        has_https = 1 if raw.startswith("https://") else 0
        has_http  = 1 if raw.startswith("http://") else 0
        has_at = 1 if "@" in raw else 0
        is_ip = 1 if _is_ip(host) else 0
        startswith_www = 1 if host.lower().startswith("www.") else 0

        tld = _tld_of_host(host)
        num_params = query.count("&") + (1 if query else 0)
        special_char_count = sum(ch in "-_?=&%." for ch in raw)
        url_entropy = _entropy(raw[:512])
        is_shortener = 1 if host.lower() in _SHORTENERS else 0

        kw_flags = {f"kw_{k}": (1 if k in raw_l else 0) for k in keywords}
        tld_oh = {f"tld_{tt}": (1 if tld == tt else 0) for tt in tlds_to_onehot}

        return {
            "url_length": url_length,
            "host_length": host_length,
            "path_length": path_length,
            "query_length": query_length,
            "num_slashes": num_slashes,
            "num_dots": num_dots,
            "num_digits": num_digits,
            "digit_ratio": digit_ratio,
            "has_https": has_https,
            "has_http": has_http,
            "has_at": has_at,
            "is_ip": is_ip,
            "startswith_www": startswith_www,
            "num_params": num_params,
            "special_char_count": special_char_count,
            "url_entropy": url_entropy,
            "is_shortener": is_shortener,
            "tld_raw": tld,  # dropped later
            **kw_flags,
            **tld_oh,
        }

    feats_df = df["domain"].apply(fe).apply(pd.Series)
    out = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
    return out

# =================================
# Step 3: Preprocess & Clean Data
# =================================
def preprocess_data(df: pd.DataFrame, target_col="label") -> pd.DataFrame:
    df = df.copy()

    # Drop raw text columns
    drop_cols = [c for c in ["domain", "tld_raw"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Convert all non-target columns to numeric
    for col in df.columns:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where ALL features are NaN
    feature_cols = [c for c in df.columns if c != target_col]
    df = df.dropna(subset=feature_cols, how="all").copy()

    # Fill NaNs with median
    for col in feature_cols:
        med = df[col].median()
        if pd.isna(med):
            med = 0
        df[col] = df[col].fillna(med)

    # Fix label to int 0/1; drop rows with missing/invalid labels
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").round().astype("Int64")
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    bad = set(df[target_col].unique()) - {0, 1}
    if bad:
        raise ValueError(f"Invalid labels found: {bad}")

    return df

# =========================================
# Step 2: Stratified Train/Test Split
# =========================================
def split_stratified(df: pd.DataFrame, target_col="label", test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# =========================================
# Step 5: Train & Evaluate (Decision Tree)
# =========================================
def train_decision_tree(X_train, y_train, random_state=42):
    from sklearn.tree import DecisionTreeClassifier
    # Simple, robust defaults; you can tune max_depth/min_samples later
    clf = DecisionTreeClassifier(
        random_state=random_state,
        class_weight=None,  # already balanced; set to "balanced" if you skip Step 1
    )
    clf.fit(X_train, y_train)
    return clf

def evaluate_on_test(model, X_test, y_test, save_dir: Path):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, confusion_matrix, classification_report
    )

    y_pred = model.predict(X_test)

    # Some metrics
    acc = accuracy_score(y_test, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    # ROC-AUC needs probabilities; handle gracefully if not available
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # Print to console
    print("\n=== Test Report ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (weighted): {precision_w:.4f}")
    print(f"Recall    (weighted): {recall_w:.4f}")
    print(f"F1        (weighted): {f1_w:.4f}")
    print(f"Precision (macro):    {precision_m:.4f}")
    print(f"Recall    (macro):    {recall_m:.4f}")
    print(f"F1        (macro):    {f1_m:.4f}")
    print(f"ROC-AUC:  {auc:.4f}" if not pd.isna(auc) else "ROC-AUC:  n/a")
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    # Save compact artifacts
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": acc,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(save_dir / "confusion_matrix.csv", index=True)
    (save_dir / "classification_report.txt").write_text(report)
# =========================================
# Step 6: Tune max_depth and plot results
# =========================================
# def tune_max_depth(X_train, y_train, X_test, y_test, save_dir: Path):
#     import matplotlib.pyplot as plt
#     from sklearn.tree import DecisionTreeClassifier
#     from sklearn.metrics import accuracy_score

#     max_depth_values = range(1, 31)  # 1 to 30
#     train_scores = []
#     test_scores = []

#     for depth in max_depth_values:
#         clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
#         clf.fit(X_train, y_train)

#         train_pred = clf.predict(X_train)
#         test_pred = clf.predict(X_test)

#         train_acc = accuracy_score(y_train, train_pred)
#         test_acc = accuracy_score(y_test, test_pred)

#         train_scores.append(train_acc)
#         test_scores.append(test_acc)

#     # Find best depth (highest test accuracy)
#     best_depth = max_depth_values[test_scores.index(max(test_scores))]
#     best_test_acc = max(test_scores)

#     print(f"\n=== Max Depth Tuning ===")
#     print(f"Best max_depth: {best_depth} with Test Accuracy: {best_test_acc:.4f}")

#     # Plot results
#     plt.figure(figsize=(10, 6))
#     plt.plot(max_depth_values, train_scores, marker='o', label="Train Accuracy")
#     plt.plot(max_depth_values, test_scores, marker='o', label="Test Accuracy")
#     plt.xlabel("max_depth")
#     plt.ylabel("Accuracy")
#     plt.title("Decision Tree Accuracy vs max_depth")
#     plt.grid(True)
#     plt.legend()

#     save_dir.mkdir(parents=True, exist_ok=True)
#     plot_path = save_dir / "max_depth_tuning.png"
#     plt.savefig(plot_path, dpi=300)
#     plt.close()

#     # Save scores to CSV
#     df_scores = pd.DataFrame({
#         "max_depth": max_depth_values,
#         "train_accuracy": train_scores,
#         "test_accuracy": test_scores
#     })
#     df_scores.to_csv(save_dir / "max_depth_scores.csv", index=False)

#     print(f"[INFO] Saved max_depth tuning plot to {plot_path}")
#     print(f"[INFO] Saved accuracy scores CSV to {save_dir / 'max_depth_scores.csv'}")

#     return best_depth, df_scores

# =========================================
# Step 6 (safer): Tune max_depth with better selection
# =========================================
def tune_max_depth(
    X_train, y_train, X_test, y_test, save_dir: Path,
    selection_method: str = "one_std",  # "one_std" or "gap_rule"
    alpha_gap: float = 0.5,
    cv_folds: int = 5
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score

    max_depth_values = list(range(1, 31))
    train_scores, test_scores, gaps, cv_means, cv_stds = [], [], [], [], []

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for depth in max_depth_values:
        # CV on training set
        clf_cv = DecisionTreeClassifier(max_depth=depth, random_state=42)
        cv_scores = cross_val_score(clf_cv, X_train, y_train, scoring="accuracy", cv=skf, n_jobs=None)
        cv_means.append(cv_scores.mean())
        cv_stds.append(cv_scores.std(ddof=1))

        # Fit on train and evaluate train/test
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc  = accuracy_score(y_test,  clf.predict(X_test))

        train_scores.append(train_acc)
        test_scores.append(test_acc)
        gaps.append(max(0.0, train_acc - test_acc))

    # --- selection rules ---
    if selection_method == "one_std":
        best_cv_mean = max(cv_means)
        best_idx = int(np.argmax(cv_means))
        margin = cv_stds[best_idx]
        threshold = best_cv_mean - margin
        # smallest depth whose CV mean >= threshold
        candidate_idxs = [i for i, m in enumerate(cv_means) if m >= threshold]
        chosen_idx = candidate_idxs[0] if candidate_idxs else best_idx
        rationale = f"one_std rule: pick smallest depth with CV ≥ (best_mean - best_std) = {threshold:.4f}"
    elif selection_method == "gap_rule":
        penalized = [t - alpha_gap * g for t, g in zip(test_scores, gaps)]
        chosen_idx = int(np.argmax(penalized))
        rationale = f"gap_rule: maximize test_acc - {alpha_gap}*(train-test)"
    else:
        # fallback: plain best test accuracy
        chosen_idx = int(np.argmax(test_scores))
        rationale = "max test accuracy (fallback)"

    best_depth = max_depth_values[chosen_idx]

    # --- Save table ---
    df_scores = pd.DataFrame({
        "max_depth": max_depth_values,
        "train_accuracy": train_scores,
        "test_accuracy": test_scores,
        "generalization_gap": gaps,
        "cv_mean_accuracy": cv_means,
        "cv_std": cv_stds,
    })
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "max_depth_scores_safe.csv"
    df_scores.to_csv(csv_path, index=False)

    # --- Plot ---
    plt.figure(figsize=(11, 6))
    plt.plot(max_depth_values, train_scores, marker='o', label="Train Acc")
    plt.plot(max_depth_values, test_scores,  marker='o', label="Test Acc")
    # CV curve + one-std band (if using one_std)
    plt.plot(max_depth_values, cv_means, linestyle='--', marker='s', label="CV Mean Acc")
    if selection_method == "one_std":
        # shade best_mean - std band
        plt.axhspan(cv_means[best_idx]-cv_stds[best_idx], cv_means[best_idx],
                    color='gray', alpha=0.15, label="Best CV ± 1 std (lower band)")
    # mark chosen depth
    plt.axvline(best_depth, color='k', linestyle=':', linewidth=1.5, label=f"Chosen depth = {best_depth}")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title(f"Decision Tree Accuracy vs max_depth ({selection_method})")
    plt.grid(True)
    plt.legend()
    plot_path = save_dir / "max_depth_tuning_safe.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== Max Depth Tuning (safer selection) ===")
    print(f"Selection method: {selection_method}")
    print(f"Chosen max_depth: {best_depth}")
    print(f"Reason: {rationale}")
    print(f"[INFO] Saved safer tuning plot to {plot_path}")
    print(f"[INFO] Saved safer scores CSV to {csv_path}")

    return best_depth, df_scores
# =========================================
# Step 7: Train Final Model & Plot Decision Tree
# =========================================
def plot_decision_tree(model, feature_names, save_dir: Path, max_depth_to_plot=None):
    import matplotlib.pyplot as plt
    from sklearn import tree

    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    plt.figure(figsize=(30, 15))
    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Benign (0)", "Phishing (1)"],
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=max_depth_to_plot  # show only top layers if specified
    )
    img_path = save_dir / "decision_tree_plot.png"
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Decision tree image saved to: {img_path}")

# =================
# Main Workflow
# =================
if __name__ == "__main__":
    CSV_PATH = "dataset.csv"
    out_dir = Path("prepared_final")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load
    df = read_csv_robust(CSV_PATH, expected_columns=EXPECTED_COLUMNS)
    print("[INFO] Original Label Distribution:")
    print(df["label"].value_counts(dropna=False))

    # Step 1: Balance
    df_bal = balance_dataset(df, method="undersample")
    print("\n[INFO] After Balancing:")
    print(df_bal["label"].value_counts())

    # Step 4: Add domain features
    df_feat = add_domain_features(df_bal, domain_col="domain")
    print("\n[INFO] Added domain features. Total columns:", len(df_feat.columns))

    # Step 3: Preprocess (drop domain/tld_raw, numeric only, fill NaNs, fix label)
    df_clean = preprocess_data(df_feat)
    print("\n[INFO] After Preprocessing:")
    print(df_clean.info())

    # Step 2: Stratified split
    X_train, X_test, y_train, y_test = split_stratified(df_clean, test_size=0.2, random_state=42)
    print("\n[INFO] Final Split Shapes:")
    print("Train:", X_train.shape, " Test:", X_test.shape)

    # Save splits for reproducibility
    pd.concat([X_train, y_train.rename("label")], axis=1).to_csv(out_dir / "train.csv", index=False)
    pd.concat([X_test,  y_test.rename("label")], axis=1).to_csv(out_dir / "test.csv", index=False)
    print(f"\n[INFO] Saved train/test & metrics to: {out_dir}")

    # Step 5: Train & Evaluate
    model = train_decision_tree(X_train, y_train, random_state=42)
    evaluate_on_test(model, X_test, y_test, save_dir=out_dir)

    # Step 6: Tune max_depth
    best_depth, scores_df = tune_max_depth(
    X_train, y_train, X_test, y_test, save_dir=out_dir,
    selection_method="one_std",   # or "gap_rule"
    alpha_gap=0.5,                # only for gap_rule
    cv_folds=5
)

    # Step 7: Train final model with best_depth
    final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    final_model.fit(X_train, y_train)

    # Plot full tree (Warning: large)
    # plot_decision_tree(final_model, X_train.columns, save_dir=out_dir, max_depth_to_plot=None)

    # Optional: Plot only top 3 levels for readability
    # plot_decision_tree(final_model, X_train.columns, save_dir=out_dir, max_depth_to_plot=3)



