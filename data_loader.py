import sys
import pandas as pd

CSV_PATH = "dataset.csv"

EXPECTED_COLUMNS = [
    "domain","ranking","mld_res","mld.ps_res","card_rem","ratio_Rrem","ratio_Arem",
    "jaccard_RR","jaccard_RA","jaccard_AR","jaccard_AA","jaccard_ARrd","jaccard_ARrem","label"
]

def _supports_on_bad_lines() -> bool:
    major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    return (major, minor) >= (1, 3)

def read_csv_robust(path: str, expected_columns=EXPECTED_COLUMNS) -> pd.DataFrame:
    """Read messy CSVs with inconsistent quoting/encodings and coerce to expected schema."""
    encodings = ["utf-8", "cp1252", "latin-1"]
    last_err = None

    base_kwargs = dict(
        sep=",",
        quotechar='"',
        doublequote=True,
        escapechar="\\",
        skipinitialspace=True,
        engine="python",
    )

    if _supports_on_bad_lines():
        base_kwargs.update(on_bad_lines="skip")
    else:
        base_kwargs.update(error_bad_lines=False, warn_bad_lines=False)

    for enc in encodings:
        for force_names in (False, True):
            try:
                kw = dict(base_kwargs, encoding=enc)
                if force_names:
                    kw.update(names=expected_columns, header=0)
                df = pd.read_csv(path, **kw)

                if len(df.columns) != len(expected_columns):
                    if not force_names and len(df.columns) >= len(expected_columns):
                        df = df.iloc[:, :len(expected_columns)]
                        df.columns = expected_columns
                    elif force_names:
                        pass
                    else:
                        while len(df.columns) < len(expected_columns):
                            df[f"_missing_{len(df.columns)}"] = pd.NA
                        df = df.iloc[:, :len(expected_columns)]
                        df.columns = expected_columns

                for c in df.select_dtypes(include=["object"]).columns:
                    df[c] = df[c].astype(str).str.strip()

                return df
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"Failed to load CSV with robust parser. Last error: {last_err}")

if __name__ == "__main__":
    try:
        df = read_csv_robust(CSV_PATH)
        print("Loaded shape:", df.shape)
        print("Columns:", list(df.columns))
        print(df.head(3))
        if "label" in df.columns:
            print("\nLabel distribution:")
            print(df["label"].value_counts(dropna=False))
    except Exception as exc:
        print("CSV load failed:", repr(exc))
        sys.exit(1)
