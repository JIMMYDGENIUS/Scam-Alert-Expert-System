import os, pandas as pd, numpy as np, pickle
from sklearn.linear_model import LogisticRegression

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data", "toy_events.csv")
OUT = os.path.join(BASE, "app", "model.pkl")

def featurize(df: pd.DataFrame):
    X = pd.DataFrame({
        "len_text": df["text"].fillna("").str.len(),
        "has_otp": df["text"].str.lower().str.contains("otp|one-time password", regex=True).astype(int),
        "has_seed": df["text"].str.lower().str.contains("seed phrase|private key|recovery phrase", regex=True).astype(int),
        "has_urgent": df["text"].str.lower().str.contains("urgent|immediately", regex=True).astype(int),
        "url_mismatch": (df["display_domain"].fillna("") != df["final_domain"].fillna("")).astype(int),
        "domain_age": df["sender_domain_age_days"].fillna(9999).astype(int),
        "reports": df["reports_last_90d"].fillna(0).astype(int),
        "blacklisted": df["global_blacklist"].fillna(False).astype(int),
    })
    y = (df["label"] == "scam").astype(int).values
    return X, y

def main():
    df = pd.read_csv(DATA)
    X, y = featurize(df)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X.values, y)
    model = {
        "clf": clf,
        "feature_order": list(X.columns)
    }
    with open(OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {OUT}")

if __name__ == "__main__":
    main()
