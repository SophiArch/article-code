"""Baseline ladder experiment behind the table in the article.

Run: uv run --with pmlb --with scikit-learn --with scipy python ladder.py adult,churn,spambase,magic,hypothyroid,phoneme 0
Seeds 0-3 were run; results.jsonl holds all four.

Rungs: majority-class dummy -> logistic regression -> default HistGradientBoosting
-> HistGradientBoosting tuned with RandomizedSearchCV (the "assistant-style" first move).
Same stratified 80/20 split per dataset; tuning uses 5-fold CV on the training set only.
"""
import sys, time, json
from pmlb import fetch_data
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import loguniform, randint

CATS = {
    "adult": ["workclass", "education", "marital-status", "occupation",
              "relationship", "race", "sex", "native-country"],
    "churn": ["state", "area code", "international plan", "voice mail plan"],
}
DROP = {"churn": ["phone number"]}

def load(name):
    df = fetch_data(name)  # drops rows with missing values by default
    df = df.drop(columns=DROP.get(name, []))
    y = df.pop("target")
    return df, y

def run(name, seed=0, n_iter=40):
    X, y = load(name)
    if y.nunique() != 2:
        return None
    y = (y == y.value_counts().idxmin()).astype(int)  # minority class = 1
    cats = CATS.get(name, [])
    nums = [c for c in X.columns if c not in cats]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", StandardScaler(), nums),
    ])
    cat_mask = [c in cats for c in X.columns]
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(pre, LogisticRegression(max_iter=2000)),
        "hgb_default": HistGradientBoostingClassifier(
            categorical_features=cat_mask if cats else None, random_state=seed),
        "hgb_tuned": RandomizedSearchCV(
            HistGradientBoostingClassifier(
                categorical_features=cat_mask if cats else None, random_state=seed),
            {
                "learning_rate": loguniform(0.01, 0.3),
                "max_leaf_nodes": randint(8, 128),
                "min_samples_leaf": randint(5, 100),
                "l2_regularization": loguniform(1e-4, 10),
                "max_iter": randint(100, 1000),
            },
            n_iter=n_iter, cv=5, scoring="roc_auc", random_state=seed, n_jobs=-1),
    }
    out = {"dataset": name, "rows": len(X), "features": X.shape[1],
           "pos_rate": round(y.mean(), 3)}
    for key, m in models.items():
        t = time.perf_counter()
        m.fit(X_tr, y_tr)
        secs = time.perf_counter() - t
        p = m.predict_proba(X_te)[:, 1]
        out[key] = {"acc": round(accuracy_score(y_te, m.predict(X_te)), 4),
                    "auc": round(roc_auc_score(y_te, p), 4),
                    "fit_s": round(secs, 1)}
    return out

if __name__ == "__main__":
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for name in sys.argv[1].split(","):
        print(json.dumps(run(name, seed=seed)), flush=True)
