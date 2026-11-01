"""Risk targeting vs. uplift targeting on a randomised campaign, behind the tables in the article.

Data: Kevin Hillstrom's MineThatData e-mail challenge (64,000 customers, randomly assigned to a
men's e-mail, a women's e-mail or no e-mail; outcome = website visit in the next two weeks).
Here "lapse" (no visit) plays the part of churn, the women's e-mail is the retention action,
and "No E-Mail" is the randomised holdout.

Run: uv run --with pandas --with scikit-learn python risk_vs_uplift.py hillstrom.csv 0 1 2 3 4 5 6 7 8 9
CSV: http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

Per seed: stratified 50/50 split. The risk model is fitted on holdout customers only
(P(lapse | no e-mail), what a churn model trained on history estimates). The uplift model is a
two-model learner, P(visit | e-mail) - P(visit | no e-mail). Both rank the test half; the
effect inside each selected group is measured with the randomised arms, not with a model.
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

TREAT = "Womens E-Mail"
SHARES = [0.1, 0.2, 0.3, 0.5, 1.0]

def load(path):
    df = pd.read_csv(path)
    df = df[df.segment.isin([TREAT, "No E-Mail"])].reset_index(drop=True)
    X = pd.get_dummies(df[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                       columns=["zip_code", "channel"], dtype=float)
    t = (df.segment == TREAT).astype(int).to_numpy()
    y = df.visit.to_numpy()
    return X, t, y

def gbm(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          min_samples_leaf=200, random_state=seed)

def effect(t, y):
    """Visits gained per 1,000 customers e-mailed: treated visit rate minus holdout visit rate."""
    return 1000 * (y[t == 1].mean() - y[t == 0].mean())

def run(X, t, y, seed):
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.5,
                                      stratify=t * 2 + y, random_state=seed)
    Xtr, ttr, ytr = X.iloc[idx_tr], t[idx_tr], y[idx_tr]
    Xte, tte, yte = X.iloc[idx_te], t[idx_te], y[idx_te]

    risk = gbm(seed).fit(Xtr[ttr == 0], 1 - ytr[ttr == 0])  # P(lapse | no e-mail)
    m1 = gbm(seed).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = gbm(seed).fit(Xtr[ttr == 0], ytr[ttr == 0])

    scores = {
        "risk": risk.predict_proba(Xte)[:, 1],
        "uplift": m1.predict_proba(Xte)[:, 1] - m0.predict_proba(Xte)[:, 1],
    }
    out = {"seed": seed, "n_test": len(yte), "overall_effect": effect(tte, yte),
           # how well the risk model predicts lapse among holdout customers it never saw
           "risk_auc_holdout": roc_auc_score(1 - yte[tte == 0], scores["risk"][tte == 0])}

    for name, s in scores.items():
        order = np.argsort(-s, kind="stable")
        out[name] = {}
        for share in SHARES:
            sel = order[: int(round(share * len(order)))]
            e = effect(tte[sel], yte[sel])
            # incremental visits per 1,000 customers in the whole base, if only this share is e-mailed
            out[name][str(share)] = {"effect_per_1000_mailed": e, "gain_per_1000_base": e * share}

    # deciles of predicted risk: lapse rate with and without the e-mail
    dec = pd.qcut(scores["risk"], 10, labels=False, duplicates="drop")
    out["risk_deciles"] = []
    for d in range(9, -1, -1):  # highest risk first
        m = dec == d
        out["risk_deciles"].append({
            "decile": 10 - d,
            "lapse_holdout": 1 - yte[m & (tte == 0)].mean(),
            "lapse_mailed": 1 - yte[m & (tte == 1)].mean(),
            "effect_per_1000": effect(tte[m], yte[m]),
        })
    return out

if __name__ == "__main__":
    X, t, y = load(sys.argv[1])
    for seed in map(int, sys.argv[2:] or ["0"]):
        print(json.dumps(run(X, t, y, seed)), flush=True)
