# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers>=3.0", "rank-bm25", "scikit-learn", "numpy"]
# ///
"""Negation experiment behind the tables in the article (test numbers match the article).

Run: uv run negation.py            (downloads ~3 GB of open models on first run)
Writes results.json next to this file. Data: items.py (40 invented policy facts).

Test 1, cache:     can any similarity threshold separate a reworded question from its negation?
Test 2, pairs:     is "X is not covered" closer to "X is covered" than a rewording of it is?
Test 3, retrieval: with "covered" and "not covered" chunks for different entities in one index,
                   does "Which plan doesn't cover X?" find the one that doesn't?
                   Dense vs BM25 vs hybrid (RRF, k=60) vs dense top-10 + cross-encoder re-rank.
"""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics import roc_auc_score

from items import ITEMS, fill

BGE_Q = "Represent this sentence for searching relevant passages: "
# name -> (prefix for symmetric sentence similarity, query prefix, passage prefix, query prompt_name)
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": ("", "", "", None),
    "BAAI/bge-base-en-v1.5": ("", BGE_Q, "", None),
    "intfloat/e5-base-v2": ("query: ", "query: ", "passage: ", None),
    "thenlper/gte-base": ("", "", "", None),
    "mixedbread-ai/mxbai-embed-large-v1": ("", BGE_Q, "", None),
    "Qwen/Qwen3-Embedding-0.6B": ("", "", "", "query"),
}
RERANKERS = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"]

# Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET — the `_english_` stop list in Elasticsearch/OpenSearch.
LUCENE_STOP = set("""a an and are as at be but by for if in into is it no not of on or such that the
their then there these they this to was will with""".split())


def tok(s, stop=frozenset()):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in stop]


def auroc(same, diff):
    """P(similarity of a same-meaning pair > similarity of a contradicting pair)."""
    y = [1] * len(same) + [0] * len(diff)
    return float(roc_auc_score(y, list(same) + list(diff)))


def cos_rows(a, b):
    return np.sum(a * b, axis=1)


# Test 3 corpus: entity A's "covered" chunk and entity B's "not covered" chunk for every item.
DOCS, TARGETS = [], []  # TARGETS[i] = (query, target doc idx, twin doc idx)
for it in ITEMS:
    ip, ineg = len(DOCS), len(DOCS) + 1
    DOCS += [fill(it["pos"], it["a"]), fill(it["neg"], it["b"])]
    TARGETS += [(it["q_pos"], ip, ineg), (it["q_neg"], ineg, ip)]


def score_rankings(score_matrix):
    """score_matrix[q, d]. Returns pairwise acc, paired (both queries of an item right), recall@1."""
    pair = np.array([score_matrix[q, t] > score_matrix[q, w] for q, (_, t, w) in enumerate(TARGETS)])
    top1 = np.array([int(np.argmax(score_matrix[q])) == t for q, (_, t, _) in enumerate(TARGETS)])
    return dict(
        pairwise=float(pair.mean()),
        paired=float((pair[0::2] & pair[1::2]).mean()),
        recall_at_1=float(top1.mean()),
        pos_pairwise=float(pair[0::2].mean()),
        neg_pairwise=float(pair[1::2].mean()),
    )


def rrf(score_lists, k=60):
    out = np.zeros_like(score_lists[0])
    for s in score_lists:
        ranks = np.argsort(np.argsort(-s, axis=1), axis=1) + 1
        out += 1.0 / (k + ranks)
    return out


def bm25_scores(stop):
    bm = BM25Okapi([tok(d, stop) for d in DOCS])
    return np.array([bm.get_scores(tok(q, stop)) for q, _, _ in TARGETS])


def main():
    results = {"n_items": len(ITEMS), "pairs": {}, "cache": {}, "retrieval": {}, "examples": {}}
    queries = [q for q, _, _ in TARGETS]

    bm_keep, bm_stop = bm25_scores(frozenset()), bm25_scores(LUCENE_STOP)
    results["retrieval"]["BM25 (keeps 'not')"] = score_rankings(bm_keep)
    results["retrieval"]["BM25 (Lucene English stop words)"] = score_rankings(bm_stop)

    dense = {}
    for name, (sym, qp, pp, prompt_name) in MODELS.items():
        m = SentenceTransformer(name, trust_remote_code=False)
        enc = lambda xs, pre="", pn=None: m.encode(  # noqa: E731
            [pre + x for x in xs], normalize_embeddings=True, prompt_name=pn, batch_size=32)

        # Test 2: statement pairs, entity A throughout.
        pos = enc([fill(it["pos"], it["a"]) for it in ITEMS], sym)
        para = enc([fill(it["para"], it["a"]) for it in ITEMS], sym)
        neg = enc([fill(it["neg"], it["a"]) for it in ITEMS], sym)
        s_para, s_neg = cos_rows(pos, para), cos_rows(pos, neg)
        results["pairs"][name] = dict(
            mean_paraphrase=float(s_para.mean()), mean_negation=float(s_neg.mean()),
            negation_closer=float((s_neg > s_para).mean()), auroc=auroc(s_para, s_neg))
        results["examples"][name] = [
            dict(pos=fill(it["pos"], it["a"]), para=float(a), neg=float(b))
            for it, a, b in zip(ITEMS, s_para, s_neg)]

        # Test 1: semantic cache keyed on question similarity.
        qpos = enc([it["q_pos"] for it in ITEMS], sym)
        c_para = cos_rows(qpos, enc([it["q_para"] for it in ITEMS], sym))
        c_neg = cos_rows(qpos, enc([it["q_neg"] for it in ITEMS], sym))
        t = float(np.percentile(c_para, 5))  # threshold that still serves 95% of true rewordings
        results["cache"][name] = dict(
            mean_paraphrase=float(c_para.mean()), mean_negation=float(c_neg.mean()),
            negation_closer=float((c_neg > c_para).mean()), auroc=auroc(c_para, c_neg),
            threshold_95=t, negated_hits_at_threshold=float((c_neg >= t).mean()),
            # RedisVL SemanticCache default: cosine distance 0.1, i.e. similarity >= 0.90
            paraphrase_hits_at_0_90=float((c_para >= 0.9).mean()),
            negated_hits_at_0_90=float((c_neg >= 0.9).mean()),
            dental=dict(paraphrase=float(c_para[0]), negation=float(c_neg[0])))

        # Test 3: retrieval.
        D = enc(DOCS, pp)
        Q = enc(queries, qp, prompt_name)
        dense[name] = Q @ D.T
        results["retrieval"][f"dense: {name}"] = score_rankings(dense[name])
        results["retrieval"][f"hybrid RRF: {name} + BM25"] = score_rankings(rrf([dense[name], bm_keep]))
        results["retrieval"][f"hybrid RRF: {name} + BM25 (stop words)"] = score_rankings(
            rrf([dense[name], bm_stop]))
        print(name, results["pairs"][name], results["cache"][name],
              results["retrieval"][f"dense: {name}"], flush=True)

    # Re-rank each dense model's top 10 with each cross-encoder.
    for rname in RERANKERS:
        ce = CrossEncoder(rname)
        ce_full = np.array([ce.predict([(q, d) for d in DOCS]) for q in queries])
        results["retrieval"][f"cross-encoder alone: {rname}"] = score_rankings(ce_full)
        for name, S in dense.items():
            top = np.argsort(-S, axis=1)[:, :10]
            R = np.full_like(S, -1e9)
            for qi in range(len(queries)):
                R[qi, top[qi]] = ce_full[qi, top[qi]]
            results["retrieval"][f"dense top-10 + {rname}: {name}"] = score_rankings(R)
        print(rname, results["retrieval"][f"cross-encoder alone: {rname}"], flush=True)

    Path(__file__).with_name("results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
