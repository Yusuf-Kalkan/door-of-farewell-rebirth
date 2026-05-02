"""
build_n_grams.py

Build 2nd-order n-gram counts and smoothed probability helpers for D -> P -> T.

Usage:
    python build_n_grams.py /path/to/dpt_json_folder /path/to/output_models_folder [--alpha 0.5]

Outputs (JSON):
    d_counts.json  -- nested dict: "context_key" -> {next_token: count}
    p_counts.json
    t_counts.json
    vocab.json     -- lists of all tokens seen (D, P, T)
"""

import os
import json
import sys
import math
import random
from collections import defaultdict, Counter
import argparse

# ---------- Helpers for key normalization ----------
SEP = "||"   # separator for turning context tuples into JSON keys

def ctx_to_key(ctx_tuple):
    return SEP.join(list(ctx_tuple))

def key_to_ctx(key):
    if key == "":
        return tuple()
    return tuple(key.split(SEP))

# ---------- Counting containers ----------
# We'll store counts as nested dicts: outer: context_key (string) -> inner: dict next_token -> count
def nested_inc(nested, ctx_tuple, next_tok, n=1):
    key = ctx_to_key(ctx_tuple)
    if key not in nested:
        nested[key] = {}
    nested[key][next_tok] = nested[key].get(next_tok, 0) + n

def nested_get_counts(nested, ctx_tuple):
    return nested.get(ctx_to_key(ctx_tuple), {})

# ---------- Loading DPT JSON sequences ----------
def load_dpt_sequences(folder):
    """
    Expects files with extension .dpt.json each containing:
      { "tpq":.., "tempo_us_per_q":.., "onsets": [...], "D": [...], "P": [...], "T": [...] }
    Returns a list of dicts with those keys.
    """
    files = [f for f in os.listdir(folder) if f.lower().endswith(".dpt.json")]
    seqs = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            with open(path, "r", encoding="utf8") as fh:
                data = json.load(fh)
            # basic validation
            assert "D" in data and "P" in data and "T" in data
            # remove empty/trivial sequences
            if len(data["D"]) == 0:
                continue
            seqs.append(data)
        except Exception as e:
            print("Skipping", f, "due to load error:", e)
    return seqs

# ---------- Model building ----------
def build_counts(seqs):
    """
    Build:
      d_counts: context (D_{t-2},D_{t-1}) -> next D counts
      p_counts: context (P_{t-2},P_{t-1}, D_t) -> next P counts
      t_counts: context (T_{t-1}, P_t, D_t) -> next T counts

    Also collects unigram counters for each stream (for backoff).
    We add two start tokens at beginning of each sequence: D_START x2, P_START x2, T_START x2.
    """
    D_START = "D_START"
    P_START = "P_START"
    T_START = "T_START"
    D_END = "D_END"
    P_END = "P_END"
    T_END = "T_END"

    d_counts = {}  # nested dict context-> next -> count
    p_counts = {}
    t_counts = {}

    d_unigram = Counter()
    p_unigram = Counter()
    t_unigram = Counter()

    # vocab sets
    D_vocab = set()
    P_vocab = set()
    T_vocab = set()

    for seq in seqs:
        D = seq["D"]
        P = seq["P"]
        T = seq["T"]
        L = len(D)
        # add start tokens: two for 2nd-order
        D_ctx = [D_START, D_START] + list(D)
        P_ctx = [P_START, P_START] + list(P)
        T_ctx = [T_START, T_START] + list(T)

        # iterate original indices t in 0..L-1, corresponding to D_ctx positions 2..L+1
        for t in range(L):
            # current tokens
            d_t = D_ctx[t+2]
            p_t = P_ctx[t+2]
            t_t = T_ctx[t+2]

            # update unigram counts
            d_unigram[d_t] += 1
            p_unigram[p_t] += 1
            t_unigram[t_t] += 1

            # update vocab
            D_vocab.add(d_t)
            P_vocab.add(p_t)
            T_vocab.add(t_t)

            # D counts: context = (D_{t-2}, D_{t-1}) predicting D_t
            ctx_d = (D_ctx[t], D_ctx[t+1])
            nested_inc(d_counts, ctx_d, d_t, n=1)

            # P counts: context = (P_{t-2}, P_{t-1}, D_t) predicting P_t
            ctx_p = (P_ctx[t], P_ctx[t+1], d_t)
            nested_inc(p_counts, ctx_p, p_t, n=1)

            # T counts: context = (T_{t-1}, P_t, D_t) predicting T_t
            ctx_t = (T_ctx[t+1], p_t, d_t)  # note T_{t-1} is at t+1
            nested_inc(t_counts, ctx_t, t_t, n=1)

        # Optionally we could add an end token counts if we want to model sequence ends;
        # for now we skip explicit END counting, generation will stop by measure/bar logic
    # return everything
    return {
        "d_counts": d_counts,
        "p_counts": p_counts,
        "t_counts": t_counts,
        "d_unigram": dict(d_unigram),
        "p_unigram": dict(p_unigram),
        "t_unigram": dict(t_unigram),
        "vocab": {
            "D": sorted(list(D_vocab)),
            "P": sorted(list(P_vocab)),
            "T": sorted(list(T_vocab))
        }
    }

# ---------- Convert counts -> smoothed probability distribution ----------
def counts_to_probs(counts_nested, alpha=0.5, vocab=None):
    """
    counts_nested: dict context_key -> {next_token: count}
    alpha: Laplace smoothing parameter (add-alpha)
    vocab: optional list/set of allowed next tokens. If not provided, use union of seen tokens under counts_nested.
    Returns dict: context_key -> {token:prob}
    """
    probs = {}
    # If vocab provided, ensure it's a list
    if vocab is None:
        vocab = set()
        for ctx_key, inner in counts_nested.items():
            vocab.update(inner.keys())
        vocab = sorted(vocab)
    else:
        vocab = sorted(list(vocab))

    V = len(vocab)
    for ctx_key, inner in counts_nested.items():
        total_counts = sum(inner.values())
        # compute smoothed denom: total_counts + alpha*V
        denom = total_counts + alpha * V
        dist = {}
        for token in vocab:
            count = inner.get(token, 0)
            prob = (count + alpha) / denom
            if prob > 0:
                dist[token] = prob
        probs[ctx_key] = dist
    return probs

# ---------- Backoff distribution query helpers ----------
def get_D_distribution(d_counts, d_unigram, D_vocab, ctx_prev2, ctx_prev1, alpha=0.5):
    """
    Backoff for D:
      1) try context (ctx_prev2, ctx_prev1)
      2) try unigram (fallback)
    Returns dict token->prob (normalized).
    """
    # try 2-gram context
    ctx = (ctx_prev2, ctx_prev1)
    inner = nested_get_counts(d_counts, ctx)
    if inner:
        # smooth over D_vocab
        V = len(D_vocab)
        denom = sum(inner.values()) + alpha * V
        return {tok: (inner.get(tok,0)+alpha)/denom for tok in D_vocab}
    # fallback to unigram smoothing
    total = sum(d_unigram.values()) + alpha * len(D_vocab)
    return {tok: (d_unigram.get(tok,0)+alpha)/total for tok in D_vocab}

def get_P_distribution(p_counts, p_unigram, P_vocab, p_prev2, p_prev1, d_t, alpha=0.5):
    """
    Backoff for P:
      1) try (p_prev2, p_prev1, d_t)
      2) try (p_prev1, d_t)
      3) try (d_t)  [i.e., context only D]
      4) unigram
    """
    # try 3-gram context
    ctx3 = (p_prev2, p_prev1, d_t)
    inner = nested_get_counts(p_counts, ctx3)
    if inner:
        V = len(P_vocab)
        denom = sum(inner.values()) + alpha * V
        return {tok: (inner.get(tok,0)+alpha)/denom for tok in P_vocab}
    # try 2-item context (p_prev1, d_t)
    ctx2 = (p_prev1, d_t)
    inner2 = nested_get_counts(p_counts, ctx2)
    if inner2:
        V = len(P_vocab)
        denom = sum(inner2.values()) + alpha * V
        return {tok: (inner2.get(tok,0)+alpha)/denom for tok in P_vocab}
    # try just D context (we'll use ctx ( '', '', d_t) equivalently)
    ctx1 = ("", "", d_t)
    inner1 = nested_get_counts(p_counts, ctx1)
    if inner1:
        V = len(P_vocab)
        denom = sum(inner1.values()) + alpha * V
        return {tok: (inner1.get(tok,0)+alpha)/denom for tok in P_vocab}
    # fallback unigram
    total = sum(p_unigram.values()) + alpha * len(P_vocab)
    return {tok: (p_unigram.get(tok,0)+alpha)/total for tok in P_vocab}

def get_T_distribution(t_counts, t_unigram, T_vocab, t_prev1, p_t, d_t, alpha=0.5):
    """
    Backoff for T:
      1) try (t_prev1, p_t, d_t)
      2) try (p_t, d_t)
      3) try (p_t)
      4) unigram
    """
    ctx1 = (t_prev1, p_t, d_t)
    inner = nested_get_counts(t_counts, ctx1)
    if inner:
        V = len(T_vocab)
        denom = sum(inner.values()) + alpha * V
        return {tok: (inner.get(tok,0)+alpha)/denom for tok in T_vocab}
    ctx2 = ("", p_t, d_t)
    inner2 = nested_get_counts(t_counts, ctx2)
    if inner2:
        V = len(T_vocab)
        denom = sum(inner2.values()) + alpha * V
        return {tok: (inner2.get(tok,0)+alpha)/denom for tok in T_vocab}
    ctx3 = ("", p_t, "")
    inner3 = nested_get_counts(t_counts, ctx3)
    if inner3:
        V = len(T_vocab)
        denom = sum(inner3.values()) + alpha * V
        return {tok: (inner3.get(tok,0)+alpha)/denom for tok in T_vocab}
    total = sum(t_unigram.values()) + alpha * len(T_vocab)
    return {tok: (t_unigram.get(tok,0)+alpha)/total for tok in T_vocab}

# ---------- Sampling helpers ----------
def sample_from_dist(dist):
    """
    dist: dict token->prob (probabilities should sum ~1)
    return single sampled token
    """
    tokens = list(dist.keys())
    probs = [dist[t] for t in tokens]
    # normalization safety
    s = sum(probs)
    if s <= 0:
        # fallback: uniform
        return random.choice(tokens) if tokens else None
    probs = [p/s for p in probs]
    return random.choices(tokens, weights=probs, k=1)[0]

# ---------- Save / load json utilities ----------
def save_json(obj, path):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf8") as fh:
        return json.load(fh)

# ---------- Main CLI logic ----------
def main(in_folder, out_folder, alpha):
    seqs = load_dpt_sequences(in_folder)
    if not seqs:
        print("No valid .dpt.json sequences found in", in_folder)
        return
    models = build_counts(seqs)
    # write raw counts
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    save_json(models["d_counts"], os.path.join(out_folder, "d_counts.json"))
    save_json(models["p_counts"], os.path.join(out_folder, "p_counts.json"))
    save_json(models["t_counts"], os.path.join(out_folder, "t_counts.json"))
    save_json({
        "d_unigram": models["d_unigram"],
        "p_unigram": models["p_unigram"],
        "t_unigram": models["t_unigram"],
        "vocab": models["vocab"]
    }, os.path.join(out_folder, "vocab.json"))
    print("Wrote raw count files to", out_folder)

    # Convert to smoothed probs for contexts we have (helpful if you want precomputed)
    d_probs = counts_to_probs(models["d_counts"], alpha=alpha, vocab=models["vocab"]["D"])
    p_probs = counts_to_probs(models["p_counts"], alpha=alpha, vocab=models["vocab"]["P"])
    t_probs = counts_to_probs(models["t_counts"], alpha=alpha, vocab=models["vocab"]["T"])
    save_json(d_probs, os.path.join(out_folder, "d_probs.json"))
    save_json(p_probs, os.path.join(out_folder, "p_probs.json"))
    save_json(t_probs, os.path.join(out_folder, "t_probs.json"))
    print("Wrote smoothed context probabilities (for seen contexts).")

    # Example: a tiny generator demonstration using backoff sampling
    D_vocab = models["vocab"]["D"]
    P_vocab = models["vocab"]["P"]
    T_vocab = models["vocab"]["T"]

    # seed
    D_prev2, D_prev1 = "D_START", "D_START"
    P_prev2, P_prev1 = "P_START", "P_START"
    T_prev1 = "T_START"

    generated = []
    # generate N steps
    N = 48
    for i in range(N):
        # sample D_t using backoff
        d_dist = get_D_distribution(models["d_counts"], models["d_unigram"], D_vocab, D_prev2, D_prev1, alpha=alpha)
        d_t = sample_from_dist(d_dist)

        # sample P_t conditioned on D_t and last two P
        p_dist = get_P_distribution(models["p_counts"], models["p_unigram"], P_vocab, P_prev2, P_prev1, d_t, alpha=alpha)
        p_t = sample_from_dist(p_dist)

        # sample T_t conditioned on last T, P_t, D_t
        t_dist = get_T_distribution(models["t_counts"], models["t_unigram"], T_vocab, T_prev1, p_t, d_t, alpha=alpha)
        t_t = sample_from_dist(t_dist)

        generated.append((d_t, p_t, t_t))

        # slide history
        D_prev2, D_prev1 = D_prev1, d_t
        P_prev2, P_prev1 = P_prev1, p_t
        T_prev1 = t_t

    # print a tiny generated example
    print("\nExample generated D,P,T sequence ({} steps):".format(N))
    for i, (d,p,t) in enumerate(generated):
        print(f"{i:02d}: D={d:15}  P={p:25}  T={t}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 2nd-order DPT n-grams")
    parser.add_argument("in_folder", help="Folder with .dpt.json files")
    parser.add_argument("out_folder", help="Folder to write models")
    parser.add_argument("--alpha", type=float, default=0.5, help="Laplace smoothing alpha (default 0.5)")
    args = parser.parse_args()
    main(args.in_folder, args.out_folder, args.alpha)
