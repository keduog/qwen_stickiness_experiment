"""
Test H3 (non-Markovianity): does predicting the next state need more than the
immediately preceding turn?

Fits order-0 (marginal), order-1 (last state), and order-2 (last two states) models
via Laplace-smoothed conditional frequency counts, evaluated with GroupKFold
cross-validation grouped by scenario_id (so held-out folds are genuinely unseen
scenarios, not just unseen samples of scenarios the model trained on — this matches
the methodology from the fabrication-probe control in the prior pivot-vs-persist
project, applied here to model selection instead of a classifier).

Metric: mean held-out log-likelihood per transition (higher = better fit, i.e. closer
to 0). If order-2 beats order-1 beats order-0 out of sample, that's evidence of
genuine non-Markovianity, not just overfitting to noise.

Usage:
  python3 markov_order_test.py --labeled ../rollouts/qwen14b_baseline_labeled.jsonl \
      --n_folds 5 --seed 0
"""
import argparse
import json
import math
import random
from collections import defaultdict

from build_transition_matrix import load_sequences, STATES

SMOOTHING = 1.0  # Laplace/add-one smoothing over the 4-state alphabet


def group_kfold(scenario_ids, n_folds, rng):
    unique_scenarios = list(sorted(set(scenario_ids)))
    rng.shuffle(unique_scenarios)
    folds = [[] for _ in range(n_folds)]
    for i, sid in enumerate(unique_scenarios):
        folds[i % n_folds].append(sid)
    fold_of = {}
    for fold_idx, sids in enumerate(folds):
        for sid in sids:
            fold_of[sid] = fold_idx
    return fold_of


def fit_order(sequences, order):
    """Returns dict: history_tuple -> {state: smoothed_prob}. history_tuple has
    length `order` (empty tuple for order 0)."""
    counts = defaultdict(lambda: {s: 0 for s in STATES})
    for _, _, states in sequences:
        for i in range(len(states)):
            if i < order:
                continue
            history = tuple(states[i - order:i]) if order > 0 else tuple()
            counts[history][states[i]] += 1

    probs = {}
    for history, c in counts.items():
        total = sum(c.values()) + SMOOTHING * len(STATES)
        probs[history] = {s: (c[s] + SMOOTHING) / total for s in STATES}
    # default distribution for histories never seen in train (backs off to uniform)
    default = {s: 1.0 / len(STATES) for s in STATES}
    return probs, default


def held_out_loglik(sequences, order, probs, default):
    total_ll = 0.0
    n = 0
    for _, _, states in sequences:
        for i in range(len(states)):
            if i < order:
                continue
            history = tuple(states[i - order:i]) if order > 0 else tuple()
            dist = probs.get(history, default)
            p = dist[states[i]]
            total_ll += math.log(p)
            n += 1
    return (total_ll / n if n > 0 else None), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sequences, n_unparsed, n_total = load_sequences(args.labeled)
    scenario_ids = [sid for sid, _, _ in sequences]
    n_unique_scenarios = len(set(scenario_ids))

    if n_unique_scenarios < args.n_folds:
        print(f"[WARNING] only {n_unique_scenarios} unique scenarios but {args.n_folds} folds "
              f"requested — reduce --n_folds or add more scenarios before trusting this.")

    rng = random.Random(args.seed)
    fold_of = group_kfold(scenario_ids, args.n_folds, rng)

    orders = [0, 1, 2]
    fold_results = {o: [] for o in orders}

    for fold_idx in range(args.n_folds):
        train = [s for s in sequences if fold_of[s[0]] != fold_idx]
        test = [s for s in sequences if fold_of[s[0]] == fold_idx]
        if not train or not test:
            continue
        for order in orders:
            probs, default = fit_order(train, order)
            ll, n = held_out_loglik(test, order, probs, default)
            if ll is not None:
                fold_results[order].append(ll)

    print(f"Loaded {len(sequences)} conversations across {n_unique_scenarios} scenarios "
          f"({n_unparsed}/{n_total} turns unparsed).\n")

    summary = {}
    for order in orders:
        vals = fold_results[order]
        mean_ll = sum(vals) / len(vals) if vals else None
        summary[order] = {"fold_mean_loglik": mean_ll, "n_folds_used": len(vals), "fold_values": vals}
        print(f"Order {order}: mean held-out log-lik = {mean_ll:.4f}  (folds: {vals})" if mean_ll is not None
              else f"Order {order}: no valid folds")

    print()
    o0 = summary[0]["fold_mean_loglik"]
    o1 = summary[1]["fold_mean_loglik"]
    o2 = summary[2]["fold_mean_loglik"]
    if o0 is not None and o1 is not None and o2 is not None:
        if o1 > o0 and o2 > o1:
            print("Order-2 > Order-1 > Order-0 on held-out likelihood: consistent with genuine "
                  "non-Markovianity (H3 supported). Report per-fold values, not just the mean — "
                  "with few scenarios, fold variance will be large.")
        elif o1 > o0 and o2 <= o1:
            print("Order-1 beats Order-0 but Order-2 doesn't beat Order-1: the process looks "
                  "first-order Markov, not higher-order. This is a clean, reportable negative "
                  "for the order-2 claim specifically.")
        else:
            print("Order-1 doesn't clearly beat Order-0: little evidence that conversation "
                  "history matters at all for next-state prediction. Report this honestly if "
                  "it's what the data shows — it's still a real finding.")

    with open("markov_order_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
