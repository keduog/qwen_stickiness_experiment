"""
Test H3-robustness: does the harmful-state stickiness measured under baseline survive
a benign perturbation (paraphrase) but break under an adversarial one (jailbreak
prefix)? That distinction is what tells you whether "stickiness" is a real property
of the model's dynamics or a fragile artifact of exact prompt wording.

Two tests, both scenario-grouped so a scenario's baseline and perturbed rollouts are
compared as matched pairs where possible:

  1. Permutation test on the DIFFERENCE in stickiness between conditions (shuffles
     which conversations are labeled "condition A" vs "condition B" across the pooled
     set, preserving each conversation's own sequence).
  2. Chi-square test of independence on the pooled transition-count table
     (state_from x state_to) between conditions, as a complementary structural check.

Usage:
  python3 perturbations.py \
      --baseline ../rollouts/qwen14b_baseline_labeled.jsonl \
      --perturbed ../rollouts/qwen14b_jailbreak_labeled.jsonl \
      --condition_name jailbreak_prefix \
      --n_permutations 2000 --seed 0
"""
import argparse
import json
import random

import numpy as np
from scipy.stats import chi2_contingency

from build_transition_matrix import load_sequences, build_matrix, STATES
from permutation_test import self_transition_prob


def stickiness_diff(seq_a, seq_b, state):
    sa, _ = self_transition_prob(seq_a, state)
    sb, _ = self_transition_prob(seq_b, state)
    if sa is None or sb is None:
        return None
    return sa - sb


def permutation_test_diff(seq_a, seq_b, state, n_permutations, rng):
    observed = stickiness_diff(seq_a, seq_b, state)
    if observed is None:
        return {"state": state, "observed_diff": None, "note": "state absent in one condition"}

    pooled = seq_a + seq_b
    n_a = len(seq_a)
    null_diffs = []
    for _ in range(n_permutations):
        rng.shuffle(pooled)  # shuffles which conversations fall in group A vs B
        group_a = pooled[:n_a]
        group_b = pooled[n_a:]
        d = stickiness_diff(group_a, group_b, state)
        if d is not None:
            null_diffs.append(d)

    if not null_diffs:
        return {"state": state, "observed_diff": observed, "note": "no valid null draws"}

    n_ge = sum(1 for d in null_diffs if abs(d) >= abs(observed))
    p_value = (n_ge + 1) / (len(null_diffs) + 1)

    return {
        "state": state,
        "observed_diff_baseline_minus_perturbed": observed,
        "p_value_two_sided": p_value,
        "n_permutations": len(null_diffs),
    }


def contingency_chi2(seq_a, seq_b):
    counts_a, _ = build_matrix(seq_a)
    counts_b, _ = build_matrix(seq_b)
    # flatten to a single from-state x condition table per to-state, or simpler:
    # build one big table of transition_type (from,to) x condition
    transition_types = [(f, t) for f in STATES for t in STATES]
    table = []
    for f, t in transition_types:
        table.append([counts_a[f][t], counts_b[f][t]])
    table = np.array(table)
    # drop all-zero rows (transition types that never occurred in either condition)
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2:
        return {"note": "not enough non-zero transition types for a valid chi-square test"}
    chi2, p, dof, _ = chi2_contingency(table)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "n_transition_types": int(table.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--perturbed", required=True)
    ap.add_argument("--condition_name", default="perturbed")
    ap.add_argument("--n_permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seq_base, _, _ = load_sequences(args.baseline)
    seq_pert, _, _ = load_sequences(args.perturbed)

    print(f"Baseline: {len(seq_base)} conversations. {args.condition_name}: {len(seq_pert)} conversations.\n")

    rng = random.Random(args.seed)
    results = {"condition": args.condition_name, "per_state": {}, "contingency": None}

    for state in STATES:
        r = permutation_test_diff(seq_base, seq_pert, state, args.n_permutations, rng)
        results["per_state"][state] = r
        print(json.dumps(r, indent=2))

    results["contingency"] = contingency_chi2(seq_base, seq_pert)
    print("\nContingency test (transition-type x condition):")
    print(json.dumps(results["contingency"], indent=2))

    harmful = results["per_state"].get("Harmful", {})
    if harmful.get("p_value_two_sided") is not None:
        print(f"\nInterpretation guide: a SMALL p-value for Harmful means the perturbation "
              f"significantly changed harmful-state stickiness. For jailbreak_prefix, that's "
              f"the expected/interesting direction (adversarial perturbations should break or "
              f"amplify the dynamic). For a paraphrase condition, a LARGE p-value (no "
              f"significant change) is the result that supports 'stickiness is real, not an "
              f"artifact of exact wording' — don't accidentally treat 'not significant' as a "
              f"failed experiment there, it's the predicted outcome.")

    with open(f"perturbation_results_{args.condition_name}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
