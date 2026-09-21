"""
Permutation (shuffled-label) test for stickiness significance.

Null hypothesis: the sequential order of states within a conversation carries no
information — i.e. observed self-transition rate for a state is just what you'd get
from the state's marginal frequency alone (order-0 null), not genuine "stickiness."

Procedure per state s:
  1. Compute observed P(s -> s) across all real transitions.
  2. For N permutations: shuffle the *order* of states within each conversation
     independently (this preserves each conversation's own marginal state counts,
     which is the correct null — it does not just re-mix everything globally, which
     would also destroy per-conversation base rates and bias the test).
  3. Recompute P(s -> s) on each shuffled dataset -> null distribution.
  4. p-value = fraction of null draws >= observed (one-sided: is it stickier than chance).

Usage:
  python3 permutation_test.py --labeled ../rollouts/qwen14b_baseline_labeled.jsonl \
      --n_permutations 2000 --seed 0
"""
import argparse
import json
import random
from collections import Counter

from build_transition_matrix import load_sequences, STATES


def self_transition_prob(sequences, state):
    same, total = 0, 0
    for _, _, states in sequences:
        for i in range(len(states) - 1):
            if states[i] == state:
                total += 1
                if states[i + 1] == state:
                    same += 1
    return (same / total if total > 0 else None), total


def shuffle_within_conversation(sequences, rng):
    shuffled = []
    for sid, idx, states in sequences:
        s = states[:]
        rng.shuffle(s)
        shuffled.append((sid, idx, s))
    return shuffled


def permutation_test(sequences, state, n_permutations, rng):
    observed, n_obs_transitions = self_transition_prob(sequences, state)
    if observed is None:
        return {"state": state, "observed": None, "note": "state never occurred, cannot test"}

    null_draws = []
    for _ in range(n_permutations):
        shuffled = shuffle_within_conversation(sequences, rng)
        val, _ = self_transition_prob(shuffled, state)
        if val is not None:
            null_draws.append(val)

    if not null_draws:
        return {"state": state, "observed": observed, "note": "no valid null draws"}

    n_ge = sum(1 for v in null_draws if v >= observed)
    p_value = (n_ge + 1) / (len(null_draws) + 1)  # add-one smoothing, standard for permutation tests
    null_mean = sum(null_draws) / len(null_draws)

    return {
        "state": state,
        "observed_stickiness": observed,
        "n_transitions_from_state": n_obs_transitions,
        "null_mean_stickiness": null_mean,
        "null_n_permutations": len(null_draws),
        "p_value": p_value,
        "effect_size_vs_null_mean": observed - null_mean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--n_permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sequences, n_unparsed, n_total = load_sequences(args.labeled)
    rng = random.Random(args.seed)

    print(f"Loaded {len(sequences)} conversations, {n_unparsed}/{n_total} turns unparsed.\n")

    results = {}
    for state in STATES:
        r = permutation_test(sequences, state, args.n_permutations, rng)
        results[state] = r
        print(json.dumps(r, indent=2))
        print()

    # Explicit primary hypothesis check
    harmful = results.get("Harmful", {})
    if harmful.get("p_value") is not None:
        sig = "SIGNIFICANT" if harmful["p_value"] < 0.05 else "not significant"
        print(f"Primary claim (H2, Harmful stickiness): p = {harmful['p_value']:.4f} -> {sig} "
              f"at alpha=0.05. Remember to correct for multiple comparisons (4 states tested) "
              f"if reporting all four, e.g. Bonferroni alpha=0.0125.")
    else:
        print("Harmful state had too few occurrences to test — this itself is worth reporting: "
              "it means the scenario set didn't generate enough harmful-state visits for power. "
              "See the scaling note in scenarios.json.")


if __name__ == "__main__":
    main()
