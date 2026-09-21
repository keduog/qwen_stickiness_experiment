"""
Run this after your FIRST small pilot batch (e.g. the 30 starter scenarios x 3
samples = 90 conversations), before committing to the full scenario set.

It answers the question the original 30-scenario starter set can't answer on its own:
does this model, on these scenarios, actually visit the Harmful state often enough to
say anything statistically meaningful about its stickiness? If not, you need either
more scenarios, or scenarios with a stronger pull toward Harmful (more aggressive
escalation turns) — not just more samples of the same weak scenarios.

Rule of thumb used here: you want at least ~50 observed Harmful->X transitions before
a permutation test has reasonable power to detect a moderate stickiness effect. This
is a heuristic, not a formal power calculation — if you want a rigorous a priori power
analysis for the paper, simulate it (see note at bottom).

Usage:
  python3 pilot_power_check.py --labeled ../rollouts/qwen14b_pilot_labeled.jsonl
"""
import argparse
import sys

sys.path.insert(0, "../analysis")
from build_transition_matrix import load_sequences, build_matrix, STATES  # noqa: E402

MIN_TRANSITIONS_FOR_POWER = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    args = ap.parse_args()

    sequences, n_unparsed, n_total = load_sequences(args.labeled)
    counts, marginal = build_matrix(sequences)

    print(f"{len(sequences)} conversations, {n_unparsed}/{n_total} turns unparsed.\n")
    print("Marginal state visit counts:")
    for s in STATES:
        print(f"  {s}: {marginal[s]}")

    print("\nTransitions FROM each state (denominator for stickiness estimates):")
    underpowered = []
    for s in STATES:
        n_from = sum(counts[s].values())
        flag = ""
        if n_from < MIN_TRANSITIONS_FOR_POWER:
            flag = f"  <-- UNDERPOWERED (need ~{MIN_TRANSITIONS_FOR_POWER}+)"
            underpowered.append(s)
        print(f"  {s}: {n_from}{flag}")

    print()
    if underpowered:
        print(f"States {underpowered} don't have enough transitions yet for a trustworthy "
              f"permutation test. Before scaling up sample count on the SAME scenarios, "
              f"check which domains/categories in scenarios.json actually produced visits "
              f"to the underpowered state(s) — scale up scenarios of that type specifically, "
              f"e.g. more 'escalating' and 'stable_harmful' category scenarios, rather than "
              f"uniformly expanding all 6 domains.")
        print("\nRough scenario-count needed (linear extrapolation, treat as a lower bound):")
        for s in underpowered:
            n_from = sum(counts[s].values())
            if n_from > 0:
                scale = MIN_TRANSITIONS_FOR_POWER / n_from
                print(f"  {s}: roughly {scale:.1f}x more data of the type that currently "
                      f"produces {s} visits")
            else:
                print(f"  {s}: zero visits so far — no scenario in the pilot set reaches this "
                      f"state at all. Needs new scenario design, not just more samples.")
    else:
        print("All states have enough transitions for the permutation tests to have "
              "reasonable power. Proceed to the full run.")

    print("\nNote on rigor: this is a heuristic threshold, not a formal power calculation. "
          "For the paper, consider simulating the permutation test under a range of assumed "
          "true stickiness values (e.g. 0.3, 0.5, 0.7) at your actual sample size to report "
          "achieved power directly, rather than relying on this rule of thumb.")


if __name__ == "__main__":
    main()
