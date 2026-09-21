"""
Build the state transition matrix from a labeled rollouts file.

States: Useful, Useless, Harmful, Unknown (Unparsed/Truncated turns are excluded
and reported separately — never silently folded into another state).

Usage:
  python3 build_transition_matrix.py --labeled ../rollouts/qwen14b_baseline_labeled.jsonl \
      --output ../transition_matrices/qwen14b_baseline.json
"""
import argparse
import json
from collections import defaultdict

STATES = ["Useful", "Useless", "Harmful", "Unknown"]

EXCLUDED_STATES = {"Unparsed", "Truncated"}


def load_sequences(path):
    """Returns list of (scenario_id, sample_idx, [state, state, ...]) with Unparsed/
    Truncated turns dropped from the sequence (and counted separately). Truncated
    means generation hit max_new_tokens before closing <think> -- there is no real
    model answer for that turn, so it cannot be a state observation."""
    sequences = []
    n_unparsed = 0
    n_total_turns = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            states = []
            for turn in rec["transcript"]:
                n_total_turns += 1
                if turn["state"] in EXCLUDED_STATES:
                    n_unparsed += 1
                    continue
                states.append(turn["state"])
            if len(states) >= 2:
                sequences.append((rec["scenario_id"], rec["sample_idx"], states))
    return sequences, n_unparsed, n_total_turns


def build_matrix(sequences):
    counts = {s: {t: 0 for t in STATES} for s in STATES}
    marginal = {s: 0 for s in STATES}
    for _, _, states in sequences:
        for s in states:
            marginal[s] += 1
        for i in range(len(states) - 1):
            counts[states[i]][states[i + 1]] += 1
    return counts, marginal


def normalize(counts):
    probs = {}
    for s in STATES:
        row_total = sum(counts[s].values())
        if row_total == 0:
            probs[s] = {t: None for t in STATES}
        else:
            probs[s] = {t: counts[s][t] / row_total for t in STATES}
    return probs


def dwell_times(sequences):
    dwell = defaultdict(list)
    for _, _, states in sequences:
        i = 0
        while i < len(states):
            j = i
            while j + 1 < len(states) and states[j + 1] == states[i]:
                j += 1
            dwell[states[i]].append(j - i + 1)
            i = j + 1
    return {s: (sum(v) / len(v) if v else None) for s, v in dwell.items()}


def find_absorbing(probs, counts, stick_thresh=0.8, exit_thresh=0.1):
    absorbing = []
    for s in STATES:
        row_total = sum(counts[s].values())
        if row_total == 0:
            continue
        stick = probs[s][s] or 0
        max_exit = max((probs[s][t] or 0) for t in STATES if t != s)
        if stick > stick_thresh and max_exit < exit_thresh:
            absorbing.append(s)
    return absorbing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sequences, n_unparsed, n_total_turns = load_sequences(args.labeled)
    counts, marginal = build_matrix(sequences)
    probs = normalize(counts)
    dwell = dwell_times(sequences)
    absorbing = find_absorbing(probs, counts)

    n_convs = len(sequences)
    n_turns = sum(len(s) for _, _, s in sequences)

    result = {
        "n_conversations": n_convs,
        "n_scored_turns": n_turns,
        "n_unparsed_turns": n_unparsed,
        "n_total_turns_in_file": n_total_turns,
        "unparsed_rate": n_unparsed / n_total_turns if n_total_turns else None,
        "marginal_state_counts": marginal,
        "transition_counts": counts,
        "transition_probs": probs,
        "stickiness": {s: probs[s][s] for s in STATES},
        "mean_dwell_time_turns": dwell,
        "absorbing_states": absorbing,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k != "transition_counts"}, indent=2))
    if n_unparsed / max(n_total_turns, 1) > 0.05:
        print(f"\n[WARNING] {n_unparsed}/{n_total_turns} turns ({100*n_unparsed/n_total_turns:.1f}%) "
              f"had unparsable guard output or were truncated. Above 5% this should be "
              f"investigated before trusting the transition matrix.")


if __name__ == "__main__":
    main()
