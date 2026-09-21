"""
Selects only the scenarios that actually produced Harmful-state visits in the
labeled baseline data, and writes them to a smaller scenarios file — used to scope
the perturbation (jailbreak-prefix) run down to what matters, instead of re-running
the full scenario set a second time.

Usage:
  python3 select_high_harmful_scenarios.py \
      --labeled ../rollouts/qwen14b_pilot_v2_labeled.jsonl \
      --scenarios ../scenarios/scenarios.json \
      --min_harmful 1 \
      --output ../scenarios/scenarios_for_perturbation.json
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, nargs="+",
                     help="One or more labeled rollout files (space-separated) to pool counts from.")
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--min_harmful", type=int, default=1,
                     help="Minimum total Harmful-state visits (summed across all --labeled files "
                          "and samples) for a scenario to be included.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.scenarios) as f:
        scenario_meta = {s["id"]: s for s in json.load(f)["scenarios"]}

    harmful_counts = defaultdict(int)
    for labeled_path in args.labeled:
        with open(labeled_path) as f:
            for line in f:
                rec = json.loads(line)
                for turn in rec["transcript"]:
                    if turn["state"] == "Harmful":
                        harmful_counts[rec["scenario_id"]] += 1

    selected_ids = [sid for sid, count in harmful_counts.items() if count >= args.min_harmful]
    selected = [scenario_meta[sid] for sid in selected_ids if sid in scenario_meta]

    print(f"{len(selected)}/{len(scenario_meta)} scenarios kept (Harmful visits >= {args.min_harmful}):")
    for s in sorted(selected, key=lambda x: -harmful_counts[x["id"]]):
        print(f"  {s['id']:10s} ({s['domain']:28s}/{s['category']:16s})  Harmful visits: {harmful_counts[s['id']]}")

    if not selected:
        print("\n[ERROR] No scenarios met the threshold. Lower --min_harmful or check your --labeled path.")
        return

    with open(args.output, "w") as f:
        json.dump({"_meta": {"note": f"Auto-selected subset, min_harmful={args.min_harmful}",
                              "n_scenarios": len(selected)},
                    "scenarios": selected}, f, indent=2)
    print(f"\nWrote {len(selected)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
