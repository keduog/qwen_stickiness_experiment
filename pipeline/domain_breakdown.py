"""
Breaks down state visits by domain and trajectory category, so scenario expansion can
target what's actually working instead of scaling everything uniformly.

Usage:
  python3 domain_breakdown.py --labeled ../rollouts/qwen14b_pilot_labeled.jsonl \
      --scenarios ../scenarios/scenarios.json
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--scenarios", required=True)
    args = ap.parse_args()

    with open(args.scenarios) as f:
        scenario_meta = {s["id"]: s for s in json.load(f)["scenarios"]}

    by_domain = defaultdict(lambda: defaultdict(int))
    by_category = defaultdict(lambda: defaultdict(int))
    by_scenario = defaultdict(lambda: defaultdict(int))

    with open(args.labeled) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["scenario_id"]
            meta = scenario_meta.get(sid, {})
            domain = meta.get("domain", "UNKNOWN")
            category = meta.get("category", "UNKNOWN")
            for turn in rec["transcript"]:
                state = turn["state"]
                by_domain[domain][state] += 1
                by_category[category][state] += 1
                by_scenario[sid][state] += 1

    print("=== By domain ===")
    for domain, counts in sorted(by_domain.items()):
        total = sum(counts.values())
        harmful = counts.get("Harmful", 0)
        print(f"{domain:30s} total={total:4d}  Harmful={harmful:3d} ({100*harmful/total:.1f}%)  "
              f"Unknown={counts.get('Unknown', 0)}")

    print("\n=== By trajectory category ===")
    for category, counts in sorted(by_category.items()):
        total = sum(counts.values())
        harmful = counts.get("Harmful", 0)
        print(f"{category:20s} total={total:4d}  Harmful={harmful:3d} ({100*harmful/total:.1f}%)  "
              f"Unknown={counts.get('Unknown', 0)}")

    print("\n=== Top individual scenarios by Harmful visits ===")
    ranked = sorted(by_scenario.items(), key=lambda kv: kv[1].get("Harmful", 0), reverse=True)
    for sid, counts in ranked[:15]:
        meta = scenario_meta.get(sid, {})
        print(f"{sid:10s} ({meta.get('domain','?'):28s} / {meta.get('category','?'):16s})  "
              f"Harmful={counts.get('Harmful', 0)}  Unknown={counts.get('Unknown', 0)}  "
              f"total={sum(counts.values())}")


if __name__ == "__main__":
    main()