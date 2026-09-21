"""Dump raw transcripts for manual spot-checking of classifier validity on a domain."""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--labeled", required=True)
ap.add_argument("--scenario_id", required=True)
ap.add_argument("--sample_idx", type=int, default=0)
args = ap.parse_args()

with open(args.labeled) as f:
    for line in f:
        rec = json.loads(line)
        if rec["scenario_id"] == args.scenario_id and rec["sample_idx"] == args.sample_idx:
            for turn in rec["transcript"]:
                print(f"\n--- Turn {turn['turn_index']} | state={turn['state']} "
                      f"(guard raw: {turn.get('guard_safety_raw')}, refusal: {turn.get('guard_refusal_raw')}) ---")
                print(f"USER: {turn['user_text']}")
                print(f"MODEL: {turn['model_text'][:500]}")
            break
    else:
        print("Not found")