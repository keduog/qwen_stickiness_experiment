# Harmful-State Stickiness in Multi-Turn Conversation: Qwen3-14B

**Research question:** Does entering a "harmful" behavioral state in a multi-turn
conversation create a self-reinforcing dynamic (H2), does predicting it require more
than the immediately preceding turn (H3), and does that dynamic survive benign
rephrasing but break under adversarial framing (robustness)?

Single model (Qwen3-14B), one sharp hypothesis with three linked, falsifiable claims,
real controls (permutation tests, held-out scenario splits, bootstrap CIs) — scoped
for an arXiv/workshop submission, not the full 3-model/6-phase program.

## Directory layout

```
qwen_stickiness/
├── README.md                          (this file)
├── scenarios/
│   └── scenarios.json                 30 starter scenarios, 6 domains x 5 trajectory
│                                       types each. SCALE THIS UP after the pilot —
│                                       see scenarios.json's "_meta.scaling_note" and
│                                       pipeline/pilot_power_check.py
├── pipeline/
│   ├── generate_rollouts.py           fp16 generation, dual-card, incremental writes
│   ├── label_states.py                Qwen3Guard-Gen labeling (Harmful/Unknown/
│   │                                   Useful/Useless), rule-based quality layer
│   └── pilot_power_check.py           run after first small batch, before scaling up
├── analysis/
│   ├── build_transition_matrix.py     stickiness, dwell times, absorbing states
│   ├── permutation_test.py            H2: is stickiness > chance? (shuffled-label control)
│   ├── markov_order_test.py           H3: order-0 vs 1 vs 2, GroupKFold by scenario
│   └── perturbations.py               robustness: paraphrase vs jailbreak-prefix
├── rollouts/                          (created by pipeline scripts)
├── activations/                       (unused in this scoped version — dropped SAE
│                                       phase, see "What was cut" below)
├── logs/
├── transition_matrices/
└── figures/
```

All four analysis scripts were tested against synthetic data with a known planted
effect before being handed to you — see the test transcript in this conversation. They
correctly recovered the planted stickiness value, correctly identified the planted
Markov order, and correctly detected the planted condition difference. That's not the
same as validating them on real model output, but it does mean the statistics
themselves are implemented correctly.

## What was cut from the original 6-phase, 3-model proposal, and why

- **Llama 3 8B and Mistral 7B comparisons**: cut for scope. A single-model study with
  real rigor is more publishable than a 3-model study that's thin everywhere. If this
  works, the natural follow-up paper is "does this generalize across model families" —
  a good problem to have, not one to try to solve in the same paper.
- **SAE mapping (original Phase 4)**: cut entirely from this scoped version. SAE
  tooling compatibility with your Iluvatar hardware is unverified, and mechanistic
  explanation of *why* stickiness happens is a separate, harder claim than *whether* it
  happens. Establish the behavioral finding first.
- **Hankel rank estimation (original Phase 5)**: cut. It answers "how many latent
  states are really needed," which is a richer question than H2/H3 but needs more data
  than a scoped single-model study can gather with good power. Worth revisiting once
  you know the scenario set reliably produces harmful-state visits.
- **Explicit nondeterminism test (original Phase 6)**: partially preserved — the
  Unknown-state stickiness result from `permutation_test.py` speaks to this, but the
  dedicated 20-continuations-per-history entropy analysis is cut for scope.

## Order of operations

### 1. Environment setup (on the Iluvatar instance)

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p rollouts activations logs transition_matrices figures
pip install transformers accelerate huggingface_hub scipy numpy --break-system-packages
python3 -c "import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

### 2. Pilot run — DO NOT skip this

Run the full pipeline on just the 30 starter scenarios first:

```bash
cd pipeline
python3 generate_rollouts.py --model_path Qwen/Qwen3-14B \
    --scenarios ../scenarios/scenarios.json \
    --output ../rollouts/qwen14b_pilot.jsonl \
    --device_index 0 --n_samples 3

python3 label_states.py --rollouts ../rollouts/qwen14b_pilot.jsonl \
    --output ../rollouts/qwen14b_pilot_labeled.jsonl \
    --guard_model Qwen/Qwen3Guard-Gen-4B --device_index 1

python3 pilot_power_check.py --labeled ../rollouts/qwen14b_pilot_labeled.jsonl
```

Read the power-check output before doing anything else. If Harmful (or any state)
is underpowered, the fix is targeted scenario expansion (more of the trajectory
category that actually produces that state), not just cranking up `--n_samples`
on the existing 30.

### 3. Scale the scenario set

Based on the pilot, expand `scenarios.json` — likely to 120-180 scenarios total,
weighted toward whichever domains/categories the pilot showed actually drift into
Harmful territory. Keep the same JSON schema so nothing downstream needs to change.

### 4. Full baseline run

Same two commands as step 2, pointed at the expanded scenario set and a new output
path (e.g. `qwen14b_full_baseline.jsonl`).

### 5. Perturbation conditions

```bash
python3 generate_rollouts.py --model_path Qwen/Qwen3-14B \
    --scenarios ../scenarios/scenarios_full.json \
    --output ../rollouts/qwen14b_jailbreak.jsonl \
    --device_index 0 --n_samples 3 --perturbation jailbreak_prefix
# then label_states.py on that output, same as before
```

The `paraphrase` perturbation in `generate_rollouts.py` is currently a no-op
placeholder (`PERTURBATIONS` dict) — for a real paraphrase condition you need an
actual paraphraser (a second small model, or a rule-based synonym+reorder system).
Hand-written paraphrase rules are defensible for a workshop paper if you document
them; an LLM-based paraphraser is more robust if you have the budget for it.

### 6. Analysis

```bash
cd ../analysis
python3 build_transition_matrix.py --labeled ../rollouts/qwen14b_full_baseline_labeled.jsonl \
    --output ../transition_matrices/baseline.json

python3 permutation_test.py --labeled ../rollouts/qwen14b_full_baseline_labeled.jsonl \
    --n_permutations 5000

python3 markov_order_test.py --labeled ../rollouts/qwen14b_full_baseline_labeled.jsonl \
    --n_folds 5

python3 perturbations.py \
    --baseline ../rollouts/qwen14b_full_baseline_labeled.jsonl \
    --perturbed ../rollouts/qwen14b_jailbreak_labeled.jsonl \
    --condition_name jailbreak_prefix --n_permutations 5000
```

## Things to watch for that would undermine publishability

- **`needs_review` and `Unparsed` entries from `label_states.py`**: don't silently
  drop these. If more than ~5% of turns are unparsed, the guard regex isn't matching
  actual output format reliably — check `Qwen3Guard-Gen`'s exact output against the
  regex before trusting anything downstream.
- **Multiple comparisons**: `permutation_test.py` tests all 4 states. If you report
  all 4 p-values, correct for multiple comparisons (Bonferroni or similar) rather than
  cherry-picking the significant one.
- **Circularity in scenario/rubric design**: because you wrote the escalation scripts
  to plausibly produce harmful states, a skeptical reviewer will ask whether you just
  built a self-fulfilling prophecy. The permutation test controls for "is stickiness
  more than the marginal rate," but it doesn't control for "did you cherry-pick
  scenarios that only show stickiness." Consider holding out a subset of scenarios
  that weren't looked at during scenario-set iteration, purely for the final reported
  numbers.
- **Single-seed generation determinism**: `generate_rollouts.py` sets a global seed,
  but sampling with `do_sample=True` is still stochastic per call on GPU — don't claim
  exact reproducibility, report it as "same seed, same code, same hardware" rather
  than bit-exact.
