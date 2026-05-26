# TDCG - Test Driven Code Generation
## Coding Hypothesis — Test-Repair Loop Ablation

Hypothesis: small coding models become much more effective when wrapped in a
test-execution-repair loop. We measure how much of agentic coding performance
comes from execution feedback vs raw one-shot model intelligence.

## Modes
| Mode | Tests              | Execution | Measures                          |
|------|--------------------|-----------|-----------------------------------|
| A    | none               | no        | raw codegen                       |
| B    | model-written      | no        | "test thinking" alone             |
| C    | public (reliable)  | yes       | feedback use with reliable tests  |
| D    | model-written      | yes       | full self-loop (the hypothesis)   |
| D_sep| model-written      | yes       | solution first, tests second      |
| D_dual | model-written    | yes       | code model + test model           |
| D_val | validated model-written | yes   | code model + test model + validator + optional reference oracle |
| E    | public + model     | yes       | practical ceiling                 |

A and B run with `k=1` (no iteration). C/D/D_sep/D_dual/D_val/E iterate up to `k`.

## Key decomposition
```
C - A = execution-feedback value (with reliable tests)
D - A = self-loop value
D_sep - A = separated self-loop value
D_dual - A = two-model self-loop value
D_val - A = validated self-loop value
C - D = test-writing weakness
D_val - D_dual = test-validation value
E - C = marginal value of model-written tests on top of public
```

## Layout
```
tasks/
  task_001_sum_evens/
    prompt.md           # spec shown to the model
    solution.py         # starter the model edits
    public_tests.py     # visible feedback (modes C, E)
    hidden_tests.py     # final scoring only — never enters the sandbox
    reference_solution.py # optional trusted oracle for validating self_tests.py
  ...
harness/
  sandbox.py            # tempdir + subprocess pytest
  models.py             # Ollama client
  agent.py              # per-mode prompts + iteration loop
  log.py                # JSONL records
  plot.py               # matplotlib plots from JSONL
run.py                  # CLI
results/
  runs.jsonl            # one line per run (appended forever)
  plots/<batch_tag>/    # auto-saved plots per run-batch
```

## Plots per batch
Every `run.py` invocation tags rows with `batch_tag` (default = timestamp) and
saves PNGs to `results/plots/<batch_tag>/`:

- `pass_rate_by_mode_k.png` — bar chart, hidden pass rate per (mode, k)
- `baseline_vs_tests.png` — headline comparison: no tests (A/k=1) vs public-test feedback (C/k=max)
- `baseline_vs_self_tests.png` — headline comparison: no tests (A/k=1) vs model-written self-test loop (D/k=max)
- `baseline_vs_separate_self_tests.png` — headline comparison: no tests (A/k=1) vs separated self-test loop (D_sep/k=max)
- `baseline_vs_dual_self_tests.png` — headline comparison: no tests (A/k=1) vs two-model self-test loop (D_dual/k=max)
- `baseline_vs_validated_self_tests.png` — headline comparison: no tests (A/k=1) vs validated self-test loop (D_val/k=max)
- `pass_rate_vs_iterations.png` — k-sweep line plot
- `repair_outcomes.png` — when `--score-hidden-each-iter` is enabled, counts hidden-fail→hidden-pass repairs
- `overfit_rate_by_mode.png` — visible pass + hidden fail (kill metric)
- `tokens_vs_pass.png` — compute spent vs outcome
- `delta_by_model_size.png` — only when JSONL has ≥2 models
- `per_task_heatmap.png` — task × mode/k grid
- `summary.json` — counts + filter info
- `ablation_summary.json` — paired A-vs-test outcomes plus self-test/hidden confusion matrix

Regenerate anytime: `python -m harness.plot --batch-tag <tag>`
Across all batches: `python -m harness.plot --out results/plots/all`
Skip auto-plot: `--no-plot`

## Quick start

```bash
# 1. start Ollama and pull the model
ollama pull qwen2.5-coder:1.5b

# 2. dry-run wiring without any model calls
python run.py --model qwen2.5-coder:1.5b --all --modes A,C --ks 1,3,5 --dry-run

# 3. real pilot: Qwen-1.5B, modes A + C, k=1,3,5, all 3 tasks (= 12 runs)
python run.py --model qwen2.5-coder:1.5b --all --modes A,C --ks 1,3,5

# 4. inspect
cat results/runs.jsonl
```

## Official benchmarks

Don't use plain HumanEval/MBPP for main proof — contamination + weak tests. Use
the stack below (phase 1 supported by current harness; phase 2 needs repo-level
extensions).

| Benchmark | Use for | Loader |
|-----------|---------|--------|
| **HumanEval / MBPP** | Sanity check only | `humaneval`, `mbpp`, `mbpp_san` |
| **EvalPlus HumanEval+ / MBPP+** | First real eval (80x / 35x stronger tests) | `humaneval_plus`, `mbpp_plus` |
| **LiveCodeBench** | Contamination-resistant main bench (time-window) | `livecodebench` |
| **BigCodeBench-Hard** | Realistic function/library-use (148 hard tasks) | `bigcodebench_hard` |
| BigCodeBench full | 1140 tasks | `bigcodebench` |
| SWE-bench Pro Public | Repo-level (731 tasks) | `swebench_pro` |
| SWE-bench-Live | Monthly fresh repo-level | `swebench_live_lite` / `_verified` / `_full` |
| SWT-Bench | Test-generation on real bugs | `swtbench_lite` / `swtbench_verified` |
| Terminal-Bench | Shell tool-use control (80 tasks) | `terminal_bench` |

### Materialize + run

```bash
# EvalPlus
python -m harness.load_benchmark --name humaneval_plus
python run.py --model qwen2.5-coder:1.5b --benchmark humaneval_plus --modes A,C --ks 1,3,5 --limit 50

# Self-test ablation with repeated seeds.
# At temperature 0, repeated seeds may produce identical candidates; use a small
# positive temperature when you want seed-to-seed variation.
python run.py \
  --model qwen2.5-coder:7b \
  --test-model qwen2.5-coder:7b \
  --validator-model gemma4:e2b \
  --benchmark humaneval_plus \
  --modes A,D,D_sep,D_dual,D_val \
  --ks 1,3,5 \
  --limit 164 \
  --hidden-timeout 120 \
  --score-hidden-each-iter \
  --temperature 0.2 \
  --seeds 1,2,3,4,5 \
  --batch-tag humaneval_plus_repair_qwen7b_full

# Two-model self-test agent only:
# code model writes and repairs solution.py; test model writes frozen self_tests.py.
python run.py \
  --model qwen2.5-coder:7b \
  --test-model qwen2.5-coder:7b \
  --benchmark humaneval_plus \
  --modes A,D_dual \
  --ks 1,3 \
  --limit 20 \
  --hidden-timeout 120 \
  --score-hidden-each-iter \
  --batch-tag humaneval_plus_dual_qwen7b_test7b_20

# Validated self-test agent:
# code model writes/repairs solution.py; test model writes self_tests.py;
# validator model rejects weak or invalid tests before execution.
# If reference_solution.py exists, D_val also runs self_tests.py against it
# and rejects tests with wrong expected values.
python run.py \
  --model qwen2.5-coder:7b \
  --test-model qwen2.5-coder:7b \
  --validator-model gemma4:e2b \
  --benchmark humaneval_plus \
  --modes A,D_val \
  --ks 1,3 \
  --limit 20 \
  --hidden-timeout 120 \
  --score-hidden-each-iter \
  --batch-tag humaneval_plus_validated_qwen7b_test7b_gemma_validator_20

# Longer full-validation runs can parallelize independent cases and raise the
# Ollama HTTP timeout separately from pytest hidden-test timeout.
python run.py \
  --model qwen2.5-coder:7b \
  --test-model qwen2.5-coder:7b \
  --validator-model gemma4:e2b \
  --repair-model gemma4:e2b \
  --benchmark humaneval_plus \
  --modes A,C,D_val \
  --ks 3 \
  --hidden-timeout 360 \
  --model-timeout 600 \
  --repair-model-timeout 240 \
  --jobs 8 \
  --score-hidden-each-iter \
  --save-artifacts \
  --self-test-candidates 3 \
  --code-candidates 2 \
  --repair-candidates 3 \
  --max-bash-calls 40 \
  --log results/humaneval_plus_full_v3.jsonl \
  --batch-tag humaneval_plus_full_v3 \
  --resume

# Cheaper D_val profile for cost/timeout-control experiments. This keeps the
# same A/C/D_val comparison shape but uses one self-test suite, one code
# candidate, one repair candidate, max 12 bash calls, and 180s repair timeout.
python run.py \
  --model qwen2.5-coder:7b \
  --test-model qwen2.5-coder:7b \
  --validator-model gemma4:e2b \
  --repair-model gemma4:e2b \
  --benchmark humaneval_plus \
  --modes A,C,D_val \
  --ks 3 \
  --hidden-timeout 360 \
  --model-timeout 600 \
  --jobs 8 \
  --score-hidden-each-iter \
  --cheap-dval \
  --log results/humaneval_plus_cheap_dval_v1.jsonl \
  --batch-tag humaneval_plus_cheap_dval_v1 \
  --resume

# Portfolio selector: choose between completed C/k=3 and D_val/k=3 artifacts
# using only visible signals, then score the selected solution as P_select.
python -m harness.portfolio_select \
  --log results/humaneval_plus_full_v3.jsonl \
  --batch-tag humaneval_plus_full_v3 \
  --out-log results/humaneval_plus_full_v3_portfolio.jsonl \
  --out-batch-tag humaneval_plus_full_v3_portfolio \
  --benchmark humaneval_plus \
  --score-policy rescore \
  --hidden-timeout 360 \
  --save-artifacts \
  --resume

# LiveCodeBench — pick problems after the model's likely training cutoff
python -m harness.load_benchmark --name livecodebench --since 2024-06-01 --difficulty easy --limit 50
python run.py --model qwen2.5-coder:1.5b --benchmark livecodebench --modes A,C --ks 1,3,5

# BigCodeBench-Hard
python -m harness.load_benchmark --name bigcodebench_hard
python run.py --model qwen2.5-coder:1.5b --benchmark bigcodebench_hard --modes A,C --ks 1,3,5 --limit 30
```

### Verify materialized benchmark

After materializing, run the canonical solutions through public + hidden tests
to confirm the loader produced a correct harness:

```bash
python -m harness.smoke_bench --name humaneval_plus --limit 20 --timeout 60
python -m harness.smoke_bench --name mbpp_plus     --limit 20 --timeout 60
```

(LiveCodeBench does not ship canonical solutions; the loader writes a starter
and pipes test inputs via subprocess.)

## Phase 2: repo-level + shell benchmarks (Docker required)

These benchmarks operate on real repositories or shell sessions. The current
Python pytest sandbox is not sufficient; execution uses Docker (and tmux for
Terminal-Bench). Loaders are implemented; the diff-generating agent + scoring
wrapper are next.

### Sanity: Docker primitive

```bash
python -c "from harness.docker_sandbox import DockerSandbox, ensure_image; \
  ensure_image('alpine:3'); sb = DockerSandbox(image='alpine:3'); \
  r = sb.run(['sh','-c','echo hello']); print(r.returncode, r.stdout); sb.cleanup()"
```

### Materialize SWE-bench / SWT-Bench instances

```bash
# SWE-bench Pro Public (731 tasks)
python -m harness.load_swebench --variant swebench_pro --limit 50

# SWE-bench-Live lite (300 monthly-fresh tasks)
python -m harness.load_swebench --variant swebench_live_lite

# SWT-Bench Lite (300 instances — same SWE-bench Lite repo space)
python -m harness.load_swebench --variant swtbench_lite
```

Each instance becomes a directory with `prompt.md`, `instance.json`, `gold.diff`.
Aggregated `instances.jsonl` is written to the variant root.

### Materialize Terminal-Bench

```bash
python -m harness.load_tbench --limit 20
```

Writes thin pointer dirs (`tbench_path.txt`) into `tasks_bench/terminal_bench/`.
Source assets stay in the official `terminal-bench-core` cache.

### Scoring (TODO — wire to official runners)

- SWE-bench / Pro / Live / SWT: invoke
  `python -m swebench.harness.run_evaluation --predictions_path … --dataset_name … --split …`
- Terminal-Bench: invoke `tb run --task <id> --agent <wrapper>`

Diff-generating agent for these (mode A vs C semantics on patch generation)
is the next milestone.

## Budget (locked, equal across modes)
- max iterations: from `--ks`
- max bash calls per run: 20 by default (`--max-bash-calls`)
- visible public/self pytest timeout: 10s (`--pytest-timeout`)
- final hidden benchmark scoring timeout: 60s (`--hidden-timeout`)
- model HTTP request timeout: 120s (`--model-timeout`)
- optional repair-only HTTP timeout: defaults to `--model-timeout`, override with `--repair-model-timeout`
- repair model errors are fail-fast by default and preserve the current solution; use `--continue-repair-after-model-error` to keep trying
- cheap validated profile: `--cheap-dval`
- visible-only portfolio selector: `python -m harness.portfolio_select`
- parallel independent cases: 1 worker (`--jobs`)
- optional per-iteration hidden scoring for repair analysis: `--score-hidden-each-iter` (never shown to the model)
- dual/validated skill files: `agent_skills/code_writer/skills.md`, `agent_skills/test_writer/skills.md`, and `agent_skills/test_validator/skills.md`
- temperature: 0
- model sampling seed: 0 by default (`--seed` or repeated with `--seeds`)
- max output tokens per call: 4096

## Gate
If `C(k=5) - A(k=1) >= 15 percentage points` over the pilot set, expand to
modes D/D_sep/D_dual/D_val + E and scale to more tasks / model sizes.

## Results

HumanEval+ (164 tasks), `qwen2.5-coder:7b`, batch `humaneval_plus_full_v3`:

| Mode      | k | Hidden pass | Pass rate | Δ vs A/k=1 |
|-----------|---|-------------|-----------|------------|
| A         | 1 | 119/164     | 72.56%    | —          |
| C         | 3 | 139/164     | 84.76%    | +3.66 pp   |
| D_val     | 3 | 141/164     | 85.98%    | +4.88 pp   |

See [RESULTS.md](RESULTS.md) for run command and provenance.
