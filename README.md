# Coding Hypothesis — Test-Repair Loop Ablation

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
| E    | public + model     | yes       | practical ceiling                 |

A and B run with `k=1` (no iteration). C/D/E iterate up to `k`.

## Key decomposition
```
C - A = execution-feedback value (with reliable tests)
D - A = self-loop value
C - D = test-writing weakness
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
  --benchmark humaneval_plus \
  --modes A,D \
  --ks 1,3,5 \
  --limit 164 \
  --hidden-timeout 120 \
  --score-hidden-each-iter \
  --temperature 0.2 \
  --seeds 1,2,3,4,5 \
  --batch-tag humaneval_plus_repair_qwen7b_full

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
- max bash calls per run: 10
- visible public/self pytest timeout: 10s (`--pytest-timeout`)
- final hidden benchmark scoring timeout: 60s (`--hidden-timeout`)
- optional per-iteration hidden scoring for repair analysis: `--score-hidden-each-iter` (never shown to the model)
- temperature: 0
- model sampling seed: 0 by default (`--seed` or repeated with `--seeds`)
- max output tokens per call: 4096

## Gate
If `C(k=5) - A(k=1) >= 15 percentage points` over the pilot set, expand to
modes D + E and scale to more tasks / model sizes.
