# Results

## HumanEval+ — `qwen2.5-coder:7b`

Batch tag: `humaneval_plus_full_v3`
Tasks: 164 (full EvalPlus HumanEval+)
Code/test/validator/repair model: `qwen2.5-coder:7b` (validator/repair: `gemma4:e2b`)
Temperature: 0 · seed: 0 · k: 3

| Mode    | k | Hidden pass | Pass rate | Δ vs A/k=1 |
|---------|---|-------------|-----------|------------|
| A       | 1 | 119/164     | 72.56%    | —          |
| C       | 3 | 139/164     | 84.76%    | +3.66 pp   |
| D_val   | 3 | 141/164     | 85.98%    | +4.88 pp   |

### Decomposition

- `C − A` = execution-feedback value with reliable public tests: **+3.66 pp** (6 tasks).
- `D_val − A` = validated self-loop value: **+4.88 pp** (8 tasks).
- `D_val − C` = marginal value of validated model-written tests on top of public tests: **+1.22 pp** (2 tasks).

### Run command

```bash
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
```

### Notes

- Hidden tests scored in an isolated tempdir; `hidden_tests.py` never enters the model-visible sandbox.
- `D_val` self-tests gated by static lint + reference-solution oracle + LLM validator + prune/fallback.
- Raw per-run rows live in `results/humaneval_plus_full_v3.jsonl` (gitignored).
