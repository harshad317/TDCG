"""Download official coding benchmarks and materialize them as tasks/ entries.

Supports:
  humaneval — 164 problems (OpenAI HumanEval, MIT license)
  mbpp      — 974 problems (Google MBPP, CC-BY 4.0)
  mbpp_san  — 427-problem sanitized subset

After running, each problem becomes a directory under tasks_bench/<bench>/
containing prompt.md, solution.py, public_tests.py, hidden_tests.py, and
reference_solution.py when the benchmark ships a canonical solution — fully
compatible with run.py.

Usage:
  python -m harness.load_benchmark --name humaneval --limit 50
  python -m harness.load_benchmark --name mbpp --limit 100 --out tasks_bench
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import sys
import os

# Some EvalPlus / LCB problems produce ints far past Python 3.11's default
# 4300-digit str-conversion cap (factorials, large powers). Bump it up front so
# `repr(expected)` doesn't blow up during materialization.
try:
    sys.set_int_max_str_digits(1_000_000)
except AttributeError:
    pass
import json
import pickle
import re
import subprocess
import urllib.request
from pathlib import Path

HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
MBPP_SAN_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "coding-hypothesis/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_humaneval() -> list[dict]:
    raw = _http_get(HUMANEVAL_URL)
    text = gzip.decompress(raw).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def fetch_mbpp() -> list[dict]:
    text = _http_get(MBPP_URL).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def fetch_mbpp_sanitized() -> list[dict]:
    return json.loads(_http_get(MBPP_SAN_URL).decode("utf-8"))


# ----------------------------------------------------------------------------- humaneval


def _humaneval_entry_point(prompt: str) -> str:
    """Pull the function name out of the prompt's `def ...(` line."""
    m = re.search(r"^def\s+([A-Za-z_]\w*)\s*\(", prompt, re.MULTILINE)
    return m.group(1) if m else "candidate"


def _humaneval_split_tests(test_src: str, entry_point: str) -> tuple[str, str]:
    """HumanEval test files define `def check(candidate): assert ...; assert ...`.

    We turn each top-level assert into its own pytest function. The first 1
    becomes the public test; the rest become hidden.
    """
    lines = [ln for ln in test_src.splitlines()]
    asserts: list[list[str]] = []
    cur: list[str] = []
    in_check = False
    base_indent = None
    for ln in lines:
        stripped = ln.lstrip()
        if not in_check:
            if stripped.startswith("def check("):
                in_check = True
            continue
        if stripped == "":
            if cur:
                cur.append(ln)
            continue
        indent = len(ln) - len(stripped)
        if base_indent is None and stripped:
            base_indent = indent
        # leaving the check function?
        if indent < (base_indent or 0):
            break
        if stripped.startswith("assert "):
            if cur:
                asserts.append(cur)
            cur = [ln]
        else:
            if cur:
                cur.append(ln)
    if cur:
        asserts.append(cur)

    def dedent_block(block: list[str]) -> str:
        # Strip the leading indent so each assertion sits at module level.
        if not block:
            return ""
        leading = len(block[0]) - len(block[0].lstrip())
        return "\n".join(line[leading:] if len(line) >= leading else line for line in block)

    def remap(block: str) -> str:
        # HumanEval test bodies reference `candidate`; rewrite to the real fn.
        return re.sub(r"\bcandidate\b", entry_point, block)

    pub_assert = remap(dedent_block(asserts[0])) if asserts else f"assert {entry_point}"
    hid_asserts = [remap(dedent_block(a)) for a in asserts] if asserts else []

    pub_src = (
        f"from solution import {entry_point}\n\n\n"
        f"def test_public_0():\n"
        + _indent_lines(pub_assert)
        + "\n"
    )
    hid_body_parts = []
    for i, a in enumerate(hid_asserts):
        hid_body_parts.append(f"def test_hidden_{i}():\n" + _indent_lines(a))
    hid_src = f"from solution import {entry_point}\n\n\n" + "\n\n\n".join(hid_body_parts) + "\n"
    return pub_src, hid_src


def _indent_lines(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + ln if ln else ln for ln in text.splitlines())


def materialize_humaneval(out_root: Path, limit: int | None = None) -> int:
    problems = fetch_humaneval()
    if limit is not None:
        problems = problems[:limit]
    n = 0
    for p in problems:
        task_id = p["task_id"].replace("/", "_")  # "HumanEval/0" -> "HumanEval_0"
        entry = p.get("entry_point") or _humaneval_entry_point(p["prompt"])
        task_dir = out_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        starter = p["prompt"] + "    raise NotImplementedError\n"
        prompt_md = (
            f"# Task: {task_id} ({entry})\n\n"
            f"Implement the function described by the docstring in `solution.py`.\n\n"
            f"```python\n{p['prompt'].rstrip()}\n```\n\n"
            "A starter file `solution.py` is provided. Public tests are in "
            "`public_tests.py` — run them with `python -m pytest public_tests.py`.\n"
        )
        pub_tests, hid_tests = _humaneval_split_tests(p["test"], entry)

        (task_dir / "prompt.md").write_text(prompt_md)
        (task_dir / "solution.py").write_text(starter)
        (task_dir / "public_tests.py").write_text(pub_tests)
        (task_dir / "hidden_tests.py").write_text(hid_tests)
        (task_dir / "reference_solution.py").write_text(p["prompt"] + p["canonical_solution"])
        n += 1
    return n


# ----------------------------------------------------------------------------- mbpp


_MBPP_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _mbpp_entry_from_tests(test_list: list[str]) -> str | None:
    """Find the function name being tested by inspecting the first assertion."""
    if not test_list:
        return None
    for t in test_list:
        # strip 'assert' prefix; look at first call expression
        body = re.sub(r"^\s*assert\s+", "", t.strip())
        # skip 'math.isclose(' style by preferring the first plain identifier
        for m in _MBPP_CALL_RE.finditer(body):
            name = m.group(1)
            if name in {"abs", "len", "set", "list", "dict", "tuple", "sorted",
                        "math", "isclose"}:
                continue
            return name
    return None


def _mbpp_signature(code: str, target_name: str | None = None) -> tuple[str, str]:
    """Return (function_name, starter). Prefer the def whose name matches the
    target (extracted from the test assertions), so we skip helper functions.
    """
    defs = list(re.finditer(r"^def\s+([A-Za-z_]\w*)\s*\([^)]*\):", code, re.MULTILINE))
    chosen = None
    if target_name:
        for m in defs:
            if m.group(1) == target_name:
                chosen = m
                break
    if chosen is None and defs:
        # fall back to the last def (often the public entry point in MBPP)
        chosen = defs[-1]
    if chosen is None:
        name = target_name or "candidate"
        return name, f"def {name}(*args, **kwargs):\n    raise NotImplementedError\n"
    name = chosen.group(1)
    sig = chosen.group(0)
    starter = f"{sig}\n    # TODO: implement\n    raise NotImplementedError\n"
    return name, starter


def _mbpp_make_tests(test_list: list[str], entry_point: str, setup: str = "") -> tuple[str, str]:
    """Convert MBPP's `assert ...` strings into pytest functions.

    First assertion -> public test. All assertions -> hidden tests. If the
    function name starts with `test_`, pytest would auto-collect it; we alias
    on import and rewrite the assertions to use the alias.
    """
    setup_block = (setup + "\n") if setup.strip() else ""
    if entry_point.startswith("test_"):
        alias = f"_impl_{entry_point}"
        import_line = f"from solution import {entry_point} as {alias}\n"
        pattern = re.compile(rf"\b{re.escape(entry_point)}\b")
        rewritten = [pattern.sub(alias, t) for t in test_list]
    else:
        import_line = f"from solution import {entry_point}\n"
        rewritten = list(test_list)

    pub_body = rewritten[0] if rewritten else "pass"
    pub_src = (
        f"{import_line}"
        f"{setup_block}\n"
        f"def test_public_0():\n"
        f"    {pub_body}\n"
    )
    hid_parts = [f"def test_hidden_{i}():\n    {t}" for i, t in enumerate(rewritten)]
    hid_src = (
        f"{import_line}"
        f"{setup_block}\n"
        + "\n\n\n".join(hid_parts)
        + "\n"
    )
    return pub_src, hid_src


def _materialize_mbpp_records(records: list[dict], out_root: Path, limit: int | None) -> int:
    if limit is not None:
        records = records[:limit]
    n = 0
    for p in records:
        # Records have either {task_id, text, code, test_list, test_setup_code}
        # (raw MBPP) or {task_id, prompt, code, test_list, test_imports} (sanitized).
        task_id = f"MBPP_{p['task_id']}"
        code = p["code"]
        tests = p["test_list"]
        target = _mbpp_entry_from_tests(tests)
        entry, starter = _mbpp_signature(code, target_name=target)
        text = p.get("text") or p.get("prompt") or ""
        setup = p.get("test_setup_code") or "\n".join(p.get("test_imports") or [])

        prompt_md = (
            f"# Task: {task_id}\n\n"
            f"{text.strip()}\n\n"
            "Implement the function in `solution.py`. Public tests are in "
            "`public_tests.py` — run them with `python -m pytest public_tests.py`.\n"
        )
        pub_tests, hid_tests = _mbpp_make_tests(tests, entry, setup)

        task_dir = out_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.md").write_text(prompt_md)
        (task_dir / "solution.py").write_text(starter)
        (task_dir / "public_tests.py").write_text(pub_tests)
        (task_dir / "hidden_tests.py").write_text(hid_tests)
        (task_dir / "reference_solution.py").write_text(code)
        n += 1
    return n


def materialize_mbpp(out_root: Path, limit: int | None = None) -> int:
    return _materialize_mbpp_records(fetch_mbpp(), out_root, limit)


def materialize_mbpp_sanitized(out_root: Path, limit: int | None = None) -> int:
    return _materialize_mbpp_records(fetch_mbpp_sanitized(), out_root, limit)


# ----------------------------------------------------------------------------- cli


def materialize_humaneval_plus(out_root: Path, limit: int | None = None) -> int:
    return _materialize_evalplus(out_root, kind="humaneval", limit=limit)


def materialize_mbpp_plus(out_root: Path, limit: int | None = None) -> int:
    return _materialize_evalplus(out_root, kind="mbpp", limit=limit)


def _materialize_evalplus(out_root: Path, kind: str, limit: int | None) -> int:
    """EvalPlus distributes (base_input, plus_input, canonical_solution, atol).
    We execute the canonical solution to compute the expected output for each
    input, then emit pytest assertions. Public = all usable base inputs; hidden
    = all base + plus inputs.

    Using only the first base input made public feedback too weak on tasks like
    HumanEval_10, where the first case is just the empty string. Public tests
    should catch basic docstring/example mistakes; EvalPlus plus inputs remain
    the stronger hidden generalization check.
    """
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    import math

    if kind == "humaneval":
        data = get_human_eval_plus()
    elif kind == "mbpp":
        data = get_mbpp_plus()
    else:
        raise ValueError(kind)

    items = list(data.items())
    if limit is not None:
        items = items[:limit]

    n = 0
    skipped = 0
    for task_key, p in items:
        task_id = task_key.replace("/", "_")
        entry = p["entry_point"]
        prompt = p["prompt"]
        canonical_module = prompt + p["canonical_solution"]
        atol = p.get("atol") or 0

        base = list(p.get("base_input") or [])
        plus = list(p.get("plus_input") or [])
        all_inputs = base + plus

        # Run canonical in a child process with a fixed hash seed so generated
        # expected values are reproducible across future pytest subprocesses.
        case_results = _evalplus_run_cases(canonical_module, entry, all_inputs)
        if case_results is None:
            skipped += 1
            continue
        base_cases = []
        plus_cases = []
        for inputs, (ok, value) in zip(base, case_results[:len(base)]):
            if ok:
                base_cases.append((inputs, value))
        for inputs, (ok, value) in zip(plus, case_results[len(base):]):
            if not ok:
                continue
            plus_cases.append((inputs, value))
        cases = base_cases + plus_cases
        if not cases:
            skipped += 1
            continue

        public_cases = base_cases or cases[:1]
        if kind == "mbpp" and not re.search(rf"^def\s+{re.escape(entry)}\s*\(", prompt, re.MULTILINE):
            _, starter_func = _mbpp_signature(p["canonical_solution"], target_name=entry)
            starter = prompt.rstrip() + "\n\n" + starter_func
        else:
            starter = prompt + "    raise NotImplementedError\n"
        prompt_md = (
            f"# Task: {task_id} ({entry})\n\n"
            f"Implement the function described by the docstring in `solution.py`.\n\n"
            f"```python\n{prompt.rstrip()}\n```\n\n"
            "A starter file `solution.py` is provided. Public tests are in "
            "`public_tests.py` — run them with `python -m pytest public_tests.py`.\n"
        )
        pub_src = _format_evalplus_tests(public_cases, entry, atol, prefix="public")
        hid_src = _format_evalplus_tests(cases, entry, atol, prefix="hidden")

        task_dir = out_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.md").write_text(prompt_md)
        (task_dir / "solution.py").write_text(starter)
        (task_dir / "public_tests.py").write_text(pub_src)
        (task_dir / "hidden_tests.py").write_text(hid_src)
        (task_dir / "reference_solution.py").write_text(canonical_module)
        n += 1

    if skipped:
        print(f"  skipped {skipped} tasks (canonical exec failed or no usable inputs)")
    return n


def _evalplus_run_cases(canonical_module: str, entry: str, inputs: list) -> list[tuple[bool, object]] | None:
    script = r"""
import base64
import pickle
import sys

try:
    sys.set_int_max_str_digits(1_000_000)
except AttributeError:
    pass

payload = pickle.loads(base64.b64decode(sys.stdin.buffer.read()))
try:
    ns = {}
    exec(compile(payload["module"], "<canonical>", "exec"), ns)
    fn = ns[payload["entry"]]
    results = []
    for args in payload["inputs"]:
        try:
            results.append((True, fn(*args)))
        except Exception as exc:
            results.append((False, repr(exc)))
    out = {"ok": True, "results": results}
except Exception as exc:
    out = {"ok": False, "error": repr(exc)}

sys.stdout.buffer.write(base64.b64encode(pickle.dumps(out)))
"""
    payload = {
        "module": canonical_module,
        "entry": entry,
        "inputs": inputs,
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=base64.b64encode(pickle.dumps(payload)),
            capture_output=True,
            timeout=60,
            env={
                **os.environ,
                "PYTHONHASHSEED": "0",
                "PYTHONINTMAXSTRDIGITS": "1000000",
            },
        )
    except (subprocess.TimeoutExpired, pickle.PickleError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        decoded = pickle.loads(base64.b64decode(proc.stdout))
    except Exception:
        return None
    if not decoded.get("ok"):
        return None
    return decoded["results"]


def _format_evalplus_tests(cases: list, entry: str, atol: float, prefix: str) -> str:
    if entry.startswith("test_"):
        candidate = f"_candidate_{entry}"
        import_line = f"from solution import {entry} as {candidate}"
    else:
        candidate = entry
        import_line = f"from solution import {entry}"
    lines = [
        "from collections import Counter, defaultdict, deque",
        "import math",
        "import sys",
        "try:",
        "    sys.set_int_max_str_digits(1_000_000)",
        "except AttributeError:",
        "    pass",
        import_line,
        "",
        "inf = math.inf",
        "nan = math.nan",
        f"_ATOL = max(float({atol!r}), 1e-6)",
        "",
        "",
        "def _approx_eq(a, b, atol):",
        "    if isinstance(a, bool) or isinstance(b, bool):",
        "        return a is b",
        "    if isinstance(b, float):",
        "        if math.isnan(b):",
        "            return isinstance(a, float) and math.isnan(a)",
        "        return isinstance(a, (int, float)) and math.isclose(a, b, rel_tol=atol, abs_tol=atol)",
        "    if isinstance(a, float):",
        "        return isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=atol, abs_tol=atol)",
        "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):",
        "        return len(a) == len(b) and all(_approx_eq(x, y, atol) for x, y in zip(a, b))",
        "    if isinstance(a, dict) and isinstance(b, dict):",
        "        return a.keys() == b.keys() and all(_approx_eq(a[k], b[k], atol) for k in a)",
        "    return a == b",
        "",
    ]
    for i, (inp, expected) in enumerate(cases):
        lines.append(f"def test_{prefix}_{i}():")
        lines.append(f"    inp = {inp!r}")
        lines.append(f"    expected = {expected!r}")
        lines.append(f"    actual = {candidate}(*inp)")
        lines.append("    assert _approx_eq(actual, expected, _ATOL)")
        lines.append("")
    return "\n".join(lines)


def materialize_livecodebench(
    out_root: Path,
    limit: int | None = None,
    since: str | None = None,
    until: str | None = None,
    difficulty: str | None = None,
    version: str = "release_v6",
) -> int:
    """LiveCodeBench: contest problems with `contest_date` for time-window
    filtering (contamination control). Mixes stdin/stdout (codeforces) and
    function-call (leetcode) test types.
    """
    import base64
    import pickle
    import zlib
    from datasets import load_dataset

    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=version,
        trust_remote_code=True,
    )

    def decode_private(blob: str) -> list:
        if not blob:
            return []
        try:
            raw = zlib.decompress(base64.b64decode(blob.encode("utf-8")))
            return json.loads(pickle.loads(raw))
        except Exception:
            try:
                return json.loads(zlib.decompress(base64.b64decode(blob)).decode("utf-8"))
            except Exception:
                return []

    n = 0
    skipped = 0
    for row in ds:
        date = (row.get("contest_date") or "")[:10]
        if since and date and date < since:
            continue
        if until and date and date > until:
            continue
        if difficulty and (row.get("difficulty") or "").lower() != difficulty.lower():
            continue
        if limit is not None and n >= limit:
            break

        pub = json.loads(row["public_test_cases"]) if row.get("public_test_cases") else []
        priv = decode_private(row.get("private_test_cases") or "")
        cases = pub + priv
        if not cases:
            skipped += 1
            continue
        testtype = (cases[0].get("testtype") or "stdin").lower()
        if testtype not in ("stdin", "functional"):
            skipped += 1
            continue

        qid = str(row["question_id"]).replace("/", "_").replace(" ", "_")
        task_id = f"LCB_{qid}"
        task_dir = out_root / task_id

        starter = row.get("starter_code") or ""
        if not starter.strip():
            # codeforces / atcoder stdin problems usually have no starter.
            starter = "# Read input from stdin, write answer to stdout.\n"
        if not starter.endswith("\n"):
            starter += "\n"

        prompt_md = (
            f"# Task: {task_id}\n\n"
            f"Platform: {row.get('platform')}  Difficulty: {row.get('difficulty')}  Date: {date}\n\n"
            f"{row['question_content']}\n\n"
            f"Test type: {testtype}\n"
        )
        if testtype == "stdin":
            prompt_md += (
                "\nWrite a Python program in `solution.py` that reads from stdin and "
                "writes to stdout. The harness will run `python solution.py` and pipe "
                "each test case to stdin.\n"
            )
        else:
            prompt_md += (
                "\nA starter signature is provided in `solution.py`. The harness will "
                "import and call your function with the given arguments.\n"
            )

        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.md").write_text(prompt_md)
        (task_dir / "solution.py").write_text(starter)

        # Split: first public case -> public_tests; everything -> hidden_tests.
        first_pub = pub[0] if pub else cases[0]
        all_cases = cases

        if testtype == "stdin":
            pub_src = _lcb_stdin_tests([first_pub], prefix="public")
            hid_src = _lcb_stdin_tests(all_cases, prefix="hidden")
        else:
            entry = row.get("entry_point") or _lcb_extract_entry(starter)
            pub_src = _lcb_functional_tests([first_pub], entry, prefix="public")
            hid_src = _lcb_functional_tests(all_cases, entry, prefix="hidden")
        (task_dir / "public_tests.py").write_text(pub_src)
        (task_dir / "hidden_tests.py").write_text(hid_src)
        n += 1

    if skipped:
        print(f"  skipped {skipped} tasks (no test cases or unknown testtype)")
    return n


def _lcb_extract_entry(starter: str) -> str:
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", starter)
    return m.group(1) if m else "solve"


def _lcb_stdin_tests(cases: list, prefix: str) -> str:
    """Each test runs `python solution.py` with the case's input on stdin and
    compares trimmed stdout to expected output."""
    lines = [
        "import subprocess",
        "from pathlib import Path",
        "",
        "_SOLUTION = str(Path(__file__).parent / 'solution.py')",
        "",
        "",
        "def _run(inp, timeout=10):",
        "    r = subprocess.run(",
        "        ['python', _SOLUTION], input=inp, capture_output=True,",
        "        text=True, timeout=timeout,",
        "    )",
        "    return r.stdout, r.stderr, r.returncode",
        "",
        "",
        "def _normalize(s):",
        "    return '\\n'.join(line.rstrip() for line in s.strip().splitlines())",
        "",
    ]
    for i, c in enumerate(cases):
        inp = c.get("input", "")
        out = c.get("output", "")
        lines.append(f"def test_{prefix}_{i}():")
        lines.append(f"    inp = {inp!r}")
        lines.append(f"    expected = {out!r}")
        lines.append("    stdout, stderr, rc = _run(inp)")
        lines.append("    assert rc == 0, f'exit {rc}: {stderr[-400:]}'")
        lines.append("    assert _normalize(stdout) == _normalize(expected)")
        lines.append("")
    return "\n".join(lines)


def _lcb_functional_tests(cases: list, entry: str, prefix: str) -> str:
    lines = [
        f"from solution import {entry} if {entry!r} != 'Solution' else Solution",
        "",
    ]
    # LCB leetcode-style functional inputs are typically JSON lists of args.
    lines = [
        f"from solution import *  # functional LCB; entry: {entry}",
        "import json",
        "",
        "",
        f"def _call(*args, **kwargs):",
        f"    return {entry}(*args, **kwargs) if '{entry}' in globals() else Solution().{entry}(*args, **kwargs)" if entry != "Solution" else "",
        "",
    ]
    body = []
    for i, c in enumerate(cases):
        raw_in = c.get("input", "")
        raw_out = c.get("output", "")
        try:
            args = json.loads(raw_in) if raw_in.strip().startswith("[") else [json.loads(raw_in)]
        except Exception:
            args = [raw_in]
        try:
            expected = json.loads(raw_out)
        except Exception:
            expected = raw_out
        body.append(f"def test_{prefix}_{i}():")
        body.append(f"    args = {args!r}")
        body.append(f"    expected = {expected!r}")
        body.append(f"    actual = _call(*args)")
        body.append("    assert actual == expected")
        body.append("")
    return "\n".join(lines + body)


def materialize_bigcodebench(out_root: Path, limit: int | None = None, hard: bool = True) -> int:
    """BigCodeBench-Hard (148 tasks) or full BigCodeBench (1140).

    Each problem has complete_prompt + canonical_solution + a unittest TestCase
    class. We emit the prompt as solution.py starter, the full TestCase as
    hidden_tests.py, and the first test method as public_tests.py.
    """
    from datasets import load_dataset

    ds_name = "bigcode/bigcodebench-hard" if hard else "bigcode/bigcodebench"
    ds = load_dataset(ds_name, split="v0.1.4", trust_remote_code=False)
    rows = list(ds)
    if limit is not None:
        rows = rows[:limit]

    n = 0
    skipped = 0
    for r in rows:
        task_id = r["task_id"].replace("/", "_")
        entry = r["entry_point"]
        prompt = r["complete_prompt"]
        test_src = r["test"]

        starter = prompt + "    raise NotImplementedError\n"
        prompt_md = (
            f"# Task: {task_id} ({entry})\n\n"
            f"Implement the function described by the docstring in `solution.py`.\n\n"
            f"```python\n{prompt.rstrip()}\n```\n\n"
            "A starter file `solution.py` is provided. Public tests are in "
            "`public_tests.py` — run them with `python -m pytest public_tests.py`.\n"
        )

        # Hidden = whole TestCase, importing entry from solution.
        hid_src = f"from solution import {entry}\n\n{test_src}\n"

        # Public = isolate first test method by string slicing the class body.
        pub_src = _bcb_first_test(test_src, entry)
        if pub_src is None:
            skipped += 1
            continue

        task_dir = out_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.md").write_text(prompt_md)
        (task_dir / "solution.py").write_text(starter)
        (task_dir / "public_tests.py").write_text(pub_src)
        (task_dir / "hidden_tests.py").write_text(hid_src)
        canonical_solution = r.get("canonical_solution")
        if canonical_solution:
            (task_dir / "reference_solution.py").write_text(prompt + canonical_solution)
        n += 1
    if skipped:
        print(f"  skipped {skipped} tasks (could not extract first test method)")
    return n


def _bcb_first_test(test_src: str, entry: str) -> str | None:
    """Extract the first `def test_*` method from the TestCase class and emit
    a minimal pytest module that runs only it."""
    # Capture everything before `class TestCases` (imports etc) and the class
    # header, plus only the first test method body.
    m_class = re.search(r"^class\s+TestCases?\b.*?:\s*$", test_src, re.MULTILINE)
    if not m_class:
        return None
    pre = test_src[: m_class.end()] + "\n"
    body = test_src[m_class.end():]

    # Find first method whose name starts with test_; include setUp/tearDown if present.
    methods = list(re.finditer(r"^( {4}|\t)def\s+(\w+)\s*\(", body, re.MULTILINE))
    if not methods:
        return None

    def method_start_with_decorators(idx: int) -> int:
        """Walk backwards from methods[idx].start() over preceding decorator lines."""
        pos = methods[idx].start()
        # Step to start of line.
        line_starts = [i for i in range(pos) if i == 0 or body[i - 1] == "\n"]
        if not line_starts:
            return pos
        # Iterate from current line backwards while previous line is a decorator.
        cur_line_start = max(ls for ls in line_starts if ls <= pos)
        out = cur_line_start
        while True:
            prev_newline = body.rfind("\n", 0, out - 1) if out > 0 else -1
            prev_line_start = prev_newline + 1
            prev_line = body[prev_line_start:out - 1] if out > 0 else ""
            stripped = prev_line.lstrip()
            if stripped.startswith("@"):
                out = prev_line_start
            else:
                break
        return out

    chosen_spans: list[tuple[int, int]] = []
    first_test_taken = False
    for i, m in enumerate(methods):
        name = m.group(2)
        start = method_start_with_decorators(i)
        end = method_start_with_decorators(i + 1) if i + 1 < len(methods) else len(body)
        if name in ("setUp", "tearDown", "setUpClass", "tearDownClass"):
            chosen_spans.append((start, end))
        elif name.startswith("test_") and not first_test_taken:
            chosen_spans.append((start, end))
            first_test_taken = True
    if not first_test_taken:
        return None

    fragments = [body[s:e] for s, e in chosen_spans]
    cls_body = "".join(fragments).rstrip() + "\n"
    return f"from solution import {entry}\n\n{pre}{cls_body}"


LOADERS = {
    "humaneval": materialize_humaneval,
    "mbpp": materialize_mbpp,
    "mbpp_san": materialize_mbpp_sanitized,
    "humaneval_plus": materialize_humaneval_plus,
    "mbpp_plus": materialize_mbpp_plus,
    "livecodebench": materialize_livecodebench,
    "bigcodebench_hard": lambda out, limit=None: materialize_bigcodebench(out, limit=limit, hard=True),
    "bigcodebench": lambda out, limit=None: materialize_bigcodebench(out, limit=limit, hard=False),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, choices=sorted(LOADERS.keys()))
    p.add_argument("--out", default="tasks_bench", help="root dir to materialize into")
    p.add_argument("--limit", type=int, default=None, help="cap number of problems")
    p.add_argument("--since", default=None, help="LCB: keep only contest_date >= YYYY-MM-DD")
    p.add_argument("--until", default=None, help="LCB: keep only contest_date <= YYYY-MM-DD")
    p.add_argument("--difficulty", default=None, help="LCB: easy|medium|hard")
    p.add_argument("--version", default="release_v6", help="LCB: HF dataset version tag (v1..v6)")
    args = p.parse_args()

    out_root = Path(args.out) / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    loader = LOADERS[args.name]
    print(f"fetching {args.name}...")
    kwargs = {"limit": args.limit}
    if args.name == "livecodebench":
        kwargs.update(since=args.since, until=args.until, difficulty=args.difficulty, version=args.version)
    n = loader(out_root, **kwargs)
    print(f"wrote {n} tasks to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
