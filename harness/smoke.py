"""Smoke test: bypass the model, stuff known-good/known-bad solutions into a
sandbox, and verify public + hidden scoring works end-to-end.
Run: python -m harness.smoke
"""
from __future__ import annotations

from pathlib import Path

from .sandbox import Sandbox, score_hidden


CORRECT = {
    "task_001_sum_evens": "def sum_evens(nums):\n    return sum(n for n in nums if n % 2 == 0)\n",
    "task_002_balanced_parens": (
        "def is_balanced(s):\n"
        "    pairs = {')': '(', ']': '[', '}': '{'}\n"
        "    stack = []\n"
        "    for ch in s:\n"
        "        if ch in '([{':\n"
        "            stack.append(ch)\n"
        "        elif ch in ')]}':\n"
        "            if not stack or stack.pop() != pairs[ch]:\n"
        "                return False\n"
        "    return not stack\n"
    ),
    "task_003_merge_intervals": (
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    sorted_iv = sorted([list(p) for p in intervals], key=lambda x: x[0])\n"
        "    out = [sorted_iv[0]]\n"
        "    for start, end in sorted_iv[1:]:\n"
        "        if start <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], end)\n"
        "        else:\n"
        "            out.append([start, end])\n"
        "    return out\n"
    ),
    "task_004_roman_to_int": (
        "def roman_to_int(s):\n"
        "    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
        "    total = 0\n"
        "    prev = 0\n"
        "    for ch in reversed(s):\n"
        "        v = vals[ch]\n"
        "        if v < prev:\n"
        "            total -= v\n"
        "        else:\n"
        "            total += v\n"
        "        prev = v\n"
        "    return total\n"
    ),
    "task_005_caesar_decode": (
        "def caesar_decode(s, shift):\n"
        "    out = []\n"
        "    for ch in s:\n"
        "        if 'a' <= ch <= 'z':\n"
        "            out.append(chr((ord(ch) - ord('a') - shift) % 26 + ord('a')))\n"
        "        elif 'A' <= ch <= 'Z':\n"
        "            out.append(chr((ord(ch) - ord('A') - shift) % 26 + ord('A')))\n"
        "        else:\n"
        "            out.append(ch)\n"
        "    return ''.join(out)\n"
    ),
    "task_006_flatten_nested": (
        "def flatten(nested):\n"
        "    out = []\n"
        "    def go(x):\n"
        "        if isinstance(x, list):\n"
        "            for e in x:\n"
        "                go(e)\n"
        "        else:\n"
        "            out.append(x)\n"
        "    go(nested)\n"
        "    return out\n"
    ),
    "task_007_group_anagrams": (
        "def group_anagrams(words):\n"
        "    groups = {}\n"
        "    for w in words:\n"
        "        key = ''.join(sorted(w))\n"
        "        groups.setdefault(key, []).append(w)\n"
        "    return [groups[k] for k in sorted(groups.keys())]\n"
    ),
    "task_008_count_islands": (
        "def count_islands(grid):\n"
        "    if not grid or not grid[0]:\n"
        "        return 0\n"
        "    rows = len(grid)\n"
        "    cols = len(grid[0])\n"
        "    seen = [[False] * cols for _ in range(rows)]\n"
        "    count = 0\n"
        "    for i in range(rows):\n"
        "        for j in range(cols):\n"
        "            if grid[i][j] == 1 and not seen[i][j]:\n"
        "                count += 1\n"
        "                stack = [(i, j)]\n"
        "                while stack:\n"
        "                    r, c = stack.pop()\n"
        "                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 1 and not seen[r][c]:\n"
        "                        seen[r][c] = True\n"
        "                        stack.extend([(r+1,c),(r-1,c),(r,c+1),(r,c-1)])\n"
        "    return count\n"
    ),
    "task_009_valid_sudoku_row": (
        "def is_valid_sudoku(board):\n"
        "    rows = [set() for _ in range(9)]\n"
        "    cols = [set() for _ in range(9)]\n"
        "    boxes = [set() for _ in range(9)]\n"
        "    for r in range(9):\n"
        "        for c in range(9):\n"
        "            v = board[r][c]\n"
        "            if v == '.':\n"
        "                continue\n"
        "            b = (r // 3) * 3 + (c // 3)\n"
        "            if v in rows[r] or v in cols[c] or v in boxes[b]:\n"
        "                return False\n"
        "            rows[r].add(v); cols[c].add(v); boxes[b].add(v)\n"
        "    return True\n"
    ),
}

BROKEN = "def sum_evens(*a, **k):\n    return 999\n"  # used as a generic stub


def check(label: str, condition: bool) -> None:
    mark = "ok" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    tasks_root = Path("tasks")
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
        name = task_dir.name
        print(f"== {name}")

        # correct solution -> public + hidden should pass
        sb = Sandbox(task_dir)
        try:
            sb.write("solution.py", CORRECT[name])
            pub = sb.run_pytest("public_tests.py")
            check("correct: public passes", pub.returncode == 0)
            hid = score_hidden(task_dir, sb)
            check("correct: hidden passes", hid.returncode == 0)
        finally:
            sb.cleanup()

        # untouched starter -> tests should fail
        sb = Sandbox(task_dir)
        try:
            pub = sb.run_pytest("public_tests.py")
            check("starter: public fails", pub.returncode != 0)
            hid = score_hidden(task_dir, sb)
            check("starter: hidden fails", hid.returncode != 0)
        finally:
            sb.cleanup()

    print("\nall smoke checks passed.")


if __name__ == "__main__":
    main()
