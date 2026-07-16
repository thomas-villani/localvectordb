"""Keep README.md's Python examples honest.

Nothing else in the suite reads the README, so its snippets can drift from the
API indefinitely -- and have. Every ```python fence must carry a directive
comment on the line above it saying how it is checked:

    <!-- test: run -->
        Executed verbatim in a temp cwd. Must not raise. Use for self-contained
        snippets that work offline (i.e. against the `mock` provider).

    <!-- test: verify-api -->
        Not executed -- parsed. Every ``db.<method>(...)`` call must name a real
        method with real keyword arguments. Use for illustrative fragments that
        cannot run standalone (no live provider, undefined `db`, or a sequence
        that is deliberately not a program).

    <!-- test: skip reason="..." -->
        Neither. Use only when the snippet makes no library calls to check
        (raw HTTP, a third-party client). The reason is required.

Directives are HTML comments, so they are invisible on GitHub and PyPI.

`verify-api` exists because ``compile()`` would be near-worthless here: a
renamed parameter or a deleted method compiles perfectly and only fails at call
time. Signature checking catches exactly the drift these fences suffer from,
without forcing illustrative fragments to become runnable programs.
"""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

from localvectordb.database import LocalVectorDB

README = Path(__file__).resolve().parents[1] / "README.md"

# A directive comment immediately above a ```python fence, plus the fence body.
_BLOCK_RE = re.compile(
    r"^<!--\s*test:\s*(?P<directive>run|verify-api|skip)(?P<args>[^>]*?)-->\s*\n" r"```python\n(?P<code>.*?)^```",
    re.MULTILINE | re.DOTALL,
)
# Every ```python fence, directive or not -- used to prove none escaped marking.
_ANY_FENCE_RE = re.compile(r"^```python\n", re.MULTILINE)
_REASON_RE = re.compile(r'reason\s*=\s*"([^"]+)"')


class Block:
    def __init__(self, directive: str, code: str, line: int, reason: Optional[str]):
        self.directive = directive
        self.code = code
        self.line = line
        self.reason = reason

    def __repr__(self) -> str:  # pytest test id
        return f"L{self.line}-{self.directive}"


def _load_blocks() -> list[Block]:
    text = README.read_text(encoding="utf-8")
    blocks = []
    for m in _BLOCK_RE.finditer(text):
        args = m.group("args")
        reason_m = _REASON_RE.search(args)
        blocks.append(
            Block(
                directive=m.group("directive"),
                code=m.group("code"),
                line=text.count("\n", 0, m.start()) + 1,
                reason=reason_m.group(1) if reason_m else None,
            )
        )
    return blocks


BLOCKS = _load_blocks()


@pytest.mark.docs
def test_every_python_fence_is_marked():
    """A new example must declare how it is checked, or the suite fails.

    This is what stops the drift from silently returning: an unmarked fence is
    an unverified claim.
    """
    text = README.read_text(encoding="utf-8")
    total_fences = len(_ANY_FENCE_RE.findall(text))
    assert total_fences == len(BLOCKS), (
        f"{total_fences} ```python fences in README.md but only {len(BLOCKS)} carry a "
        f"'<!-- test: ... -->' directive. Every Python fence needs one "
        f'(run | verify-api | skip reason="..."). See this module\'s docstring.'
    )


@pytest.mark.docs
def test_skips_explain_themselves():
    for b in BLOCKS:
        if b.directive == "skip":
            assert b.reason, f"README.md:{b.line}: 'skip' requires reason=\"...\""


@pytest.mark.docs
@pytest.mark.parametrize("block", [b for b in BLOCKS if b.directive == "run"], ids=repr)
def test_runnable_examples_run(block: Block, tmp_path: Path):
    """Execute the snippet exactly as a reader would, in a clean directory.

    Run in a subprocess so a snippet's own imports/state cannot leak into the
    test session, and so a stray sys.exit() doesn't take pytest with it.
    """
    script = tmp_path / f"readme_l{block.line}.py"
    script.write_text(block.code, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"README.md:{block.line} example failed to run:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def _iter_db_calls(code: str):
    """Yield (method_name, [keyword names], lineno) for every `db.<method>(...)`."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "db":
            continue
        kwargs = [kw.arg for kw in node.keywords if kw.arg is not None]
        yield func.attr, kwargs, node.lineno


@pytest.mark.docs
@pytest.mark.parametrize("block", [b for b in BLOCKS if b.directive == "verify-api"], ids=repr)
def test_illustrative_examples_use_the_real_api(block: Block):
    """Check documented calls against real signatures without executing them.

    Catches the drift class that bit us: a method that was renamed or removed,
    or a keyword argument that no longer exists.
    """
    for method, kwargs, lineno in _iter_db_calls(block.code):
        where = f"README.md:{block.line + lineno} (db.{method})"
        assert hasattr(LocalVectorDB, method), f"{where}: LocalVectorDB has no method '{method}'"

        sig = inspect.signature(getattr(LocalVectorDB, method))
        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue  # **kwargs accepts anything; nothing to check
        for kw in kwargs:
            assert kw in params, f"{where}: '{kw}' is not a parameter of LocalVectorDB.{method}{sig}"
