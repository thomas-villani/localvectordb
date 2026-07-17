"""Guard the one docs failure that is completely silent.

``LocalVectorDB`` is assembled entirely from mixins and defines nothing in its
own ``__dict__``. Autodoc only documents inherited members when asked, so the
``:inherited-members:`` option on ``docs/source/modules/localvectordb.database.rst``
is the single thing standing between a full API page and a page listing the
class docstring and zero methods.

Both ways of losing it are invisible:

* Removing the option from the .rst builds cleanly and produces an empty class.
* Re-adding ``"inherited-members": False`` to ``autodoc_default_options`` *also*
  produces an empty class, from a config file that reads like it is merely
  restating a default. Sphinx's ``process_documenter_options`` overwrites a
  directive option with the config default whenever that default is not a
  string, so the config silently wins over the .rst.

Neither is caught by ``sphinx-build -W``: nothing warns, the build succeeds, and
the page simply stops describing the library. That is why this is a test and not
a comment. It is deliberately static -- no Sphinx build, no backend -- so it runs
on every PR alongside the fast lane.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from localvectordb.database import LocalVectorDB

DOCS = Path(__file__).resolve().parents[1] / "docs" / "source"
CONF = DOCS / "conf.py"
DATABASE_RST = DOCS / "modules" / "localvectordb.database.rst"


def _autodoc_default_options() -> dict:
    """Read the option dict out of conf.py without importing it."""
    tree = ast.parse(CONF.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "autodoc_default_options" in targets:
            return ast.literal_eval(node.value)
    pytest.fail("conf.py no longer defines autodoc_default_options")


@pytest.mark.docs
def test_localvectordb_defines_no_members_of_its_own():
    """The premise of the other two tests. If this ever changes, re-read them."""
    own = [n for n in vars(LocalVectorDB) if not n.startswith("_")]
    assert not own, (
        f"LocalVectorDB now defines its own public members ({own}); it used to "
        "inherit all of them from mixins, which is why the docs need "
        ":inherited-members:. Re-check that assumption."
    )


@pytest.mark.docs
def test_database_page_asks_for_inherited_members():
    """Without this option the class documents nothing at all.

    Match an indented directive option, not the bare string: the page's own
    prose explains ``:inherited-members:``, so a substring check would pass on
    the explanation alone after someone deleted the option it describes.
    """
    lines = DATABASE_RST.read_text(encoding="utf-8").splitlines()
    assert any(re.match(r"\s+:inherited-members:", line) for line in lines), (
        f"{DATABASE_RST.name} lost its :inherited-members: option. LocalVectorDB "
        "inherits every method from a mixin, so the page now documents zero "
        "methods -- and still builds clean."
    )


@pytest.mark.docs
def test_autodoc_defaults_do_not_clobber_inherited_members():
    """`"inherited-members": False` in conf.py silently overrides the .rst."""
    options = _autodoc_default_options()
    assert "inherited-members" not in options, (
        "conf.py sets 'inherited-members' in autodoc_default_options. A non-string "
        "default there overwrites the :inherited-members: option on "
        f"{DATABASE_RST.name}, so LocalVectorDB documents zero methods. Omit the "
        "key -- absent already means off for every other page."
    )
