<!--
Thanks for contributing! See CONTRIBUTING.md for setup and conventions.
Keep PRs focused — a bug fix and a feature belong in separate PRs.
-->

## What and why

<!-- What changed, and what problem it solves. Link any related issue. -->

## How it was verified

<!--
Not "tests pass" — what did you actually run, and what did you observe?
If it changes retrieval (search, fusion, reranking, document scoring), say so:
those changes must be measured against benchmarks/RETRIEVAL_BASELINE.md with
`python benchmarks/eval_retrieval.py --check`. Tests using MockEmbeddings cannot
tell whether the right document ranks first.
-->

## Checklist

- [ ] `pytest` passes and `ruff check .` is clean
- [ ] Tests added or updated for the change
- [ ] Docs updated if behaviour or the public API changed
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for any user-visible change
- [ ] Any new README Python example carries a `<!-- test: ... -->` directive
      (see `tests/test_readme_examples.py`)
- [ ] Retrieval-affecting changes measured against `RETRIEVAL_BASELINE.md`
