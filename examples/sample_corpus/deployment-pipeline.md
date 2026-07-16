# Deployment Pipeline

This document describes how code reaches production, and the controls that sit
between a merged commit and a customer request.

## Continuous Integration

Every pull request runs the full test suite, the linters, and a build of the
distributable artifact. The artifact built at pull request time is the same one
that will eventually be promoted, so a green build is a statement about the
thing that ships, not about a similar thing.

Tests run against a matrix of supported runtimes. The matrix is deliberately
small: every entry is a runtime someone actually uses, and entries are removed
when that stops being true. A matrix that grows without pruning tests
combinations no customer has.

Flaky tests are quarantined rather than retried. A retry hides the flake and
lets it spread; quarantine makes it visible and gives it an owner. A test that
has been quarantined for a full quarter is deleted, because a test nobody has
fixed in three months is not protecting anything.

## Promotion and Environments

There are three environments and they differ only in the traffic they serve.
Configuration drift between environments is the usual reason a deploy behaves
differently in production, so the environments are built from one definition.

Staging receives every merged commit automatically. It serves synthetic traffic
and a mirror of real read traffic. Nothing in staging is customer visible, which
means staging is where a bad change should be discovered.

Canary receives a small fraction of real production traffic. A change sits in
canary for at least thirty minutes, long enough for a slow leak or a cache
effect to show up. Metrics are compared against the un-canaried fleet rather
than against absolute thresholds, because absolute thresholds drift.

Production receives the change only after canary metrics are clean. Promotion is
automatic when the comparison passes, which removes the temptation to eyeball a
dashboard and declare it fine.

## Rollback

Every deploy is reversible, and reversibility is tested rather than assumed. A
deploy that cannot be rolled back is not a deploy, it is a migration, and
migrations follow a different and slower process.

Rollback is a single command and takes under two minutes. This number is a
design constraint rather than an observation: mitigation time bounds incident
duration, so the rollback path is optimised the way a hot code path would be.

A rollback never requires a decision about the cause. If the question "should we
roll back" is being debated, the answer is yes, and the debate should happen
afterwards with the service healthy.

Database changes are the exception that shapes the rest of the process. Because
a schema change cannot generally be reversed, schema changes are decoupled from
code changes: the schema is made compatible with both the old and new code
first, the code ships second, and the cleanup ships third.

## Schema Migrations

A migration is expand, then migrate, then contract. Each phase ships
independently and each phase is individually reversible, which is what makes the
sequence as a whole safe despite no single step being undoable.

Expand adds the new structure without removing the old one. Both the old and the
new code must work against the expanded schema, which is the property that lets
the code deploy roll back freely.

Migrate backfills data into the new structure. Backfills run in bounded batches
with a kill switch, because a backfill that saturates the database is
indistinguishable from an outage.

Contract removes the old structure, and only after the new code has been stable
in production long enough that rolling back to the old code is no longer
plausible. Contract is routinely deferred and occasionally forgotten, which is
untidy but harmless, whereas contracting early is neither.
