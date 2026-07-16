# Data Retention Policy

This policy sets out how long each class of data is kept, where it is kept, and
how it is disposed of. It applies to production data, backups, and analytics
exports alike.

## Classification

Data is classified at the point of collection. Reclassifying later is expensive
and frequently incomplete, because copies have already spread.

Customer content is anything a customer uploads, types, or generates through
normal use of the product. It is the most tightly controlled class and the
default assumption for anything ambiguous.

Operational telemetry is data the system emits about itself: request logs,
traces, metrics, and error reports. It often contains fragments of customer
content by accident, which is why it is treated as sensitive rather than
public.

Derived aggregates are statistics computed across many customers such that no
individual record is recoverable. Aggregates are only genuinely derived if the
underlying join keys have been discarded; keeping the keys means keeping
customer content under a different name.

## Retention Periods

Customer content is retained for the life of the account plus thirty days. The
thirty day tail exists so that an accidental deletion or a disputed termination
can be reversed. After that window the data is unrecoverable by design, and no
amount of escalation will retrieve it.

Operational telemetry is retained for ninety days in queryable storage and a
further nine months in cold storage. Ninety days is chosen to cover a full
quarterly cycle of investigations; the cold tail exists for security forensics,
not for debugging.

Derived aggregates are retained indefinitely. Because they cannot be resolved to
an individual, no retention clock applies to them.

Backups follow the retention period of the most sensitive data they contain,
which in practice always means customer content. A backup is not a separate
class of data; it is a copy of an existing class.

## Deletion Requests

A customer may request deletion of their content at any time. The request is
honoured within thirty days, which is the same window as the account tail rather
than a coincidence.

Deletion means the data is removed from production stores immediately and from
backups as those backups age out. It is not feasible to surgically excise a
record from an immutable backup, so the guarantee offered is expiry, not
excision. This distinction is stated plainly in the customer agreement, and it
should be stated plainly in support conversations too.

Derived aggregates are not affected by a deletion request, because they contain
no recoverable individual record. If an aggregate would be affected, it was not
an aggregate.

## Access and Audit

Access to customer content in production requires a named business reason and is
logged. The log is written to a store that the accessing engineer cannot modify,
which is the entire point of it.

Standing access is not granted. Access is requested, time-boxed, and expires
automatically. An engineer who needs access every day should be asking why the
tooling requires it, rather than requesting a permanent grant.

Audit logs are themselves operational telemetry and follow that retention
period, with one exception: access logs for customer content are kept for the
full seven years required by the compliance regime.

## Disposal

Disposal is the deliberate destruction of data at the end of its retention
period, and it is a scheduled job, not a manual task. Manual disposal is
unreliable because it depends on someone remembering.

Physical media is destroyed rather than wiped when it leaves the estate. Wiping
is adequate in principle and unverifiable in practice, and the cost difference
does not justify the argument.

Disposal is verified quarterly by sampling: pick records that should have
expired, confirm they are gone, and record the result. A retention policy that
is never verified is a statement of intent rather than a control.
