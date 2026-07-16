# Incident Response Handbook

This handbook describes how the platform team declares, runs, and closes out
production incidents. It applies to all services in the production estate.

## Declaring an Incident

Anyone may declare an incident. You do not need permission, and you will never
be criticised for declaring one that turns out to be minor. Declaring early is
cheap; declaring late is expensive.

An incident is warranted whenever customer-visible behaviour is degraded, data
integrity is in doubt, or a security boundary may have been crossed. Internal
tooling being broken is not, by itself, an incident, unless it blocks the
resolution of a customer-visible problem.

To declare, post in the operations channel with the word "declaring" and a one
line description. This creates the incident record automatically and pages the
current incident commander. Do not wait until you understand the cause. The
declaration is a request for help, not a diagnosis.

## Severity Levels

Severity is set at declaration time and revised freely as understanding
improves. It drives who is paged, not how seriously the problem is taken.

Sev1 means a total outage of a customer-facing surface, confirmed data loss, or
an active security breach. Sev1 pages the incident commander, the service owner,
and the leadership on-call, immediately and at any hour.

Sev2 means substantial degradation with a workaround available, or a failure
confined to a subset of customers. Sev2 pages the service owner during business
hours and the on-call engineer outside them.

Sev3 means a defect that is contained and not spreading. Sev3 creates a ticket
and is picked up during the next working day. Most declarations settle at Sev3
after investigation, which is the system working as intended.

## Roles During an Incident

The incident commander owns the incident. They do not debug. Their job is to
maintain a coherent picture, decide what is tried next, and keep the timeline
accurate. If the commander is also debugging, the incident has no commander.

The communications lead owns everything the outside world sees. They post the
status page, draft customer messaging, and shield the responders from requests
for updates. Without a dedicated communications lead, responders spend the
incident answering questions instead of resolving it.

Responders investigate and act. A responder should announce what they are about
to do before they do it, and announce what they observed afterwards. Silent
action is the most common cause of an incident lasting longer than it needed to.

## Mitigation Before Diagnosis

Restore service first. Understand it afterwards. This is the single most
frequently violated rule in this handbook, because engineers find diagnosis more
interesting than mitigation.

If a rollback is available, roll back. Do not spend twenty minutes determining
whether the deploy was truly the cause when rolling it back takes two. A
rollback that turns out to have been unnecessary costs almost nothing; a
diagnosis conducted while customers are down costs a great deal.

Feature flags, traffic shedding, and regional failover are all legitimate
mitigations. So is turning a feature off entirely. The bar for mitigation is
"does this stop the bleeding", not "is this the correct long term design".

## Closing Out and Review

An incident closes when customer impact has ended and been confirmed ended, not
when the cause is understood. Leaving an incident open pending a root cause
analysis obscures how long customers were actually affected.

Every Sev1 and Sev2 gets a written review within five working days. The review
is blameless: it describes what people knew and when, not what they should have
known. A review that concludes "engineer should have been more careful" has
found a symptom, not a cause.

Reviews produce action items with owners and dates. An action item without an
owner is a wish. The most valuable action items usually make the next incident
easier to detect or mitigate, rather than attempting to make it impossible.
