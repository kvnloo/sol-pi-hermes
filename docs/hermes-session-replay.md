# Offline replay from a copied Hermes SQLite session

Use an operator-created, consistent SQLite backup of a database you have
permission to analyze. Do not point this at the live profile or copy only the
main file of a live WAL database. Provide exact session IDs; the script does
not search profiles, open a provider, install anything or run tools.

```sh
python -B scripts/replay_hermes_session.py /path/to/snapshot.sqlite SESSION_ID [ANOTHER_SESSION_ID]
python -B -m unittest discover -s tests -p test_session_replay.py
```

The input is opened in SQLite read-only mode with a single read transaction.
ObservationPack stores its replay archives in temporary per-session directories,
deleted on exit. The JSON report omits message content but includes session,
assistant-row and observation IDs; keep it private as appropriate. Report is
written to stdout; redirect it to a file if needed. Exit 2 means input or
evaluation was rejected; exit 1 means at least one simulated recall was not
byte-exact. The replay performs work proportional to all measured prefixes,
so it rejects sessions over 2,000 active rows rather than truncating history.

## Reconstruction and limits

At Hermes main `32dbc1dcda2255fd35969ec99cfc0170988c3eca`, the
[SQLite DDL](https://github.com/NousResearch/hermes-agent/blob/32dbc1dcda2255fd35969ec99cfc0170988c3eca/persistence/sqlite/_schema_ddl.py)
defines the sessions, messages and usage fields used here. Its
[conversation reader](https://github.com/NousResearch/hermes-agent/blob/32dbc1dcda2255fd35969ec99cfc0170988c3eca/persistence/sqlite/conversation/_store/messages.py)
orders by row ID, not timestamp, and decodes the NUL-prefixed JSON content
format. The script reads only a selected session's active system, user,
assistant and tool rows, keeping each session separate. It uses nonempty
stored user/assistant `api_content` sidecars where available; this is not an
implementation of Hermes's conversation reader or sanitation.

An inferred request opportunity is the durable prefix before an assistant
response row immediately following a supported user or tool row. It is not a
logged API request. Consecutive assistant rows and a pending tail do not add
opportunities. Inactive/compacted history, parent/ancestor segments and their
compaction handoffs, spill/prune file contents, historical intermediate
states, retries and unrecorded requests are not reconstructed.

The real SoL-Pi `project_request` compares before and after at each inferred
opportunity. Byte totals are exact UTF-8 lengths for this script's compact,
ASCII-escaped JSON message arrays, not the provider request. Text-token
figures are a per-message ceil(characters/4) estimate, excluding tool schemas,
images, reasoning and JSON framing. Recalls are simulated once per distinct
observation that became a placeholder: the report records pages, full
serialized recall JSON bytes (including envelope/escaping), largest page size
and whether the recovered tool text was byte-exact. These are not observed
recall calls or an estimate of actual recall frequency.

Historical `sessions` token/cache counters are reported separately, including
nulls. They cannot be attached to individual inferred boundaries or used to
compute actual counterfactual savings; actual projection-token delta is null.
Hermes [request assembly](https://github.com/NousResearch/hermes-agent/blob/32dbc1dcda2255fd35969ec99cfc0170988c3eca/agent/turn_request_assembly.py)
and [wire-copy building](https://github.com/NousResearch/hermes-agent/blob/32dbc1dcda2255fd35969ec99cfc0170988c3eca/agent/turn_context.py)
apply system prompts (which may partly be stored), ephemeral prefill and context,
history cleanup, reasoning routing, current tool schemas, provider middleware
and cache decoration. A transcript snapshot does not establish their exact
per-request state, transport bytes, model quality, provider billing or cache
reuse. This is a local message-level estimate, not a provider replay.
