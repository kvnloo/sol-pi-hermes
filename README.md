# sol-pi-hermes

A standalone Hermes plugin that ports two mechanisms from [NVlabs/SoL-Pi](https://github.com/NVlabs/SoL-Pi) to Hermes.

- **ObservationPack.** A tool result over 10 KiB is sent to the model in full twice. On later provider requests it is replaced by a short placeholder: an id plus a 1 KiB excerpt. The model reads it back with `obs_recall` (up to 16 KiB or 400 lines per call, continuing from `next_offset`). Only the outgoing request is changed. Stored history keeps the full result.
- **Action Fusion.** `sol_pi_write` and `sol_pi_patch` run Hermes' own `write_file` / `patch` and, when the call carries `then_run`, run one follow-up command through Hermes' `terminal` tool. The edit and the check then cost one model turn instead of two.

## Install

```bash
hermes plugins install kvnloo/sol-pi-hermes
hermes plugins enable sol-pi
```

Both mechanisms are **off** until you turn them on, as in SoL-Pi. Write `sol-pi.json` in your Hermes home (the active profile's home):

```json
{ "version": 1, "observationPack": true, "actionFusion": true }
```

## How it plugs in

| Mechanism | Hermes surface |
| --- | --- |
| ObservationPack | `llm_request` middleware rewrites `messages` in the provider request copy; `obs_recall` tool |
| Action Fusion | `sol_pi_write` / `sol_pi_patch` tools, dispatched through Hermes' tool registry |

The plugin never registers a context engine, so Hermes' compressor stays in charge. It never patches Hermes code.

SoL-Pi's Evidence-Preserving Reducer and Online Context Compact are ported in `epr.py` / `occ.py` with tests, but they are **not wired** in this release. EPR needs a reducer model, and OCC needs an `agent_settled` hook that Hermes doesn't have.

## What it touches

- **Writes**: archived tool results under `<hermes home>/sol-pi/<session-slug>-<session-SHA256>/observation-pack/` (directory mode 0700, new files 0600), plus a newly created 0600 `ledger.jsonl` next to them. Existing archives/ledgers are not permission-migrated; installations upgrading from older versions should inspect their permissions before enabling ObservationPack. Action Fusion additionally writes the requested target through Hermes write_file/patch in the originating task/session backend.
- **Reads**: `sol-pi.json` in the Hermes home, no host-side reads of Action Fusion target files. PluginContext does not attest backend locality, so the plugin does not hash a host path or promise an atomic edit/check transaction.
- **Commands**: `then_run` runs the model's command through Hermes' `terminal` tool, with the originating task/session identity. Dangerous-command approval and the write denylist still apply, but nested dispatch currently bypasses ACP/Zed editor edit approval, file checkpoints (`/rollback`), and other plugins' `pre_tool_call` guards. Do not enable Action Fusion where those protections are required. Core PR NousResearch/hermes-agent#138933 addresses the hook portion, not all these protections.
- **Network**: none.
- **Credentials**: none.

Action Fusion tools are registered only when `actionFusion` is enabled. Returned mutation failures skip the command; returned command failures are labeled failed. Command results without a completed integer exit code are labeled `then_run:unknown`, not success; no background polling is added. Restart/reload the plugin after changing configuration.

Session directory names use a bounded ASCII slug followed by the SHA-256 of the **original** session ID, so sanitized names cannot accidentally mix different sessions. Old slug-only archives are not consulted after upgrade; full tool observations retained in the transcript are re-archived when a later provider request projects them.

## Provenance

Failure-result handling and task/session forwarding reuse the focused logic from retained downstream commits `015d10fc` (#4) and `12e03247`, without restoring their older host hooks.

The algorithms follow NVlabs/SoL-Pi @ `22277b7e` (MIT).

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

ObservationPack requires a nonempty host-provided session ID (or task ID fallback). Without one, request projection returns the full unchanged observation without archiving it, and recall returns an identity error. An explicit session named `default` remains valid but is never an anonymous fallback. These identifiers must come from the trusted host callback, not tool arguments.
### Ledger contention

The optional observation ledger uses a bounded lock attempt (50 non-blocking
attempts, with 2 ms sleeps on contention). If locking times out, fails, or is
unavailable on the platform, that telemetry record is omitted, rather than
appended unsafely. Projection and recall remain available; the ledger is not
a complete audit log. Successful cooperating writers hold the lock across
torn-tail recovery and all short writes. Non-cooperating writers are not
protected. Filesystem I/O itself is not time-bounded.

Ledger appends open nonblocking and verify that the opened descriptor is a
regular file before acquiring a lock or writing. Existing or substituted FIFOs
and other nonregular descriptors are rejected and closed. Platforms without
nonblocking-open support skip optional ledger telemetry without opening a file.
This is not a hostile-parent-directory sandbox or a filesystem I/O deadline;
normal regular-file locking, torn-tail repair and short-write handling remain.
### Observation archive publication

New observation objects are written completely to a private 0600 temporary file,
then published with an atomic, no-replace hard link in the same directory.
Short writes are retried; zero-progress or failed writes do not expose a partial
object. An existing object must still match its expected bytes and hash.
If the filesystem cannot create hard links or the archive write fails, packing
fails open and retains the full tool result. Normal failures remove temporary
files; process termination can leave unpublished `.observation-*` files. This
provides atomic visibility, not an fsync/power-loss durability guarantee. Ledger
behavior and pre-existing archive permissions are unchanged.

An already-published object is fully checked for regular-file type, expected size
and content hash, then reused without creating or writing a temporary file.
This avoids unnecessary duplicate writes and allows reuse when temporary-file
creation fails. New objects still require atomic publication; corrupt existing
objects are rejected rather than overwritten. Ledger I/O remains separate, so
this does not promise that packing works on a wholly unwritable filesystem.

Existing ObservationPack archives are validated through one nonblocking file
descriptor: regular-file identity and expected size are checked before a bounded
content/hash read. Path replacement and FIFO substitution are rejected rather
than reopening an unchecked path. Platforms without nonblocking file opens
refuse archive reuse, leaving projection unchanged through the existing
fail-open path. This is not a sandbox against hostile parent directories or
in-place modification after validation, nor a durability guarantee.
### Fused command admission

A fused follow-up command requires an affirmative mutation result: Hermes patch
results report `success: true`; write results report a nonnegative integer
`bytes_written` (including a successful empty write). Reported errors override
these fields. Missing, malformed, pending or unknown results skip the command
and preserve the mutation response for inspection. The mutation may already
have happened; a skipped follow-up is not a rollback. Calls without `then_run`
still return the original mutation result unchanged.

`sol_pi_write` validates its required inputs before any nested dispatch: `path`
must be a nonempty string and `content` must be a string. Missing/null/boolean or
structured content is rejected, never coerced into an empty-file write. An
explicit empty string remains a valid deliberate write. Hermes advertises plugin
schemas but does not apply its built-in required-argument checks to plugins, so
this boundary is enforced by the plugin itself.

Fused follow-up specifications are checked before either write or patch is
attempted. A provided `then_run` must contain a nonblank string `command` and,
when supplied, a finite non-negative numeric `timeout` (not a boolean). Timeout
0 retains Hermes's default-timeout behavior. Omitted/null `then_run` retains the
existing mutation-only behavior; omitted/null timeout uses the host default.
Malformed specifications return an error without attempting either nested tool.
This preflight does not guarantee that a valid command will succeed or provide
rollback if a later backend operation fails.

Action Fusion rejects an actual NUL character in `then_run.command` before any
mutation or nested command dispatch. Native command spawning cannot accept that
argument, so discovering it after a successful write/patch would create an
avoidable partial action. Literal backslash-zero, Unicode and newlines remain
unchanged. This is argument validation, not shell-command safety analysis or a
promise that valid commands will succeed or that failed checks roll back edits.
### Recall response budget

Successful `obs_recall` responses fit the 16 KiB limit as serialized UTF-8 JSON,
including its envelope and escaped content. Unicode, control characters, quotes
and backslashes can require smaller pages than plain text. The returned `bytes`
and `next_offset` still count original UTF-8 content bytes, not JSON escaping;
continue with `next_offset` until `eof` to recover the complete observation.
The page selection conservatively halves the content budget when needed rather
than maximizing each page. This does not claim a model-token or billing limit.

Malformed recall IDs return a fixed, bounded format error rather than echoing
arbitrarily large caller input into the tool response. Non-string IDs are also
refused before path construction; valid but unknown IDs retain their short
identifier in the diagnostic. This bounds this admission-error path, not all
possible host-generated errors or the size of incoming tool arguments.
### Failed tool observations

ObservationPack preserves explicit failed JSON tool-result envelopes in full,
including a nonempty top-level `error`, `success: false`, or a nonzero integer
`exit_code`. The explicit top-level ActionFusion `then_run: "[then_run:failed]"`
status is also preserved, because its command diagnostics live inside `output`.
Successful or unknown fused-command statuses remain eligible for packing. Hermes
may put these fields inside content without a message-level
`isError` flag. These results are not archived, counted or replaced by a
placeholder. Successful/unstructured output remains eligible; nested data fields
are not interpreted as failure signals. Wrapped or non-JSON error prose is not
heuristically classified. This deliberately trades savings for visible diagnostics.
### Recall file opening

Recall validates a regular archive file with `lstat`, opens it with no-follow and
nonblocking flags where available, then checks the opened descriptor's type and
device/inode identity before reading. Size and content come from that descriptor,
not separate pathname lookups. A substituted symlink, FIFO or different file is
rejected; rejected descriptors are closed. Without a no-follow flag, identity
validation still rejects a link to a different file before reading its content.
This is not a sandbox against arbitrary modifications to archive directories or
in-place writes to an already-open file. Native non-POSIX behavior is unqualified.

Recall refuses an empty page before the descriptor's observed EOF rather than
returning the same next offset with `eof: false`. This catches truncation between
the size check and read, and a byte budget too small for the next UTF-8 character.
Legitimate empty EOF responses remain valid. This progress check does not make
mutable archives immutable or detect every concurrent in-place content change.
The standalone `~/.hermes` fallback applies only when `hermes_constants` itself
is unavailable. A present host with a missing resolver, broken dependency or
failing profile resolver now fails registration rather than silently loading a
default-profile policy or choosing its archive location. Operators must repair
the host/profile failure before retrying; no alternate profile is inferred.
ObservationPack send accounting advances once per observation ID per projected
provider request, even when that ID appears repeatedly in the message list.
Duplicate appearances use the same prior count and packing decision; distinct
call IDs keep independent counts. This does not change the existing cold-history
assistant-turn estimate, or turn projection into an acknowledgement that a
provider actually received a request. Ledger records remain per appearance.
ObservationPack leaves a tool result unchanged when any text block carries
extra fields (including citations, annotations, IDs or cache controls), or has
missing/non-string text. The archive is text-only and cannot restore those
fields. Plain strings and exact `{type: "text", text: <string>}` blocks remain
eligible, including mixed lists of those two plain forms. Rich results incur no
archive or send-count side effects; this conservative gate favors retaining
provider-visible metadata over token savings.



## Offline CaptureResult shadow evaluation (issue #2)

From a source checkout (Python stdlib only):

```bash
python -m sol_pi_hermes.observation_eval --output /tmp/sol-pi-observation-shadow.json
python -m unittest discover -s tests
```

The evaluator replays frozen, **synthetic** `CaptureResult`-shaped fixtures alongside
Hermes model-visible text JSON or multimodal capture blocks, using the existing
`project_request` and `obs_recall` implementation. It **does not** replace any
capture, send a provider request, issue computer-use actions, or enable a policy.
A custom `--fixtures` input must contain at least one case; empty replays fail
without emitting a success report.
The first two full sends remain control observations; later sends are projected
only when the current ObservationPack middleware supports their content type.

The JSON report records original/projected UTF-8 bytes, estimated text-token
savings, element counts, surviving indices and opaque tokens, structured fields
hidden from the immediate model view, exact field recovery after paged recall,
and locally measured projection latency. The estimator uses four characters per
token, **not** a model tokenizer; image token costs, cache effects, and action
quality are not measured. Timings vary by machine. A native image-bearing
multimodal result is intentionally passed through and reports **zero** savings.
Tokens present only in the original `CaptureResult` but never serialized
by Hermes are not blamed on ObservationPack.

The snapshots follow the `CaptureResult` / `_text_capture_payload` contract in
NousResearch/hermes-agent at `3f5a11f624c8c3c7d0e99ddd8b4b7da97fcb7940`. They are not
real-device recordings. Real, consented/redacted capture traces and live agent
equivalence remain separate gates before considering model-visible activation.

The default synthetic shadow-evaluation fixtures ship inside the installed
`sol_pi_hermes` package. `python -m sol_pi_hermes.observation_eval` therefore runs
outside a source checkout too; `--fixtures` still accepts an explicit JSON file.

The shadow evaluator's passthrough invariant includes metadata-bearing text
blocks, not just images: exact recall of text cannot certify retained citations,
annotations, cache controls or unknown block fields. Such loss fails the CLI even
when text recall is byte-exact. Plain string blocks remain text-only controls.
This evaluator rule is independent of the candidate projection's eligibility
function; it does not itself change production packing or enable rich packing.
