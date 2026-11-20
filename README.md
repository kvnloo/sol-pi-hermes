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
