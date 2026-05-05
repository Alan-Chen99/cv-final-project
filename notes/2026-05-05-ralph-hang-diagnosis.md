# Ralph Workflow Hang Diagnosis: research3/research4 First Sessions

**Date:** 2026-05-05
**Scope:** Why research3/research4 first iterations hung 4.5-6 hours while research/research2 (~30 sessions) never hung

## Root Cause

The new PROMPT.md's "Prior findings" section references notes containing relative data paths (`external/constrained-downscaling/data/era5_sr_data`), biasing the agent toward searching the filesystem for existing data instead of downloading fresh. In a fresh worktree with era5_sr_data absent from CLAUDE.md's data table, the search escalates to `find /` on cluster NFS — which auto-backgrounds invisibly and runs for hours, blocking session exit.

## Causal Chain

```
PROMPT.md "Prior findings" references notes with data paths
  -> Agent infers data should exist, searches instead of downloading
  -> Fresh worktree: nothing at relative paths
  -> CLAUDE.md pool dir: era5_sr_data not listed (only gendiff/wassdiff)
  -> Escalation: find in pool -> find in /home -> find / (15s timeout, 3-stage pipeline)
  -> find / auto-backgrounds after 15s (invisible zombie)
  -> Agent finds data via main worktree, symlinks, moves on
  -> Cleanup: ps grep for experiment names misses find /; 0 TaskStop calls
  -> CC process blocks on find / still running -> 4.5-6hr hang
```

## Why Old Workflows Worked

```
Old PROMPT.md: no "Prior findings" section
  -> Agent starts tabula rasa
  -> Reads constrained-downscaling/README.md -> finds Google Drive URL
  -> Downloads via gdown (no filesystem search)
```

- **research iter1** (2947daee): Downloaded from Google Drive via `gdown` (ID `1IENhP1-aTYyqOkRcnmCIvxXkvUW2Qbdx`) found in `external/constrained-downscaling/README.md`. No filesystem search. 1 TaskStop call.
- **research2 iter1** (6389e0f8): Data pre-staged from research branch. No filesystem search. 23 TaskStop calls, 57 background tasks all resolved.

## Affected Sessions

| Session ID | Workflow | Symptom | Dangling task |
|---|---|---|---|
| d5049bec-bfdf-467f-8fb3-d6154b22f400 | research3 iter1 (coral-drift) | 268min hang | `be0m4d9la` = `find /` |
| e8a1256a-11f8-4125-8d1b-bb2f72e983ad | research4 iter1 (azure-quilt) | 361min hang | `bqb4zym6l` = `find /` |

Exact command from research3 iter1:
```bash
find /home/chenxy/orcd/pool/datasets/ -name "era5_sr*" -type d 2>/dev/null; \
echo "---"; \
find /home/chenxy/ -path "*/era5_sr_data*" -type d 2>/dev/null | head -5; \
echo "---"; \
find / -path "*/era5_sr_data*" -type d 2>/dev/null | head -5
```
Bash tool timeout: 15000ms. First two stages completed (no results). `find /` on NFS ran for hours.

## Three Contributing Factors

| # | Factor | Why it matters | Fix domain |
|---|---|---|---|
| 1 | "Prior findings" in PROMPT.md | Biases agent toward search-for-existing instead of download-fresh | PROMPT.md |
| 2 | era5_sr_data absent from CLAUDE.md data table | Agent can't resolve via standard data discovery path (pool dir) | CLAUDE.md |
| 3 | `find /` not prohibited in build.yml | Search escalates to catastrophic NFS traversal instead of failing fast | build.yml |

Factor 1 is the **trigger** (why new workflows diverge). Factor 2 is the **enabler** (why the search escalates). Factor 3 is the **amplifier** (why it becomes a 6-hour hang).

## Evidence

- `PROMPT.md` at commit 344d274 — "Prior findings" section with 3 note references
- `PROMPT.md` at commit 080e2b4 — no "Prior findings" section
- `build.yml` — identical across all 4 workflows (commits ab11333, 37da686, 3ae0396)
- Output styles (`alan-default` vs `alan-default-next`) — Bash Timeout section word-for-word identical; not a factor
- `notes/2026-05-02-flow-matching-downscaling.md` — contains `--data-dir external/constrained-downscaling/data/era5_sr_data` (5 occurrences)
- `external/constrained-downscaling/README.md` — Google Drive download URL for era5_sr_data
- `CLAUDE.md` data table — lists only `gendiff/` and `wassdiff/` under `/home/chenxy/orcd/pool/datasets/`

## Secondary Finding: TaskStop Behavior Divergence

Old sessions developed TaskStop discipline from using `long-running-commands` skill (explicit `run_in_background: true` + timer pairs). New sessions used `srun` with bash-level timeouts, never creating CC background tasks intentionally, so no TaskStop reflex formed. When `find /` auto-backgrounded, the agent had no habit of checking for orphaned background tasks.

| Session | TaskStop calls | Background tasks | Outcome |
|---|---|---|---|
| 286d5de4 (research iter11) | 14 | 18 | Clean |
| 6389e0f8 (research2 iter1) | 23 | 57 | Clean |
| d5049bec (research3 iter1) | 0 | 1 (find /) | Hung 268min |
| e8a1256a (research4 iter1) | 0 | 1 (find /) | Hung 361min |
