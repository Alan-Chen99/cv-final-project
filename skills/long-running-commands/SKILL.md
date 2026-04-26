---
name: long-running-commands
description: Strategy for executing long-running bash commands (training, data processing, builds). TRIGGER when about to run a command expected to take >1m or when runtime is uncertain (e.g. waiting in a queue, network transfers, external dependencies).
---

# Long-Running Bash Command

Protocol for running bash commands that may exceed the default
timeout. Backgrounds immediately, inspects early output, sets a timer,
and waits for notification.

Use this mid-task whenever a command might be long-running. No user
interaction required between steps — the protocol is self-contained.

## Before launching

### Ensure the command produces output

Many commands have verbose/progress flags. Use them so that stall
detection (Step 5) has something to work with.

If the command has no progress output and no verbose flag, wrap it:

```
bash -c 'echo "[long-bash] started"; <command>; echo "[long-bash] exit=$?"'
```

### Add timestamps to output

When the command or its logging framework supports it, enable
timestamps so you can see _when_ each line was produced — not just
what was produced. Examples:

- `make`: pipe through `ts` if available: `make 2>&1 | ts '[%Y-%m-%d %H:%M:%S]'`
- `cargo`: `CARGO_LOG_TIMESTAMP=1` (nightly) or pipe through `ts`
- `pytest`: `--tb=short` already includes timing per test
- `docker build`: `--progress=plain` includes step timing
- Generic: `<command> 2>&1 | while IFS= read -r line; do printf '[%s] %s\n' "$(date +%H:%M:%S)" "$line"; done`

This matters because the harness never injects timestamps into tool
results or notifications. The output file is raw command output with no
timing metadata. Timestamps in the output itself are the only way to
correlate lines with wall-clock time.

### Dry run / short version first

When feasible, run a fast variant before committing to the full command:

- `make -n` (dry run) to verify the dependency graph resolves
- `cargo check` before `cargo build --release`
- `npm run build -- --dry-run` if supported
- `rsync -n` (dry run) before the real sync
- A subset first (one test file, one target) to catch config errors early

### Timer calibration

When unsure how long a command takes, start with a **short timer**
(1–2 minutes), check whether output is progressing, then set a longer
timer. Prefer multiple short check-ins over one long blind wait.

## Procedure

### Step 1: Launch command + record start time

Run in parallel:

| Call | Parameters                                            |
| ---- | ----------------------------------------------------- |
| Bash | `command: "date '+%s %H:%M:%S'"`                      |
| Bash | `command: "<the command>"`, `run_in_background: true` |

The background call returns immediately with a message like:

```
Command running in background with ID: b3kx9m2p. Output is being written to: /home/user/.claude/temp/abc123/tasks/b3kx9m2p.output
```

Extract from this message:

- The task ID (after `ID: `, before the period)
- The output file path (after `written to: `)

### Step 2: Check initial output

```
Bash: sleep 1
```

Then read the output file path from Step 1.

- **Early fatal error** (command not found, permission denied, syntax
  error): TaskStop the command, compute elapsed, abort with error.
- **Normal startup** or **empty**: continue.

### Step 3: Start timeout timer

Infer time needed, or use a short timer if unsure:

```
Bash: command="sleep <TIMEOUT_SECONDS>", run_in_background=true, description="timeout timer"
```

Returns immediately with a message like:

```
Command running in background with ID: bw7q1p4r. Output is being written to: /home/user/.claude/temp/abc123/tasks/bw7q1p4r.output
```

Extract the timer's task ID.

### Step 4: Wait for notification

**CRITICAL: Do not call any more tools after this step.**

The system delivers `<task-notification>` XML when background tasks
complete. The first to arrive (command done OR timer expired) starts the
next turn.

Persist in your response text (for next-turn reference):

- Command task ID + output path (from Step 1)
- Timer task ID (from Step 3)
- Start time: both epoch and `%H:%M:%S` (from Step 1)

### Step 5: Handle notification

A `<task-notification>` arrives:

```xml
<task-notification>
<task-id>b3kx9m2p</task-id>
<output-file>/home/user/.claude/temp/abc123/tasks/b3kx9m2p.output</output-file>
<status>completed</status>
<summary>Background command "make -j8" completed (exit code 0)</summary>
</task-notification>
```

#### Command finished (task-id matches command)

1. TaskStop the timer task ID
2. Read the command's output file
3. Continue the original task with the result + elapsed time

#### Timer fired (task-id matches timer)

Timer notifications look like:

```xml
<summary>Background command "timeout timer" completed (exit code 0)</summary>
```

1. Read the command's output file (partial output so far)
2. Decide based on partial output:
   - **Progress visible** (output growing, compilation advancing): set a
     new timer (go to Step 3)
   - **Stalled** (no new output, stuck on same line): TaskStop the
     command, continue with error
   - **Unclear**: escalate to user with partial output and elapsed time

#### Multiple notifications

Both may arrive between turns. Match task IDs. TaskStop whichever task
is still running.

---

## Reference

### Why wall-clock is mandatory

The harness injects no timestamps or timing info into tool results or
notifications. The model has no built-in clock. Agent turns themselves
take variable time (depends on Anthropic server load, model thinking
time, tool execution queue) — so "I launched two tools 3 steps ago"
tells you nothing about how much real time passed. Always use
wall-clock to measure elapsed time. Never estimate
based on turn count or step count.

### Default Bash tool behavior (run_in_background=false)

By default, `run_in_background` is false and the Bash tool's timeout
applies (the agent sees the exact value in its system prompt; default
120s, max 600s, overridable via `BASH_DEFAULT_TIMEOUT_MS` env var).
When the timeout hits, the command is auto-backgrounded (not killed).
The tool result has **empty stdout/stderr** and a message:

```
Command running in background with ID: b3kx9m2p. Output is being written to: /home/user/.claude/temp/abc123/tasks/b3kx9m2p.output
```

This is the same message as explicit `run_in_background: true`. The only
difference is the 120s delay before you see it. In KAIROS assistant
mode, a separate 15s budget auto-backgrounds even earlier with a
distinct message (`"Command exceeded the assistant-mode blocking
budget..."`).

This skill uses `run_in_background: true` to skip the wait — since
auto-backgrounding returns empty stdout/stderr anyway, there is no
benefit to running foreground first.

### Other facts

- Output accumulates on disk at the path in the message. Read it anytime.
- Foreground `sleep N` (N >= 2) is blocked. Must use `run_in_background: true`.
- `sleep 1` foreground is fine (< 2s threshold).
- Task IDs: `b` prefix + 8 random alphanumeric chars.
- Notifications use `description` param if provided; otherwise truncated command.
- 5GB output watchdog kills runaway processes.
