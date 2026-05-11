#!/usr/bin/env bash
# Prints the Claude Code session ID for the calling Claude Code process.
#
# Strategy: walk the process tree from $$ upward until a PID matches
# a file in ~/.claude/sessions/<pid>.json. Subagents skip PID
# registration (concurrentSessions.ts:60), so the walk always lands
# on the top-level Claude process — correct for subagents, nested
# subprocesses, and piped commands.
#
# Usage:
#   scripts/cc-session-id.sh           # print session UUID
#   scripts/cc-session-id.sh --path    # print JSONL transcript path
#   scripts/cc-session-id.sh --json    # print {sessionId, pid, transcriptPath}

set -euo pipefail

SESSIONS_DIR="${HOME}/.claude/sessions"

die() { echo "error: $1" >&2; exit 1; }

get_ppid() {
    # /proc is always available on Linux; ps fallback for macOS
    if [[ -f "/proc/$1/stat" ]]; then
        awk '{print $4}' "/proc/$1/stat"
    else
        ps -o ppid= -p "$1" 2>/dev/null | tr -d ' '
    fi
}

encode_project_path() {
    # Mirrors sessionStoragePortable.ts:sanitizePath —
    # replace ALL non-alphanumeric chars with hyphens.
    echo "$1" | sed 's/[^a-zA-Z0-9]/-/g'
}

[[ -d "$SESSIONS_DIR" ]] || die "no sessions directory at $SESSIONS_DIR"

# Walk up process tree from self
pid=$$
found_pid=""
found_session=""

while [[ -n "$pid" && "$pid" != "0" ]]; do
    pidfile="${SESSIONS_DIR}/${pid}.json"
    if [[ -f "$pidfile" ]]; then
        found_pid="$pid"
        found_session="$(jq -r '.sessionId' "$pidfile")"
        break
    fi
    pid="$(get_ppid "$pid")"
done

[[ -n "$found_session" ]] || die "no Claude Code ancestor found in process tree"

case "${1:-}" in
    --path)
        cwd="$(jq -r '.cwd' "${SESSIONS_DIR}/${found_pid}.json")"
        encoded="$(encode_project_path "$cwd")"
        echo "${HOME}/.claude/projects/${encoded}/${found_session}.jsonl"
        ;;
    --json)
        cwd="$(jq -r '.cwd' "${SESSIONS_DIR}/${found_pid}.json")"
        encoded="$(encode_project_path "$cwd")"
        transcript="${HOME}/.claude/projects/${encoded}/${found_session}.jsonl"
        jq -n --arg sid "$found_session" --arg pid "$found_pid" --arg path "$transcript" \
            '{sessionId: $sid, pid: ($pid | tonumber), transcriptPath: $path}'
        ;;
    "")
        echo "$found_session"
        ;;
    *)
        die "unknown flag: $1 (use --path or --json)"
        ;;
esac
