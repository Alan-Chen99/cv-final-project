# Spatial-4x-30min Experiment Diagnosis: Why Agents Didn't Finetune or Catalog Papers

**Date**: 2026-05-09
**Scope**: All 4 worktrees (spatial-4x-30min-v{1,2,3,4}), ~60 sessions, 10 sessions diagnosed in detail
**Method**: /diagnose-session on 10 JSONL sessions (5 early, 5 late), HTTP log system prompt analysis, ToolSearch/Skill audit across all sessions

## Problem Statement

Across all 4 worktrees (~54 iterations total), agents:
- Never cataloged papers (0 `/arxiv-to-md` invocations)
- Never invoked `/decision-critic` (marked REQUIRED in PROMPT.md)
- Ran 50+ from-scratch experiments despite PROMPT.md saying "focus on finetuning external models, maybe training 1-2 from-scratch ones"

## Root Causes

### 1. HARD BLOCKER: User-level skills stripped from SDK sessions

The Claude Agent SDK sessions receive only **8 skills** in the system-reminder:

```
update-config, keybindings-help, simplify, loop, schedule, claude-api,
long-running-commands, slurm-preemptable
```

All 20+ user-level skills are missing, including the ones PROMPT.md and CLAUDE.md reference:

| Missing Skill | Referenced by | Impact |
|---|---|---|
| `/arxiv-to-md` | CLAUDE.md (2x) | Cannot catalog papers |
| `/decision-critic` | PROMPT.md (REQUIRED) | Cannot stress-test decisions |
| `/problem-analysis` | PROMPT.md | Cannot diagnose failures |

The system prompt says: "Only use Skill for skills listed in its user-invocable skills section - do not guess." This creates a hard block: agents are prohibited from invoking skills that CLAUDE.md tells them to use.

**Evidence**: HTTP logs at `/home/chenxy/.claude/http-logs/2026-05-09T13-04-33_6f8596/` show the skill list in `001-request.json` system-reminder. Confirmed across 5 different session HTTP logs -- all identical 8-skill list.

### 2. SOFT BLOCKER: WebSearch/WebFetch are deferred tools requiring discovery

WebSearch and WebFetch ARE present in `<available-deferred-tools>` in the request, but agents must call `ToolSearch("select:WebSearch")` to activate them. Most agents never discover this:

- **87% of sessions** (52/60) never called ToolSearch for WebSearch
- ToolSearch was used almost exclusively for `TaskStop`
- Only 1 session (v2 iter13) successfully used WebSearch after discovering it via ToolSearch
- The CLAUDE.md instruction "use web search to find recent work" assumes the tool is visible, but it's hidden

**Evidence**: Background task scanning all spatial-4x-30min sessions for ToolSearch invocations found only `TaskStop` queries from spatial sessions.

### 3. STRUCTURAL: Prompt signals that discourage finetuning

Despite explicitly requesting finetuning, PROMPT.md contains structural anti-finetuning signals:

#### 3a. "method that worked best in previous experiments"

> "You will focus on looking at more recent papers, finetuning external models, **maybe training 1-2 from-scratch ones with the method that worked best in previous experiments.**"

The organize2 report (linked as prior findings) shows flow matching OT-CFM as the best method. Agents immediately know "the method that worked best" is from-scratch flow matching. The word "maybe" makes from-scratch feel like the low-commitment safe option.

#### 3b. Scratchpad accumulation creates self-reinforcing "from-scratch is best" narrative

The scratchpad grows each iteration. By iteration 8, the leaderboard shows flow matching at CRPS 0.186-0.189. Each new agent reads this and sees the proven winner. Finetuning attempts that fail (v1 iter13: SwinIR bf16, CRPS 0.1866; v4 iter8-9: SwinIR CRPS 0.22) get recorded as negative results, poisoning future iterations.

Key poisoning entries:
- v4 scratchpad: "The 'finetuning external models' direction is effectively exhausted"
- v4 scratchpad: "Pretrained SR: Definitively Exhausted"
- v3 DEC-001 (iteration 1): "Architecture mismatch makes direct weight transfer infeasible" -- persisted for all subsequent iterations

#### 3c. 30-minute budget creates time-pressure bias

Finetuning requires: download/locate weights, adapt architecture (channels, resolution), debug AMP issues, finetune, evaluate. Steps 1-3 eat into the 30-min budget. From-scratch training has zero setup overhead -- reuse existing `src/downscaling/training/` library.

v1 iter13 proved this: SwinIR bf16 finetuning used 8.4 min for precompute+finetune, leaving fewer flow epochs. Result was worse (0.1866 vs 0.183).

#### 3d. "under-explored" interpreted as "novel from-scratch idea"

> "MUST: If possible, choose under-explored / uncertain directions"

Agents satisfy this with new loss functions (IQN-CRPS, discrete CRPS, Gaussian CRPS, multi-resolution CRPS) rather than finetuning, which feels like engineering rather than exploration.

#### 3e. No finetuning infrastructure

`src/downscaling/training/` is built for from-scratch flow matching. No pretrained weight loading utilities, no architecture adaptation code. An agent finetuning GenDiff must write everything from scratch, while from-scratch flow matching is a one-liner.

### 4. PRACTICAL: Available pretrained weights are genuinely mismatched

| Model | Size | Resolution | Channels | Compatibility |
|---|---|---|---|---|
| GenDiff UNet | 97.4M | 256x128 | 5in/3out | Incompatible (resolution, channels) |
| WassDiff NCSN++ | 129M | 256x256 | multi | Incompatible (resolution, channels) |
| SwinIR | 11.9M | any | 3ch RGB | Needs 3ch->1ch adaptation + AMP fix |

SwinIR is the only viable option. When tried (v4 iter8-9), it hit FP16 NaN in Swin Transformer attention. FP32 is too slow under 30-min budget. BF16 works but unfreezing backbone causes catastrophic feature disruption (climate data too far from DF2K natural images).

## Thinking-Block Evidence

Agents explicitly recognized the finetuning gap in thinking blocks, then abandoned it:

**v2 iter13** (most explicit):
> "I should identify the key gaps: I've done minimal literature searching despite 12 iterations, haven't attempted finetuning pretrained models despite that being explicitly suggested, and all experiments have used random initialization. These are the three main concerns to address before proceeding further."
> -- Immediately implements IQN from scratch with random init.

**v2 iter14**:
> "I'm noticing a significant gap: the objective explicitly calls for finetuning external pretrained models, but across 13 iterations I haven't attempted this at all."
> -- Pivots to discrete CRPS from scratch.

**v1 iter12**:
> "I should also scan recent literature for newer approaches since we've been circling the same ideas for a while."
> -- Never executes a web search.

**v3 iter10**: Zero mentions of finetuning in any thinking block despite iteration 8's scratchpad calling it the "#1 gap." The directive was not rationalized away -- it simply disappeared from reasoning.

## Quantified Impact

| Metric | Count |
|---|---|
| Total iterations across all worktrees | ~54 |
| `/arxiv-to-md` invocations | 0 |
| `/decision-critic` invocations | 0 |
| Papers found via web search | ~8 (PC-AFM, AIFS-CRPS, CRPS-LAM, QRE, IQN, etc.) |
| Papers cataloged to `papers/` or CLAUDE.md | 0 |
| Real finetuning attempts | 3 (v1 iter13, v4 iter8, v4 iter9) |
| Finetuning attempts that improved over best | 0 |
| From-scratch experiments | 50+ |
| Sessions that discovered WebSearch via ToolSearch | ~8/60 (13%) |

## Recommendations

### Fix 1: SDK skill configuration
Ensure Ralph's Agent SDK sessions include user-level skills (`/arxiv-to-md`, `/decision-critic`, `/problem-analysis`). The skills exist at `/home/chenxy/.claude/skills/` but are not loaded by the SDK entrypoint (`cc_entrypoint=sdk-cli`).

### Fix 2: Make WebSearch a default tool, not deferred
Or add explicit instruction: "WebSearch is a deferred tool. Call `ToolSearch('select:WebSearch')` before using it."

### Fix 3: Restructure PROMPT.md finetuning directive
- Remove "maybe" from "maybe training 1-2 from-scratch ones"
- Invert the framing: "You MUST attempt finetuning in at least N of your iterations. From-scratch training is allowed for at most 2 iterations."
- Provide concrete finetuning recipes (SwinIR 1ch adaptation, GenDiff projection layers)

### Fix 4: Prevent scratchpad poisoning
- Add instruction: "Do not write 'exhausted' or 'definitively ruled out' about any direction unless you have tried 3+ distinct approaches within that direction"
- Or: scratchpad entries older than N iterations should be summarized, not carried verbatim

### Fix 5: Provide finetuning infrastructure
- Add `src/downscaling/training/finetune.py` with pretrained weight loading
- Add channel adaptation utilities (3ch<->1ch projection)
- Pre-cache adapted checkpoint files in pool data

## Files Examined

- HTTP logs: `/home/chenxy/.claude/http-logs/2026-05-09T*` (system prompts, tool lists, skill lists)
- Session JSONLs: `/home/chenxy/.claude/projects/-workspace/*.jsonl` (10 sessions diagnosed)
- Scratchpads: `/home/chenxy/repos/workspace/spatial-4x-30min-v{1,2,3,4}/.ralph/agent/scratchpad.md`
- Decisions: `/home/chenxy/repos/workspace/spatial-4x-30min-v{1,2,3,4}/.ralph/agent/decisions.md`
- PROMPT.md: `/home/chenxy/repos/workspace/spatial-4x-30min-v1/PROMPT.md`
- Ralph config: `/workspace/ralph/build.yml`
- Training infra: `/workspace/src/downscaling/training/`
- Pool data: `/home/chenxy/orcd/pool/datasets/`
