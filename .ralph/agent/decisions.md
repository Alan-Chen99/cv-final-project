# Decision Journal

## DEC-001
- **Decision**: Use sbatch instead of salloc for GPU evaluation
- **Chosen Option**: sbatch with incremental saving
- **Confidence**: 85
- **Alternatives Considered**: salloc + srun (blocked by competing workflow), wait for ivy-ash to finish
- **Reasoning**: salloc allocations get cancelled by the ivy-ash workflow's cleanup (observed 2x in iter6, 3x in iter5). sbatch jobs also get cancelled while pending, but with `--requeue` and incremental saving, partial results are preserved. The eval script is tested (10-sample sanity passed in iter5).
- **When to re-evaluate**: When ivy-ash workflow is not running, salloc becomes viable again
- **Independent evaluation**: not-started
- **Framing Biases**: Assumes ivy-ash is the cancellation source (observed correlation, not confirmed mechanism)
- **Timestamp**: 2026-05-11T20:50:00Z

## DEC-002
- **Decision**: Write preliminary report from stdout data
- **Chosen Option**: Create METRICS_REPORT.md with iter5 stdout results marked as preliminary
- **Confidence**: 70
- **Alternatives Considered**: Wait for verified JSON results before writing any report
- **Reasoning**: Report was required since iter1 and never created. Using stdout data (manually transcribed in scratchpad) provides directional findings. Clearly marked as preliminary and pending verification.
- **When to re-evaluate**: When JSON results are available from GPU eval
- **Independent evaluation**: not-started
- **Framing Biases**: Stdout data may have transcription errors; results for only 6/15 methods shown
- **Timestamp**: 2026-05-11T20:50:00Z
