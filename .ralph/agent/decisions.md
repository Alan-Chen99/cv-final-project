# Decision Journal

## DEC-001
- **Decision**: Use correct CRPS formula instead of baseline's buggy implementation
- **Chosen Option**: Standard CRPS = E|X-y| - 0.5*E|X-X'|
- **Confidence**: 95
- **Alternatives Considered**: Use baseline's formula for consistency with paper
- **Reasoning**: Baseline's `crps_ensemble()` uses `fc.shape[-1]**2` (=128²=16384) instead of `fc.shape[0]**2` (=M²=100) in the first loop weight. This makes below-forecast contributions ~1638x too small. On 128×128 data, the buggy CRPS is ~0.5x the correct value. Using the correct formula is essential for meaningful comparisons.
- **Reversibility**: High — just need to re-evaluate with different formula
- **Timestamp**: 2026-05-02T22:10:00Z

## DEC-002
- **Decision**: Train GAN baselines (none, softmax) at 200 epochs batch_size=256
- **Chosen Option**: Follow paper's training recipe exactly for baseline
- **Confidence**: 90
- **Alternatives Considered**: Fewer epochs (100), different batch size
- **Reasoning**: Timing test shows ~96 min per GAN model at 200 epochs on L40S. Two models = ~192 min, fits within 4hr iteration. Using paper's exact hyperparameters ensures fair comparison.
- **Reversibility**: High — can retrain with different settings
- **Timestamp**: 2026-05-02T22:10:00Z
