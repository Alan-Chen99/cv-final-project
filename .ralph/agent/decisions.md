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

## DEC-003
- **Decision**: Use conditional flow matching instead of fixing GAN diversity or DDPM
- **Chosen Option**: Flow matching with small UNet, Euler ODE inference
- **Confidence**: 85
- **Alternatives Considered**: (1) Increase GAN adv_factor to fix collapse — risky, unstable training; (2) DDPM — more complex, slower inference (1000 steps vs 20); (3) Consistency models — too complex for first attempt
- **Reasoning**: Flow matching is the simplest generative framework for diverse samples: no variance schedule (unlike DDPM), fast inference (20 Euler steps), and linear interpolation training objective. Any model with calibrated spread should beat CRPS=0.307 baseline.
- **Reversibility**: High — can try other approaches in subsequent iterations
- **Timestamp**: 2026-05-03T04:30:00Z

## DEC-004
- **Decision**: Train flow model for 100 epochs (not 200) with AMP
- **Chosen Option**: 100 epochs, ~87 min training
- **Confidence**: 80
- **Alternatives Considered**: 200 epochs (~180 min, might not fit in GPU alloc)
- **Reasoning**: Loss plateaued around epoch 50 (val=0.0025), cosine schedule reaches near-zero LR by epoch 100. 200 epochs would give marginal improvement. 100 epochs fits in 3h GPU allocation with room for eval.
- **Reversibility**: High — can train longer in next iteration
- **Timestamp**: 2026-05-03T04:35:00Z

## DEC-005
- **Decision**: Use multiplicative constraint (not SmCL/softmax) for post-hoc conservation on flow model
- **Chosen Option**: `out = clamp(hr, ε) × (lr / AvgPool(clamp(hr, ε)))↑4×4`
- **Confidence**: 95
- **Alternatives Considered**: SmCL with exp() (CRPS=0.553, terrible), no constraint (CRPS=0.252, mass viol=0.058)
- **Reasoning**: SmCL's exp() distorts the flow model's calibrated [0,1] output — the model was not trained to account for exp(). Mult is a minimal adjustment (scaling factors ≈ 1.0) that preserves the flow output distribution while enforcing exact conservation. Confirmed empirically: mult CRPS=0.246, SmCL CRPS=0.553.
- **Reversibility**: High — constraint is only applied at inference time
- **Timestamp**: 2026-05-03T07:10:00Z

## DEC-006
- **Decision**: Constraint-aware auxiliary loss weight λ=0.1
- **Chosen Option**: λ=0.1 (aux_loss ≈ 0.00002 vs velocity_loss ≈ 0.002)
- **Confidence**: 75
- **Alternatives Considered**: λ=1.0 (may dominate velocity learning), λ=0.01 (may be too small)
- **Reasoning**: At λ=0.1, the aux loss contributes ~0.1% to total loss — gentle regularizer. Achieved CRPS=0.2424 vs 0.2460 post-hoc-only, confirming it helps without hurting velocity prediction (val loss comparable: 0.002068 vs 0.002009).
- **Reversibility**: High — can retrain with different weight
- **Timestamp**: 2026-05-03T07:35:00Z

## DEC-007
- **Decision**: Heun solver not recommended at 20 steps
- **Chosen Option**: Use Euler with 20 steps
- **Confidence**: 95
- **Alternatives Considered**: Heun with more steps (50+)
- **Reasoning**: Heun at 20 steps gives CRPS=0.751 (3× worse than Euler). The velocity field is not smooth enough for the 2nd-order corrector at large dt=0.05. Heun doubles NFE, so 20 Heun steps = 40 NFE ≈ 40 Euler steps. If we want more accuracy, just increase Euler steps.
- **Reversibility**: High — solver choice is inference-only
- **Timestamp**: 2026-05-03T10:10:00Z

## DEC-008
- **Decision**: Use LR-anchor flow matching (start ODE from LR + noise instead of pure noise)
- **Chosen Option**: x_0 = bicubic(LR) + 0.5 * N(0,I), with CA loss and mult constraint
- **Confidence**: 90
- **Alternatives Considered**: (1) More Euler steps with standard flow (marginal gains expected), (2) Larger model (capacity not the bottleneck), (3) Different noise_std values (0.5 worked well, tuning deferred)
- **Reasoning**: Inspired by CDSI (2603.03838) which starts from LR field. Standard flow matching transports from N(0,I) to HR — the model must reconstruct the entire image. LR-anchor reduces transport distance since bicubic LR is already close to HR (SSIM~0.98). Val loss dropped 50% (0.001034 vs 0.002068), CRPS improved 8.5% (0.2218 vs 0.2424). The simpler velocity field also needs fewer Euler steps.
- **Reversibility**: High — new model directory, old models preserved
- **Timestamp**: 2026-05-03T12:22:00Z
