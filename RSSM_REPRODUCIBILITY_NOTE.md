# RSSM reproducibility and interpretation

The release preserves five independent RSSM runs for seeds 17, 23, 31, 43, and 59. Their checkpoints have distinct SHA-256 hashes. Every run selected epoch 1 and stopped after epoch 21 under patience 20. Validation NLL values were 58.536944, 57.891570, 52.890418, 57.341452, and 58.431223 (mean 57.018321; sample SD 2.3563).

The fixed RSSM uses deterministic state 64, stochastic state 16, hidden size 64, and KL weight 0.01. Training adds the weighted KL term, while validation uses the same observed-target Gaussian NLL as other neural models and does not include KL. The output has 33 means and 33 log-variances, with log-variance clamped to [-6, 3].

At 48 hours the evaluated RSSM has lower DynamicMAE than Transformer, but also much stronger contraction of first-difference variance and poorer likelihood/coverage. This supports only a configuration-specific difference in one temporal-change statistic. It is not evidence that RSSM architectures generally forecast better, preserve realistic dynamics, or provide superior probabilistic trajectories. No sensitivity experiment or retraining was performed for this release.

