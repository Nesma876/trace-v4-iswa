# RSSM diagnostic

## Finding

The reported RSSM result is a genuine outcome of the frozen implementation, not a transcription error: five distinct checkpoints were selected at epoch 1 with validation NLLs 58.5369, 57.8916, 52.8904, 57.3415, and 58.4312 (mean 57.0183, sample SD 2.3563). Each neural seed has a distinct SHA-256 checkpoint hash. Training ran 21 epochs in every case, consistent with patience 20 after the epoch-1 optimum.

## Loss comparability

`src/trace/physionet2019/scoring.py:79-93` applies one shared observed-target Gaussian NLL and clamps log-variance to [-6,3]. `internal_training.py` adds `0.01 * KL` only to RSSM training loss; validation calls the common prior prediction and `masked_training_nll`, so KL is not included in reported validation NLL. Output dimensions are 33 means plus 33 log-variances. The same target masks and Set A training scaler are used across families.

The stored logs/checkpoint audits report finite reconstruction, KL, total loss, validation NLL, and validation MAE. Checkpoint restoration and stopping metadata are internally consistent. RSSM's poor NLL therefore is not explained by an obvious sign error, inclusion of KL at validation, a different target mask, or duplicate checkpoints.

## Interpretation

The evidence is most consistent with severe under-dispersion on common variables and a mismatch between the fixed RSSM prior rollout objective and one-step validation likelihood. RSSM simultaneously shows lower 48-h DynamicMAE and the smallest change-variance ratio, so its advantage is a smoother first-difference point estimate—not realistic temporal variability or good probabilistic fit. The current evidence cannot distinguish architectural limitation from an optimization/configuration issue. The manuscript's cautious wording is appropriate; any claim that “RSSM is better at dynamics” without this qualification would be unsupported.

## Proposed sensitivity experiment (not performed)

Scientific question: does epoch-1 selection and NLL failure persist when the fixed architecture is trained with configurations that separately address prior/posterior mismatch and variance collapse?

Proposed configurations: frozen baseline; KL weights 0, 0.001, 0.01; free-nats or KL balancing; validation of posterior reconstruction NLL versus prior predictive NLL; variance-floor sensitivity while retaining the same data splits and masks. Use five existing seeds, maximum 200 epochs, and no locked-test/Set B inspection until validation choices are frozen.

Expected cost: approximately 15–20 additional RSSM fits (about 3–4 GPU-hours if runtime matches the recorded 11–14 minutes/run, plus evaluation). Required artifacts: configs, full per-epoch component logs, checkpoint hashes, prior/posterior validation predictions, and a preregistered selection rule. It could confirm or invalidate the interpretation that the observed RSSM behavior is configuration-specific; it must not be presented as an existing result.

