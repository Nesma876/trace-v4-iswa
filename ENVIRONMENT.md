# Environment

Audited environment: Python 3.12.14 on 64-bit Windows. The operating-system and GPU CIM queries were denied by the execution sandbox, so exact OS build, GPU, driver, CUDA, and cuDNN versions are **NOT AVAILABLE** from this run.

Core frozen requirements are in `requirements-lock.txt`; the audited environment used NumPy 2.3.5, pandas 3.0.1, PyTorch 2.14.0, matplotlib 3.11.1, PyYAML 6.0.3, and pytest 9.1.1. The repository package was installed from commit `fe5110dfd8e42b5e4c4b318339f1da66044ed991`.

The short lock file is sufficient for the tested source-checkout workflow but is not a complete hardware/driver lock. For archival reproduction, record `pip freeze`, OS build, CPU/GPU model, driver, CUDA/cuDNN, and deterministic-algorithm settings on the publication machine.

