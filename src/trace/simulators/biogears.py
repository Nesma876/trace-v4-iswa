"""BioGears adapter skeleton."""

class BioGearsSimulator:
    """Placeholder adapter. Do not fake BioGears results."""

    def __init__(self, executable_path: str | None = None) -> None:
        self.executable_path = executable_path

    def reset(self, patient_state: dict[str, float]) -> None:
        raise NotImplementedError("BioGears integration is not implemented in Milestone 1.")

