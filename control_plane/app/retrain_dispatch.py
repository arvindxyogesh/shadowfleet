import logging
from typing import Protocol

logger = logging.getLogger("shadowfleet.control_plane.retrain_dispatch")


class RetrainDispatcher(Protocol):
    """How a retrain trigger actually kicks off training_pipeline. FR-5's
    acceptance criterion only requires the trigger to be recorded and a run
    invoked -- what "invoked" means is pluggable so a real deployment can
    wire in CI without the control plane needing GitHub credentials to run
    its test suite.
    """

    def dispatch(self, labeled_example_count: int, threshold: int) -> tuple[str, dict]: ...


class LoggingRetrainDispatcher:
    """Default dispatcher: just logs. Safe with zero configuration -- the
    RetrainTrigger row it returns is itself the auditable record that
    FR-5 asks for; a human or a scheduled job picks it up and runs
    training_pipeline/scripts/train.py (see that package's README).
    """

    def dispatch(self, labeled_example_count: int, threshold: int) -> tuple[str, dict]:
        logger.info(
            "retrain trigger fired: %d labeled hard examples (threshold=%d) -- "
            "run training_pipeline/scripts/train.py",
            labeled_example_count,
            threshold,
        )
        return "log_only", {}


class GitHubActionsRetrainDispatcher:
    """Fires a `repository_dispatch` event to trigger a training workflow.
    Not wired up by default -- needs a repo and a token with `repo` scope,
    which this project's test suite and free-tier demo deliberately never
    require. Construct and pass to the control plane explicitly to use it.
    """

    def __init__(self, repo: str, token: str, event_type: str = "shadowfleet-retrain"):
        self.repo = repo
        self.token = token
        self.event_type = event_type

    def dispatch(self, labeled_example_count: int, threshold: int) -> tuple[str, dict]:
        import httpx

        resp = httpx.post(
            f"https://api.github.com/repos/{self.repo}/dispatches",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "event_type": self.event_type,
                "client_payload": {"labeled_example_count": labeled_example_count, "threshold": threshold},
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        return "github_repository_dispatch", {"repo": self.repo, "event_type": self.event_type}
