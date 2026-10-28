from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).parent.parent / ".github" / "workflows").glob("*.yml"))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_a_workflow_starts_with_read_only_permissions(path):
    """The token a workflow gets can only read the repository, unless a job asks for more by name."""
    assert yaml.safe_load(path.read_text()).get("permissions") == {"contents": "read"}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_jobs_that_set_their_own_permissions_only_ask_for_the_publishing_token(path):
    for name, job in yaml.safe_load(path.read_text())["jobs"].items():
        assert job.get("permissions", {"id-token": "write"}) == {"id-token": "write"}, name
