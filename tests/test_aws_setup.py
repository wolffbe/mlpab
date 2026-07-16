"""Unit tests for the SageMaker platform setup (configs/platforms/aws/setup.py),
focused on the execution role's ECR-pull grant: a job can't pull its container
image without it, and a PRE-EXISTING under-permissioned role must be repaired
(not just freshly-created ones). Offline — a fake IAM client, no AWS."""

import importlib.util
import json
import os
import unittest
from pathlib import Path

_SETUP = Path(__file__).resolve().parents[1] / "configs" / "platforms" / "aws" / "setup.py"


def _load_setup():
    spec = importlib.util.spec_from_file_location("aws_setup", _SETUP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeIam:
    """Records calls; get_role optionally raises to drive the create path."""

    def __init__(self, role_exists: bool):
        self._role_exists = role_exists
        self.calls: list[tuple] = []

    def get_role(self, RoleName):
        self.calls.append(("get_role", RoleName))
        if not self._role_exists:
            raise RuntimeError("NoSuchEntity")
        return {"Role": {"Arn": f"arn:aws:iam::111:role/{RoleName}"}}

    def create_role(self, **kw):
        self.calls.append(("create_role", kw["RoleName"]))
        return {"Role": {"Arn": f"arn:aws:iam::111:role/{kw['RoleName']}"}}

    def attach_role_policy(self, **kw):
        self.calls.append(("attach_role_policy", kw["PolicyArn"]))

    def put_role_policy(self, **kw):
        self.calls.append(("put_role_policy", kw["PolicyName"], kw["PolicyDocument"]))


class _FakeSession:
    def __init__(self, iam):
        self._iam = iam

    def create_client(self, name, region_name=None):
        assert name == "iam"
        return self._iam


class EcrPullPolicyTests(unittest.TestCase):
    def setUp(self):
        self.setup = _load_setup()
        self._saved = os.environ.pop("SAGEMAKER_ROLE_ARN", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["SAGEMAKER_ROLE_ARN"] = self._saved

    def _names(self, iam):
        return [c[0] for c in iam.calls]

    def _ecr_doc(self, iam):
        put = next(c for c in iam.calls if c[0] == "put_role_policy")
        self.assertEqual(put[1], self.setup.ECR_PULL_POLICY_NAME)
        return json.loads(put[2])

    def test_existing_role_is_repaired_with_ecr_pull(self):
        # The bug this fixes: the "already exists" path used to return without
        # ensuring ANY policy, leaving an under-permissioned role unrepaired.
        iam = _FakeIam(role_exists=True)
        arn = self.setup._ensure_execution_role(_FakeSession(iam), "eu-north-1")
        self.assertIn("put_role_policy", self._names(iam))
        self.assertNotIn("create_role", self._names(iam))
        self.assertTrue(arn and arn.endswith(self.setup.DEFAULT_ROLE_NAME))

    def test_created_role_also_gets_ecr_pull(self):
        iam = _FakeIam(role_exists=False)
        self.setup._ensure_execution_role(_FakeSession(iam), "eu-north-1")
        names = self._names(iam)
        self.assertIn("create_role", names)
        self.assertIn("attach_role_policy", names)
        self.assertIn("put_role_policy", names)

    def test_ecr_policy_grants_the_three_pull_actions(self):
        iam = _FakeIam(role_exists=True)
        self.setup._ensure_execution_role(_FakeSession(iam), "eu-north-1")
        doc = self._ecr_doc(iam)
        actions = set()
        for stmt in doc["Statement"]:
            a = stmt["Action"]
            actions.update([a] if isinstance(a, str) else a)
        self.assertTrue(
            {
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
            }
            <= actions,
            actions,
        )

    def test_ensure_policies_is_best_effort(self):
        # Setup runs via the `serve:` step whose runner ignores failures —
        # nothing here may raise (e.g. an IAM user lacking iam:PutRolePolicy),
        # and a failure of one grant must not skip the other.
        class _Boom:
            def attach_role_policy(self, **kw):
                raise RuntimeError("AccessDenied")

            def put_role_policy(self, **kw):
                raise RuntimeError("AccessDenied")

        self.setup._ensure_policies(_Boom(), "role")  # must not raise


if __name__ == "__main__":
    unittest.main()
