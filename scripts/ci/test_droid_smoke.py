#!/usr/bin/env python3
"""In-process unit tests for droid_smoke.py.

Patches build_docker_cmd so the "container" is `sys.executable` running a
temp fake that speaks App Server JSON-RPC on stdout. No docker, no network,
no real key.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


DUMMY_KEY = "test-key-not-real-0123456789"
SCRIPT = Path(__file__).resolve().parent / "droid_smoke.py"

FAKE_CONTAINER = r"""#!/usr/bin/env python3
import json
import os
import sys
import time

mode = os.environ.get("FAKE_MODE", "pong")
sys.stdin.readline()
if mode == "pong":
    print(
        json.dumps(
            {
                "method": "item/agentMessage/delta",
                "params": {"delta": "PONG"},
            }
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "PONG"}],
                    }
                },
            }
        ),
        flush=True,
    )
elif mode == "error":
    print(
        json.dumps(
            {
                "method": "error",
                "params": {"error": {"message": "Authentication required"}},
            }
        ),
        flush=True,
    )
    time.sleep(30)
elif mode == "leak":
    key = os.environ.get("FACTORY_API_KEY", "")
    print(key, flush=True)
    print(
        json.dumps(
            {
                "method": "item/agentMessage/delta",
                "params": {"delta": "PONG"},
            }
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "PONG"}],
                    }
                },
            }
        ),
        flush=True,
    )
elif mode == "silent":
    time.sleep(30)
"""


def load_droid_smoke():
    if not SCRIPT.is_file():
        raise ModuleNotFoundError("droid_smoke")
    spec = importlib.util.spec_from_file_location("droid_smoke", SCRIPT)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError("droid_smoke")
    module = importlib.util.module_from_spec(spec)
    sys.modules["droid_smoke"] = module
    spec.loader.exec_module(module)
    return module


class DroidSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_droid_smoke()
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="droid-smoke-test-")
        cls.fake_path = Path(cls._tmpdir.name) / "fake_container.py"
        cls.fake_path.write_text(FAKE_CONTAINER, encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        self.out_dir = tempfile.mkdtemp(prefix="droid-smoke-out-")

    def _builder(self, image: str, pass_key: bool):
        del image, pass_key
        return [sys.executable, "-u", str(self.fake_path)]

    def _run(self, extra_argv: list[str], env: dict[str, str | None], builder=None) -> int:
        full_env = os.environ.copy()
        for key, value in env.items():
            if value is None:
                full_env.pop(key, None)
            else:
                full_env[key] = value
        argv = [
            "droid_smoke.py",
            "--image",
            "centaur-agent:test",
            "--out-dir",
            self.out_dir,
            *extra_argv,
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, full_env, clear=True):
            if builder is None:
                return self.mod.main()
            with mock.patch.object(self.mod, "build_docker_cmd", builder):
                return self.mod.main()

    def test_pong_turn_exits_zero(self) -> None:
        code = self._run(
            ["--timeout", "10"],
            env={"FACTORY_API_KEY": DUMMY_KEY, "FAKE_MODE": "pong"},
            builder=self._builder,
        )
        self.assertEqual(code, 0)

    def test_protocol_error_fails_fast(self) -> None:
        started = time.monotonic()
        code = self._run(
            ["--timeout", "20"],
            env={"FACTORY_API_KEY": DUMMY_KEY, "FAKE_MODE": "error"},
            builder=self._builder,
        )
        elapsed = time.monotonic() - started
        self.assertEqual(code, 1)
        self.assertLess(elapsed, 5.0)

    def test_key_leak_exits_four(self) -> None:
        code = self._run(
            ["--timeout", "10"],
            env={"FACTORY_API_KEY": DUMMY_KEY, "FAKE_MODE": "leak"},
            builder=self._builder,
        )
        self.assertEqual(code, 4)

    def test_no_turn_completed_times_out(self) -> None:
        code = self._run(
            ["--timeout", "2"],
            env={"FACTORY_API_KEY": DUMMY_KEY, "FAKE_MODE": "silent"},
            builder=self._builder,
        )
        self.assertEqual(code, 2)

    def test_missing_key_exits_three_without_container(self) -> None:
        builder = mock.Mock(side_effect=AssertionError("build_docker_cmd should not be called"))
        code = self._run(
            [],
            env={"FACTORY_API_KEY": None, "FAKE_MODE": None},
            builder=builder,
        )
        self.assertEqual(code, 3)
        builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
