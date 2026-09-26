from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ENTRYPOINT = Path(__file__).with_name("entrypoint.sh")


def _state_dir_block() -> str:
    text = ENTRYPOINT.read_text()
    start = text.find('if [ -d "$STATE_DIR" ] && [ -w "$STATE_DIR" ]; then')
    if start < 0:
        raise AssertionError("entrypoint.sh is missing the STATE_DIR linking block")
    end = text.find("\nfi\n", start)
    if end < 0:
        raise AssertionError("entrypoint.sh STATE_DIR linking block is unclosed")
    return text[start : end + 4]


class EntrypointStateLinksTest(unittest.TestCase):
    def test_factory_dir_linked_to_state_volume(self) -> None:
        block = _state_dir_block()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            state = root / "state"
            home.mkdir()
            state.mkdir()
            (home / ".codex").mkdir()
            (home / ".claude").mkdir()
            (home / ".factory").mkdir()
            (home / "uploads").mkdir()
            (home / "branches").mkdir()

            script = f"""
set -euo pipefail
HOME_DIR={home.as_posix()!r}
STATE_DIR={state.as_posix()!r}
{block}
"""
            subprocess.run(["bash", "-c", script], check=True)

            self.assertTrue((state / "factory").is_dir())
            self.assertTrue((home / ".factory").is_symlink())
            self.assertEqual(os.readlink(home / ".factory"), str(state / "factory"))
            self.assertTrue((home / ".codex").is_symlink())
            self.assertEqual(os.readlink(home / ".codex"), str(state / "codex"))
            self.assertTrue((home / ".claude").is_symlink())
            self.assertEqual(os.readlink(home / ".claude"), str(state / "claude"))

    def test_factory_dir_linked_when_state_dir_is_missing(self) -> None:
        text = ENTRYPOINT.read_text()
        mkdir_at = text.find('mkdir -p "$STATE_DIR"')
        if mkdir_at < 0:
            raise AssertionError("entrypoint.sh must create STATE_DIR before linking ~/.factory")
        end = text.find("\nfi\n", mkdir_at)
        if end < 0:
            raise AssertionError("entrypoint.sh STATE_DIR linking block is unclosed")
        block = text[mkdir_at : end + 4]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            state = root / "state"
            home.mkdir()

            script = f"""
set -euo pipefail
HOME_DIR={home.as_posix()!r}
STATE_DIR={state.as_posix()!r}
{block}
"""
            subprocess.run(["bash", "-c", script], check=True)

            self.assertTrue(state.is_dir())
            self.assertTrue((state / "factory").is_dir())
            self.assertTrue((home / ".factory").is_symlink())
            self.assertEqual(os.readlink(home / ".factory"), str(state / "factory"))


if __name__ == "__main__":
    unittest.main()
