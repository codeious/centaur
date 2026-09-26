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

    def test_no_state_volume_keeps_ephemeral_layout(self) -> None:
        text = ENTRYPOINT.read_text()
        prelude_marker = (
            'install-tool-shims || echo "warning: failed to install Centaur tool CLI shims" >&2\nfi\n'
        )
        start = text.find(prelude_marker)
        if start < 0:
            raise AssertionError("entrypoint.sh is missing the STATE_DIR setup region")
        start += len(prelude_marker)
        end = text.find("\nfi\n", start)
        if end < 0:
            raise AssertionError("entrypoint.sh STATE_DIR linking block is unclosed")
        block = text[start : end + 4]

        ws_start = text.find('if [ "${CENTAUR_PERSISTENT_STATE:-0}" = "1" ]; then')
        if ws_start < 0:
            raise AssertionError("entrypoint.sh is missing the WORKSPACE_DIR assignment")
        ws_end = text.find("\nfi\n", ws_start)
        if ws_end < 0:
            raise AssertionError("entrypoint.sh WORKSPACE_DIR assignment is unclosed")
        workspace_block = text[ws_start : ws_end + 4]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            home.mkdir()
            state = home / "state"

            persist_file = root / "persist"
            workspace_file = root / "workspace"
            script = f"""
set -euo pipefail
HOME_DIR={home.as_posix()!r}
STATE_DIR={state.as_posix()!r}
unset CENTAUR_PERSISTENT_STATE
{block}
{workspace_block}
printf '%s\\n' "${{CENTAUR_PERSISTENT_STATE:-}}" > {persist_file.as_posix()!r}
printf '%s\\n' "$WORKSPACE_DIR" > {workspace_file.as_posix()!r}
"""
            subprocess.run(["bash", "-c", script], check=True)

            self.assertFalse(state.exists(), "$HOME/state must not be created without a volume")
            self.assertNotEqual(persist_file.read_text().strip(), "1")
            for name in (".codex", ".claude", ".factory", "uploads", "branches"):
                path = home / name
                if path.is_symlink():
                    target = os.readlink(path)
                    self.assertFalse(
                        os.path.abspath(target).startswith(os.path.abspath(state) + os.sep)
                        or os.path.abspath(target) == os.path.abspath(state),
                        f"{name} must not be a symlink into STATE_DIR (target={target!r})",
                    )
            self.assertEqual(workspace_file.read_text().strip(), str(home / "workspace"))


if __name__ == "__main__":
    unittest.main()
