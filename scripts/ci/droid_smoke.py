#!/usr/bin/env python3
"""CI smoke: drive `harness-server droid` inside the agent image.

Stdlib only. Writes one Centaur user block to stdin, waits for a
`turn/completed` App Server notification, then closes stdin.

Exit 0 only if the turn completed and collected assistant text contains
"PONG". Never prints FACTORY_API_KEY. Exits non-zero if the key value
appears in container stdout or stderr.

Exit codes:
  0  PONG
  1  protocol error / container exit / no PONG
  2  timeout
  3  FACTORY_API_KEY missing from the environment
  4  key value seen in container stdout or stderr (wins over every other result)

Usage:
  python3 scripts/ci/droid_smoke.py --image ghcr.io/<owner>/<repo>/centaur-agent@sha256:<digest>
  python3 scripts/ci/droid_smoke.py --image centaur-agent:latest --no-key
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


REDACTED = "***REDACTED***"
PROMPT = "Reply with exactly: PONG"
USER_BLOCK = {
    "type": "user",
    "message": {"role": "user", "content": [{"type": "text", "text": PROMPT}]},
}


class SmokeError(Exception):
    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def default_out_dir() -> str:
    runner_temp = os.environ.get("RUNNER_TEMP")
    if runner_temp:
        return str(Path(runner_temp) / "droid-smoke")
    return tempfile.mkdtemp(prefix="droid-smoke-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI Droid smoke against the agent image")
    parser.add_argument(
        "--image",
        default=os.environ.get("CENTAUR_AGENT_IMAGE"),
        required=not bool(os.environ.get("CENTAUR_AGENT_IMAGE")),
        help="agent image ref (or CENTAUR_AGENT_IMAGE)",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--exit-timeout", type=float, default=15.0)
    parser.add_argument("--no-key", action="store_true", help="do not pass FACTORY_API_KEY")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for redacted logs (default: $RUNNER_TEMP/droid-smoke or a fresh temp dir)",
    )
    args = parser.parse_args()
    if not args.image:
        parser.error("--image is required (or set CENTAUR_AGENT_IMAGE)")
    if not args.out_dir:
        args.out_dir = default_out_dir()
    return args


def redact(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, REDACTED)


def json_method(parsed: Any) -> str | None:
    if isinstance(parsed, dict) and isinstance(parsed.get("method"), str):
        return parsed["method"]
    return None


def collect_assistant_text(parsed: dict[str, Any], bucket: list[str]) -> None:
    method = json_method(parsed)
    params = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}

    if method == "item/agentMessage/delta":
        delta = params.get("delta")
        if isinstance(delta, str) and delta:
            bucket.append(delta)
        return

    items: list[Any] = []
    if method == "turn/completed":
        turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
        raw_items = turn.get("items")
        if isinstance(raw_items, list):
            items.extend(raw_items)
    elif method == "item/completed":
        item = params.get("item")
        if isinstance(item, dict):
            items.append(item)

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "agentMessage":
            text = item.get("text")
            if isinstance(text, str) and text:
                bucket.append(text)


def reader_thread(stream, dest: queue.Queue[str | None]) -> None:
    try:
        for raw in iter(stream.readline, b""):
            dest.put(raw.decode("utf-8", "replace"))
    finally:
        dest.put(None)


def build_docker_cmd(image: str, pass_key: bool) -> list[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "-e",
        "FACTORY_DROID_AUTO_UPDATE_ENABLED=false",
        "-e",
        "DROID_MODEL=gpt-5.4-mini-fast",
        "-e",
        "DROID_REASONING_EFFORT=low",
    ]
    if pass_key:
        # Inherit from the caller environment; do not put the value on argv.
        cmd.extend(["-e", "FACTORY_API_KEY"])
    cmd.extend([image, "harness-server", "droid"])
    return cmd


def write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    secret = None if args.no_key else os.environ.get("FACTORY_API_KEY")
    if not args.no_key and not secret:
        sys.stderr.write("FACTORY_API_KEY is not set in the environment\n")
        return 3

    cmd = build_docker_cmd(args.image, pass_key=not args.no_key)
    started = time.monotonic()
    proc: subprocess.Popen[bytes] | None = None
    stdout_lines: list[str] = []
    stderr_chunks: list[str] = []
    methods: list[str] = []
    method_counts: dict[str, int] = {}
    assistant_parts: list[str] = []
    turn_completed = False
    leak = False
    child_exit: int | None = None
    error_message = ""
    non_json = 0

    stdout_q: queue.Queue[str | None] = queue.Queue()
    stderr_q: queue.Queue[str | None] = queue.Queue()

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        threading.Thread(target=reader_thread, args=(proc.stdout, stdout_q), daemon=True).start()
        threading.Thread(target=reader_thread, args=(proc.stderr, stderr_q), daemon=True).start()

        line = json.dumps(USER_BLOCK, separators=(",", ":"))
        proc.stdin.write(line.encode("utf-8") + b"\n")
        proc.stdin.flush()

        stdout_done = False
        stderr_done = False
        deadline = time.monotonic() + args.timeout
        while not turn_completed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SmokeError(
                    f"timeout after {args.timeout:g}s waiting for turn/completed; "
                    f"methods={methods}",
                    exit_code=2,
                )
            if proc.poll() is not None and stdout_q.empty() and stderr_q.empty():
                child_exit = proc.returncode
                raise SmokeError(
                    f"container exited {child_exit} before turn/completed; "
                    f"methods={methods}",
                    exit_code=1,
                )
            try:
                item = stdout_q.get(timeout=min(0.2, remaining))
            except queue.Empty:
                item = "empty"  # type: ignore[assignment]
                # Drain stderr opportunistically.
                while True:
                    try:
                        err = stderr_q.get_nowait()
                    except queue.Empty:
                        break
                    if err is None:
                        stderr_done = True
                        break
                    stderr_chunks.append(err)
                    if secret and secret in err:
                        leak = True
                continue
            if item is None:
                stdout_done = True
                if proc.poll() is not None:
                    child_exit = proc.returncode
                    raise SmokeError(
                        f"container exited {child_exit} before turn/completed; "
                        f"methods={methods}",
                        exit_code=1,
                    )
                continue
            stdout_lines.append(item)
            if secret and secret in item:
                leak = True
            stripped = item.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                non_json += 1
                continue
            if not isinstance(parsed, dict):
                non_json += 1
                continue
            method = json_method(parsed)
            if method:
                methods.append(method)
                method_counts[method] = method_counts.get(method, 0) + 1
            collect_assistant_text(parsed, assistant_parts)
            if method == "error":
                params = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}
                err = params.get("error") if isinstance(params.get("error"), dict) else {}
                msg = err.get("message") if isinstance(err.get("message"), str) else str(params)
                raise SmokeError(f"protocol error before turn/completed: {msg}", exit_code=1)
            if method == "turn/completed":
                turn_completed = True

        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass

        exit_deadline = time.monotonic() + args.exit_timeout
        while time.monotonic() < exit_deadline:
            while True:
                try:
                    item = stdout_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    stdout_done = True
                    break
                stdout_lines.append(item)
                if secret and secret in item:
                    leak = True
                stripped = item.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    non_json += 1
                    continue
                if isinstance(parsed, dict):
                    method = json_method(parsed)
                    if method:
                        methods.append(method)
                        method_counts[method] = method_counts.get(method, 0) + 1
                    collect_assistant_text(parsed, assistant_parts)
            while True:
                try:
                    err = stderr_q.get_nowait()
                except queue.Empty:
                    break
                if err is None:
                    stderr_done = True
                    break
                stderr_chunks.append(err)
                if secret and secret in err:
                    leak = True
            if proc.poll() is not None:
                child_exit = proc.returncode
                break
            time.sleep(0.05)
        else:
            proc.kill()
            child_exit = proc.wait(timeout=5)
            error_message = f"killed after turn/completed; did not exit within {args.exit_timeout:g}s"

        # Final drain
        while not stdout_done:
            try:
                item = stdout_q.get(timeout=0.2)
            except queue.Empty:
                break
            if item is None:
                break
            stdout_lines.append(item)
            if secret and secret in item:
                leak = True
        while not stderr_done:
            try:
                err = stderr_q.get(timeout=0.2)
            except queue.Empty:
                break
            if err is None:
                break
            stderr_chunks.append(err)
            if secret and secret in err:
                leak = True

        if proc.poll() is None:
            proc.kill()
            child_exit = proc.wait(timeout=5)

    except SmokeError as exc:
        error_message = str(exc)
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                child_exit = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child_exit = -9
        drain_deadline = time.monotonic() + 1.0
        while time.monotonic() < drain_deadline:
            progressed = False
            try:
                item = stdout_q.get_nowait()
                progressed = True
                if item is not None:
                    stdout_lines.append(item)
                    if secret and secret in item:
                        leak = True
            except queue.Empty:
                pass
            try:
                err = stderr_q.get_nowait()
                progressed = True
                if err is not None:
                    stderr_chunks.append(err)
                    if secret and secret in err:
                        leak = True
            except queue.Empty:
                pass
            if not progressed:
                break
        elapsed = time.monotonic() - started
        assistant = "".join(assistant_parts)
        stderr_text = "".join(stderr_chunks)
        if secret and (secret in stderr_text or any(secret in line for line in stdout_lines)):
            leak = True
        _write_artifacts(
            out_dir,
            args,
            cmd,
            stdout_lines,
            stderr_text,
            methods,
            method_counts,
            assistant,
            turn_completed,
            leak,
            child_exit,
            elapsed,
            non_json,
            error_message,
            secret,
        )
        _print_summary(
            args=args,
            elapsed=elapsed,
            turn_completed=turn_completed,
            assistant=assistant,
            methods=methods,
            method_counts=method_counts,
            child_exit=child_exit,
            leak=leak,
            non_json=non_json,
            stderr_text=stderr_text,
            error_message=error_message,
            secret=secret,
        )
        if leak:
            sys.stderr.write("secret value appeared in container output\n")
            return 4
        return exc.exit_code
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            proc.kill()
        sys.stderr.write("interrupted\n")
        return 130

    elapsed = time.monotonic() - started
    assistant = "".join(assistant_parts)
    stderr_text = "".join(stderr_chunks)
    if secret and (secret in stderr_text or any(secret in line for line in stdout_lines)):
        leak = True
    pong = "PONG" in assistant
    if leak:
        error_message = error_message or "secret value appeared in container output"
    elif not turn_completed:
        error_message = error_message or "turn/completed was not observed"
    elif not pong:
        error_message = error_message or "assistant text did not contain PONG"

    _write_artifacts(
        out_dir,
        args,
        cmd,
        stdout_lines,
        stderr_text,
        methods,
        method_counts,
        assistant,
        turn_completed,
        leak,
        child_exit,
        elapsed,
        non_json,
        error_message,
        secret,
    )
    _print_summary(
        args=args,
        elapsed=elapsed,
        turn_completed=turn_completed,
        assistant=assistant,
        methods=methods,
        method_counts=method_counts,
        child_exit=child_exit,
        leak=leak,
        non_json=non_json,
        stderr_text=stderr_text,
        error_message=error_message,
        secret=secret,
    )
    if leak:
        return 4
    if turn_completed and pong:
        return 0
    return 1


def _write_artifacts(
    out_dir: Path,
    args: argparse.Namespace,
    cmd: list[str],
    stdout_lines: list[str],
    stderr_text: str,
    methods: list[str],
    method_counts: dict[str, int],
    assistant: str,
    turn_completed: bool,
    leak: bool,
    child_exit: int | None,
    elapsed: float,
    non_json: int,
    error_message: str,
    secret: str | None,
) -> None:
    suffix = "nokey" if args.no_key else "key"
    write_log(
        out_dir / f"stdout-{suffix}.jsonl",
        "".join(
            redact(line, secret) if line.endswith("\n") else redact(line, secret) + "\n"
            for line in stdout_lines
        ),
    )
    write_log(out_dir / f"stderr-{suffix}.log", redact(stderr_text, secret))
    summary = {
        "image": args.image,
        "no_key": args.no_key,
        "argv": cmd,
        "elapsed_s": round(elapsed, 3),
        "turn_completed": turn_completed,
        "pong": "PONG" in assistant,
        "assistant_text": redact(assistant, secret),
        "methods": methods,
        "method_counts": method_counts,
        "child_exit": child_exit,
        "secret_leak": leak,
        "non_json_stdout_lines": non_json,
        "stderr_bytes": len(stderr_text.encode("utf-8")),
        "error": redact(error_message, secret) if error_message else None,
    }
    write_log(out_dir / f"summary-{suffix}.json", json.dumps(summary, indent=2) + "\n")


def _print_summary(
    *,
    args: argparse.Namespace,
    elapsed: float,
    turn_completed: bool,
    assistant: str,
    methods: list[str],
    method_counts: dict[str, int],
    child_exit: int | None,
    leak: bool,
    non_json: int,
    stderr_text: str,
    error_message: str,
    secret: str | None,
) -> None:
    pong = "PONG" in assistant
    unique_methods = list(dict.fromkeys(methods))
    preview = redact(assistant, secret)
    if len(preview) > 120:
        preview = preview[:120] + "…"
    stderr_preview = redact(stderr_text, secret).strip()
    if len(stderr_preview) > 400:
        stderr_preview = stderr_preview[:400] + "…"
    print(f"image={args.image}")
    print(f"no_key={args.no_key}")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"turn_completed={turn_completed} pong={pong} child_exit={child_exit}")
    print(f"secret_leak={leak} non_json_stdout_lines={non_json}")
    print(f"methods={unique_methods}")
    print(f"method_counts={method_counts}")
    print(f"assistant_preview={preview!r}")
    print(f"stderr_bytes={len(stderr_text.encode('utf-8'))}")
    if stderr_preview:
        print("stderr_preview:")
        print(stderr_preview)
    if error_message:
        print(f"error={redact(error_message, secret)}")


if __name__ == "__main__":
    sys.exit(main())
