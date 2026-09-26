#!/usr/bin/env python3
"""Scripted ACP agent for `harness-server droid` integration tests.

Selected as the child via DROID_BIN. Scenario comes from DROID_FAKE_SCENARIO
(default: pong). Optional JSONL log at DROID_FAKE_LOG. Stdlib only.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import queue
from typing import Any

SESSION_ID = "11111111-1111-4111-8111-111111111111"


def _log_path() -> str | None:
    path = os.environ.get("DROID_FAKE_LOG", "").strip()
    return path or None


def log_event(event: dict[str, Any]) -> None:
    path = _log_path()
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def write_msg(obj: dict[str, Any]) -> None:
    log_event({"dir": "out", "msg": obj})
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def respond(req: dict[str, Any], result: dict[str, Any]) -> None:
    write_msg({"jsonrpc": "2.0", "id": req["id"], "result": result})


def notify_update(session_id: str, update: dict[str, Any]) -> None:
    write_msg(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session_id, "update": update},
        }
    )


def stream_text(session_id: str, text: str) -> None:
    notify_update(
        session_id,
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        },
    )


def end_turn(req: dict[str, Any]) -> None:
    respond(req, {"stopReason": "end_turn"})


def wait_for(incoming: queue.Queue, predicate, timeout: float = 8.0) -> dict[str, Any]:
    leftover: list[dict[str, Any]] = []
    try:
        while True:
            msg = incoming.get(timeout=timeout)
            if msg is None:
                raise TimeoutError("stdin closed while waiting")
            if predicate(msg):
                for item in leftover:
                    incoming.put(item)
                return msg
            leftover.append(msg)
    except queue.Empty as exc:
        raise TimeoutError("timed out waiting for ACP message") from exc


def wait_response(incoming: queue.Queue, req_id: Any, timeout: float = 8.0) -> dict[str, Any]:
    return wait_for(
        incoming,
        lambda msg: msg.get("id") == req_id and "method" not in msg,
        timeout,
    )


def permission_params(session_id: str) -> dict[str, Any]:
    return {
        "sessionId": session_id,
        "options": [
            {"optionId": "proceed_once", "name": "Allow", "kind": "allow_once"},
            {
                "optionId": "proceed_always",
                "name": "Allow always",
                "kind": "allow_always",
            },
            {"optionId": "cancel", "name": "No", "kind": "reject_once"},
        ],
        "toolCall": {"toolCallId": "call_perm_1", "title": "echo hi", "kind": "execute"},
    }


def send_permission(session_id: str, req_id: Any) -> None:
    write_msg(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "session/request_permission",
            "params": permission_params(session_id),
        }
    )


def run_terminal_roundtrip(
    incoming: queue.Queue,
    session_id: str,
    command: str,
    first_id: Any,
    output_byte_limit: int | None = None,
) -> str:
    create: dict[str, Any] = {
        "sessionId": session_id,
        "command": command,
    }
    if output_byte_limit is not None:
        create["outputByteLimit"] = output_byte_limit
    write_msg(
        {
            "jsonrpc": "2.0",
            "id": first_id,
            "method": "terminal/create",
            "params": create,
        }
    )
    created = wait_response(incoming, first_id)
    terminal_id = created["result"]["terminalId"]
    wait_id = 1 if first_id == 0 else first_id + 1
    write_msg(
        {
            "jsonrpc": "2.0",
            "id": wait_id,
            "method": "terminal/wait_for_exit",
            "params": {"sessionId": session_id, "terminalId": terminal_id},
        }
    )
    wait_response(incoming, wait_id)
    return terminal_id


def handle_prompt(
    incoming: queue.Queue,
    req: dict[str, Any],
    scenario: str,
    prompt_index: int,
) -> None:
    session_id = req.get("params", {}).get("sessionId", SESSION_ID)
    if scenario == "unknown_update":
        notify_update(
            session_id,
            {"sessionUpdate": "vendor_extension_xyz", "extra": {"n": 1}},
        )
        stream_text(session_id, "PONG")
        end_turn(req)
        return
    if scenario == "permission_id_zero":
        send_permission(session_id, 0)
        wait_response(incoming, 0)
        stream_text(session_id, "PONG")
        end_turn(req)
        return
    if scenario == "colliding_ids":
        send_permission(session_id, req["id"])
        wait_response(incoming, req["id"])
        stream_text(session_id, "PONG")
        end_turn(req)
        return
    if scenario == "terminal_truncate":
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_trunc",
                "title": "printf 0123456789",
                "kind": "execute",
                "status": "pending",
                "rawInput": {"command": "printf '0123456789'"},
            },
        )
        terminal_id = run_terminal_roundtrip(
            incoming,
            session_id,
            "printf '0123456789'",
            first_id=0,
            output_byte_limit=4,
        )
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_trunc",
                "status": "completed",
                "content": [{"type": "terminal", "terminalId": terminal_id}],
            },
        )
        stream_text(session_id, "DONE")
        end_turn(req)
        return
    if scenario == "execute":
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_ok",
                "title": "echo CENTAUR-OK",
                "kind": "execute",
                "status": "pending",
                "rawInput": {"command": "echo CENTAUR-OK"},
            },
        )
        terminal_id = run_terminal_roundtrip(
            incoming, session_id, "echo CENTAUR-OK", first_id=0
        )
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_ok",
                "status": "completed",
                "content": [{"type": "terminal", "terminalId": terminal_id}],
            },
        )
        stream_text(session_id, "DONE")
        end_turn(req)
        return
    if scenario == "execute_fail":
        command = "sh -c 'echo FAIL-MARKER; exit 7'"
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_fail",
                "title": command,
                "kind": "execute",
                "status": "pending",
                "rawInput": {"command": command},
            },
        )
        terminal_id = run_terminal_roundtrip(incoming, session_id, command, first_id=0)
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_fail",
                "status": "completed",
                "content": [{"type": "terminal", "terminalId": terminal_id}],
            },
        )
        stream_text(session_id, "FAILED")
        end_turn(req)
        return
    if scenario == "read_tool":
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_read",
                "title": "Read note.txt",
                "kind": "read",
                "status": "pending",
                "rawInput": {"file_path": "note.txt"},
            },
        )
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_read",
                "status": "completed",
                "rawOutput": {"text": "ALPHA"},
                "content": [
                    {
                        "type": "content",
                        "content": {"type": "text", "text": "ALPHA"},
                    }
                ],
            },
        )
        stream_text(session_id, "READ-OK")
        end_turn(req)
        return
    if scenario == "two_turns":
        if prompt_index == 1:
            stream_text(session_id, "OK")
        else:
            stream_text(session_id, "KIWI")
        end_turn(req)
        return

    stream_text(session_id, "PONG")
    end_turn(req)


def reader_thread(incoming: queue.Queue) -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log_event({"dir": "in", "unparseable": line})
            continue
        log_event({"dir": "in", "msg": msg})
        incoming.put(msg)
    incoming.put(None)


def settings_info() -> dict[str, Any]:
    settings_path = None
    argv = sys.argv[1:]
    if "--settings" in argv:
        idx = argv.index("--settings")
        if idx + 1 < len(argv):
            settings_path = argv[idx + 1]
    info: dict[str, Any] = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "auto_update": os.environ.get("FACTORY_DROID_AUTO_UPDATE_ENABLED"),
        "settings_path": settings_path,
    }
    if settings_path and os.path.isfile(settings_path):
        mode = stat.S_IMODE(os.stat(settings_path).st_mode)
        info["settings_mode"] = mode
        with open(settings_path, encoding="utf-8") as handle:
            info["settings"] = json.load(handle)
    return info


def main() -> int:
    scenario = os.environ.get("DROID_FAKE_SCENARIO", "pong").strip() or "pong"
    log_event({"event": "start", "scenario": scenario, **settings_info()})
    incoming: queue.Queue = queue.Queue()
    thread = threading.Thread(target=reader_thread, args=(incoming,), daemon=True)
    thread.start()
    prompt_index = 0
    while True:
        msg = incoming.get()
        if msg is None:
            return 0
        method = msg.get("method")
        if method == "initialize":
            respond(
                msg,
                {
                    "protocolVersion": 1,
                    "agentCapabilities": {"loadSession": True},
                    "agentInfo": {"name": "fake-acp", "title": "Fake ACP", "version": "0"},
                },
            )
        elif method == "session/new":
            respond(msg, {"sessionId": SESSION_ID, "configOptions": []})
        elif method == "session/load":
            respond(msg, {"configOptions": []})
        elif method == "session/set_config_option":
            respond(msg, {})
        elif method == "session/prompt":
            prompt_index += 1
            handle_prompt(incoming, msg, scenario, prompt_index)
        elif method == "session/cancel":
            continue
        elif method is None:
            continue
        else:
            write_msg(
                {
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # pragma: no cover - test failure path
        log_event({"event": "crash", "error": str(error)})
        raise
