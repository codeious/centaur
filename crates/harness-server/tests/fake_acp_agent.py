#!/usr/bin/env python3
"""Scripted ACP agent for `harness-server droid` integration tests.

Selected as the child via DROID_BIN. Scenario comes from DROID_FAKE_SCENARIO
(default: pong). Optional JSONL log at DROID_FAKE_LOG. Stdlib only.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
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


def prompt_text(req: dict[str, Any]) -> str:
    prompt = (req.get("params") or {}).get("prompt") or []
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(parts)


def emit_execute_tool(session_id: str, tool_id: str, command: str) -> None:
    notify_update(
        session_id,
        {
            "sessionUpdate": "tool_call",
            "toolCallId": tool_id,
            "title": command,
            "kind": "execute",
            "status": "pending",
            "rawInput": {"command": command},
        },
    )


def wait_cancel(incoming: queue.Queue, timeout: float = 30.0) -> dict[str, Any] | None:
    try:
        return wait_for(
            incoming,
            lambda msg: msg.get("method") == "session/cancel",
            timeout,
        )
    except TimeoutError:
        return None


def reply_for_text(session_id: str, req: dict[str, Any], text: str) -> None:
    lowered = text.lower()
    if "steered-ok" in lowered:
        stream_text(session_id, "STEERED-OK")
    elif "after-cancel" in lowered:
        stream_text(session_id, "AFTER-CANCEL")
    elif "pong2" in lowered:
        stream_text(session_id, "PONG2")
    elif "pong3" in lowered:
        stream_text(session_id, "PONG3")
    elif "word i asked you to remember" in lowered:
        if "papaya" in lowered:
            stream_text(session_id, "PAPAYA")
        elif "mango" in lowered:
            stream_text(session_id, "MANGO")
        else:
            stream_text(session_id, "KIWI")
    elif "remember the word papaya" in lowered:
        stream_text(session_id, "OK")
    elif "remember the word mango" in lowered:
        stream_text(session_id, "OK")
    elif "remember the word kiwi" in lowered:
        stream_text(session_id, "OK")
    else:
        stream_text(session_id, "PONG")
    end_turn(req)


def handle_cancellable(
    incoming: queue.Queue,
    req: dict[str, Any],
    session_id: str,
    late_updates: bool,
) -> None:
    emit_execute_tool(session_id, "call_sleep", "sleep 20")
    cancel = wait_cancel(incoming, timeout=30.0)
    if cancel is None:
        stream_text(session_id, "DONE-ORIGINAL")
        end_turn(req)
        return
    if late_updates:
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_sleep",
                "status": "failed",
                "rawOutput": {"text": "Error: Tool execution cancelled by user"},
            },
        )
        notify_update(
            session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_sleep",
                "title": "sleep 20",
                "kind": "other",
                "status": "pending",
                "rawInput": {},
            },
        )
    respond(req, {"stopReason": "cancelled"})


def handle_prompt(
    incoming: queue.Queue,
    req: dict[str, Any],
    scenario: str,
    prompt_index: int,
    used_load: bool,
) -> None:
    session_id = req.get("params", {}).get("sessionId", SESSION_ID)
    text = prompt_text(req)
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
    if scenario in {"cancellable_sleep", "late_updates_after_cancel"}:
        lowered = text.lower()
        if "done-original" in lowered or (
            "run sleep" in lowered and "steered" not in lowered and "after-cancel" not in lowered
        ):
            handle_cancellable(
                incoming,
                req,
                session_id,
                late_updates=scenario == "late_updates_after_cancel",
            )
            return
        reply_for_text(session_id, req, text)
        return
    if scenario == "hang_on_cancel":
        if used_load:
            reply_for_text(session_id, req, text)
            return
        emit_execute_tool(session_id, "call_hang", "sleep 20")
        while True:
            msg = incoming.get()
            if msg is None:
                return
        return
    if scenario == "crash_mid_turn":
        if used_load:
            stream_text(session_id, "PAPAYA")
            end_turn(req)
            return
        if prompt_index == 1:
            stream_text(session_id, "OK")
            end_turn(req)
            return
        emit_execute_tool(session_id, "call_crash", "sleep 25")
        os._exit(9)
    if scenario == "orphan_sleep":
        subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        stream_text(session_id, "PONG")
        end_turn(req)
        return
    if scenario == "load_replay":
        reply_for_text(session_id, req, text)
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


MODELS = ["gpt-5.4-mini-fast", "gpt-test"]
REASONING = ["low", "medium", "high", "xhigh"]


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


def advertised_config(settings: dict[str, Any] | None) -> list[dict[str, Any]]:
    session = (settings or {}).get("sessionDefaultSettings") or {}
    model = session.get("model")
    if model not in MODELS:
        model = "gpt-5.4-mini-fast"
    reasoning = session.get("reasoningEffort")
    if reasoning not in REASONING:
        reasoning = "low"
    return [
        {
            "id": "model",
            "name": "Model",
            "category": "model",
            "type": "select",
            "currentValue": model,
            "options": [{"value": value, "name": value} for value in MODELS],
        },
        {
            "id": "reasoning_effort",
            "name": "Reasoning",
            "category": "thought_level",
            "type": "select",
            "currentValue": reasoning,
            "options": [{"value": value, "name": value} for value in REASONING],
        },
    ]


def apply_config_option(options: list[dict[str, Any]], config_id: str, value: str) -> None:
    for option in options:
        if option.get("id") == config_id:
            option["currentValue"] = value
            return


def main() -> int:
    scenario = os.environ.get("DROID_FAKE_SCENARIO", "pong").strip() or "pong"
    start = settings_info()
    log_event({"event": "start", "scenario": scenario, **start})
    config_options = advertised_config(start.get("settings") if isinstance(start.get("settings"), dict) else None)
    incoming: queue.Queue = queue.Queue()
    thread = threading.Thread(target=reader_thread, args=(incoming,), daemon=True)
    thread.start()
    prompt_index = 0
    used_load = False
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
            respond(msg, {"sessionId": SESSION_ID, "configOptions": config_options})
        elif method == "session/load":
            used_load = True
            params = msg.get("params") or {}
            session_id = str(params.get("sessionId") or SESSION_ID)
            if scenario == "load_unknown":
                write_msg(
                    {
                        "jsonrpc": "2.0",
                        "id": msg.get("id"),
                        "error": {
                            "code": -32602,
                            "message": "Invalid params: Unknown session identifier",
                            "data": {"sessionId": session_id},
                        },
                    }
                )
            else:
                if scenario == "load_replay":
                    notify_update(
                        session_id,
                        {
                            "sessionUpdate": "user_message_chunk",
                            "content": {
                                "type": "text",
                                "text": "Remember the word MANGO. Reply OK.",
                            },
                        },
                    )
                    notify_update(
                        session_id,
                        {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "REPLAY-HISTORY"},
                        },
                    )
                respond(msg, {"configOptions": config_options})
        elif method == "session/set_config_option":
            if scenario == "config_set_error":
                write_msg(
                    {
                        "jsonrpc": "2.0",
                        "id": msg.get("id"),
                        "error": {
                            "code": -32602,
                            "message": "Invalid params: unknown config value",
                        },
                    }
                )
            else:
                params = msg.get("params") or {}
                apply_config_option(
                    config_options,
                    str(params.get("configId") or ""),
                    str(params.get("value") or ""),
                )
                respond(msg, {"configOptions": config_options})
        elif method == "session/prompt":
            prompt_index += 1
            handle_prompt(incoming, msg, scenario, prompt_index, used_load)
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
