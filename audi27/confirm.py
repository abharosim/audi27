"""Confirm or cancel a task that is waiting. Voice in, one `confirm` event out.

This module does not call an adapter and does not import the dispatcher. It
reads the log and appends `confirm` events (and, once, a "say that again"
brief). The dispatcher is the only thing that turns `approve` into a sent task.

A 达达 turn after the read-back counts until the task expires. A line with no
wake word counts only inside `[confirm] window_s` of that read-back. A word
spoken before the read-back is not an answer. `[confirm] enabled` defaults
off; `sync` and `serve` then write nothing.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .events import append, make_event, read_new, replay, snapshot_offsets
from .turns import DEFAULT_TRIGGERS, _strip_trigger, match_trigger

log = logging.getLogger("audi27.confirm")

CONFIRM_SOURCE = "confirm"
_CURSOR_NAME = "confirm.cursor.json"
_ASK = "确认还是取消？"
_LABEL = {
    "grok": "Grok",
    "muse": "Muse",
    "claude_code": "Claude Code",
    "local": "本地模型",
}
# Longer phrases first when matching. A one-character word only counts when
# the next character does not continue a different word (别人, 好好, 行李).
_APPROVE = ("确认", "好的", "可以", "发吧", "执行", "okay", "approve", "yes", "ok", "同意", "批准", "行吧", "好吧", "行", "好")
_REJECT = ("取消", "算了", "不要", "别发", "不行", "不用", "cancel", "停止", "no", "别")
_PHRASES = tuple((phrase, "approve") for phrase in _APPROVE) + tuple((phrase, "reject") for phrase in _REJECT)
_PHRASES = tuple(sorted(_PHRASES, key=lambda item: len(item[0]), reverse=True))
_FILLER_CHARS = frozenset("吧了啊呀呢嘛哦嗯呃额的")
_GLUE = ("发给", "发到", "那个", "这个", "那件", "这件", "那条", "这条", "一下", "一个",
         "帮我", "给我", "那就", "的话", "please", "thanks", "thankyou")
# Stuck to the front of a yes-word, these flip it to no ("不确认", "别确认").
# Longest first so "不要" wins over "不".
_NEGATION = ("不要", "并不", "并未", "尚未", "无需", "无法", "不能", "不是", "不可", "没有",
             "先别", "先不", "别", "没", "不", "未", "非", "勿")
# Leftover that means the speaker is still thinking, not naming a task.
_HESITATION = ("想想", "看看", "等一下", "等会", "再说", "考虑")
# After `say` returns, a bare line this soon can still be the tail of the read-back.
ECHO_GRACE_S = 0.5
_ANSWER_TYPES = ("turn", "utterance")


@dataclass(frozen=True)
class Answer:
    """`approve`, `reject`, or `unclear`. `hint` is leftover text that may name a task."""

    decision: str
    hint: str


def _parse_ts(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None
    return dt


def _cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF


def _prepare(text, triggers) -> str:
    raw = text.strip() if isinstance(text, str) else ""
    surface = match_trigger(raw, triggers)
    if surface is not None:
        raw = _strip_trigger(raw, surface)
    folded = unicodedata.normalize("NFKC", raw).casefold()
    body = "".join(ch for ch in folded if not ch.isspace() and unicodedata.category(ch)[:1] != "P")
    for tail in ("please", "thanks", "thankyou"):
        if body.endswith(tail) and len(body) > len(tail):
            body = body[:-len(tail)]
    return body


def _clean_hint(text: str) -> str:
    prev = None
    while text and text != prev:
        prev = text
        for glue in _GLUE:
            if text.startswith(glue):
                text = text[len(glue):]
            if text.endswith(glue) and len(text) >= len(glue):
                text = text[:-len(glue)]
        text = text.strip("".join(_FILLER_CHARS))
    return text


def _negation_before(body: str, index: int) -> int:
    """Length of a negation stuck to the front of the match at `index`, or 0."""
    for neg in _NEGATION:
        if index >= len(neg) and body[index - len(neg):index] == neg:
            return len(neg)
    return 0


def _hits(body: str):
    """Yes/no hits, leftover text, and `(start, end, side)` of each hit.

    A yes-word stuck to a negation is a no ("不确认", "不可以"). A no-word stuck
    to a negation is not a decision ("别取消"). "好了" / "可以了" are not a clear
    yes. An ASCII word inside a longer word ("book", "yesterday") is not a hit.
    """
    occupied = [False] * len(body)
    found = []
    spans = []
    for phrase, side in _PHRASES:
        start = 0
        single = len(phrase) == 1 and not phrase.isascii()
        while True:
            i = body.find(phrase, start)
            if i < 0:
                break
            end = i + len(phrase)
            if any(occupied[i:end]):
                start = i + 1
                continue
            if phrase.isascii():
                before = body[i - 1] if i else ""
                after = body[end] if end < len(body) else ""
                if (before.isascii() and before.isalnum()) or (after.isascii() and after.isalnum()):
                    start = i + 1
                    continue
            if single and end < len(body) and _cjk(body[end]) and body[end] not in _FILLER_CHARS:
                start = i + 1
                continue
            # "好了" / "行了" / "可以了" mean "that's done", not "yes, do it".
            # "确认了" stays a yes.
            if side == "approve" and phrase != "确认" and end < len(body) and body[end] == "了":
                start = i + 1
                continue
            neg = _negation_before(body, i)
            if neg and any(occupied[max(0, i - neg):end]):
                # The negation was already taken ("不要确认": 不要 is its own no).
                # Drop the yes-word so it cannot approve, and don't let it name a task.
                if side == "approve" and not any(occupied[i:end]):
                    for j in range(i, end):
                        occupied[j] = True
                start = end
                continue
            if neg and side == "approve":
                for j in range(i - neg, end):
                    occupied[j] = True
                found.append("reject")
                spans.append((i - neg, end, "reject"))
                start = end
                continue
            if neg and side == "reject":
                # "别取消" is not a cancellation.
                for j in range(i - neg, end):
                    occupied[j] = True
                start = end
                continue
            for j in range(i, end):
                occupied[j] = True
            found.append(side)
            spans.append((i, end, side))
            start = end
    leftover = _clean_hint("".join(ch for i, ch in enumerate(body) if not occupied[i]))
    return found, leftover, spans


def _is_question(raw: str, body: str) -> bool:
    """A question is not a decision. `?` is stripped before matching, so check the raw line."""
    if "?" in raw or "？" in raw:
        return True
    return body.endswith(("吗", "么", "呢"))


def _embedded(body: str, spans) -> bool:
    """True when a real word sits in front of the yes/no ("那就好", "帮我确认一下")."""
    if not spans:
        return False
    first = min(span[0] for span in spans)
    return any(ch not in _FILLER_CHARS for ch in body[:first])


def _hesitating(hint: str) -> bool:
    return any(piece in hint for piece in _HESITATION)


def classify_answer(text: str, triggers=None) -> Answer | None:
    """What this line says about a pending task, or None when it is not an answer.

    `unclear` is a short or mixed reply ("嗯", "确认取消", "可以吗"). A longer
    sentence that merely contains no yes/no word is None: it is a new command.
    A question, a negated yes, or a yes buried after other words is not an
    approval. Ambiguous lines stay `unclear` so the caller does not approve.
    """
    triggers = tuple(triggers) if triggers else DEFAULT_TRIGGERS
    raw = text.strip() if isinstance(text, str) else ""
    body = _prepare(text, triggers)
    if not body:
        return None
    found, leftover, spans = _hits(body)
    question = _is_question(raw, body)
    buried = _embedded(body, spans)
    if not found:
        if question or len(body) <= 4:
            return Answer("unclear", "")
        return None
    # A question or a yes/no stuck behind other words is not a clear answer.
    # Short ones are "say that again"; a long sentence is a new command.
    if question or buried:
        if len(body) <= 8 or (question and len(body) <= 12):
            return Answer("unclear", "")
        return None
    if "approve" in found and "reject" in found:
        return Answer("unclear", leftover)
    if "approve" in found and (_hesitating(leftover) or body.endswith("的话") or leftover.endswith("的话")):
        return Answer("unclear", leftover)
    return Answer("approve" if "approve" in found else "reject", leftover)


def _confirm_event(task_id: str, decision: str, source: str, ts) -> dict:
    return make_event("confirm", CONFIRM_SOURCE, {
        "task_id": task_id,
        "decision": decision,
        "source": source,
    }, ts=ts, ref=task_id, id=f"confirm-{task_id}-{decision}-{source}")


def _reask_event(task: dict, source: str, ts) -> dict:
    label = _LABEL.get(task["target"], task["target"] or "它")
    return make_event("brief", CONFIRM_SOURCE, {
        "priority": "decision",
        "summary": f"没听清。发给 {label} 的这件事，确认还是取消？",
        "text_ref": task["id"],
        "kind": "confirm",
        "item_id": task["id"],
        "task_id": task["id"],
    }, ts=ts, ref=source, id=f"brief-reask-{task['id']}")


def _hint_hits(hint: str, task: dict) -> bool:
    if len(hint) < 2:
        return False
    blob = _prepare(f"{task['text']} {task['target']} {_LABEL.get(task['target'], '')}", ())
    return hint in blob


def _choose(answer: Answer, ready: list):
    """The task this answer names, the most recent one, or `miss` when the name matches nothing.

    A name that matches more than one pending task is `miss`. Guessing the
    latest one would approve a side effect the speaker did not pick.
    """
    if answer.hint:
        matched = [task for _, task in ready if _hint_hits(answer.hint, task)]
        if len(matched) == 1:
            return matched[0]
        return "miss"
    return max(ready, key=lambda pair: (pair[0], pair[1]["ts"], pair[1]["id"]))[1]


def overlaps_playback(start: datetime, briefs: dict, spoken_at: dict) -> bool:
    """True when `start` is during a confirm read-back, or within `ECHO_GRACE_S` after it.

    `briefs` maps a task id to `(read-back time, brief id)` pairs. `spoken_at`
    maps a brief id to the time `say` finished. A bare line in that span is the
    speaker's own audio, not a person approving. No `spoken` event means nothing
    was played, so this returns false.
    """
    if not isinstance(start, datetime):
        return False
    grace = timedelta(seconds=ECHO_GRACE_S)
    for rows in briefs.values():
        for when, brief_id in rows:
            spoken = spoken_at.get(brief_id)
            if spoken is None or when > start:
                continue
            if start < spoken + grace:
                return True
    return False


def plan(events, *, window_s: float = 20.0, triggers=None) -> list[dict]:
    """Confirm events and at most one re-ask brief implied by `events`.

    The same events always produce the same list. An answer whose audio start
    is before that task's read-back is ignored. Bare lines (no wake word) have
    to fall inside `window_s` of the latest read-back; a 达达 turn does not.
    """
    triggers = tuple(triggers) if triggers else DEFAULT_TRIGGERS
    tasks: dict[str, dict] = {}
    briefs: dict[str, list] = {}
    spoken_at: dict[str, datetime] = {}
    answers = []
    in_turn: set[str] = set()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        kind = ev.get("type")
        if kind == "task" and data.get("status") == "awaiting_confirmation" and isinstance(ev.get("id"), str):
            tasks[ev["id"]] = {
                "id": ev["id"],
                "route_id": data.get("route_id") if isinstance(data.get("route_id"), str) else "",
                "target": data.get("target") if isinstance(data.get("target"), str) else "",
                "text": data.get("text") if isinstance(data.get("text"), str) else "",
                "ts": ev.get("ts") if isinstance(ev.get("ts"), str) else "",
            }
        elif kind == "brief" and data.get("kind") == "confirm" and isinstance(data.get("task_id"), str):
            when = _parse_ts(ev.get("ts"))
            if when is not None and isinstance(ev.get("id"), str):
                briefs.setdefault(data["task_id"], []).append((when, ev["id"]))
        elif kind == "spoken" and isinstance(ev.get("ref"), str):
            when = _parse_ts(ev.get("ts"))
            if when is not None:
                prev = spoken_at.get(ev["ref"])
                if prev is None or when < prev:
                    spoken_at[ev["ref"]] = when
        elif kind == "turn" and isinstance(ev.get("id"), str):
            ids = data.get("utterance_ids") if isinstance(data.get("utterance_ids"), list) else []
            for uid in ids:
                if isinstance(uid, str):
                    in_turn.add(uid)
            start = _parse_ts(data.get("start") or ev.get("ts"))
            if start is not None:
                answers.append(("turn", ev["id"], data.get("text") if isinstance(data.get("text"), str) else "", start))
        elif kind == "utterance" and isinstance(ev.get("id"), str) and data.get("status") in ("ok", "suspect"):
            start = _parse_ts(data.get("start") or ev.get("ts"))
            if start is not None:
                text = data.get("text") if isinstance(data.get("text"), str) else ""
                answers.append(("bare", ev["id"], text, start))

    answers.sort(key=lambda row: (row[3], row[1]))
    unclear: dict[str, int] = {}
    out = []
    seen: set[str] = set()

    def add(event: dict) -> None:
        if event["id"] in seen:
            return
        seen.add(event["id"])
        out.append(event)

    for kind, ident, text, start in answers:
        if kind == "bare" and (ident in in_turn or match_trigger(text, triggers)):
            continue
        answer = classify_answer(text, triggers)
        if answer is None:
            continue
        # The mic hears the read-back. A bare "确认" during playback, or in the
        # short tail after `say` returns, is the speaker, not an approval.
        if kind == "bare" and overlaps_playback(start, briefs, spoken_at):
            continue
        ready = []
        for task in tasks.values():
            times = [item for item in briefs.get(task["id"], []) if item[0] <= start]
            if not times:
                continue
            latest = max(times)
            if kind == "bare" and (start - latest[0]).total_seconds() > window_s:
                continue
            ready.append((latest[0], task))
        if not ready:
            continue
        miss = False
        chosen = None
        if answer.decision != "unclear":
            chosen = _choose(answer, ready)
            miss = chosen == "miss"
        if answer.decision == "unclear" or miss:
            if kind == "bare":
                continue
            task = max(ready, key=lambda pair: (pair[0], pair[1]["ts"], pair[1]["id"]))[1]
            count = unclear.get(task["id"], 0)
            if count == 0:
                add(_reask_event(task, ident, start))
                briefs.setdefault(task["id"], []).append((start, f"brief-reask-{task['id']}"))
            unclear[task["id"]] = count + 1
            continue
        add(_confirm_event(chosen["id"], answer.decision, ident, start))
    return out


def _caused_by(event: dict):
    if event.get("type") == "confirm":
        source = (event.get("data") or {}).get("source")
        return source if isinstance(source, str) else None
    ref = event.get("ref")
    return ref if isinstance(ref, str) else None


def _write(data_dir, events) -> dict:
    wrote = skipped = 0
    for event in events:
        if append(data_dir, event, if_absent=True) is None:
            skipped += 1
        else:
            wrote += 1
    return {"wrote": wrote, "skipped": skipped}


def sync(data_dir, cfg) -> dict:
    """Append the confirm events implied by the whole log. Off writes nothing."""
    if not cfg.confirm.enabled:
        return {"wrote": 0, "skipped": 0}
    planned = plan(list(replay(data_dir)), window_s=cfg.confirm.window_s, triggers=cfg.turns.triggers)
    return _write(data_dir, planned)


_MOVED = frozenset({"sent", "running", "done", "failed", "cancelled", "expired", "ignored"})


def _is_pending(data_dir, task_id: str) -> bool:
    """True when `task_id` is an `awaiting_confirmation` task nothing has released yet."""
    route_id = None
    for ev in replay(data_dir, types="task"):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if ev.get("id") == task_id and data.get("status") == "awaiting_confirmation":
            route_id = data.get("route_id") if isinstance(data.get("route_id"), str) else ""
            break
    if not route_id:
        return False
    for ev in replay(data_dir, types=("task", "reply")):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if data.get("route_id") != route_id:
            continue
        if ev.get("type") == "reply" or data.get("status") in _MOVED:
            return False
    return True


def record_decision(data_dir, task_id: str, decision: str, *, source: str = "dashboard") -> dict:
    """Append one dashboard (or other explicit) confirm. A repeat is the same id.

    The task has to be waiting. A confirm for an unknown or already-finished id
    is refused, so a guessed future id cannot pre-approve the next side effect.
    """
    if decision not in ("approve", "reject"):
        raise ValueError("decision must be approve or reject")
    if (not isinstance(task_id, str) or not task_id or any(ch.isspace() for ch in task_id)
            or len(task_id) > 200 or any(ord(ch) < 32 for ch in task_id)):
        raise ValueError("bad task_id")
    if not isinstance(source, str) or not source or any(ch.isspace() for ch in source):
        raise ValueError("bad source")
    if not _is_pending(data_dir, task_id):
        raise ValueError("task is not waiting for confirmation")
    event = _confirm_event(task_id, decision, source, datetime.now().astimezone())
    stored = append(data_dir, event, if_absent=True)
    return stored if stored is not None else event


def _cursor_path(data_dir) -> Path:
    return Path(data_dir) / _CURSOR_NAME


def _valid_offsets(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in value.items()):
        return None
    return dict(value)


def _load_book(data_dir):
    path = _cursor_path(data_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        log.warning("ignoring unreadable confirm cursor %s", path)
        return None
    if not isinstance(raw, dict):
        return None
    epoch = _valid_offsets(raw.get("epoch"))
    offsets = _valid_offsets(raw.get("offsets"))
    if epoch is None or offsets is None:
        log.warning("ignoring unreadable confirm cursor %s", path)
        return None
    return epoch, offsets


def _save_cursor(data_dir, offsets: dict, epoch: dict) -> None:
    path = _cursor_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"epoch": epoch, "offsets": offsets}, ensure_ascii=False, sort_keys=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _apply_since(data_dir, cfg, epoch: dict) -> dict:
    """Write decisions for answers that arrived after `epoch`. Earlier ones are left alone."""
    after, _ = read_new(data_dir, epoch)
    allowed = {ev["id"] for ev in after if ev.get("type") in _ANSWER_TYPES and isinstance(ev.get("id"), str)}
    if not allowed:
        return {"wrote": 0, "skipped": 0}
    planned = [
        ev for ev in plan(list(replay(data_dir)), window_s=cfg.confirm.window_s, triggers=cfg.turns.triggers)
        if _caused_by(ev) in allowed
    ]
    return _write(data_dir, planned)


def serve(data_dir, cfg, *, poll_s: float = 0.2, stop: threading.Event | None = None,
          ready: threading.Event | None = None) -> None:
    """Follow the log and append confirm events. Blocks until `stop`.

    The first start does not read history. A restart keeps the original epoch,
    so an answer that arrived while this process was down is still decided,
    and one that was already decided is not written again.
    """
    stop = stop or threading.Event()
    if not cfg.confirm.enabled:
        log.info("confirm gate is off")
        if ready is not None:
            ready.set()
        try:
            while not stop.wait(poll_s):
                pass
        except KeyboardInterrupt:
            stop.set()
        return

    saved = _load_book(data_dir)
    if saved is None:
        epoch = snapshot_offsets(data_dir)
        offsets = dict(epoch)
    else:
        epoch, _old = saved
        _apply_since(data_dir, cfg, epoch)
        _, offsets = read_new(data_dir, epoch)
    try:
        _save_cursor(data_dir, offsets, epoch)
    except OSError:
        log.exception("confirm cursor was not written")
    durable = dict(offsets)

    def save(cursor: dict) -> None:
        nonlocal durable
        if cursor == durable:
            return
        try:
            _save_cursor(data_dir, cursor, epoch)
        except OSError:
            log.exception("confirm cursor was not written")
            return
        durable = dict(cursor)

    log.info("confirm gate watching %s", data_dir)
    if ready is not None:
        ready.set()
    try:
        while not stop.is_set():
            found, offsets = read_new(data_dir, offsets)
            if found:
                _apply_since(data_dir, cfg, epoch)
                _, offsets = read_new(data_dir, epoch)
                save(offsets)
            if stop.wait(poll_s):
                break
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()
    finally:
        try:
            found, offsets = read_new(data_dir, offsets)
            if found:
                _apply_since(data_dir, cfg, epoch)
                _, offsets = read_new(data_dir, epoch)
                save(offsets)
        except Exception:
            log.exception("confirm gate did not flush on the way out")


# Re-export the question so the queue and this module ask it the same way.
ASK = _ASK
TARGET_LABEL = _LABEL
