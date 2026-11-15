# Searching the transcripts (for agents)

Everything spoken near the Mac is transcribed into `data/transcripts/`. Read-only access is safe while the recorder runs.

## Files

- `data/transcripts/YYYY-MM-DD.md`: one line per utterance, local time. `- 14:03:22 text`. A `[?]` after the time means low confidence.
- `data/transcripts/YYYY-MM-DD.jsonl`: one JSON record per utterance, including rejected ones. Source of truth.

Record fields: `id`, `start`, `end` (ISO 8601 with offset), `duration` (s), `text`, `status`, `reasons`, `language`,
`language_prob`, `avg_logprob`, `no_speech_prob`, `compression_ratio`, `backend`, `model`, `audio` (path under `data/` or null),
`raw_text` (only when validation removed part of the backend output).

`status`:
- `ok`: passed validation.
- `suspect`: possibly wrong; treat as a hint and say so when you rely on it.
- `rejected`: judged a hallucination or noise. Do not use as evidence of what was said.

## Fast path: grep

```bash
rg -i "connector" data/transcripts/*.md
rg -i -C2 "deploy" data/transcripts/2026-11-15.md
```

## CLI

```bash
uv run audi27 search connector deploy            # all terms must match, case-insensitive
uv run audi27 search --since 2h                  # everything from the last 2 hours
uv run audi27 search 报销 --since yesterday --until today --context 2
uv run audi27 search fetcher --json --limit 5    # 5 most recent matches as JSON lines
uv run audi27 search --since today --ok-only     # drop suspect lines
```

`--since` / `--until` accept `YYYY-MM-DD`, an ISO datetime, `today`, `yesterday`, or `30m` / `2h` / `3d` (ago).
Text output marks each line with ` ` (ok), `?` (suspect) or `x` (rejected, only with `--all`). With `--context`, JSON records carry `"match": true|false`.

Transcripts are speech recognition output: names, numbers and mixed Chinese/English terms are the most likely to be wrong.

## Event stream

Later modules talk only through `data/events/YYYY-MM-DD.jsonl` (one JSON object per line, local date). `data/transcripts/` stays the source of truth for what was said. Each transcript record is also copied to an `utterance` event; `audi27 events backfill` copies transcripts that were written before the stream existed.

Fixed fields: `id`, `ts` (ISO 8601 with a timezone offset), `type`, `source`, `ref` (id of the event that caused this one, or null), `data` (object). Known types: `utterance`, `screen`, `presence`, `turn`, `route`, `task`, `reply`, `brief`, `spoken`, `confirm`, `cue_played`. Any other type is valid — do not reject a type you do not recognise.

```bash
uv run audi27 events replay --since 2h --type utterance
uv run audi27 events tail --since 30m --type route --type reply
uv run audi27 events backfill
uv run audi27 events chain <id>          # origin first, then each event that followed
```

`--since` / `--until` use the same formats as `search`. From Python, `audi27.events.replay` reads a range and `audi27.events.follow` polls for new lines (default every 0.1 s). `audi27.events.ref_chain` walks `ref` back to the origin. Several processes can append. A torn last line is ignored until it is a complete event.

## Screen

`[screen] enabled` defaults off. Off means no sampling, no `screen` events, and turns and routes are unchanged. On, `audi27 supervise` / `start.sh` starts `audi27 screen run`.

The sampler records only the frontmost app name, its bundle id, and the front window title. It does not take screenshots, run OCR, read window contents, or walk the accessibility tree. On macOS one `osascript` reads all three from the same front process, so a title cannot be filed under a different app. The script is a fixed string; the app name and the title are never interpolated into it. Each call times out (3 s) and the process group is killed if it hangs; partial output is discarded. `lsappinfo` is the fallback when Accessibility is denied or the app has no window. A denial is logged once, with no title in the log, and later polls do not call `osascript`. On any other OS the process logs once and writes nothing.

A `screen` event is written only when the recorded snapshot changes, and only after it has stayed the same for `debounce_s` (default 1 s), so a quick alt-tab does not flood the log. Poll interval is `poll_s` (default 2 s). Redaction happens before that comparison and before the event is written. `[screen] exclude_apps` matches the app name or bundle id, case-insensitively; a name with spaces also matches those letters in the bundle id (`Keychain Access` matches `com.apple.keychainaccess`, including a localized display name). `1Password` also matches `com.agilebits.onepassword7`. The default list is 1Password, Bitwarden, and Keychain Access (plus those bundle ids and `钥匙串访问`). A match stores the app name and drops the title. Banking apps are not in the default list; add them. A title containing a `private_title_markers` substring (incognito, inprivate, private browsing, 无痕, …) or matching `redact_title_patterns` is dropped whole. Titles are cut to `max_title_chars` (default 200). A title that merely contains `1002` or "not authorized" is not treated as a permission error.

While the sampler is running it rewrites `data/screen.json` with `started` and `updated` only (no app, no title). The turn assembler stores the latest `screen` event at the turn's end. If that heartbeat shows this process was already running and had checked in within a few poll intervals of the turn, the event may be older than `stale_s` (default 3600 s): a change-only log means the window is still front. If the heartbeat shows the sampler had stopped, or this process started after the turn, nothing is attached. With no heartbeat file, `stale_s` is the age limit. The router copies the turn's snapshot (or reads the log if the turn has none) and drops an excluded title again. Events after the turn are ignored. `[grokbot] include_screen` defaults off; only then does the grokbot POST put the snapshot in `context`, and an excluded or redacted title is removed again before the POST. With recent utterances as well, `context` is `{"utterances", "screen"}`; otherwise it stays the utterance list.

```bash
uv run audi27 screen run
```

## Turns

The segmenter cuts on 0.8 s of silence, so one command arrives as several utterances. `audi27 turns` merges the ones that **start** with the trigger into a single `turn` event (`ref` = first utterance id). Anything else produces no event.

Default triggers, config key `turns.triggers`: 达达, 哒哒, 大大, DaDa, dada, Dada. Latin forms are case-insensitive, including fullwidth letters, and must end on a word boundary (`dadapple` and `dada's`, straight or curly apostrophe, do not match). 大大 only counts at the start, and not as the head of an AABB word such as 大大小小 or 大大方方. Do not add 大大 to the context.md glossary, which rewrites every occurrence.

`rejected` utterances are ignored and do not extend the gap. `suspect` utterances are kept (a low-confidence trigger is still a command) and the turn's `has_suspect` is true.

```bash
uv run audi27 turns replay --since today          # print only; --json, --output FILE, or --write
uv run audi27 turns run                           # follow the stream and emit turns
```

Utterances are merged by audio time, not file order: a retried segment or a backfill can be appended after a later one. Close reasons: `gap` (audio gap greater than `gap_s`, default 2 s), `max` (audio span reached `max_duration_s`, default 120 s), `new_trigger`, `timeout` (live idle `wall_close_s`, default 8 s, measured from when the last utterance was seen; the batch is ordered before the close), `eof` (replay ended, or the live process is stopping). The first `turns run` starts at the end of the stream. A restart resumes from `data/turns.cursor.json` and does not write a turn id again. Replay is deterministic and does not append to the live stream unless `--write`. `[turns] enabled = true` makes `audi27 supervise` start `turns run` as well; if that process crashes, only it is restarted.

## Cue

`[cue] enabled` defaults off, same as the other live loops. On, `audi27 supervise` / `start.sh` starts `audi27 cue run`. That process follows `utterance` events and plays a short local sound as soon as one would open a turn: status `ok` or `suspect`, and `match_trigger` hits `turns.triggers` (the configured list, so a homophone such as 答答 counts when it is listed). It does not wait for `wall_close_s`. A continuation, a `rejected` line, and a blank line do not play. It does not call an agent. macOS uses `afplay` of `[cue] sound` (default `/System/Library/Sounds/Tink.aiff`) for at most one second, in another process, so a missing player never blocks transcription. A missing file, a missing `afplay`, or any other OS is logged and the loop continues. Each hit appends `cue_played` (`id` = `cue-<utterance id>`, `ref` = that utterance, `data.ok`, `data.backend`, `data.trigger`, `data.until` when it played). The same utterance is not played twice; if the append fails, the id is remembered and the next poll writes the event without playing again. The first start does not replay history (`data/cue.cursor.json`). `audi27 cue run` plays even when the flag is off; `audi27 cue test` plays once and writes nothing.

A bare yes/no whose audio start falls inside that `cue_played` window is ignored by confirm and dispatch, the same way a TTS read-back is. A 达达 turn during the ding still counts. A cue with `ok` false was not heard, so it does not swallow an answer. Turn this on together with `[turns]`; a config copied from an older `config.example.toml` has no `[cue]` section and stays silent until you add `enabled = true`.

```bash
uv run audi27 cue test
uv run audi27 cue run
```

## Routes

`audi27 route` reads `turn` events and writes one `route` (`id` = `route-<turn id>`, `ref` = the turn id). It does not message any agent. Rules are `routes.toml` when that file exists, otherwise `routes.example.toml`, re-read on every turn. Explicit agent names win (including ASR spellings such as Cloud Code and grog), then a whole-turn queue command (`command`: details / skip / later / pending / close), then a whole-turn "not a command" (`target` `none`), then keyword rules (career topics → muse), then the default (grok). A name immediately after 别用 / 不要用 / don't use is not a selection; if the fallback target is itself one of those negated names (别用 Grok), the route is `none` with rule `negated:<target>`. Claude Code and the local model are chosen only when named. `[route] llm = true` asks the corrector's local model about leftovers, with `timeout_s` and a fallback to the default; tests inject `classify`. `needs_confirmation` is a keyword hit for send-email / pay / delete / push / post. The dashboard 路由 tab stores ✓/✗ in `data/route_verdicts.jsonl` (last line per id wins) and shows the running accuracy. Replay prints only; `--write` appends. The first `route run` starts at the end of the stream (`data/route.cursor.json`). `[route] enabled = true` (default off) starts it from `audi27 supervise`; turns need to be enabled too or there is nothing to route.

```bash
uv run audi27 route test "帮我问一下 Muse 下个月面试怎么准备"
uv run audi27 route replay --since today --json   # print only; add --write to append
uv run audi27 route run
```

## Dispatch, queue, speaker

These three close the loop and default to off (`[dispatch]`, `[queue]`, `[speak]` `enabled`). `[confirm] enabled` is also off. `audi27 supervise` / `start.sh` starts one only when its flag is on. Each `run` command executes even when the flag is off, same as turns and route; `confirm run` with the flag off writes nothing. Modules only pass events: route → task/reply → brief → spoken, and `confirm` sits between the read-back and the adapter. `ref` links each one to the previous, so `audi27 events chain` on a `spoken` id walks back to the utterance.

`audi27 dispatch run` hands a `route` to the adapter named by `[dispatch.adapters]` (missing targets use `fake`). Targets `none` and `queue` are not dispatched. `needs_confirmation` writes one `task` with status `awaiting_confirmation` and does not call the adapter. With `[confirm]` off that status stays. With it on, the dispatcher releases the original route only after a `confirm` event with `decision` `approve` (exactly once, including across restarts); `reject` writes `cancelled`; past `[confirm] ttl_s` (default 600) it writes `expired` and never calls the adapter. An approve that arrives after `expired` is ignored and one brief says so (`这条确认已经过期，没有执行。`). A confirm whose source turn started before the read-back brief is ignored. A route that is only a yes/no word while something is awaiting is not sent to an agent. `fake` echoes `收到：…`, sleeps `[fake] delay_s`, fails with no reply when the text contains `fail_marker` (default 模拟失败), and sets `needs_decision` when it contains `decision_marker` (default 需要你决定). An adapter implements `Agent.handle(route, emit)` and emits `sent` before any side effect, then `running`, then `done` plus a `reply`, or `failed`. The reply may be emitted after `handle` returns. `handle` is called at most once per route; a restart that finds `sent` / `running` / `done` without a reply or `failed` writes one `failed` task and does not send the work again. The first live start does not backfill (`data/dispatch.cursor.json`).

`grokbot` is that adapter for Grok Bot, and only when `[dispatch.adapters] grok = "grokbot"` (otherwise grok stays `fake`). `handle` emits `sent`, then POSTs `{task_id, route_id, text, ts}` and, when `[grokbot] context_utterances` is greater than 0, `context` (the last N non-rejected utterance texts, max 20). `[grokbot] include_screen` (default false) adds the route's screen snapshot to that POST; with both set, `context` is an object with `utterances` and `screen`. URL and key are `AUDI27_GROKBOT_WEBHOOK_URL` and `AUDI27_GROKBOT_WEBHOOK_KEY` (environment or `.env`, never config). The auth header is `[grokbot] header_name` and `header_value` (default `Authorization` / `Bearer {key}`). Timeout is `[grokbot] timeout_s` (default 10); one retry, and only on a network error; a non-2xx fails the task. The bot writes `data/inbox/grokbot/<task_id>.json` (`status` `done`, `needs_decision`, or `failed`). `dispatch run` polls it (so does `audi27 grokbot inbox`). A reply's `ref` is the task, so `events chain` walks back to the route. `done` is a normal reply, `needs_decision` sets `needs_decision` (queue priority decision), `failed` is a failed task (queue priority error). The same `task_id` is not emitted twice. Malformed files and unknown task ids are logged and moved to `inbox/grokbot/quarantine/`. A JSON parse error is retried `[grokbot] parse_retries` times (default 3), then quarantined. No reply within `[grokbot] reply_timeout_s` (default 1800) fails the task with `Grok Bot 没有回复`; a file that arrives after that still emits `reply`. A grokbot task left `sent` or `running` is not closed as interrupted on restart — it is not sent again, and the inbox watcher keeps waiting until the timeout or the file. `audi27 grokbot test "文本"` POSTs one line and prints the HTTP status. The key and URL are never logged or stored on an event. Reply text is stripped of control characters and capped at 4000 characters before the queue speaks it. A symlink under the inbox is not followed, and a file over 1MB is quarantined. A response body that keeps arriving does not hold dispatch past `timeout_s`.

`audi27 queue run` turns a `reply` (and a `task` with status `failed`) into a `brief`: priority `decision` > `error` > `done` > `info`, a one-sentence summary (first sentence, truncated to `[queue] summary_chars`; pass `summarize_fn` later for an LLM, it is not called now), and `text_ref` pointing at the full text. When `[confirm]` is on, an `awaiting_confirmation` task is also a `decision` brief: who would do it, a short paraphrase, then `确认还是取消？` (`kind` `confirm`, `task_id` set). The brief's timestamp is when it is released, which is the read-back. One brief is released at a time; the next waits for `spoken`. A queue-target route runs a command: `details` (full text of the current or last item), `skip`, `later` (back of the queue), `pending` (one line listing what is left), `close`. Replaying the log rebuilds the queue; there is no second store. `audi27 queue show` prints that fold and writes nothing. The first `queue run` does not brief history; a restart replays from the epoch stored in `data/queue.cursor.json`.

`audi27 speak run` speaks briefs in log order with macOS `say` (`[speak] voice`, default `Tingting`). It writes `spoken` (`text`, `completed`, `interrupted`, `backend`). Where `say` is missing, the line is logged (`backend` `log`) and the process does not crash. `say` does not report partial progress; `text` is the whole line it was given. The cursor stays put when a `spoken` event was not written, so a later brief in that batch is not skipped. `audi27 speak test "文字"` speaks one line and prints the result; it does not write an event. The first live start does not read old briefs.

`[presence] enabled` defaults off, and then the speaker still will not start while `data/status.json` says `speaking`, and still kills `say` if that flag turns true. The recorder writes that heartbeat about every 0.4 s. There is no presence file and no presence event. `start.sh` does not start a presence process either way: with the flag on, the recorder itself (it already has the frames) publishes `data/presence.json` by atomic replace, immediately on change and at least every `publish_s` (default 0.1 s, so the signal is available within 0.2 s). The bit is the segmenter's existing energy decision. Speech must last `onset_ms` (default 60) to become `speaking`; silence must last `hangover_ms` (default 400) to become `quiet`, so a short pause does not flap. Only that transition is appended as a `presence` event (`data.state`, source `capture`). The speaker reads the file: it does not start while the state is `speaking` or for `grace_s` (default 0.3 s) after the quiet transition, and a fresh `speaking` kills `say` within about 0.3 s. The spoken event has `interrupted` true. The queue then releases a new brief id for the same item (`brief-<id>-<n>`); the brief that already has a spoken event is not read again. A missing, unreadable, stale (`stale_s`), or future-dated file is unknown and does not block, including after a clock jump. An open 达达 turn is not a presence state: turn events are written when the turn closes, so there is nothing on the stream that says a turn is still open.

## Confirm

`[confirm] enabled` defaults off. On, `audi27 confirm run` reads the log and writes `confirm` events (`data.task_id`, `data.decision` `approve`|`reject`, `data.source` the turn or utterance id, or `dashboard`). It does not call an adapter. Only the latest pending read-back is answered, unless the line names exactly one other pending task (its text or its target). A name that matches two tasks is not an approval. Yes words, and only as the whole answer: 确认 / 好的 / 可以 / 发吧 / 执行 / yes / ok. No words: 取消 / 算了 / 不要 / 别 / no. A question (`?` / `？` / 吗 / 呢 / 么), a negated yes (不确认, 别确认, 不可以), or a yes stuck behind other words (那就好, 帮我确认一下, 好的我想想) is not an approval. `ok` inside an English word (`book`) is not one either. A 达达 turn after the read-back counts until the task expires. A line with no wake word counts only within `[confirm] window_s` (default 20) of that read-back, and a word from before the read-back never does. A bare line whose audio start falls during that read-back, or within 0.5 s after its `spoken` event, is the speaker's own playback and is ignored. A bare line inside a successful `cue_played` window is ignored the same way. A 达达 turn that is mixed or empty ("嗯", "确认取消") asks again once (`brief-reask-<task id>`, `没听清。……确认还是取消？`) and then leaves the task pending. A bare line that is not a clear yes or no is ignored, so ambient speech inside the window does not ask again. Nothing is approved because time passed. The dashboard buttons POST `/api/confirm` only while the flag is on, and only for a task that is still waiting. The request must be JSON (`Content-Type` media type `application/json`), the Host must be loopback, and a cross-site `Sec-Fetch-Site` is refused. GET does not approve. The first `confirm run` does not backfill (`data/confirm.cursor.json`). A confirm timestamped before the task, or a same-timestamp approve and reject, does not run the adapter (reject wins the tie).

```bash
uv run audi27 dispatch run
uv run audi27 grokbot test "你好"
uv run audi27 grokbot inbox
uv run audi27 queue show
uv run audi27 queue run
uv run audi27 confirm run
uv run audi27 speak test "你好"
uv run audi27 speak run
```

## Replays

`data/evals/<label>.jsonl` holds re-transcriptions of the same audio by other models (same `id` as the live record).
They are for model evaluation; `data/transcripts/` remains the record of what was said.
