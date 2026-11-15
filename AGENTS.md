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

Fixed fields: `id`, `ts` (ISO 8601 with a timezone offset), `type`, `source`, `ref` (id of the event that caused this one, or null), `data` (object). Known types: `utterance`, `screen`, `presence`, `turn`, `route`, `task`, `reply`, `brief`, `spoken`. Any other type is valid — do not reject a type you do not recognise.

```bash
uv run audi27 events replay --since 2h --type utterance
uv run audi27 events tail --since 30m --type route --type reply
uv run audi27 events backfill
uv run audi27 events chain <id>          # origin first, then each event that followed
```

`--since` / `--until` use the same formats as `search`. From Python, `audi27.events.replay` reads a range and `audi27.events.follow` polls for new lines (default every 0.1 s). `audi27.events.ref_chain` walks `ref` back to the origin. Several processes can append. A torn last line is ignored until it is a complete event.

## Turns

The segmenter cuts on 0.8 s of silence, so one command arrives as several utterances. `audi27 turns` merges the ones that **start** with the trigger into a single `turn` event (`ref` = first utterance id). Anything else produces no event.

Default triggers, config key `turns.triggers`: 达达, 哒哒, 大大, DaDa, dada, Dada. Latin forms are case-insensitive, including fullwidth letters, and must end on a word boundary (`dadapple` and `dada's`, straight or curly apostrophe, do not match). 大大 only counts at the start, and not as the head of an AABB word such as 大大小小 or 大大方方. Do not add 大大 to the context.md glossary, which rewrites every occurrence.

`rejected` utterances are ignored and do not extend the gap. `suspect` utterances are kept (a low-confidence trigger is still a command) and the turn's `has_suspect` is true.

```bash
uv run audi27 turns replay --since today          # print only; --json, --output FILE, or --write
uv run audi27 turns run                           # follow the stream and emit turns
```

Utterances are merged by audio time, not file order: a retried segment or a backfill can be appended after a later one. Close reasons: `gap` (audio gap greater than `gap_s`, default 2 s), `max` (audio span reached `max_duration_s`, default 120 s), `new_trigger`, `timeout` (live idle `wall_close_s`, default 8 s, measured from when the last utterance was seen; the batch is ordered before the close), `eof` (replay ended, or the live process is stopping). The first `turns run` starts at the end of the stream. A restart resumes from `data/turns.cursor.json` and does not write a turn id again. Replay is deterministic and does not append to the live stream unless `--write`. `[turns] enabled = true` makes `audi27 supervise` start `turns run` as well; if that process crashes, only it is restarted.

## Routes

`audi27 route` reads `turn` events and writes one `route` (`id` = `route-<turn id>`, `ref` = the turn id). It does not message any agent. Rules are `routes.toml` when that file exists, otherwise `routes.example.toml`, re-read on every turn. Explicit agent names win (including ASR spellings such as Cloud Code and grog), then a whole-turn queue command (`command`: details / skip / later / pending / close), then a whole-turn "not a command" (`target` `none`), then keyword rules (career topics → muse), then the default (grok). Claude Code and the local model are chosen only when named. `[route] llm = true` asks the corrector's local model about leftovers, with `timeout_s` and a fallback to the default; tests inject `classify`. `needs_confirmation` is a keyword hit for send-email / pay / delete / push / post. The dashboard 路由 tab stores ✓/✗ in `data/route_verdicts.jsonl` (last line per id wins) and shows the running accuracy. Replay prints only; `--write` appends. The first `route run` starts at the end of the stream (`data/route.cursor.json`). `[route] enabled = true` (default off) starts it from `audi27 supervise`; turns need to be enabled too or there is nothing to route.

```bash
uv run audi27 route test "帮我问一下 Muse 下个月面试怎么准备"
uv run audi27 route replay --since today --json   # print only; add --write to append
uv run audi27 route run
```

## Replays

`data/evals/<label>.jsonl` holds re-transcriptions of the same audio by other models (same `id` as the live record).
They are for model evaluation; `data/transcripts/` remains the record of what was said.
