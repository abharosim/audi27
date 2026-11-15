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

## Replays

`data/evals/<label>.jsonl` holds re-transcriptions of the same audio by other models (same `id` as the live record).
They are for model evaluation; `data/transcripts/` remains the record of what was said.
