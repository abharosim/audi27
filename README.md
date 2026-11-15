# audi27

Mac 常驻录音 → 分段 → 转写（本地 Whisper 或云端）→ 校验 → 落盘，供其他 agent 搜索。

```
麦克风 → Segmenter(能量 VAD) → data/spool/*.wav → Transcriber(可换/可回退) → validate → transcripts/*.jsonl + *.md
```

转写文件仍是“说了什么”的依据。每条转写同时写成一条 `utterance` 事件到 `data/events/`；以后的模块只通过这条事件流通信。

## 运行

```bash
cd ~/Code/audi27
uv sync --extra dev
cp config.example.toml config.toml   # 可选，不建也能跑默认值
uv run audi27 devices                # 确认输入设备
uv run audi27 run                    # 开始常驻录音，Ctrl-C 停止
uv run audi27 web                    # 另开一个终端：仪表盘 http://127.0.0.1:8727
uv run pytest                        # 不需要麦克风和模型
```

默认后端是 `mlx_whisper`，直接用本机已缓存的 `mlx-community/whisper-turbo`（`~/.cache/huggingface/hub`），不需要再下载。
首次 `run` 时 macOS 会弹出终端的麦克风授权。

## 命令

| 命令 | 作用 |
|---|---|
| `audi27 run` | 常驻录音并转写 |
| `audi27 devices` | 列出输入设备 |
| `audi27 file a.wav` | 把现成的 16 kHz 单声道 WAV 走一遍完整管线，用来测后端和校验 |
| `audi27 retry` | 重试 `data/failed/` 和 `data/spool/` 里的片段 |
| `audi27 search 关键词 ...` | 搜索转写，见 AGENTS.md |
| `audi27 events replay` | 按时间、类型、来源回放 `data/events/` |
| `audi27 events tail` | 轮询并打印新事件 |
| `audi27 events backfill` | 把已有转写补成 `utterance` 事件 |
| `audi27 events chain <id>` | 顺着 `ref` 追到起因事件 |
| `audi27 turns run` | 跟着事件流，把以触发词开头的一轮话写成 `turn` |
| `audi27 turns replay --since today` | 用历史 utterance 重算 turn，默认只打印 |
| `audi27 eval run <后端>` | 把存下来的录音用另一个后端重跑一遍，结果存到 `data/evals/<标签>.jsonl` |
| `audi27 eval compare --ref gold` | 以某个标签为参考答案，给其他标签打分 |
| `audi27 web` | 本地仪表盘 http://127.0.0.1:8727 ：运行状态、实时波形、转写列表 |

## 事件流

模块之间只追加写入 `data/events/YYYY-MM-DD.jsonl`（本地日期，一行一个 JSON）。固定字段：`id`、`ts`（带时区）、`type`、`source`、`ref`（引起它的事件，没有则为 null）、`data`。已知类型：`utterance`、`screen`、`presence`、`turn`、`route`、`task`、`reply`、`brief`、`spoken`。不认识的类型也会原样保存，读的时候不要拒绝。

```bash
uv run audi27 events replay --since 2h --type utterance
uv run audi27 events tail
uv run audi27 events backfill          # 给事件流出现之前的转写补 utterance
uv run audi27 events chain <id>
```

`--since` / `--until` 的写法和 `search` 相同。多个进程可以同时追加。最后一行如果只写了一半，读者会跳过它，等它写完。

## 对话轮次

分段器在 0.8 秒静音处切开，一句「达达，……（停一下）……」会落成好几条 utterance。`turns` 把它们收成一条 `turn`，以后的路由只消费 turn。没以触发词开头的话不产生事件，仍只是环境录音。

- **句首触发**。默认认 `达达`、`哒哒`、`大大`、`DaDa`、`dada`、`Dada`（拉丁字母不区分大小写）。触发词后面可以跟空格或标点。只看去掉开头空白后的开头：句中的「大大」不算，「dadapple」也不算。
- **`suspect` 算进去，`rejected` 丢掉**。存疑多半是真说过、只是置信度低，漏掉一句命令比多收一句更糟，所以照常并入，并在 turn 上记 `has_suspect`。`rejected` 是幻觉或噪声，整条忽略，也不拿它来延长间隔。其它不认识的 status 同样忽略。
- **收口**。下一条的开始比上一条的结束晚超过 `gap_s`（默认 2 秒，刚好等于仍合并）就以 `gap` 结束；这一轮的音频时长到了 `max_duration_s`（默认 120 秒，要长过一段的 30 秒硬切）就以 `max` 结束，跨过上限的那一条仍算在里面。间隔还没到又听到触发词，上一轮以 `new_trigger` 结束。间隔已经超过才听到触发词，上一轮记 `gap`。
- **实时**。`audi27 turns run` 跟着事件流写 `turn`，从启动时的末尾开始看，不补历史。一条话说完如果没有下一条，不能干等，所以看到 utterance 之后再等 `wall_close_s`（默认 8 秒）就以 `timeout` 结束。这 8 秒是在等转写落盘，合并不合并仍看音频上的 `gap_s`。进程退出时还开着的一轮记 `eof`。
- **回放**。`audi27 turns replay` 只用 utterance 的时间戳，同一份输入结果相同，默认只打印。`--output` 写到另一个 JSONL，`--write` 才追加进事件流；turn 的 id 由第一条 utterance 决定（`ref` 也指向它），重复跑不会写重。回放结束时还开着的一轮记 `eof`。

不要把「大大」加进 `context.md` 的术语替换：那里是整句替换，句中的「大大」也会被改掉。

```bash
uv run audi27 turns replay --since today          # 先拿真实转写看误触发和漏触发
uv run audi27 turns replay --since today --json
uv run audi27 turns run                           # 常驻写 turn
```

`config.toml` 里 `[turns] enabled = true` 时，`./start.sh`（`audi27 supervise`）会把 `turns run` 和录音、仪表盘一起拉起来。默认关着。

## 换转写后端

`config.toml` 里 `transcribe.backends` 按顺序尝试，后面的是回退：

```toml
[transcribe]
backends = ["mlx_whisper", "openai"]
```

新增后端：在 `audi27/transcribers/` 加一个文件，继承 `Transcriber` 实现 `transcribe()`，
失败时抛 `TranscribeError`，然后在 `transcribers/__init__.py` 的 `REGISTRY` 里登记一行。
`[backend.<name>]` 下的键原样作为构造参数传入。

## 错误处理与防幻觉

| 问题 | 处理 |
|---|---|
| 静音/噪音被转成字 | 能量 VAD 先过滤；faster-whisper 内再跑一遍 Silero VAD；`no_speech_prob` + `avg_logprob` 双阈值丢弃 |
| Whisper 惯用幻觉句（"谢谢观看"、"Thanks for watching"、字幕组署名） | 黑名单，整句基本只有该短语时判 `rejected` |
| 复读循环 | 同一 2–30 字单元连续 4 次以上，或 `compression_ratio > 2.4` |
| 低置信度、语言不确定、字数远超时长、与上一句完全重复 | 判 `suspect`，保留但打标 |
| 检测出的语言不在 `languages`（默认中文+英文）里 | 分别强制每种允许的语言重转，取置信度高的，并判 `suspect` |
| initial_prompt 被原样吐回 | 判 `rejected` |
| 后端报错/超时/断网 | 绝不写入文字；音频留在 spool，指数退避重试 5 次，仍失败移到 `data/failed/` 并写 `.err` |
| 进程崩溃 | spool 是磁盘队列，重启自动续转；按 id 去重，不会重复记录 |
| 麦克风没权限（全零信号 10 秒） | 日志报错并提示去系统设置授权 |
| 设备拔出/切换 | 自动重开音频流，退避 1–30 秒 |
| 转写跟不上 | 每 5 分钟打印积压数，超过 20 段告警 |

校验只打标不删除：`rejected` 也写进 JSONL，默认连同音频保留（`keep_audio = "suspect"`），方便回查阈值是否过严。

## 已知限制

- 能量 VAD 分不清人声和其他响声（键盘、音乐），靠后面的校验兜底。
- 没有说话人识别，录到的别人说话、视频声音都会进转写。
- 云端 `gpt-4o-transcribe` 类模型不返回置信度，只能做文本层校验（记录里会有 `no_confidence_metrics`）。

## 仪表盘

`audi27 web` 是只读的，只绑定 127.0.0.1。录音进程每 0.4 秒把心跳写到 `data/status.json`
（音量、是否在说话、积压、丢帧），仪表盘据此判断是否在运行：心跳超过 3 秒没更新即显示"未运行"。

## 回放与模型对比

每段录音默认都保留（`keep_audio = "all"`，约每小时语音 115 MB），所以任何时候都能换模型重跑同一批输入。

```bash
uv run audi27 eval run gold                      # 用 [backend.gold] 重跑全部录音，作为参考答案
uv run audi27 eval run mlx_whisper --label mlx-v2 --since 3d   # 改了实现或参数后，换个标签再跑
uv run audi27 eval compare --ref gold            # live 和其他标签相对 gold 的错误率，并列出分歧最大的 10 段
uv run audi27 eval list
```

- **标签**：一次回放的名字，默认等于后端名。`live` 是保留标签，指录音当时实际写入的结果。
- **后端别名**：`[backend.<名字>]` 里写 `type = "openai"`，同一种实现可以配置成多个名字（不同模型、不同参数）。
- **错误率**：中文按字、英文按词（忽略大小写和标点）算编辑距离，除以参考答案的总字词数。被校验拒绝的输出按空文本计，所以评的是"模型 + 校验"整体。
- **可续跑**：已跑过的片段自动跳过，失败的片段不写入，下次再跑会补上；`--force` 全部重跑。
- **人工修正参考答案**：`data/evals/gold.jsonl` 是普通 JSONL，直接改 `text` 字段即可。
- 仪表盘的"模型对比"页显示同样的结果，并可以逐段回放录音。

已知偏差：繁体和简体算不同的字；数字写法不同（"十点" 和 "10 点"）也算错。

## 守护进程与重启

`./start.sh` 启动的是 `audi27 supervise`：它同时拉起录音和仪表盘，任一进程崩溃会自动重启（退避 1–30 秒）。

改完代码后重启，不需要碰那个终端：

```bash
echo "改了什么" > data/restart.request
```

守护进程每秒检查一次这个文件：先让录音进程把最后一段写完再退出，`pyproject.toml` 变了就先 `uv sync`，
然后重新执行自己，所以所有代码改动（包括守护进程本身）都会生效。`data/supervisor.json` 记录 pid、启动时间和重启次数。

## 对比页（仪表盘 → 模型对比）

- **重跑**：选一个 `[backend.*]`，起个标签，点两次"运行"。改 `config.toml` 里的后端参数不用重启，下次运行自动读取。
- **表格**：一行一段录音，一列一个标签。和基准列不同的字词高亮；各列完全一致的行默认隐藏。列多了表格横向滚动，每列不窄于 300 px，文字始终完整换行。
- **判定**：点哪一格就是判哪一格正确（可多选），或点"都不对"。结果存在 `data/verdicts.jsonl`，保存的是被判正确的文本，
  所以以后新跑的标签会自动按这些答案计分。
- **密钥**：云端后端的 key 写在项目根目录的 `.env`（已 gitignore），例如 `OPENAI_API_KEY=...`，重启后生效。

## 语气词与上下文纠错

- **语气词**：整段只有"嗯 / 呃 / um / Mm-hmm"之类的，标为 `rejected`（原因 `filler_only`），不进搜索结果。
  句子里夹的语气词保留原样；做模型对比时忽略它们，同时忽略 的/地/得、儿化和口吃式重复（"他他"）。
- **上下文**：`context.md`，纯文本，随时可改，保存即生效。"术语"一节里 `- 正确写法 = 听错的写法` 会被直接替换（不经过模型）；
  术语还会附加到 Whisper 的提示词里，让识别阶段就更容易写对。
- **本地模型纠错**：替换之后，本地 LLM（默认 `mlx-community/Qwen2.5-7B-Instruct-4bit`，只用本机缓存，不会自动下载）
  结合上下文和前几句话提出修改，必须同时满足以下条件才采纳，否则保留识别原文：
  只有一行；改动不超过 30% 的字词；新出现的英文词必须在 `context.md` 里出现过。
- **可追溯**：被改过的记录保留 `asr_text`（识别原文）和 `corrections`（每处改动及来源 alias / llm）。
- **验证纠错有没有用**：对比页会多出 `live-asr` 一列（纠错前）。改了 `context.md` 后，在对比页选"只重跑上下文纠错"，
  或运行 `uv run audi27 eval correct --label ctx-v2`，不需要录音、不花钱，结果是新的一列。
- 模型加载失败或连续出错 3 次，本次运行内自动停用 LLM 纠错，术语替换继续工作，转写不受影响。
