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
| `audi27 route run` | 跟着 turn，写成 `route`（只显示，不发给 agent） |
| `audi27 route replay --since today` | 用历史 turn 重算 route，默认只打印 |
| `audi27 route test "一句话"` | 看这一句会分给谁 |
| `audi27 dispatch run` | 把 route 交给适配器，写出 `task` / `reply`（默认关） |
| `audi27 queue run` | 跟着回复，按优先级写 `brief` |
| `audi27 queue show` | 用事件流重算队列，只打印 |
| `audi27 confirm run` | 跟事件流，把确认或取消写成 `confirm`（`[confirm]` 关着就不写） |
| `audi27 speak run` | 按顺序念 `brief`，写出 `spoken` |
| `audi27 speak test "文字"` | 念一句，用来试声音；没有 `say` 就打印 |
| `audi27 cue test` | 响一声唤醒提示音，不写事件；没有 `afplay` 就打印 |
| `audi27 cue run` | 一听到唤醒词就播放一声短提示，不等这一轮收口 |
| `audi27 screen run` | 记录最前面的应用和窗口标题（默认关，仅 macOS 真正采样） |
| `audi27 grokbot test "文本"` | 向 Grok Bot webhook 发一条测试 POST，打印 HTTP 状态码 |
| `audi27 grokbot inbox` | 只盯 `inbox/grokbot` 的回复文件（`dispatch run` 也会盯） |
| `audi27 eval run <后端>` | 把存下来的录音用另一个后端重跑一遍，结果存到 `data/evals/<标签>.jsonl` |
| `audi27 eval compare --ref gold` | 以某个标签为参考答案，给其他标签打分 |
| `audi27 web` | 本地仪表盘 http://127.0.0.1:8727 ：运行状态、实时波形、转写列表、任务状态 |

## 事件流

模块之间只追加写入 `data/events/YYYY-MM-DD.jsonl`（本地日期，一行一个 JSON）。固定字段：`id`、`ts`（带时区）、`type`、`source`、`ref`（引起它的事件，没有则为 null）、`data`。已知类型：`utterance`、`screen`、`presence`、`turn`、`route`、`task`、`reply`、`brief`、`spoken`、`confirm`、`cue_played`。不认识的类型也会原样保存，读的时候不要拒绝。

```bash
uv run audi27 events replay --since 2h --type utterance
uv run audi27 events tail
uv run audi27 events backfill          # 给事件流出现之前的转写补 utterance
uv run audi27 events chain <id>
```

`--since` / `--until` 的写法和 `search` 相同。多个进程可以同时追加。最后一行如果只写了一半，读者会跳过它，等它写完。

## 对话轮次

分段器在 0.8 秒静音处切开，一句「达达，……（停一下）……」会落成好几条 utterance。`turns` 把它们收成一条 `turn`，以后的路由只消费 turn。没以触发词开头的话不产生事件，仍只是环境录音。

- **句首触发**。默认认 `达达`、`哒哒`、`大大`、`DaDa`、`dada`、`Dada`（拉丁字母不区分大小写，全角 `ＤＡＤＡ` 也算）。触发词后面可以跟空格或标点。只看去掉开头空白后的开头：句中的「大大」不算；「大大小小」「大大方方」这种叠词不算；「dadapple」和「dada's」（直引号或弯引号）也不算。
- **`suspect` 算进去，`rejected` 丢掉**。存疑多半是真说过、只是置信度低，漏掉一句命令比多收一句更糟，所以照常并入，并在 turn 上记 `has_suspect`。`rejected` 是幻觉或噪声，整条忽略，也不拿它来延长间隔。其它不认识的 status 同样忽略。
- **收口**。下一条的开始比上一条的结束晚超过 `gap_s`（默认 2 秒，刚好等于仍合并）就以 `gap` 结束；这一轮的音频时长到了 `max_duration_s`（默认 120 秒，要长过一段的 30 秒硬切）就以 `max` 结束，跨过上限的那一条仍算在里面。间隔还没到又听到触发词，上一轮以 `new_trigger` 结束。间隔已经超过才听到触发词，上一轮记 `gap`。
- **实时**。`audi27 turns run` 第一次从事件流末尾开始看，不补历史；读到哪里记在 `data/turns.cursor.json`，崩溃或重启从那里继续，已经写过的 turn 按 id 去重。收到的 utterance 先按音频时间排好，再等 `wall_close_s`（默认 8 秒，从最后一条被看到开始算）才收口：转写重试或补写可能把较早的一段写在后面，不能按文件顺序把话说丢或并错。收口时音频上已经隔开就记 `gap`，没有下一条就记 `timeout`。进程退出时还开着的一轮记 `eof`。
- **回放**。`audi27 turns replay` 按音频时间而不是文件顺序合并，同一份输入结果相同，默认只打印。`--output` 写到另一个 JSONL，`--write` 才追加进事件流；turn 的 id 由第一条 utterance 决定（`ref` 也指向它），重复跑不会写重。回放结束时还开着的一轮记 `eof`。

不要把「大大」加进 `context.md` 的术语替换：那里是整句替换，句中的「大大」也会被改掉。

```bash
uv run audi27 turns replay --since today          # 先拿真实转写看误触发和漏触发
uv run audi27 turns replay --since today --json
uv run audi27 turns run                           # 常驻写 turn
```

`config.toml` 里 `[turns] enabled = true` 时，`./start.sh`（`audi27 supervise`）会把 `turns run` 和录音、仪表盘一起拉起来。默认关着。

## 唤醒提示音

一听到唤醒词就在本机响一声，不用等这一轮收口（`wall_close_s`，默认 8 秒），路由和分派也都还没动手。`cue` 直接看 `utterance`：状态是 `ok` 或 `suspect`，并且句首对上 `[turns] triggers`（用你配置的那份，所以「答答」「搭搭」这种自己加的同音词也算）。后面接着说的话、被丢掉的 `rejected`、空文本，都不响。默认关，和其它常驻开关一样。要听到，在自己的 `config.toml` 里加上（从旧的 `config.example.toml` 抄来的文件没有这一节，缺了就保持安静）：

```toml
[cue]
enabled = true
sound = "/System/Library/Sounds/Tink.aiff"   # 可省略，这是默认值
```

`./start.sh` 会拉起 `audi27 cue run`。macOS 用 `afplay` 播放，最长 1 秒，不等它播完，所以不会拖住转写。文件不存在、没有 `afplay`、或者不是 macOS，只记一条日志，进程不退出。每条命中的 utterance 追加一条 `cue_played`（`id` 是 `cue-<utterance id>`，`ref` 指向那条 utterance）。同一条不会响第二次；事件没写上也会先记住，下一轮只补写、不再响。第一次启动不补历史。`audi27 cue test` 只响一声，不写事件。

提示音很短。麦克风如果把这一声听成没有触发词的「确认 / 好的 / ok」，确认门会把它当成自己的回声丢掉，和念读回时的处理一样。带「达达」的一轮仍然算数。`ok` 为 false 的记录表示没响成，不会吞掉后面的回答。

```bash
uv run audi27 cue test
uv run audi27 cue run
```

## 意图路由

`route` 只读 `turn`，每条写一个 `route`（`id` 为 `route-<turn id>`，`ref` 指向那条 turn）。它不给任何 agent 发消息。仪表盘「路由」页列出最近的结果，✓ / ✗ 记在 `data/route_verdicts.jsonl`，并显示累计准确率。

规则在 `routes.toml`（不存在就用仓库里的 `routes.example.toml`），每条 turn 重新读，改完保存下一条就生效。顺序是：句中点名的 agent（含 Cloud Code、Crowed Code、grog 这类听写）优先，然后整句的队列指令（详细说说 / 跳过 / 稍后 / 还有什么没处理 / 关掉），然后不像指令的整句（算了、不用了），然后关键词（职业规划一类给 Muse），剩下默认 Grok。名字紧挨着「别用 / 不要用 / don't use」时不当成点名；如果因此落到的目标正是被排除的那个（比如「别用 Grok」），结果是不分派。Claude Code 和本地模型只有被点名才用。`[route] llm = true` 时，没撞上规则的句子才交给纠错用的那个本地模型，超时或答非所问就退回默认；测试可以注入一个假分类器，不需要模型。像发邮件、付款、删除、push、发帖这种句子会标 `needs_confirmation`。

```bash
uv run audi27 route test "帮我问一下 Muse 下个月面试怎么准备"
uv run audi27 route replay --since today          # 只打印
uv run audi27 route replay --since today --json --write
uv run audi27 route run                           # 常驻写 route
```

`[route] enabled = true` 时 `./start.sh` 会把 `route run` 也拉起来。默认关着。要有 turn 才有 route，所以一起开 `[turns] enabled`。

## 分派、队列、播报

这三步把回路接上，但默认都不执行。`[dispatch] enabled`、`[queue] enabled`、`[speak] enabled` 都是 `false`，`./start.sh` 不会拉起它们。路由继续只显示。要跑的时候再开，或者手动执行下面的命令。

- **分派** `audi27 dispatch run` 读 `route`，按 `[dispatch.adapters]` 找适配器（没写就是 `fake`）。`none` / `queue` 不发。`needs_confirmation` 只写一条 `task`（`awaiting_confirmation`），不调用适配器。`[confirm]` 关着时这条就停在那里。开着时，只有事后的 `confirm`（`approve`）才会把原来的 route 交给适配器，而且只交一次；`reject` 写成 `cancelled`；超过 `ttl_s`（默认 600 秒）写成 `expired`，不再执行。过期之后的同意不执行，并写一条 brief 说「这条确认已经过期，没有执行。」读回之前的「确认」不能批准。`fake` 把原文用「收到：…」回显，`delay_s` 是秒；正文里有 `fail_marker`（默认「模拟失败」）就失败且没有 `reply`，有 `decision_marker`（默认「需要你决定」）则回复里 `needs_decision` 为真。真实适配器按同一个 `handle(route, emit)` 加。第一次启动从事件流末尾开始，不补历史。
- **Grok Bot** 没有 SDK。只有 `[dispatch.adapters]` 里写了 `grok = "grokbot"` 才会真的发出去，默认仍是 `fake`。`dispatch run` 在 POST 之前先写 `task`（`sent`），JSON 体是 `task_id`、`route_id`、`text`（分给 Grok 的那句）、`ts`，可选 `context`（最近几条 utterance，`context_utterances`，默认 0 不带）。超时约 10 秒，只有网络错误重试一次；非 2xx 记失败。URL 和密钥不进 `config.toml`，放环境变量或项目根的 `.env`（已 gitignore）：`AUDI27_GROKBOT_WEBHOOK_URL`、`AUDI27_GROKBOT_WEBHOOK_KEY`。默认请求头是 `Authorization: Bearer <key>`。Grok Bot 要的头还没定，改 `[grokbot]` 的 `header_name` 和 `header_value`，模板里的 `{key}` 会换成那个密钥，例如 `header_name = "X-Webhook-Key"`、`header_value = "{key}"`。做完后它把回复写到 `data/inbox/grokbot/<task_id>.json`（`{"task_id","route_id","status":"done"|"failed"|"needs_decision","summary","text"}`）。`done` 是普通回复，`needs_decision` 进队列的 decision，`failed` 记一条失败（队列里是 error）。同一 `task_id` 不会重复写事件。格式不对、或不认识的 `task_id`，移到 `inbox/grokbot/quarantine/` 并在日志里记一条错误。JSON 解析失败先留着，连着几个 tick 还不行再隔离。超过 `reply_timeout_s`（默认 1800 秒）还没有回复，失败原因是「Grok Bot 没有回复」；超时之后文件才到，仍然写 reply。`dispatch run` 会一起看这个目录，所以不用再开一个进程；也可以单独跑 `audi27 grokbot inbox`。`audi27 grokbot test "文本"` 发一条测试 POST，打印 HTTP 状态码。密钥和 URL 不会出现在日志或事件里。回复会去掉控制字符，正文最长 4000 字。inbox 里的符号链接不跟随，超过 1MB 的文件直接隔离。对方一直写响应体时，等待不超过 `timeout_s`。
- **队列** `audi27 queue run` 把 `reply` 和失败的 `task` 收成 `brief`。优先级是 decision > error > done > info。`[confirm]` 开着时，`awaiting_confirmation` 也变成一条 decision brief：发给谁、一句转述，然后问「确认还是取消？」。一次只放出一条，等 `spoken` 之后才放下一条；已经在念的不会被后来的更高优先级打断。摘要是第一句，超长截断；`summarize_fn` 是以后接模型的钩子，现在不调用。队列指令（target `queue`）：详细说说念全文，跳过丢掉当前条，稍后放到队尾，还有什么列剩下的，关掉清空。状态靠重放事件恢复，不另存一份。`audi27 queue show` 重放整段日志并打印，不写 brief。`queue run` 第一次启动不汇报历史，重启从那次的位置重放。
- **确认** `[confirm] enabled` 默认关。开着时 `audi27 confirm run` 只写 `confirm` 事件，不调用适配器。跟在读回后面的达达一轮里，整句就是确认 / 好的 / 可以 / 发吧 / 执行 / yes / ok 才算同意，取消 / 算了 / 不要 / 别 / no 才算取消。问句（吗、呢、问号）、否定（不确认、不可以）、以及「那就好」「好的我想想」这种夹在别的话里的词，都不算同意。英文单词里的 ok（book）也不算。没带触发词的回答只在读回之后的 `window_s`（默认 20 秒）内算数，而且不能落在播报期间或播报结束后的 0.5 秒里（那是自己的回声）。读回之前说的「确认」不能批准。含糊的达达一轮（嗯、又要又不要）会再问一次，然后再含糊就继续挂着；窗口里没有触发词、又不是清楚的同意或取消，就当没听见。没点名时只针对最近一条；点名要对上恰好一条，对上两条就不执行。永远不会因为超时自动同意。同一时刻又同意又取消，按取消。仪表盘回路里，只有开关开着并且任务还挂着，才有「确认 / 取消」按钮。POST `/api/confirm` 必须是 JSON，Host 必须是本机，跨站请求会被拒绝；GET 不会批准。任务还不存在时写的确认，之后也不会执行。第一次启动不补历史。
- **播报** `audi27 speak run` 按 brief 的顺序调用 macOS `say`（`[speak] voice`，默认 `Tingting`）。写出 `spoken`：念了什么、是否念完、是否被打断。没有 `say`（比如 Linux 上的测试）就记一条日志，进程不退出。`say` 报不出念到第几个字，`text` 是交给它的整句。`audi27 speak test "文字"` 只试这一句，不写事件。第一次启动不补历史。
- **在不在说话** `[presence] enabled` 默认关。关掉时，播报仍看录音心跳 `data/status.json` 的 `speaking`（大约 0.4 秒写一次），为真就不开始，念到一半变成真就停 `say`，也不写 presence 文件。打开后，录音进程用分段器已有的能量判断，把 `data/presence.json` 原子替换更新：状态一变就写，另外最快每 `publish_s`（默认 0.1 秒）刷新一次，所以信号在 0.2 秒内可读。说话要连续 `onset_ms`（默认 60 毫秒）才算开口，安静要连续 `hangover_ms`（默认 400 毫秒）才算停，中间换气不会来回翻转。只有翻转才追加一条 `presence` 事件（`data.state` 为 `speaking` 或 `quiet`）。播报在 `speaking` 期间，以及变回安静后的 `grace_s`（默认 0.3 秒）内，都不开口。念到一半文件变成 `speaking`，大约 0.3 秒内停掉 `say`，`spoken` 的 `interrupted` 为真；队列用新的 brief id 把同一条再放出一次，已经有 `spoken` 的那条不会再读。文件缺失、读不了、超过 `stale_s` 没更新、或时间戳在未来（时钟跳了），都当成不知道，不卡住。没有单独的 presence 进程，`start.sh` 不会因为这个开关多拉起一个进程。达达这一轮还开着时并不额外算忙：turn 事件是轮次结束才写的，流上看不到「还开着」。

`spoken` 的 `ref` 指回 brief，再往前是 reply、task、route、turn、utterance。`confirm` 的 `ref` 指回挂起的 task。仪表盘「路由」页下面多一块「回路」，列出 task / reply / brief / spoken / confirm；挂起的任务旁边是确认和取消。

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
`[presence]` 打开时另写的 `data/presence.json` 只给播报用，仪表盘仍看这份心跳。

转写列表上面有一块「任务」，每 3 秒读一次 `GET /api/tasks`。进行中是已经分派、还没有回复的；待播报是还在队列里的回复、还没念的失败，以及 `[confirm]` 开着时还没念的确认读回；最近完成留下 10 条（已播报、已经离开队列的失败、念完的待决定、以及 `inbox/grokbot/quarantine/` 里的文件）。只重放昨天和今天的事件，和下面的回路共用这一份。不写、也不改变分派。不用改 `config.toml`。

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

`./start.sh` 启动的是 `audi27 supervise`：它同时拉起录音和仪表盘，任一进程崩溃会自动重启（退避 1–30 秒）。`[turns]`、`[route]`、`[dispatch]`、`[queue]`、`[speak]`、`[confirm]`、`[screen]`、`[cue]` 的 `enabled` 为真时，对应的常驻进程也一起拉起，默认都是关的。`[presence] enabled` 也默认关，打开后不增加进程，只让录音进程写 `data/presence.json`。

改完代码后重启，不需要碰那个终端：

```bash
echo "改了什么" > data/restart.request
```

守护进程每秒检查一次这个文件：先让录音进程把最后一段写完再退出，`pyproject.toml` 变了就先 `uv sync`，
然后重新执行自己，所以所有代码改动（包括守护进程本身）都会生效。`data/supervisor.json` 记录 pid、启动时间和重启次数。

## 屏幕上下文

`[screen] enabled` 默认关。关掉时不采样、不写 `screen` 事件，turn 和 route 跟以前一样。打开后 `./start.sh` 会拉起 `audi27 screen run`。

只记录最前面的应用名、bundle id 和最前窗口的标题。不截图、不做 OCR、不读窗口里的文字。macOS 上一次 `osascript` 同时读这三样，保证标题属于同一个进程；脚本是固定的，不会把应用名或标题拼进去。调用有超时，卡住就杀掉整个进程组，已经打出来的半截输出丢掉。辅助功能被拒绝时改用 `lsappinfo` 只记应用名，并且只警告一次，之后不再调用 `osascript`。不是 macOS 就打一条日志，不写事件。

应用或标题变了，并且稳住 `debounce_s`（默认 1 秒）才写一条，所以快速切换不会刷屏。抹掉标题发生在比较和写事件之前。`exclude_apps` 默认有 1Password（含 `com.agilebits.onepassword7`）、Bitwarden、钥匙串访问（含 `钥匙串访问` 和 `com.apple.keychainaccess`）。名字不区分大小写；带空格的名字会拿掉空格再对 bundle id，所以「Keychain Access」对得上 `com.apple.keychainaccess`。命中时只留应用名，标题抹掉。银行类应用不在默认列表里，要自己加。无痕窗口（标题里能看出 incognito、inprivate、无痕等）和 `redact_title_patterns` 会整段丢掉标题。标题最长 `max_title_chars`（默认 200）。

采样进程活着时会改写 `data/screen.json`，里面只有 `started` 和 `updated` 两个时间，没有应用名，也没有标题。轮次结束时，如果这个心跳说明采样当时还在跑，就算上一条 `screen` 比 `stale_s`（默认 3600 秒）更老，也会记在 turn 上：没变化就不写事件，窗口还是那一个。心跳说明采样已经停了，或者这个进程是轮次之后才启动的，就不附带。没有心跳文件时，才用 `stale_s` 当最长时间。route 再抄过去，排除名单里的标题会再抹一次。发给 Grok Bot 还要另开 `[grokbot] include_screen`（默认关），发出去之前再抹一次。

## 对比页（仪表盘 → 模型对比）

- **重跑**：选一个 `[backend.*]`，起个标签，点两次"运行"。改 `config.toml` 里的后端参数不用重启，下次运行自动读取。
- **表格**：一行一段录音，一列一个标签。和基准列不同的字词高亮；各列完全一致的行默认隐藏。列多了表格横向滚动，每列不窄于 300 px，文字始终完整换行。
- **判定**：点哪一格就是判哪一格正确（可多选），或点"都不对"。结果存在 `data/verdicts.jsonl`，保存的是被判正确的文本，
  所以以后新跑的标签会自动按这些答案计分。
- **密钥**：云端后端的 key 写在项目根目录的 `.env`（已 gitignore），例如 `OPENAI_API_KEY=...`，重启后生效。Grok Bot 的 webhook 也写在这里：`AUDI27_GROKBOT_WEBHOOK_URL` 和 `AUDI27_GROKBOT_WEBHOOK_KEY`。

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
