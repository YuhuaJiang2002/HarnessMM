# dsh-harnessmm

[English](./README.md) · [中文](./README.zh-CN.md)

DeepSeek Harness 桌面版插件：在原生工具面上做「带守卫的决策复用」。

本插件把 [`integrations/deepseek_harness/`](../integrations/deepseek_harness/) 里的
CHISEL / SAR 协议搬进标准的 DSH 桌面 profile。它只站在「模型 → 工具」的边界上：既不拥有
benchmark 循环，也不替换原生执行器，更不给结果打分。

## 它做什么

1. **先声明一次。** 通过注入的系统提示段落，慢思考模型要在发出唯一一次 root 调用的同时，
   给出一个 `<CHISEL_PLAN>` 动作库。每个候选都是一个完全字面的原生动作，外加一个
   *只依赖 receipt* 的守卫，例如 `{"exit_code": 0}`。
2. **按 receipt 判定。** root 的结果回来之后，`projectTurn` 用请求消息重建当前的决策状态，
   并且只拿公开 receipt 去评估每个候选的守卫。
3. **替换回调。** 当某个候选存活、且净节省超过准入阈值时，插件直接发出那次原生工具调用，
   而不再问模型下一步该做什么。多个存活候选由一个单字符选择器挑出；只剩一个时判定是
   确定性的，不花任何调用。
4. **出证。** 每次替换都写成一条收据，绑定动作库哈希、候选、被旁路的回调和精确的原生结果。
   动作库耗尽后，控制权回到普通慢路径。

`tools/execute` 会把每个 `chisel_*` 调用重新对照作者声明的字面动作，所以被篡改的参数
即使出现在流里，也到不了原生执行器。

## 为什么这不等同论文配置

参考实现对接的是 SDK 的 *terminal bridge*。桌面 profile 没有这种桥：它的 shell 面就是
原生的 `bash` / `pwsh` 工具，可靠的控制点只有 `llm/stream`、`tools/execute`、
`tools/result` 和系统提示。因此本插件复现的是协议，不是 benchmark 装置：

- 守卫只依赖 receipt（`ok`、`exit_code`、`stdout_contains`、`stdout_equals`、
  `stderr_empty`、`json_equals`），从不做语义判断；
- 准入阈值（论文默认 `net > 1024` token）可由 `policy.admission_threshold_tokens`
  配置，token 数量来自宿主自己的用量上报；
- 选择器调用被限制在 `FAST_CALL_UPPER_TOKENS`（512）token，而不是按作者自己给出的
  重决策估算，因此一个悲观的 `expected_local_cost` 不会把所有步骤都否掉；
- 收据 schema 是 `harnessmm-desktop-certificate/v1`，只声称桌面循环内旁路了一次回调 ——
  不做 episode 级的反事实声明。

不要把这些收据拿去和历史 HumanEval 的 run identity 比较。仓库自己的 README 正是就
这一点做了警告。

## 配置

出厂这一行默认把控制器打开（`policy.enabled: true`）。设为 `false` 就退回纯原生循环，
同时保留插件安装。在 profile 的 patch 层里改这一行：

```yaml
- insert:
    - id: harnessmm
      name: 'dsh-harnessmm'
      config:
        policy:
          enabled: true
          maximum_candidates: 16      # A-P 标签，1..16
          maximum_fast_steps: 8       # 动作库耗尽前允许的替换次数
          maximum_public_bytes: 65536 # 选择器载荷的字节上限
          admission_threshold_tokens: 1024 # 论文默认值，见下文
        selector:
          enabled: true
          provider: ''                # '' = 复用边界请求自己的路由
          model: ''                   # '' = 复用边界请求自己的模型
          maxTokens: 1024             # 推理模型要先思考再给标签
          timeoutMs: 20000
        shellTools: []                # [] = 从工具目录里自动识别 bash/pwsh
        planSection: true             # 注入 CHISEL_PLAN 声明段落
        receiptsFile: ''              # '' = <DSH_HOME 或 ~/.dsh>/harnessmm/receipts.jsonl
```

`selector.provider` / `selector.model` 留空时，复用边界请求自己走的那条路由 —— 也就是
这个宿主刚刚证明过能被回答的那条。这个默认值在「只服务 thinking 请求」的网关上很关键：
完全不带推理控制的调用，会在模型读到 receipt 之前就被拒绝
（`400 ... "The value of the enable_thinking parameter is restricted to True."`）。
因此选择器会复用边界请求自己的推理档位（边界没带时退回到模型声明的最高档位），保证线上
真的带上推理控制。把 `selector.enabled` 设为 `false` 则完全不花这次调用：多候选步骤
直接交回慢模型，收据里记 `selector-abstain`，`reason: "selector-disabled"`。

桌面宿主不导出 `DSH_HOME`，所以 `receiptsFile` 留空时会落到
`~/.dsh/harnessmm/receipts.jsonl`，而不是静默关掉收据。写不进去的 `receiptsFile`
也绝不会弄坏一轮对话：持久化按设计是尽力而为。

### 准入阈值

只有当被替换的那次慢调用，比重放它的快调用至少多花 `policy.admission_threshold_tokens`
时，才允许准入。具体地说，设 `n` 为该慢调用自身的 token 用量：

- 只剩一个存活候选 → 确定性提交，`n` 无关；
- 有多个存活候选 → 需要 `n - 选择器上限 > 阈值`，其中选择器上限被钳在 512 token。

论文的 1024 是默认值。调高是保守方向；调低则接受期望节省更小的替换。单个存活候选不受
影响 —— 正是确定性提交让控制器在低于阈值的步骤上也有用。

公开 receipt 只承载当前被判定的这一轮的目标，不含整个 transcript。长会话累积的用户文本
远超 `maximum_public_bytes`；把它们全部投影会因为字节数本身，否掉后面每一步替换。历史
也按宽松方式读取：只是引用了 `<CHISEL_PLAN>` 片段的消息（文档、bug 报告）不算声明，
也永远不会污染后续轮次。

## 在真实桌面宿主上验证过的内容

一个真实会话里，针对单候选 `A`（它的 root 调用已经成功）的一份计划，产生了下面这串收据
（节选）：

```json
{"type":"bank","candidates":1,"public_bytes":337}
{"type":"decision","kind":"commit","eligible":["A"],"rejected":[]}
{"type":"substitution-issued","call_id":"chisel_e6c66019cfa84ddd97a91ba8_A","tool":"pwsh","selector":"deterministic"}
{"type":"substitution","schema":"harnessmm-desktop-certificate/v1",
 "exact_arguments":"{\"description\":\"HMM15 substituted step\",\"command\":\"Write-Output 'HMM15-GUARD-MATCHED'\"}",
 "native_receipt":{"status":"PARSED","is_error":false,
   "value":{"tool":"pwsh","ok":true,"exit_code":0,"stdout":"HMM15-GUARD-MATCHED"}},
 "valid_substitution":true,"amount_substituted_slow_callbacks":1}
```

那份计划的守卫是 `{"exit_code":0,"stdout_equals":"HMM15-ROOT-MARK"}` —— 它与 root 调用的
输出精确相等，而这只有在 receipt 被规范化之后才可能发生（shell 给的是
`HMM15-ROOT-MARK\r\n`，永远不可能直接相等）。相邻的两次真实运行从两侧夹住了这个行为：

- 一份守卫比错字符串的计划被正确拒绝：`{"kind":"slow","reason":"no_eligible_candidate","rejected":[{"id":"A","reason":"guard_false"}]}`，
  慢模型继续掌握控制权；
- 一份命令非零退出的计划记录了 `"exit_code":3` 与 `"stdout":"HMM13-FAIL-OUT"` ——
  渲染器自己加的 `[exit code: 3]` 尾巴只被用来读状态，随后就被剥掉，所以守卫看到的是
  程序输出本身。

那次原生调用走的是普通的 `pwsh` 工具，它的输出回到了循环里；下一轮投影得到
`no-candidates:position=A`，控制权交还慢模型。这正是设计中的形状：替换执行、动作库耗尽、
判断权归还。

### 真正跑过的行为

下表每一行都在这个桌面宿主里观察到过，而不只是在 fake-host 单测里：

| 行为 | 证据 |
| --- | --- |
| 单存活候选的确定性提交 | `decision {"kind":"commit","eligible":["A"]}` → `chisel_…_A`，没有任何选择器调用 |
| 守卫比错时干净拒绝 | `{"kind":"slow","reason":"no_eligible_candidate","rejected":[{"id":"A","reason":"guard_false"}]}` |
| 非零退出的 receipt 保真 | `"exit_code":3`、`"stdout":"HMM13-FAIL-OUT"`（渲染器尾巴被剥掉） |
| 多候选步骤由选择器裁决 | `{"kind":"select","eligible":["A","B"]}` → `substitution-issued`，`"selector":"fast-selector"`，`"control_usage":{"inputTokens":912,"outputTokens":39,"totalTokens":951}` |
| 同一轮内的链式替换 | 同一个 `turn_id`：先提交 `A`（`predecessor_id":"root"`），再提交 `B`（`predecessor_id":"A"`），相隔 4 ms，都是 `deterministic` |
| 两个并发会话互不干扰 | 两个会话的 root 在同一个墙上时刻屏障放行（`…T01:48:00Z`）；两次提交相隔 74 ms，各自的证书保留了自己的 session id 和 marker |

并发这一段，节选自同一个收据文件：

```json
{"at":"…01:48:00.340Z","type":"decision","turn_id":"ac3fcd41-…:…","kind":"commit"}
{"at":"…01:48:00.341Z","type":"substitution-issued","turn_id":"ac3fcd41-…:…","candidate_id":"A"}
{"at":"…01:48:00.414Z","type":"decision","turn_id":"232cf615-…:…","kind":"commit"}
{"at":"…01:48:00.415Z","type":"substitution-issued","turn_id":"232cf615-…:…","candidate_id":"A"}
{"at":"…01:48:00.851Z","type":"substitution","turn_id":"ac3fcd41-…:…","native_receipt":{"value":{"stdout":"V3-MARK-SEVEN"}}}
{"at":"…01:48:00.930Z","type":"substitution","turn_id":"232cf615-…:…","native_receipt":{"value":{"stdout":"V3-MARK-EIGHT"}}}
```

轮次状态以 `sessionId` 为键，所以重叠的会话之间不会共享待决动作库、步数计数器或证书。

## 安装

`dsh plugin add`（或插件管理器）会把这个目录链接进某个 profile：

```sh
dsh plugin --profile desktop add /path/to/dsh-plugin/harnessmm-desktop
```

patch 层是整块替换一行的 `config`，不做深合并。要改一个字段，就得把上面那份完整的
`config` 块重述一遍。

## 目录结构

| 路径 | 作用 |
| --- | --- |
| `lib/controller.mjs` | 纯协议：动作库解析、receipt、守卫、准入、投影、派发校验 |
| `lib/index.js` | 宿主插件：`systemPrompt` 段落、`llm/stream`、`tools/execute`、`tools/result`、收据 |
| `cordis.patch.yml` | 覆盖在 profile 根上的 bundle 行 |
| `locale/{en,zh}.json` | 配置字段的设置页文案 |
| `test/*.test.mjs` | 协议测试，以及驱动 `lib/index.js` 的 fake-host 测试 |

## 收据

一行一个 JSON 对象：

| 类型 | 含义 |
| --- | --- |
| `bank` | 某次慢轮声明了一份合法的动作库 |
| `decision` | 闸门收到了 receipt；列出可用与被拒的候选及原因 |
| `substitution-issued` | 用一次原生调用替换了慢回调 |
| `substitution` | 证书，绑定精确的原生结果 |
| `dispatch-rejected` | `tools/execute` 拒绝了被篡改或越出动作库的调用 |
| `selector-abstain` / `selector-unmapped` | 选择器弃权；慢路径恢复 |

收据里的 `stdout` 与 `stderr` 在守卫看到之前会先规范化：CRLF 变 LF、去掉一个尾部换行；
渲染器自带的状态尾巴（`[exit code: N]`、`[output truncated; …]`）在退出码解读完成之后
被移除。因此守卫比较的是程序输出，而不是 shell 的渲染结果。

## 测试

```sh
node --test test/*.test.mjs
```

`test/controller.test.mjs` 覆盖纯协议（动作库解析、receipt、守卫、准入、投影、派发）。
`test/host.test.mjs` 用一个 fake cordis context 驱动 `lib/index.js`，断言：确定性提交
从不调用慢模型；两候选步骤执行的是选择器答复的那个标签；选择器复用边界路由并请求一个
推理档位；选择器失败时把这一轮交还（两次，每种请求形状各一次）；禁用选择器时不花任何
调用；证书携带的是原生结果而不只是执行对象；轮次中途重启恢复的是可执行的动作库而不只是
动作库哈希；配置的 `maximum_candidates` 约束真实解析路径；提示词里被代码围栏包住的示例
不会被当成声明；被篡改的参数会被拒绝。共 42 个测试通过。

`host.test.mjs` 会从磁盘解析 `@deepseek-ai/schemastery`，因此本地需要这个包。在 DSH
profile 内部，宿主通过自己的内部 loader 提供它；在 profile 之外，跑 host 测试前要自己
准备：

```sh
mkdir -p node_modules/@deepseek-ai/schemastery
cp -r <extracted-schemastery>/* node_modules/@deepseek-ai/schemastery/
```

没有它时，只有 `test/controller.test.mjs`（28 个纯协议测试）能跑。