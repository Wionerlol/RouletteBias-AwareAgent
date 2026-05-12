# Roulette Bias-Aware Agent

一个**学习用**的云端 AI agent 项目。技术目标是搭一套有状态、带工具调用、能自我演化的云端 agent；业务场景借用轮盘赌作为练手对象。

> ⚠️ **数学事实，必须先看清楚**
>
> 美式轮盘（含 0 和 00）所有押法的数学期望都是 **-5.26%**（five-line 押法是 -7.89%），欧式轮盘是 **-2.70%**。这意味着无论 agent 如何"优化"，**长期 bankroll 期望必然单调下降**。
>
> "过去十几或二十几轮 60% 黑、59% 奇、35 出现 3 次"在公平轮盘上是**完全正常的统计波动**，**不构成下注优势**。前 200 轮基本没法在统计上区分公平轮盘和不公平轮盘。**桌面显示屏给出的红黑/奇偶比例样本量未知**——可能是几十轮也可能是几千轮——所以它的统计意义不能直接当作"已观测 N 次"来用，详见 §3.3。
>
> 真正能在轮盘上挣到正期望的唯一方法是**发现物理上有偏的轮盘**（轮子磨损、倾斜），并且通常需要**数千轮**观测样本。
>
> 本项目的真正学习价值：
> 1. 搭一个**完整闭环**的云端 agent（有状态、记忆、工具、反思、部署）
> 2. 亲眼看到"agent 不能违背环境的数学约束"——这套架构换到正期望场景就能赚钱
> 3. 用代码实现并验证统计检验、贝叶斯混合、Kelly 准则等概率工具

---

## 目录

1. [项目目标](#1-项目目标)
2. [轮盘数据结构](#2-轮盘数据结构)
3. [核心算法](#3-核心算法)
4. [Agent 架构](#4-agent-架构)
5. [技术栈](#5-技术栈)
6. [数据模型](#6-数据模型)
7. [API 设计](#7-api-设计)
8. [用户流程](#8-用户流程)
9. [关键设计决策](#9-关键设计决策)
10. [里程碑与落地步骤](#10-里程碑与落地步骤)

---

## 1. 项目目标

### 1.1 功能目标

- **手机为唯一前端**。首次输入：初始 bankroll、押注单位、可选剔除区域、**最近若干轮的开奖数字序列**（数量可变，常见 10-30 轮）、**可选的桌面聚合统计**（红黑比例、奇偶比例，样本量未知）。之后每轮**只输入开奖数字**。
- **agent 输出每轮策略**：一组 `(bet_type, numbers, amount)`。
- **agent 维护状态**：bankroll、历史序列、外部聚合统计、信念模型超参、累积经验笔记。
- **每轮做偏向检测**：判断轮盘是否"有问题"，输出 verdict 和置信权重。**两类输入分开处理**：内部历史序列做正式统计检验，外部聚合统计作为辅助 sanity check。
- **bias-aware 信念模型**：把偏向检测的置信度作为权重，混入下注决策。
- **服务部署在云端**：用户不需要自己开机器。

### 1.2 学习目标

- 用 Claude Agent SDK 实现 tool-calling 循环
- 在 LLM 决策外，让数学工具承担可验证的部分
- 状态持久化（Postgres + JSONB）
- 让 LLM 做"软优化"（反思 → 更新经验笔记）和"硬优化"（数值超参更新）
- 部署到云、做手机端 UI（PWA 或 Telegram bot）

---

## 2. 轮盘数据结构

### 2.1 数字属性

```
RED   = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
BLACK = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
GREEN = {0, 37}     # 37 代表 00，方便整数索引
```

每个数字 n 的派生属性：`color(n)`、`parity(n)`、`high_low(n)`、`dozen(n)`、`column(n)`。

### 2.2 桌面网格（3 行 × 12 列 + 0/00）

```
   ┌─────┬─────┐
   │  0  │ 00  │
   ├──┬──┼──┬──┼──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
列1│ 1│ 4│ 7│10│13│16│19│22│25│28│31│34│ 2:1
列2│ 2│ 5│ 8│11│14│17│20│23│26│29│32│35│ 2:1
列3│ 3│ 6│ 9│12│15│18│21│24│27│30│33│36│ 2:1
   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
     1st 12          2nd 12         3rd 12
```

`grid_pos(n)` 返回 `(col_index ∈ [0,11], row_index ∈ [0,2])`。

### 2.3 轮盘物理顺序（重要——用于扇区偏差检测）

美式轮盘顺时针顺序：

```
0, 28, 9, 26, 30, 11, 7, 20, 32, 17, 5, 22, 34, 15, 3, 24, 36, 13, 1,
00, 27, 10, 25, 29, 12, 8, 19, 31, 18, 6, 21, 33, 16, 4, 23, 35, 14, 2
```

欧式轮盘单独存。**这个顺序和桌面上的网格顺序完全不同**——物理偏差在轮盘上是连续扇区，在桌面上是离散的。

### 2.4 下注类型表

| 类型 | 净赔率 | 覆盖数字数 | house edge |
|---|---|---|---|
| straight | 35 | 1 | 5.26% |
| split | 17 | 2 | 5.26% |
| street | 11 | 3 | 5.26% |
| corner | 8 | 4 | 5.26% |
| five_line（美式独有）| 6 | 5 | **7.89%** |
| six_line | 5 | 6 | 5.26% |
| red / black | 1 | 18 | 5.26% |
| odd / even | 1 | 18 | 5.26% |
| low / high | 1 | 18 | 5.26% |
| dozen | 2 | 12 | 5.26% |
| column | 2 | 12 | 5.26% |

**净赔率**：押 1 中了拿回 (1 + payout)。EV 公式：`EV = (payout+1) × size/38 - 1`。

### 2.5 校验

数据结构写完必须配单元测试：
- 红黑各 18、奇偶各 18、各 dozen 12、各 column 12
- 每个 corner 的 4 个数字在网格上构成 2×2
- 每种 bet type 的期望值 = -house edge（误差 < 1e-9）

---

## 3. 核心算法

### 3.1 信念模型（bias-aware）

```
p(n) = w · p_bias(n) + (1 - w) · p_uniform(n)
```

其中：
- `p_uniform(n) = 1/38`（公平轮盘真实概率）
- `p_bias(n) = (count(n) + k) / (N + 38·k)`（Dirichlet 后验，k 是先验强度）
- `w ∈ [0,1]` 来自偏向检测器（见 3.2），统计越显著 w 越大
- **N 是"内部历史序列"的总轮数**，即用户初始输入的 12-23 轮 + 本次 session 开始后每轮观察到的开奖数字之和。**不包含**桌面显示屏的外部聚合统计（见 3.3）

**关键性质**：N 小时 w≈0，p(n)≈1/38，所有 EV 为负 → agent 自动选择保守或不下注；只有强证据下 w 才会大，p_bias 才会真正影响决策。

### 3.2 偏向检测（bias detector）

每轮调用一次，多层次跑统计检验：

| 层 | 检验 | 何时可信 |
|---|---|---|
| 单数字 | χ² 拟合优度（df=37） | N ≥ ~200，且每格期望 ≥ 5 |
| 轮盘扇区（4 或 8 段） | χ² 拟合优度 | N ≥ 50 |
| 红黑 / 奇偶 / 高低 | 二项检验 | N ≥ 30 |
| dozen / column | χ² 或多项检验 | N ≥ 60 |
| 单热数字 | 二项检验，**Bonferroni 修正 38 倍** | N ≥ 100 |

**verdict 阈值**：

| verdict | 条件 | weight |
|---|---|---|
| no_evidence | 所有可用检验 p > 0.1，或样本太小 | 0.00 |
| weak | 某检验 p ∈ [0.01, 0.1] | 0.15 |
| moderate | 某检验 p ∈ [0.001, 0.01] **且** N ≥ 200 | 0.45 |
| strong | 某检验 p < 0.001 **且** N ≥ 500 | 0.80 |

阈值映射作为可调超参，可由"硬优化"模块调整。

**多重比较**：单数字检验做了 38 次，p-value 必须 Bonferroni 修正后才用于 verdict；未修正值仅供展示。

### 3.3 两类数据源的处理

项目有**两种本质不同**的输入：

**(I) 内部历史序列** —— `recent_history: list[int]`
- 用户初始输入的 12-23 个数字 + session 开始后每轮的开奖数字
- 是**有序、完整、样本量已知**的原始数据
- **唯一**用于：信念模型的 `count(n)`、所有正式统计检验、bias verdict 的判定

**(II) 外部聚合统计** —— `external_stats: dict | None`
- 桌面显示屏给出的红黑比例、奇偶比例等
- 样本量 **N_ext 未知**（可能 50 轮，也可能 5000 轮）
- 用户**可选**输入。如果输入，可附带一个估计的 N_ext（"我猜 ~100 轮"），但不强求

外部统计的样本量是关键不确定性。同一个"62% 黑"在不同 N_ext 下统计意义完全不同：

| N_ext | 62% 黑 vs 公平 50% 的双尾 p-value |
|---|---|
| 30 | 0.20（毫无意义）|
| 100 | 0.018（弱证据）|
| 500 | 8e-8（强证据）|
| 5000 | < 1e-50（极强证据）|

**所以本项目坚决不把外部聚合统计当成"已观测 N 次"直接融进卡方**，而是采用以下两层处理：

**默认模式（保守）：external_stats 仅作辅助 sanity check**
- 不参与 verdict 判定
- 在 bias_report 里单独有一个 `external_check` 字段，告诉 agent 和用户："显示屏 62% 黑，内部 15 轮 60% 黑，方向一致" 或 "显示屏 62% 黑，内部 8 轮 38% 黑，方向相反，怀疑显示屏数据陈旧"
- 这样能给 Claude 一个软提示，但不污染数学

**进阶模式（可选）：用户输入 N_ext 估计值**
- 用户在 init 时填 `external_stats_n_estimate: int | null`
- 如果填了，额外跑一个辅助二项检验，**结果明确标注为"基于用户估计的样本量，置信度打折"**
- verdict 仍由内部历史主导；外部辅助检验只能把已有 verdict **降级**（用作 sanity check 反驳），不能升级

**为什么外部辅助只能降级不能升级**：用户估计的 N_ext 不可信，如果用它升级 verdict 等于"基于用户瞎猜的样本量给出强信号"——典型的 garbage in, garbage out。但如果内部检验说 weak 而外部对应数据完全不一致，那是个有用的反驳信号，可以下调 weight。

**对边界情况的处理**：
- 用户没提供 external_stats：`external_check = None`，完全跳过
- 用户提供了 external_stats 但没填 N_ext：只做方向一致性 sanity check，不做检验
- 用户的 recent_history 极短（< 5 轮）：禁用所有统计检验，verdict=no_evidence

### 3.4 下注组合优化

三种策略，agent 自选或用户指定：

**A. 期望值最大化**
对每种合法下注 b 计算 `EV(b) = (payout+1) × Σp(n) for n∈covered(b) - 1`，挑 EV>0 的按权重分配；全为负则**不下注**。

**B. Fractional Kelly**（推荐默认 0.25× Kelly）
```
f* = (b·p - q) / b
allocation = max(0, fraction × f* × bankroll)
```

**C. 固定 baseline**（对照组）
1/64 bankroll 押在频率前 3 高的单数字。

### 3.5 离散化约束

押注单位 2.5。所有 amount 向下取整到 2.5 的倍数。总下注额 ≤ bankroll。

### 3.6 剔除区域

用户开局可指定 `excluded_dozens: list[int]`（`[]`、`[1]`、`[3]`，**不允许 `[1,3]` 因为没意义**）。

- **信念模型不变**：球还是会落在被剔除区，p(n) 仍然计算所有 38 格
- **合法下注被裁剪**：optimizer 过滤掉任何覆盖被剔除区域数字的 bet
- **跨界 corner/split**（如 12-15 的 split）：严格模式禁用

### 3.7 自我优化

两层：

**硬优化（数值）**
- 信念模型 α / k（响应预测准确度）
- 策略选择权重（A/B/C 三策略的近期收益率，ε-greedy 或 UCB）
- bias verdict → weight 映射阈值
- 风险阈值（bankroll 下降时收紧 EV 门槛）

**软优化（LLM 反思）**
每 N 轮或亏损超阈值时，让 Claude 跑一次反思：
- 输入：近 N 轮的 bets / results / pnl / bias_report
- 输出：自然语言"经验笔记"，写进 `session.notes`，下次作为 system prompt 的一部分

---

## 4. Agent 架构

### 4.1 模块图

```
┌─────────────────────────────────────────┐
│  Phone (PWA / Telegram bot)             │
│  初次输入，之后只输数字                  │
└────────────────┬────────────────────────┘
                 │ HTTPS
┌────────────────▼────────────────────────┐
│  FastAPI (cloud)                        │
│  POST /session/new                      │
│  POST /session/{id}/spin                │
│  GET  /session/{id}/state               │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Agent core (Claude Agent SDK)          │
│  tool-calling loop, reflection          │
└──┬─────────┬─────────┬─────────┬────────┘
   │         │         │         │
┌──▼──┐  ┌──▼──┐  ┌──▼──┐  ┌──▼──┐
│Stats│  │Bias │  │Opt. │  │Settle│
│  /  │  │Detec│  │     │  │      │
│Belief│ │tor  │  │     │  │      │
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │         │        │         │
┌──▼─────────▼────────▼─────────▼────────┐
│  Postgres                               │
│  session / spin / reflection            │
└─────────────────────────────────────────┘
```

### 4.2 一次 `/spin` 请求的完整流程

1. 加载 session 状态（bankroll、recent_history、external_stats、external_stats_n_estimate、超参、notes、excluded_dozens）
2. **结算**上一轮（settle 工具）
3. 把新开奖数字追加到 recent_history
4. 跑 **bias_detector**（recent_history + external_stats）→ 拿到 verdict、weight、external_check
5. 跑 **compute_belief**（只用 recent_history）→ 拿到 p(n)
6. **Claude agent loop**：
   - System prompt 含规则、bankroll、超参、notes、bias_report（含 external_check 字段）
   - Claude 调用 `kelly_allocation` / `greedy_ev_allocation`（带 excluded_dozens）
   - Claude 输出最终策略 + rationale
7. **可选反思**（每 N 轮或亏损超阈值）：Claude 再被调用，更新 notes 和超参
8. 写库
9. 返回手机

### 4.3 Claude 的工具签名

```python
@tool
def compute_belief(recent_history: list[int],
                   bias_report: dict,
                   k_prior: float = 10.0) -> dict[int, float]:
    """bias-aware 信念模型，返回 p(n) for n in 0..37
    
    只用 recent_history（内部已知样本量序列）计算 count(n)；不使用外部聚合统计。
    """

@tool
def detect_bias(recent_history: list[int],
                wheel_type: str = "american",
                external_stats: dict | None = None,
                external_stats_n_estimate: int | None = None) -> dict:
    """跑所有统计检验，返回 verdict / weight / 细节
    
    recent_history: 已知样本量的内部历史序列（用户初始输入 + session 内观察）
    external_stats: 可选，桌面显示屏给出的比例，如 {"black": 0.62, "odd": 0.59}
    external_stats_n_estimate: 可选，用户对外部样本量的估计；不影响 verdict 升级，仅用于辅助降级
    """

@tool
def compute_ev(p: dict[int, float],
               bet_type: str,
               numbers: list[int]) -> float:
    """主观 EV"""

@tool
def kelly_allocation(p: dict[int, float],
                     bankroll: float,
                     bet_unit: float,
                     excluded_dozens: list[int],
                     fraction: float = 0.25) -> list[dict]:
    """Fractional Kelly，EV<=0 不分配"""

@tool
def greedy_ev_allocation(p: dict[int, float],
                         bankroll: float,
                         bet_unit: float,
                         excluded_dozens: list[int],
                         max_bet_fraction: float = 0.1) -> list[dict]:
    """EV 加权分配"""

@tool
def settle(bets: list[dict], result_number: int) -> dict:
    """返回 {pnl, detail[]}"""
```

---

## 5. 技术栈

| 组件 | 选型 |
|---|---|
| 语言 | Python 3.11+ |
| Agent 框架 | Claude Agent SDK (Python) |
| Web 框架 | FastAPI |
| 数据库 | Postgres（生产）/ SQLite（本地） |
| ORM | SQLAlchemy 2.x + Alembic |
| 部署 | Railway 或 Fly.io |
| 前端 MVP | PWA（HTML + 少量 JS） |
| 前端升级 | Telegram bot |
| 认证 | API Key（header）|
| 监控 | Logfire 或 Sentry |
| 测试 | pytest + hypothesis |

---

## 6. 数据模型

### 6.1 `session`

```sql
CREATE TABLE session (
    id                          UUID PRIMARY KEY,
    created_at                  TIMESTAMPTZ DEFAULT now(),
    wheel_type                  TEXT,                  -- 'american' | 'european'
    bankroll_init               NUMERIC,
    bet_unit                    NUMERIC,
    bankroll_now                NUMERIC,
    excluded_dozens             JSONB DEFAULT '[]',    -- e.g. [1] or [3] or []
    initial_history             JSONB,                 -- 用户初始输入的 12-23 轮，原样保留以备审计
    external_stats              JSONB,                 -- e.g. {"black": 0.62, "odd": 0.59} 或 null
    external_stats_n_estimate   INT,                   -- 用户估计的外部样本量；可空
    hyperparams                 JSONB,                 -- k_prior, fraction, verdict→weight 映射等
    notes                       TEXT DEFAULT ''        -- LLM 累积的经验笔记
);
```

注：`session.recent_history` 不单独存——它等于 `initial_history + 所有 spin.result_number`，需要时通过 join 重建。这避免数据冗余和不一致。

### 6.2 `spin`

```sql
CREATE TABLE spin (
    id              SERIAL PRIMARY KEY,
    session_id      UUID REFERENCES session(id),
    spin_index      INT,
    bets            JSONB,                   -- [{type, numbers, amount}, ...]
    bets_total      NUMERIC,
    result_number   INT,                     -- 0..37（37=00）
    pnl             NUMERIC,
    bankroll_after  NUMERIC,
    bias_report     JSONB,                   -- 当时的 bias detector 输出
    rationale       TEXT,                    -- Claude 给的简短理由
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### 6.3 `reflection`

```sql
CREATE TABLE reflection (
    id              SERIAL PRIMARY KEY,
    session_id      UUID REFERENCES session(id),
    at_spin_index   INT,
    observation     TEXT,
    hyperparam_diff JSONB,                   -- 改了哪些超参
    notes_diff      TEXT,                    -- notes 增量
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

---

## 7. API 设计

### 7.1 `POST /session/new`

请求：
```json
{
  "wheel_type": "american",
  "bankroll": 800,
  "bet_unit": 2.5,
  "excluded_dozens": [3],
  "recent_history": [35, 17, 35, 29, 22, 31, 35, 11, 29, 4, 13, 26, 8, 31, 20, 17, 23, 0, 11],
  "external_stats": {
    "black_pct": 0.62,
    "red_pct":   0.37,
    "odd_pct":   0.59,
    "even_pct":  0.40
  },
  "external_stats_n_estimate": null,
  "strategy_pref": "auto"
}
```

字段说明：
- `recent_history` 必填，至少 1 个数字，长度自由（典型 10-30 轮）
- `external_stats` 可选；如果填，至少包含 black_pct/red_pct 或 odd_pct/even_pct 中的一组
- `external_stats_n_estimate` 可选；用户对外部样本量的估计，null 表示"不知道"

响应：
```json
{
  "session_id": "uuid-...",
  "spin_index": 0,
  "bankroll_now": 800,
  "bias_report": {
    "verdict": "no_evidence",
    "weight": 0.0,
    "n_internal": 19,
    "external_check": {
      "status": "consistent",
      "note": "显示屏 62% 黑 vs 内部 19 轮 58% 黑，方向一致但样本量都不足。"
    },
    "summary": "样本量不足，无统计证据，按公平轮盘对待。"
  },
  "next_strategy": [
    {"type": "black",    "numbers": null,                 "amount": 50},
    {"type": "odd",      "numbers": null,                 "amount": 50},
    {"type": "corner",   "numbers": [28, 29, 31, 32],     "amount": 15},
    {"type": "straight", "numbers": [31],                 "amount": 25}
  ],
  "rationale": "..."
}
```

### 7.2 `POST /session/{id}/spin`

请求：
```json
{ "result_number": 31 }
```

响应：
```json
{
  "spin_index": 16,
  "pnl_last_spin": -25,
  "bankroll_now": 775,
  "bias_report": {
    "verdict": "weak",
    "weight": 0.15,
    "n_internal": 20,
    "external_check": {"status": "consistent", "note": "..."},
    "summary": "..."
  },
  "next_strategy": [...],
  "rationale": "..."
}
```

### 7.3 `GET /session/{id}/state`

返回当前 session 全状态：bankroll、最近 K 轮历史、当前超参、notes 摘要、最近 bias_report。用于手机端刷新。

---

## 8. 用户流程

1. 打开 PWA / 给 Telegram bot 发 `/new`
2. 填入桌面信息：
   - bankroll=800, bet_unit=2.5, exclude=3rd_12
   - 粘贴你能看清的最近开奖数字（数量自由，10-30 轮都行）
   - **可选**：填入显示屏上的红黑/奇偶比例；样本量估计可以留空
3. 收到第一轮策略 + bias verdict + external check
4. 实际去下注 → 看开奖数字 → 输入数字
5. 收到本轮 pnl + 新 bankroll + bias verdict 更新 + 下一轮策略
6. 循环 4-5 直到用户停手或 bankroll 归零

---

## 9. 关键设计决策

### 9.1 为什么 LLM 不直接算策略

让 LLM 心算赔率会出错。**LLM 的角色是编排者**：决定调用哪些工具、用什么参数、综合结果。所有数学计算都在确定性的 Python 工具里，可验证。

### 9.2 为什么 bias detector 默认极保守

十几二十轮 60% 黑在公平轮盘上是常见波动。如果 verdict 阈值太松，weight 立刻飙高，agent 会基于噪声押注，亏得更快。**保守阈值确保 agent 在没证据时不"自欺"**。

### 9.3 为什么外部聚合统计不参与正式检验

桌面显示屏的红黑/奇偶比例**样本量未知**。同一个"62% 黑"，N=30 时 p=0.20（无意义），N=5000 时 p<1e-50（极强证据）——两者结论天差地别。把它当成已知样本量塞进卡方等于让用户猜出来的数字决定 verdict 升级，garbage in garbage out。

所以外部统计**只做两件事**：(a) 给 agent 一个软提示（方向是否一致），(b) 当用户主动估计 N_ext 时，作为**辅助降级**信号——内部检测说 weak、外部数据完全相反 → 下调 weight。永远不参与 verdict 升级。

### 9.4 为什么剔除区域不影响信念模型

球落点的物理过程不受用户押注偏好影响。剔除只是"这片我不碰"，是合法 bet 集的约束，不是概率分布的修改。

### 9.5 为什么用 JSONB 存 bets 而不是关系表

bets 是异构的——straight 用单数字，corner 用 4 个数字，颜色押注没有数字。JSONB 比建一堆关联表更直接，查询用 Postgres 的 JSONB operators 也够用。

### 9.6 为什么默认 Fractional Kelly = 0.25

全 Kelly 方差极大、即使有正期望也容易破产。0.25× Kelly 是文献和实践里的常见保守值，bankroll 增长率约为全 Kelly 的 7/16，但破产概率远低。

---

## 10. 里程碑与落地步骤

详细的多轮 Claude Code prompts 见 `claude_code_prompts.md`。

| 里程碑 | 交付物 | 预计耗时 |
|---|---|---|
| **M1** 数据结构 | `roulette_layout.py` + 完整单元测试（含轮盘物理顺序）| 0.5 天 |
| **M2** 结算与统计基础 | `settler.py`, `stats.py`，本地能算账，无 LLM | 0.5 天 |
| **M3** 偏向检测器 | `bias_detector.py` + 在公平/不公平模拟数据上验证 | 1 天 |
| **M4** 信念模型 + 优化器 | `belief.py`, `optimizer.py`（A/B/C 三策略 + excluded_dozens 支持）| 1 天 |
| **M5** 蒙特卡洛模拟器 | `simulator.py`，画 bankroll 曲线，看 -5.26% 长什么样 | 0.5 天 |
| **M6** Claude Agent 接入 | `agent.py`，本地 CLI 跑通完整一轮 | 1 天 |
| **M7** FastAPI 服务 | `app.py` + SQLAlchemy 模型 + 三个端点 | 1 天 |
| **M8** 部署到云 | Railway 部署 + Postgres + 公网 HTTPS | 0.5 天 |
| **M9** 手机前端（MVP）| 简单 PWA（一个页面 + 输入框 + 卡片）| 0.5 天 |
| **M10** 反思机制 | 每 N 轮触发反思 + notes 更新 | 0.5 天 |
| **M11**（可选）Telegram bot | 替代 PWA 的对话式前端 | 0.5 天 |
| **M12**（可选）硬优化 | 超参自动调整（ε-greedy 选策略等）| 1 天 |

**关键路径**：M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9。其余可后置或跳过。

总计 MVP（M1-M9）约 6-7 天工作量。
