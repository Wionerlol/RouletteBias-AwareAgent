# Claude Code 多轮 Prompts

把这个文件和 `README.md` 一起放进项目根目录。然后**按顺序**把下面每个 prompt 块独立喂给 Claude Code，每完成一轮后再发下一轮。每轮 prompt 后面都标明了**里程碑、依赖、产出、验收标准**。

> **使用建议**
> - 第一次进入项目目录时，先让 Claude Code 读 `README.md`：`请先完整阅读 README.md，然后等待我的下一条指令。`
> - 每个 prompt 都明确指定了文件名和函数签名，尽量不改。
> - 每完成一轮，跑一次该轮的"验收标准"再进下一轮。
> - 如果某轮失败，把错误信息丢回给 Claude Code 修复，不要带着错误进下一轮。

> ⚠️ **架构里最关键的一个区分（请确保 Claude Code 理解）**：
>
> 项目有**两种本质不同**的输入数据：
> 1. `recent_history: list[int]` —— 用户初始输入的开奖序列（典型 10-30 轮，长度可变）+ session 内所有 spin 的 result_number。**样本量已知、有序、用于所有正式统计检验**。
> 2. `external_stats: dict | None` —— 桌面显示屏给出的红黑/奇偶比例。**样本量未知**，可能 50 轮可能 5000 轮。只做辅助 sanity check，**不参与 verdict 升级**。
>
> 详见 README.md §3.3 和 §9.3。Round 3 测试明确覆盖这一点。

---

## 总览

| 轮次 | 里程碑 | 依赖 | 时间 |
|---|---|---|---|
| **Round 0** | 项目初始化 | — | 10 min |
| **Round 1** | M1 轮盘数据结构 | R0 | 0.5 天 |
| **Round 2** | M2 结算与统计 | R1 | 0.5 天 |
| **Round 3** | M3 偏向检测器 | R1, R2 | 1 天 |
| **Round 4** | M4 信念模型 + 优化器 | R1, R3 | 1 天 |
| **Round 5** | M5 蒙特卡洛模拟器 | R1, R2, R4 | 0.5 天 |
| **Round 6** | M6 Claude Agent 接入 | R1-R4 | 1 天 |
| **Round 7** | M7 FastAPI 服务 | R6 | 1 天 |
| **Round 8** | M8 云部署 | R7 | 0.5 天 |
| **Round 9** | M9 手机前端 PWA | R7, R8 | 0.5 天 |
| **Round 10** | M10 反思机制 | R6, R7 | 0.5 天 |
| **Round 11**（可选）| M11 Telegram bot | R7, R8 | 0.5 天 |
| **Round 12**（可选）| M12 硬优化 | R5, R6 | 1 天 |

**关键路径**：R0 → R1 → R2 → R3 → R4 → R5 → R6 → R7 → R8 → R9。MVP 到 R9 即完成。

---

## Round 0：项目初始化

**目标**：建立项目骨架、依赖、git、基础配置。

**Prompt**:

```
请先完整阅读 README.md，然后初始化项目骨架。

要求：
1. 创建以下目录结构：
   roulette_agent/
   ├── pyproject.toml          # uv 或 poetry，Python 3.11+
   ├── README.md               # 已存在
   ├── claude_code_prompts.md  # 已存在
   ├── .gitignore
   ├── .env.example            # ANTHROPIC_API_KEY, DATABASE_URL 占位
   ├── src/roulette_agent/
   │   ├── __init__.py
   │   ├── layout.py           # 占位
   │   ├── stats.py            # 占位
   │   ├── bias_detector.py    # 占位
   │   ├── belief.py           # 占位
   │   ├── optimizer.py        # 占位
   │   ├── settler.py          # 占位
   │   ├── simulator.py        # 占位
   │   ├── agent.py            # 占位
   │   ├── tools.py            # 占位（Claude tool 定义）
   │   ├── models.py           # 占位（SQLAlchemy）
   │   └── app.py              # 占位（FastAPI）
   └── tests/
       ├── __init__.py
       └── （后续每个模块对应一个 test_*.py）

2. pyproject.toml 依赖：
   - anthropic (官方 SDK，包含 Agent SDK 能力)
   - fastapi
   - uvicorn[standard]
   - sqlalchemy>=2.0
   - alembic
   - psycopg[binary]
   - pydantic>=2
   - scipy            # 卡方、二项检验
   - numpy
   - pytest
   - hypothesis       # 属性测试
   - httpx            # FastAPI 测试

3. .gitignore 包含 Python 标准忽略 + .env + *.db + .venv/

4. 初始化 git 仓库，做一个 initial commit。

5. 在每个空模块顶端加一行模块 docstring 说明它将承担什么职责。

完成后告诉我：(a) 目录树；(b) Python 版本；(c) 怎么用 uv 或 poetry 创建虚拟环境并安装依赖。
```

**验收**：
- 目录结构完整，所有 stub 文件存在
- `pip install -e .` 或 `uv sync` 能成功
- `pytest` 能跑（即使没用例）

---

## Round 1：M1 轮盘数据结构（数据层基石）

**目标**：写完所有静态数据 + 校验。这是地基，必须 100% 正确。

**Prompt**:

```
实现 src/roulette_agent/layout.py，提供完整的轮盘数据结构。请严格按下面的规格：

1. 常量
   - RED, BLACK: set[int]，按 README.md §2.1 的定义
   - GREEN = {0, 37}，其中 37 代表 00
   - WHEEL_ORDER_AMERICAN: list[int]，按 README.md §2.3 顺时针顺序
   - WHEEL_ORDER_EUROPEAN: list[int]，只有 0（单零），顺时针顺序：
     0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26

2. 数字属性函数（n: int 输入，None for 不适用）
   - color(n) -> "red" | "black" | "green"
   - parity(n) -> "odd" | "even" | None      # 0, 00 不分奇偶
   - high_low(n) -> "low" | "high" | None    # 1-18 low, 19-36 high
   - dozen(n) -> 1 | 2 | 3 | None             # 1st/2nd/3rd 12
   - column(n) -> 1 | 2 | 3 | None            # 桌面"列1/2/3"，对应 n%3 的映射
   - grid_pos(n) -> tuple[int, int] | None    # (col_idx 0..11, row_idx 0..2)
     注意：col_idx = (n-1) // 3, row_idx = 2 - (n-1) % 3

3. 用 dataclass 或 TypedDict 定义 BetType:
   class BetType:
       name: str
       payout: int          # 净赔率
       size: int            # 覆盖数字数
       edge: float          # house edge（美式默认）
       covered_fn: Callable[[list[int]], set[int]]
       # covered_fn 接收 bet 的 numbers 参数，返回实际覆盖的数字集合
       # 例如 straight(numbers=[31]) -> {31}
       # 例如 corner(numbers=[28,29,31,32]) -> {28,29,31,32}
       # 例如 red(numbers=None) -> RED
   
   BET_TYPES: dict[str, BetType]
   完整覆盖 README.md §2.4 列出的所有类型。

4. 工具函数
   - is_valid_bet(bet_type: str, numbers: list[int] | None) -> bool
     校验下注合法性。例如 corner 的 4 个数字必须在网格上是 2x2；split 必须相邻。
   - get_covered_numbers(bet_type: str, numbers: list[int] | None) -> set[int]
   - expected_value(bet_type: str, numbers: list[int] | None, p: dict[int, float] = None) -> float
     默认 p 为均匀分布 1/38（美式）；可传主观概率。

5. 配套 tests/test_layout.py：
   - 红黑各 18 个，奇偶各 18 个，high/low 各 18 个
   - 各 dozen 12 个，各 column 12 个
   - WHEEL_ORDER_AMERICAN 长度 38、含且仅含 0..37 各一次
   - WHEEL_ORDER_EUROPEAN 长度 37、含且仅含 0..36 各一次
   - 每种 BET_TYPES 在均匀概率下 expected_value 等于 -edge（误差 < 1e-9）
   - 经典 corner 合法性：[28,29,31,32] 合法；[1,2,3,4] 不合法（不构成 2x2）
   - 用 hypothesis 写一个属性测试：随机 n ∈ 1..36，验证 (color, parity, dozen, column) 的组合一致性
   - grid_pos 反向：从 (col,row) 应能算回原始 n

6. five_line 是美式独有，欧式调用时应抛 ValueError 或返回 False。先按美式实现，欧式标 TODO。

完成后跑 pytest，把测试结果贴出来。任何失败必须修复。
```

**验收**：
- `pytest tests/test_layout.py -v` 全绿
- 跑一个简单 sanity check（如下）应得到 -0.0526：
  ```python
  from roulette_agent.layout import expected_value
  print(expected_value("straight", [17]))   # -0.0526...
  ```

---

## Round 2：M2 结算 + 基础统计

**目标**：能算账（给定 bets + 开奖数字 → pnl）+ 历史数据基础统计。

**Prompt**:

```
依赖 layout.py，实现两个模块。

A. src/roulette_agent/settler.py

   def settle(bets: list[dict], result_number: int) -> dict:
       """
       bets: [{"type": str, "numbers": list[int] | None, "amount": float}, ...]
       result_number: 0..37 (37 = 00)
       
       返回 {
           "result_number": int,
           "total_staked": float,
           "total_payout": float,        # 含本金的回收
           "pnl": float,                 # = total_payout - total_staked
           "detail": [
               {"bet": {...}, "won": bool, "payout": float}
               ...
           ]
       }
       
       规则：
       - 中了：返还 amount * (payout + 1)
       - 没中：返还 0（amount 已损失）
       - covered_numbers 由 layout.get_covered_numbers 决定
       """

B. src/roulette_agent/stats.py

   def frequency_counts(history: list[int], wheel_type: str = "american") -> dict[int, int]:
       """每个数字出现次数，包括出现 0 次的"""
   
   def basic_stats(history: list[int], wheel_type: str = "american") -> dict:
       """
       返回 {
           "n_spins": int,
           "red_count": int, "red_pct": float,
           "black_count": int, "black_pct": float,
           "green_count": int,
           "odd_count": int, "odd_pct": float,    # 注意分母是非 green 总数
           "even_count": int, "even_pct": float,
           "low_count": int, "low_pct": float,
           "high_count": int, "high_pct": float,
           "dozen_counts": {1: int, 2: int, 3: int},
           "column_counts": {1: int, 2: int, 3: int},
           "frequency": dict[int, int],            # = frequency_counts
           "hot_numbers_top5": list[tuple[int, int]],  # 按出现次数降序
       }
       """

C. 测试 tests/test_settler.py + tests/test_stats.py
   - settle: 单数字命中、外围押注命中、未命中、多注混合
   - 注意 0/00 的特殊处理（不算红黑、不算奇偶）
   - 用 hypothesis：随机生成 bets 和 result，验证 pnl 在合理范围
   - stats: 空 history 返回正确零值；含 0/00 时分母处理正确

完成后跑测试并贴出结果。
```

**验收**：
- `pytest tests/test_settler.py tests/test_stats.py -v` 全绿
- 手动测一个例子：bets=[{"type":"corner","numbers":[28,29,31,32],"amount":15}]，result=31 → pnl = 15×8 = 120

---

## Round 3：M3 偏向检测器（最关键的统计模块）

**目标**：把"轮盘是否有偏"这件事做对。**这个模块决定 agent 会不会基于噪声乱押**。

**Prompt**:

```
依赖 layout.py 和 stats.py，实现 src/roulette_agent/bias_detector.py。

详细规格见 README.md §3.2 和 §3.3。核心要求：

== 重要概念区分 ==
本项目有两种输入：
- recent_history: list[int]，内部有序、样本量已知的开奖数字序列
- external_stats: dict | None，桌面显示屏的聚合统计，样本量未知

所有正式统计检验只用 recent_history。external_stats 只做辅助 sanity check，
且只能用于"降级"verdict（弱化已经检测到的弱信号），不能升级。
详见 README.md §3.3 和 §9.3。

1. 内部检验函数（用 scipy.stats，只用 recent_history）
   - chi_square_single(recent_history, wheel_type) -> dict
     38 格的 χ² 拟合优度。返回 stat, df, p_value, usable (bool), reason (str)。
     usable 标准：每格期望次数 ≥ 5（即 N ≥ 190）。否则 usable=False，p_value 仍计算但标 reason="E<5"。
   
   - chi_square_sector(recent_history, wheel_type, n_sectors=8) -> dict
     按 WHEEL_ORDER_AMERICAN（或 EUROPEAN）分成 n_sectors 段，每段 4-5 个数字。
     做 χ² 拟合优度。同样返回 stat, df, p_value, usable。
     usable 标准：N ≥ 50。
   
   - binomial_test(recent_history, group: str) -> dict
     group ∈ {"red", "black", "odd", "even", "low", "high"}
     返回 observed_pct, n_effective (排除 green), p_value (双尾), usable (N≥30)。
   
   - hot_numbers_test(recent_history, wheel_type) -> list[dict]
     对每个数字单独做二项检验（H0: p=1/38），输出未修正 p-value。
     返回按 p-value 升序的前 5 个，每条 {n, observed, expected, p_uncorrected, p_bonferroni}。
     p_bonferroni = min(1, p_uncorrected * 38)。

2. 外部统计辅助检验（不参与 verdict 升级）
   
   def external_consistency_check(recent_history: list[int],
                                  external_stats: dict | None,
                                  external_n_estimate: int | None,
                                  wheel_type: str = "american") -> dict | None:
       """
       返回 None（如果 external_stats 是 None）或：
       {
           "status": "consistent" | "inconsistent" | "unknown",
           "details": {
               "black": {"internal_pct": 0.58, "external_pct": 0.62, "diff": 0.04},
               "odd": {...},
               ...
           },
           "auxiliary_p_values": {  # 只在 external_n_estimate 不为 None 时填
               "black": 0.10,
               "odd": 0.25,
           },
           "note": str    # 给 agent 看的一句话
       }
       
       逻辑：
       - 对每个外部统计字段（black/red/odd/even/low/high），算"内部对应比例"
       - 如果方向一致（都偏黑或都偏红）→ consistent
       - 如果方向相反 → inconsistent
       - 如果差异 < 0.05 → unknown（噪声范围内）
       - 如果 external_n_estimate 给了，对每个字段额外算一个二项检验 p-value，
         明确这是"基于用户估计样本量"，仅供 agent 参考
       """

3. 主入口
   
   def detect_bias(recent_history: list[int],
                   wheel_type: str = "american",
                   external_stats: dict | None = None,
                   external_n_estimate: int | None = None) -> dict:
       """
       返回完整 bias report：
       {
           "n_internal": int,                 # len(recent_history)
           "wheel_type": str,
           "tests": {
               "chi2_single": {...},
               "chi2_sector_8": {...},
               "binomial_red": {...},
               "binomial_odd": {...},
               "binomial_low": {...},
               "dozen": {...},
               "column": {...},
               "hot_numbers": [...]
           },
           "external_check": dict | None,     # external_consistency_check 输出
           "verdict": "no_evidence" | "weak" | "moderate" | "strong",
           "weight": float,                   # 0.0 / 0.15 / 0.45 / 0.80
           "suspected_bias": {
               "type": str | None,
               "details": str | None
           },
           "summary": str                     # 一句话给人看
       }
       
       verdict 决定逻辑：
       1. 先根据内部检验得出初步 verdict
       2. 如果 external_check.status == "inconsistent" 且初步 verdict 是 weak：
          降级为 no_evidence（外部数据反驳了弱信号）
       3. 永远不基于 external_check 升级 verdict
       """
   
4. verdict 阈值，写成模块级常量：
   
   VERDICT_THRESHOLDS = {
       "weak":     {"p_max": 0.1,   "p_min": 0.01,  "n_min": 30},
       "moderate": {"p_max": 0.01,  "p_min": 0.001, "n_min": 200},
       "strong":   {"p_max": 0.001, "p_min": 0.0,   "n_min": 500},
   }
   
   VERDICT_WEIGHTS = {
       "no_evidence": 0.0,
       "weak":        0.15,
       "moderate":    0.45,
       "strong":      0.80,
   }
   
   单数字检验使用 Bonferroni 修正后的 p-value 参与 verdict 判定。

5. 测试 tests/test_bias_detector.py：必须包含
   
   A. 公平轮盘模拟（关键！）：
      用 numpy 随机模拟 10000 轮公平轮盘，每 100 轮调一次 detect_bias（无 external_stats）。
      断言：> 95% 的调用返回 no_evidence 或 weak。
      （检测假阳性率。如果失败说明阈值太松。）
   
   B. 注入偏差：
      模拟一个"35 出现概率 = 3/38、其他 35 个数字均分剩余"的不公平轮盘。
      在 N = 100, 500, 2000 时各调一次 detect_bias。
      断言：N=2000 时 verdict 至少 weak；N=500 时大多数会捕获到。
   
   C. 外部统计降级：
      构造一个内部 recent_history 让初步 verdict=weak（比如 60 轮里红色占 65%）。
      喂一个 inconsistent 的 external_stats（比如 black_pct=0.70）。
      断言：最终 verdict 被降级为 no_evidence。
   
   D. 外部统计不能升级：
      内部 recent_history=20 轮（验证 verdict=no_evidence）。
      喂一个看似强烈的 external_stats={"black_pct": 0.95}（即使 n_estimate=1000）。
      断言：verdict 仍是 no_evidence。
   
   E. 边界用例：
      - recent_history = [] → verdict=no_evidence, weight=0
      - recent_history = [17] → 同上
      - recent_history 全是同一个数字 → 检测出极强偏差
      - external_stats = None → external_check 字段为 None
      - external_stats 给了但 external_n_estimate=None → auxiliary_p_values 字段为空

6. 写一个 examples/bias_demo.py 脚本，跑场景 A、B、C，把结果用 print 表格化输出。

完成后跑测试 + 跑 examples/bias_demo.py，把两个输出都贴出来。
```

**验收**：
- 测试全绿
- 公平轮盘假阳性率明确 < 5%
- bias_demo.py 能给出"N vs verdict"的表
- 外部统计降级和不升级两种行为都被测试覆盖

---

## Round 4：M4 信念模型 + 优化器（决策核心）

**Prompt**:

```
依赖前三轮，实现 belief.py 和 optimizer.py。

A. src/roulette_agent/belief.py

   def compute_belief(recent_history: list[int],
                      bias_report: dict,
                      wheel_type: str = "american",
                      k_prior: float = 10.0) -> dict[int, float]:
       """
       bias-aware mixture（README.md §3.1）：
       p(n) = w * p_bias(n) + (1 - w) * p_uniform(n)
       其中 p_bias(n) = (count(n) + k_prior) / (N + 38 * k_prior)
       N = len(recent_history)。只用 recent_history，不使用 external_stats。
       w 来自 bias_report["weight"]
       """
   
   写测试：
   - bias_report.weight=0 时，p(n) 全为 1/38（误差 < 1e-9）
   - bias_report.weight=1 时，p(n) 等于 Dirichlet 后验
   - sum(p.values()) ≈ 1.0
   - 较大 k_prior 时分布趋向均匀

B. src/roulette_agent/optimizer.py

   def excluded_numbers(excluded_dozens: list[int]) -> set[int]:
       """[1] -> {1..12}, [3] -> {25..36}, [] -> {}, [1,3] -> 报错"""
   
   def enumerate_legal_bets(excluded_dozens: list[int]) -> list[dict]:
       """
       枚举所有合法 bet "模板"（type + numbers，未含 amount）。
       严格模式：任何 covered_numbers 与 excluded 有交集的 bet 都剔除。
       为了组合爆炸，建议先生成所有外围 + 所有 straight + 所有 corner + 
       所有 street + 所有 split + 所有 six_line + 所有 column + 所有 dozen。
       注意 corner/split 的几何合法性（layout.is_valid_bet）。
       """
   
   def greedy_ev_allocation(p: dict[int, float],
                            bankroll: float,
                            bet_unit: float,
                            excluded_dozens: list[int],
                            max_bet_fraction: float = 0.1,
                            top_k: int = 5) -> list[dict]:
       """
       1. 枚举所有合法 bets
       2. 算每个 bet 的主观 EV
       3. 取 EV > 0 的，按 EV 大小排序取前 top_k
       4. 按 EV 加权，总分配额 = bankroll * max_bet_fraction
       5. 每个 amount 向下取整到 bet_unit 的倍数
       6. 全 EV <= 0 时返回空 list（不下注）
       """
   
   def kelly_allocation(p: dict[int, float],
                        bankroll: float,
                        bet_unit: float,
                        excluded_dozens: list[int],
                        fraction: float = 0.25) -> list[dict]:
       """
       对每个合法 bet 单独算 Kelly f* = (b*p - q) / b
       其中 b = 净赔率, p = sum(p[n] for n in covered)
       f* > 0 才下注；amount = fraction * f* * bankroll，向下取整到 bet_unit
       注意：多个 bet 同时下时，Kelly 公式严格说是组合优化问题。本项目用简化版（每个独立算），
       再把总下注额钳制在 bankroll * 0.5 以内，避免 overbet。
       """
   
   def fixed_baseline_allocation(p: dict[int, float],
                                 bankroll: float,
                                 bet_unit: float,
                                 excluded_dozens: list[int]) -> list[dict]:
       """1/64 bankroll 押在 p(n) 前 3 高的非剔除单数字"""

C. tests/test_optimizer.py
   - 公平 p(n) = 1/38 时，greedy_ev_allocation 返回空（所有 EV <= 0）
   - excluded_dozens=[3] 时，所有返回 bet 的 covered_numbers 与 {25..36} 不相交
   - excluded_dozens=[1,3] 应抛 ValueError
   - 极偏 p（如 p(31)=0.5, 其余分摊）时，kelly 应主要押 31
   - 所有 amount 都是 bet_unit 的倍数
   - 总下注额 <= bankroll

完成后跑测试并贴结果。
```

**验收**：测试全绿。手动测：均匀 p 下三个 allocator 全返回空（fixed_baseline 例外，那个无条件下注）。

---

## Round 5：M5 蒙特卡洛模拟器（直观验证负期望）

**Prompt**:

```
依赖前四轮，实现 src/roulette_agent/simulator.py。

   def simulate(strategy_name: str,
                n_spins: int,
                initial_bankroll: float,
                bet_unit: float,
                wheel_type: str = "american",
                excluded_dozens: list[int] = None,
                bias_inject: dict | None = None,
                seed: int | None = None) -> dict:
       """
       strategy_name: "greedy_ev" | "kelly" | "fixed_baseline" | "always_red" | "no_bet"
       bias_inject: 注入偏差，如 {35: 3/38} 表示让 35 出现概率为 3/38（其余按比例缩放）
       
       每轮：
       1. 用当前 history 算 bias_report、p(n)
       2. 用 strategy 拿到 bets
       3. 随机摇一个 result_number（受 bias_inject 影响）
       4. settle，更新 bankroll
       5. bankroll <= 0 时提前终止
       
       返回 {
           "bankroll_curve": list[float],   # 长度 n_spins+1
           "spin_history": list[int],
           "total_pnl": float,
           "final_bankroll": float,
           "ruined": bool,
           "bets_log": list[list[dict]]
       }
       """

   def compare_strategies(strategies: list[str],
                          n_spins: int,
                          n_runs: int,
                          initial_bankroll: float,
                          bet_unit: float,
                          wheel_type: str = "american",
                          seed: int | None = None) -> dict:
       """
       并行（用 multiprocessing 或单线程都行）跑 n_runs 次每个策略，
       返回每策略的：平均最终 bankroll、破产率、bankroll 中位数、最大/最小、std。
       """

写 examples/simulate.py：
- 跑 4 个策略各 1000 次模拟，每次 200 轮，初始 800
- 用 matplotlib 画 4 张 bankroll 曲线图（叠加 20 条样本路径 + 均值线）
- 在终端打印对比表
- 然后跑一次 bias_inject={35: 3/38}，看 kelly 策略在偏轮盘上的表现

不要依赖额外可视化库以外的东西，matplotlib 即可。把图保存为 PNG 而非 plt.show。

完成后跑 examples/simulate.py，贴出终端输出和图的文件路径。
```

**验收**：
- 公平轮盘下，所有策略平均最终 bankroll < 800（验证负期望）
- 偏轮盘下，Kelly 策略平均最终 bankroll 显著 > 公平场景

---

## Round 6：M6 Claude Agent 接入

**Prompt**:

```
依赖前五轮，实现 Claude tool-calling agent。

注意：用 anthropic Python SDK 的 Messages API + tool_use。如果项目里装的是新版 Claude Agent SDK，
就用它。先 import 看看哪个能用。

A. src/roulette_agent/tools.py
   定义 6 个 tool 的 JSON Schema（用 Anthropic 的 tool 格式）+ Python 处理函数。
   
   tools = [
       {
           "name": "detect_bias",
           "description": "...",
           "input_schema": {...},
       },
       # compute_belief, kelly_allocation, greedy_ev_allocation, 
       # fixed_baseline_allocation, settle
   ]
   
   def dispatch_tool(name: str, args: dict) -> dict:
       """Claude 调 tool 时路由到对应 Python 函数"""

B. src/roulette_agent/agent.py
   
   class RouletteAgent:
       def __init__(self, anthropic_client, model="claude-sonnet-4-5-20250929"):
           ...
       
       def decide(self, session_state: dict) -> dict:
           """
           session_state 字段：
             - bankroll: float
             - bet_unit: float
             - wheel_type: str
             - excluded_dozens: list[int]
             - recent_history: list[int]           # initial_history + 后续所有 result_number
             - external_stats: dict | None
             - external_stats_n_estimate: int | None
             - hyperparams: dict
             - notes: str
           
           跑 tool-use 循环：
           1. 构造 system prompt（含规则、bankroll、notes、当前 bias_report 摘要）
           2. user message: "根据当前状态给出下一轮押注策略"
           3. Claude 决定调哪些工具
           4. 循环直到 Claude 给出最终策略（stop_reason="end_turn"）
           5. 最终返回 {"bets": [...], "rationale": "..."}
           
           最多 10 个工具调用回合，超出报错。
           """

C. system prompt 模板（写进 agent.py 顶部）：
   - 介绍轮盘规则
   - 告知当前 bankroll、bet_unit、wheel_type、excluded_dozens
   - 列出可用工具
   - 重申"如果所有 EV 为负，建议小注或不下注"
   - **明确告诉 Claude：external_stats 仅供参考，不能据此升级押注信心。**
   - 注入 session.notes（agent 的累积经验笔记）

D. CLI 入口 src/roulette_agent/cli.py：
   - `python -m roulette_agent.cli init --bankroll 800 \
       --recent-history "35,17,35,29,22,31,35,11,29,4,13,26,8,31,20" \
       --external-stats '{"black_pct":0.62,"odd_pct":0.59}' \
       --excluded-dozens 3` 
     创建本地 session（存 JSON 文件 ~/.roulette_agent/sessions/<uuid>.json）
   - `python -m roulette_agent.cli spin <session_id> <result_number>`
     更新 session，跑 agent，输出策略

E. tests/test_agent.py：用 mock 的 anthropic client，验证 tool dispatch 正确、循环不死。
   不需要真的调 API（节省 token）。

完成后：
1. 跑 tests 贴结果
2. 设置 ANTHROPIC_API_KEY 环境变量
3. 用 CLI 真跑一次 init + 3 次 spin，把对话 + 返回贴出来
```

**验收**：
- 测试全绿
- CLI 能跑通完整循环
- agent 给出的策略合法（amount 是 bet_unit 倍数、不押被排除区域）

---

## Round 7：M7 FastAPI 服务

**Prompt**:

```
依赖 R6，实现 HTTP 服务层。

A. src/roulette_agent/models.py
   SQLAlchemy 2.x ORM 模型，对应 README.md §6 的三张表。
   - Session, Spin, Reflection
   - 用 UUID 主键，JSONB 用 SQLAlchemy 的 JSON 类型（兼容 SQLite 和 Postgres）

B. Alembic 初始化
   - alembic init alembic
   - 写一个 initial migration 创建三张表
   - alembic/env.py 配置从环境变量读 DATABASE_URL，默认 sqlite:///./dev.db

C. src/roulette_agent/app.py
   FastAPI 应用，三个端点（详见 README.md §7）：
   
   - POST /session/new
   - POST /session/{id}/spin
   - GET  /session/{id}/state
   
   每个端点都用 Pydantic 模型做请求/响应校验。
   
   一个简单的 API key 中间件：从 header X-API-Key 读，对比环境变量 API_KEY，不匹配 401。

D. spin 端点的核心逻辑：
   1. 加载 session（含 initial_history、external_stats、external_stats_n_estimate）
   2. 从 session 重建 recent_history = initial_history + [s.result_number for s in spins]
   3. 如果不是第一轮，先 settle 上一轮（用上一轮的 bets + 本次 result_number）
   4. 把 result_number 追加到 recent_history（仅在内存中，DB 里通过 spin 行重建）
   5. 调 RouletteAgent.decide(session_state) 拿到下一轮策略
   6. 写一行 spin 记录（含 bias_report）
   7. 返回响应

E. tests/test_app.py：用 FastAPI TestClient + httpx
   - 完整跑一遍 new session → 3 次 spin
   - 测 401（缺 API key）
   - 测无效 session_id 404
   - 测无效 result_number（39）400
   - 测 external_stats 字段为 None 时正常工作
   - 测 external_stats 给了但 external_stats_n_estimate=null 时正常工作

F. 写一个 scripts/run_dev.sh：
   #!/bin/bash
   export DATABASE_URL=sqlite:///./dev.db
   export API_KEY=dev-key-123
   export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
   alembic upgrade head
   uvicorn roulette_agent.app:app --reload --port 8000

完成后：
1. 跑 alembic upgrade head 建表
2. 跑 scripts/run_dev.sh
3. 用 curl 完整跑一遍 new + 3 次 spin，贴出请求和响应
```

**验收**：
- API 完整跑通
- 数据库里能看到 session 和 spin 记录
- 测试全绿

---

## Round 8：M8 云部署

**Prompt**:

```
依赖 R7，部署到 Railway。

A. 准备部署配置
   - 创建 railway.toml 或 nixpacks.toml
   - 创建 Procfile（如 Railway 需要）:
     web: alembic upgrade head && uvicorn roulette_agent.app:app --host 0.0.0.0 --port $PORT
   - 在 README.md 加一节"Deployment"，列出需要的环境变量：
     ANTHROPIC_API_KEY, DATABASE_URL (Railway 自动注入), API_KEY

B. 配置 Postgres
   - Railway 加一个 Postgres 服务
   - 验证 DATABASE_URL 注入

C. 验证部署
   - 部署成功后，用公网 URL + API key 跑一遍完整流程：
     curl -X POST https://<your-app>.up.railway.app/session/new \
       -H "X-API-Key: ..." \
       -H "Content-Type: application/json" \
       -d '{...}'
   - 把响应贴出来

D. 加一个 /health 端点（GET，无 auth），返回 {"status":"ok","db":"connected"}

E. 写一个 docs/DEPLOY.md 把整个部署流程记下来，含截图位置和命令。

注意：
- Railway 的免费层有限额，提醒我如果跑超限怎么办
- Anthropic API 是按 token 收费的，提醒我观察成本
- 别把 .env 提交到 git

完成后告诉我公网 URL（不含 API key）和一个 curl 测试命令模板。
```

**验收**：
- 公网 URL 可访问
- /health 返回 ok
- 能跑完整 session 流程

---

## Round 9：M9 手机前端 PWA

**Prompt**:

```
依赖 R8，做一个最小可用的 PWA 前端，挂在 FastAPI 同一个域名下。

A. src/roulette_agent/static/
   - index.html        # 主页面
   - manifest.json     # PWA manifest
   - service-worker.js # 最小化 SW（仅 offline 提示）
   - app.js
   - style.css

B. UI 设计（极简单页面）
   两个视图，由按钮切换：
   
   View 1: New Session
   - 输入：wheel_type（select: american/european）、bankroll、bet_unit
   - excluded_dozens：三个 radio（none / 1st 12 / 3rd 12）
   - **Section A: Recent draws**（必填）：
     - 一个 textarea，逗号分隔最近的开奖数字（如 "35,17,35,29,..."；数量自由，10-30 轮典型）
     - 帮助文字："输入你能确认的最近开奖数字，按从早到晚顺序。00 用 '00' 表示。"
   - **Section B: Table stats**（可选，可折叠展开）：
     - 帮助文字："桌面显示屏给出的统计，样本量通常未知，仅供 agent 参考。"
     - 4 个数字输入：black_pct、red_pct、odd_pct、even_pct（0-1 范围；任何字段空白都接受）
     - 一个数字输入 external_stats_n_estimate（用户对样本量的估计，可空）
     - 一行小字："如果不知道样本量就留空——agent 会更保守地处理这些数据。"
   - API key：一个 password input，存 localStorage
   - 提交 → POST /session/new → 跳到 View 2，session_id 存 localStorage
   
   View 2: Spin
   - 顶部卡片：bankroll、spin_index、上轮 pnl
   - 中部卡片：bias verdict（带颜色：灰=no_evidence，黄=weak，橙=moderate，红=strong）、weight、一句话摘要
   - **如果 bias_report.external_check 不为 null，加一行小字显示 external_check.status 和 note**（用淡色字体，强调"仅供参考"）
   - 主要卡片：本轮 next_strategy 一条条列出（很大的字号方便手机读）
   - 底部：一个大输入框 + 0/00 按钮 + 提交
   - 历史区域：折叠显示最近 10 轮的 (result, pnl)

C. app.js
   - 用 fetch，X-API-Key 从 localStorage
   - 错误处理：网络错、401、404、400
   - 没有任何前端框架，原生 JS + DOM

D. FastAPI 挂载 static
   from fastapi.staticfiles import StaticFiles
   app.mount("/", StaticFiles(directory="static", html=True), name="static")
   
   注意把这行放在 API 路由之后，不然会拦截。

E. 手机适配
   - viewport meta 标签
   - 大按钮、大字号（最小 16px）
   - touch-friendly 输入

F. PWA 基础
   - manifest.json: name, short_name, start_url, display: standalone, theme_color
   - 让用户能"添加到主屏"
   - 一个简单的 192x192 PNG icon（你可以用 SVG 转 PNG，或者直接用一个色块）

部署后用手机访问 https://<your-app>.up.railway.app/ 测试。

完成后贴出主要文件内容。
```

**验收**：
- 手机能打开网页并完成完整流程
- 能添加到主屏

---

## Round 10：M10 反思机制（软优化）

**Prompt**:

```
依赖 R6 和 R7，加上反思机制。

A. 反思触发条件（在 spin 端点里检查）：
   - 每 N 轮触发一次（默认 N=10，写在 session.hyperparams）
   - 或者 bankroll 相对上次反思下降超过 15%

B. src/roulette_agent/agent.py 增加方法
   
   def reflect(self, session_state: dict, recent_spins: list[dict]) -> dict:
       """
       让 Claude 看过去 N 轮的 bets/results/pnl/bias_report，
       输出：
       {
           "observation": str,           # 自然语言总结
           "notes_addition": str,        # 要加进 session.notes 的内容
           "hyperparam_diff": dict       # 建议改的超参（可空）
       }
       
       这是一次独立的 Claude 调用，不用 tools，纯 prompt + 结构化输出。
       让 Claude 返回 JSON，用 json mode。
       """

C. 在 POST /session/{id}/spin 里：
   - decide 之后检查是否触发 reflect
   - 触发则 reflect、写 reflection 表、更新 session.notes 和 hyperparams
   - 响应里加一个字段 "reflection": null | {observation, notes_addition_excerpt}

D. PWA 加一个折叠面板"Agent notes"，能看完整 session.notes，从 GET /session/{id}/state 拉。

E. tests/test_reflection.py：mock client，验证触发条件和写库逻辑。

注意：
- reflect 的 prompt 里明确告诉 Claude：负期望游戏里，"经验"主要是关于 bias detection 的灵敏度、
  Kelly 仓位大小、何时停手——别让它学出"我感觉 35 会出"这种迷信
- notes 不要无限增长，超过 4000 字符时让 Claude 自己浓缩一次

完成后跑 20 轮模拟（用 simulator 喂数据，跳过手动），看 notes 长什么样。贴出来。
```

**验收**：reflection 表里有记录、notes 在合理增长、PWA 能看到。

---

## Round 11（可选）：M11 Telegram bot

**Prompt**:

```
增加一个 Telegram bot 作为替代前端。依赖 R7 / R8。

A. 用 python-telegram-bot 库
B. 命令：
   /start             - 欢迎 + 说明
   /new               - 进入新 session 对话流（一步步问 bankroll、history 等）
   /spin <number>     - 上轮结果
   /state             - 当前 bankroll + bias verdict
   /notes             - 显示 agent notes
   /end               - 结束 session

C. bot 调用同一套 FastAPI 服务（用 BOT_API_KEY 这个内部 key 调）
D. 部署：Railway 加一个 worker process 跑 bot，跟 web service 共用 DB

完成后给我 bot 的 @username 和测试步骤。
```

---

## Round 12（可选）：M12 硬优化（数值超参自动调整）

**Prompt**:

```
依赖 R5、R6、R10。

A. src/roulette_agent/hard_optim.py
   - 三个策略 (greedy_ev / kelly / fixed_baseline) 各维护一个 reward 历史
   - ε-greedy 选策略（ε=0.1）
   - bias verdict→weight 映射阈值：滚动评估"verdict=weak 之后那一轮的 pnl"，
     如果连续 K 次为负 → 收紧 weight (e.g., weak 的 weight 从 0.15 → 0.05)
   
B. 把这些超参写进 session.hyperparams，每次 spin 后更新

C. tests + 在 simulator 里跑长程对比：有 hard optim vs 没有，看 ruined rate 和 bankroll 中位数。

D. 告诉我："硬优化能不能让 ruined rate 下降？如果不能，为什么？"
   （提示：负期望游戏里，硬优化只能降方差，不能改变期望符号。）
```

---

## 反向检查（每完成一个里程碑都跑一次）

每轮完成后，让 Claude Code 跑：

```
请：
1. 跑全部 pytest，确保前面里程碑的测试没被破坏
2. 跑 examples/ 下所有脚本，确保没崩
3. 列出本轮新增/修改的文件
4. 列出已知的 TODO 和潜在问题
5. 不要进入下一轮，等我指示
```

这是回归测试的最低门槛。

---

## 调试时的提示模板

如果某轮卡住，用这个模板：

```
我在 Round X 的 [具体步骤] 卡住了。

错误信息：
<贴 traceback>

我已经检查的：
- <列出排查过的方向>

请：
1. 解释这个错误的根本原因（不是表面现象）
2. 给出最小修复 patch
3. 加一个测试用例防止这个错误回归
```

---

## 成本控制

- Claude API 调用每次 spin 大约消耗 2k-5k tokens（含 system prompt、tool definitions、tool 调用回合）
- 假设 Sonnet 4.5 定价水平，每次 spin 几分到一毛钱人民币
- Railway 免费层 $5/月额度，MVP 阶段够用
- 监控建议：在 agent.py 里记录每次调用的 input/output tokens，写进 spin 表（加一列 token_usage JSONB）。M7 完成后加上这一列。

---

## 完成定义（MVP）

到 R9 完成时，你应该能做到：
1. 在手机上打开一个网址，添加到主屏
2. 输入桌面初始信息，点提交
3. 看到第一轮押注策略 + bias verdict
4. 每轮只输入一个数字，看到更新的策略、bankroll、verdict
5. 关掉手机过一天再打开，session 还在
6. 整个服务跑在云上，不依赖你的电脑

到 R10 完成时，agent 会"记笔记"，下次响应会引用以前学到的东西。

到 R12 完成时，你能拿出数据回答："hard optim 有用吗？为什么轮盘场景里有上限？"——这个问题的答案就是这个项目最大的学习价值。
