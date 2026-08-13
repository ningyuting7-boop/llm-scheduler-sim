# Week 2–3 详细计划：ARRS Scheduler + LMSYS 数据接入 + 实验框架

> **[更新] ARRS 已从代码库中移除，不再使用。** 下文 Phase 2/3.3、Phase 4 的 exp2/exp4、第九节整段记录的是 ARRS 从设计到被放弃的完整过程，作为决策历史保留，但对应的 `src/schedulers/arrs.py`、`exp2_prediction_robustness.py`、`exp4_starvation.py`、`plot_skew_ablation.py`、`plot_skew_at_k7.py` 等文件已删除。ARRS 想解决的问题（tail-risk-aware 调度）由第十一节的 **TIEScheduler** 取代，见该节。

本文档只做设计，不含最终实现代码。所有函数签名/类结构都是"契约"（接口 + 关键行为约定），具体实现留给写代码的时候。目标是让两个人可以照着这份文档并行开工，不用互相对齐接口。

沿用 Week 1 的原则：**先改动最小的公共基础设施，再加算法，再接数据，最后才是实验脚本和画图**——顺序反了会导致后面的人一边写实验一边发现底层框架要改。

---

## 〇、总体目标回顾

核心问题：output length 预测存在误差和不确定性时，ARRS（预测 + uncertainty + aging）是否比 SJF / Predicted-SJF 更能降低平均 latency，同时避免长请求被饿死。

四个 scheduler 全部继承已有的 `Scheduler` 基类（[src/scheduler.py](../src/scheduler.py)），只重写 `_priority_key`：

| Scheduler | 用什么信息排序 |
|---|---|
| FCFS（已完成） | `arrival_time` |
| Oracle SJF | 真实 `output_len`（upper bound，不该被实际系统使用，只是理论参照） |
| Predicted SJF | 预测的 `predicted_output_len`（不知道真实值） |
| ARRS | `predicted_output_len + β·uncertainty − α·waiting_time` |

---

## 一、项目结构变化总览

```
llm-scheduler-sim/
├── data/
│   └── lmsys_output_lengths.csv        [新增] LMSYS 预处理后的 output length 数据（只有数字，不含原始对话文本）
├── scripts/
│   └── extract_lmsys_lengths.py        [新增] 一次性预处理脚本：HF 数据集 -> CSV
├── src/
│   ├── request.py                      [改动] 加 prediction_uncertainty 字段
│   ├── scheduler.py                     [改动] _priority_key 签名加 current_time
│   ├── predictor.py                     [新增] 预测误差注入模块
│   ├── metrics.py                       [改动] percentile 通用化，加 p50/p95/max_waiting_time
│   ├── workload.py                      [改动] 加 from_lengths / bimodal 生成器
│   └── schedulers/
│       ├── fcfs.py                      （已完成，signature 需跟着基类改，逻辑不变）
│       ├── oracle_sjf.py                [新增]
│       ├── predicted_sjf.py             [新增]
│       └── arrs.py                      [新增]
├── experiments/
│   ├── run_experiment.py                [新增] 通用 runner，替代 run_fcfs_baseline.py 的角色
│   ├── exp1_overall_performance.py      [新增]
│   ├── exp2_prediction_robustness.py    [新增]
│   ├── exp3_congestion.py               [新增]
│   ├── exp4_starvation.py               [新增]
│   └── plotting.py                      [新增] 读取上面 4 个实验的 CSV，出图
└── tests/
    ├── test_simulator.py                （已完成，不动）
    └── test_schedulers.py               [新增] Oracle/Predicted SJF、ARRS 的手算回归测试 + starvation 测试
```

---

## 二、Phase 1：基础设施改动（先做，其他所有东西都依赖它）

### 2.1 `src/scheduler.py`：`_priority_key` 加 `current_time`

现状（[scheduler.py:28-44](../src/scheduler.py#L28-L44)）：`schedule()` 内部用 `min(self.waiting, key=self._priority_key)`，而 `_priority_key(self, request)` 拿不到当前时间。ARRS 的 aging 项 `-α·waiting_time` 必须知道 `current_time`，这是本次改动的唯一原因。

```python
class Scheduler(ABC):
    def schedule(self, current_time: float) -> List[Request]:
        admitted: List[Request] = []
        while self.waiting and len(self.running) < self.max_batch_size:
            request = min(self.waiting, key=lambda r: self._priority_key(r, current_time))
            self.waiting.remove(request)
            self.running[request.request_id] = request
            admitted.append(request)
        return admitted

    @abstractmethod
    def _priority_key(self, request: Request, current_time: float) -> Any:
        raise NotImplementedError
```

`FCFSScheduler._priority_key` 只需要加一个未使用的 `current_time` 形参，逻辑不变。

**注意**：这个改动不影响算法复杂度和现有 FCFS 测试的结果（`test_simulator.py` 里的手算数字不会变），只是签名变化，纯重构。改完先跑一遍 `tests/test_simulator.py` 确认没破坏东西。

### 2.2 `src/request.py`：加 `prediction_uncertainty`

```python
predicted_output_len: Optional[float] = None       # 已有
prediction_uncertainty: Optional[float] = None      # 新增，ARRS 用；PredictedSJF/FCFS/OracleSJF 不用
```

不需要改 `__post_init__`，允许是 `None`（意味着"这个 request 还没被预测过"，ARRS 里 `None` 当 0 处理）。

### 2.3 `src/predictor.py`（新增模块）

这是 Experiment 2（prediction robustness）和 ARRS 的核心依赖，必须单独抽出来，不要写在 `workload.py` 或某个 scheduler 里——否则 4 个实验的噪声模型会不一致。

**设计决策（需要写进注释里，因为不是显然的）**：`uncertainty` 不能等于"这次预测实际的误差绝对值 `|ε|`"，因为调度器在真实系统里不可能知道真实长度、也就不可能知道这次预测到底错了多少。`uncertainty` 应该是**预测器对自己这次输出的置信区间宽度估计**，和"预测器整体质量档位"绑定，而不是和某一次采样到的噪声绑定。所以用配置的 `error_level` 本身作为 uncertainty 的比例系数，不用实际抽到的 `ε`。

```python
def predict_length(
    true_length: int,
    error_level: float,      # 0.0=完美预测, 0.25=~25%相对误差, 1.0=误差与真实值同量级
    rng: random.Random,
) -> Tuple[float, float]:
    """
    返回 (predicted_length, uncertainty)。

    predicted_length = true_length * (1 + eps), eps ~ N(0, error_level^2)
    下限 clip 到 1.0，避免 error_level 很大时抽出负数/零。

    uncertainty = error_level * predicted_length
    —— 用配置的误差档位而不是这次实际抽到的 eps，理由见模块 docstring。
    """
```

`error_level` 取值就是 Experiment 2 里扫的那几档：`0.0 / 0.10 / 0.25 / 0.50 / 1.00`。

FCFS、Oracle SJF 不调用这个函数（它们不需要预测）。Predicted SJF 和 ARRS 在 workload 生成后、request 入队前统一跑一遍 `predict_length`，把结果写回 `request.predicted_output_len` / `request.prediction_uncertainty`。**这一步应该放在 `run_experiment.py` 里（构造完 workload 之后，创建 scheduler 之前），不要放进 `workload.py`**——因为同一批 request 需要能在"不同 error_level"下重跑（Experiment 2 就是同一个 workload、扫不同 error_level），如果预测误差在 `workload.py` 生成时就写死，就没法复用同一批 request 做扫描实验了。

### 2.4 `src/metrics.py`：percentile 通用化

现状只硬编码了 p99（[metrics.py:54](../src/metrics.py#L54)）。改成：

```python
def percentile(values: List[float], q: float) -> float:
    """q in [0, 100]. values 会被排序，不要求传入前先排序。"""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * q / 100))
    return ordered[idx]
```

`MetricsSummary` 加字段：

```python
p50_response_time: float
p95_response_time: float
p99_response_time: float          # 保留，别删，Week1 测试可能间接依赖
max_waiting_time: float           # Experiment 4 (starvation) 要用
```

`compute_summary` 里对应用 `percentile(response_times, 50/95/99)` 和 `max(waiting_times)`。CSV 导出（`write_csv`）不用改，那个是逐 request 明细，百分位是汇总层面的东西，走 `print_summary`/单独的汇总 CSV（见 Phase 4）。

---

## 三、Phase 2：三个新 Scheduler

### 3.1 `src/schedulers/oracle_sjf.py`

```python
class OracleSJFScheduler(Scheduler):
    def _priority_key(self, request: Request, current_time: float) -> Tuple[int, float, int]:
        return (request.output_len, request.arrival_time, request.request_id)
```

用真实 `output_len`（这是"upper bound"参照组，实际系统不可能有这个信息）。`arrival_time`/`request_id` 作为 tie-break，避免长度相同时排序不稳定。

### 3.2 `src/schedulers/predicted_sjf.py`

```python
class PredictedSJFScheduler(Scheduler):
    def _priority_key(self, request: Request, current_time: float) -> Tuple[float, float, int]:
        predicted = request.predicted_output_len
        if predicted is None:
            raise ValueError(f"Request {request.request_id} has no predicted_output_len; "
                              f"did you forget to run predictor.predict_length on the workload?")
        return (predicted, request.arrival_time, request.request_id)
```

显式抛异常而不是静默 fallback 到 0——这样如果实验脚本忘了跑 predictor，会立刻报错，不会悄悄跑出一组无意义的数据。

### 3.3 `src/schedulers/arrs.py`

```python
class ARRSScheduler(Scheduler):
    def __init__(self, max_batch_size: int, alpha: float, beta: float) -> None:
        super().__init__(max_batch_size)
        self.alpha = alpha   # aging 权重：waiting_time 每增加1单位，score 降多少
        self.beta = beta     # uncertainty 权重：uncertainty 每增加1单位，score 升多少

    def _priority_key(self, request: Request, current_time: float) -> float:
        predicted = request.predicted_output_len
        uncertainty = request.prediction_uncertainty or 0.0
        if predicted is None:
            raise ValueError(...)  # 同上
        waiting_time = current_time - request.arrival_time
        return predicted + self.beta * uncertainty - self.alpha * waiting_time
```

**单位一致性问题（容易被忽略，实现时要注意）**：`predicted`/`uncertainty` 的单位是 "token 数"（decode steps），`waiting_time` 的单位是仿真时间（如果 `decode_time_per_step != 1.0`，两者不是同一个量纲）。建议在构造 `alpha`/`beta` 时就把 `waiting_time` 先除以 `decode_time_per_step` 转成"等价 step 数"再乘 `alpha`，这样 `alpha` 的取值含义在不同 `decode_time_per_step` 下才是可比的：

```python
        waiting_steps = (current_time - request.arrival_time) / self.decode_time_per_step
        return predicted + self.beta * uncertainty - self.alpha * waiting_steps
```

这意味着 `ARRSScheduler.__init__` 还需要接收 `decode_time_per_step`（或者干脆让 `run_experiment.py` 在构造时把它传进去）。写代码时定下来，别让 `alpha` 的实际含义随 `--decode-time-per-step` 参数变化而漂移。

**`alpha`/`beta` 怎么定**（Phase 4 会用到）：先手动试几组量级（比如 `alpha ∈ {0.1, 0.5, 1, 2}`，`beta ∈ {0, 0.2, 0.5}`），在 Experiment 4 的 starvation workload 上看长请求的 `max_waiting_time` 是否明显低于 Predicted SJF、同时 Experiment 1 的 `avg_response_time` 别比 Predicted SJF 差太多——这是个 tradeoff，没有"正确答案"，能讲清楚 tradeoff 就是加分项。不建议在 3 周项目里做完整 grid search + 交叉验证，写一个小脚本手动扫 3×3 组合、挑一组代表性的就够（可选：`experiments/tune_arrs_hyperparams.py`，非核心，时间够才做）。

---

## 四、Phase 3：LMSYS-Chat-1M 数据预处理管线（重点，详细写）

### 4.1 前置条件（你要做的，我做不了）

1. 注册免费 HuggingFace 账号
2. 打开 https://huggingface.co/datasets/lmsys/lmsys-chat-1m ，点击接受使用协议（gated dataset，条款大意是仅用于研究、不能重新分发原始对话内容）
3. 生成一个 Access Token（Settings → Access Tokens），本地跑 `huggingface-cli login` 粘贴 token，或者设置环境变量 `HF_TOKEN=xxx`
4. `pip install datasets huggingface_hub`

### 4.2 数据集里到底是什么

`lmsys-chat-1m` 每一行是一条完整对话：

```json
{
  "conversation_id": "...",
  "model": "vicuna-13b",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "turn": 2,
  "language": "English",
  ...
}
```

**关键设计决策：什么算一个 "request"？** 我们要模拟的是"一次 LLM 推理请求"，粒度应该是**单条 assistant 回复**，不是整个多轮对话。所以预处理时要把每条对话拆开，每个 `role == "assistant"` 的 turn 单独算一条样本、贡献一个 output length 数字。一条 1000 轮的长对话不应该被当成"一个超长 request"，那是完全不同的东西（多轮对话 vs 单次生成长度），会污染分布。

### 4.3 长度口径

已确认：用**单词数**近似 token 数（`len(content.split())`），不引入 tokenizer 依赖。预处理脚本里顺带把 `char_count` 和 `char_count/4` 也存一列，方便后面万一要交叉检查/换口径，不用重新跑一遍下载。

过滤规则：
- `role != "assistant"` 的 turn 跳过
- `word_count == 0`（空回复、纯符号回复分词后为空）的样本丢弃——这是异常值不是长尾，不该保留
- **不要**手动裁剪长尾的上限——长尾正是这个数据集对项目有价值的地方（Figure 1 就是要展示这个），裁剪了就白拿真实数据了

### 4.4 子采样规模

全量 100 万条对话、每条多轮，全部处理会比较慢且没必要——分布的形状用几万个样本点就能估得很稳。用 **streaming 模式**（`load_dataset(..., streaming=True)`），只取前 `N=20000` 条对话（约几万条 assistant 回复），不用把整个数据集（几 GB）下到本地：

```python
from datasets import load_dataset
import itertools

ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
subset = itertools.islice(ds, 20000)
```

### 4.5 输出格式

`data/lmsys_output_lengths.csv`，列：`word_count, char_count, char_div4`。**这个文件本身可以提交进 git**——它只包含统计出来的数字，不含任何原始对话文本，不算"重新分发数据集"，团队其他人不用自己登录 HF 就能直接用你产出的 CSV 跑后续所有实验。（这是我的判断，不是数据集条款的法律意见——如果不放心，可以在 README 里注明"派生统计量，不含原文"，或者干脆不 commit、每人各自跑一次预处理脚本，两种都行，看你们纠结程度。）

### 4.6 预处理脚本骨架 `scripts/extract_lmsys_lengths.py`

```python
def extract_lengths(num_conversations: int, hf_token: Optional[str]) -> List[Dict[str, float]]:
    """流式读取 num_conversations 条对话，拆出所有 assistant turn，
    返回 [{"word_count":.., "char_count":.., "char_div4":..}, ...]"""

def write_lengths_csv(rows: List[Dict[str, float]], path: str) -> None: ...

def print_distribution_summary(rows: List[Dict[str, float]]) -> None:
    """打印 mean/median/p50/p95/p99/max，人工核对这个分布看起来是不是合理的长尾
    （数量级上应该是几十到几百词为主，少量千词以上的长回复）。"""

def main() -> None:
    # argparse: --num-conversations, --out
    ...
```

跑完之后**必须**人工看一眼 `print_distribution_summary` 的输出，确认数字量级合理（比如 median 几十词、p99 几百到上千词），再继续往下走——这是防止"数据管道有 bug 但没人发现，最后 Figure 1 画出一个奇怪分布"的最后一道检查。

### 4.7 `workload.py` 怎么消费这份数据

#### 4.7.0 workload 生成的整体逻辑（和现有 `generate_workload` 完全一致的那部分）

不管数据来源是 lognormal / 真实 LMSYS / bimodal，**到达过程（谁、什么时候来）和长度过程（这个请求要生成多少 token）是两件独立的事**，只在"给每个 request 组装 `Request` 对象"这一步合并。现有的 [`generate_workload`](../src/workload.py#L24-L55)（lognormal 版本）已经是这个结构了：

```
arrival_time = 0.0
for request_id in range(num_requests):
    arrival_time += rng.expovariate(arrival_rate)   # 到达过程：下一个请求的到达间隔 ~ Exponential(λ)，
                                                       # 累加起来就是 Poisson process 的到达时刻序列
    output_len = <某种方式采样一个长度>                 # 长度过程：三种数据源在这里分叉
    requests.append(Request(request_id, arrival_time, output_len, ...))
```

新加的两个生成器**只改长度过程那一行**，到达过程原样复用。为了不把同一段 Poisson 到达时间生成代码抄三遍，建议先抽一个私有 helper：

```python
def _poisson_arrival_times(num_requests: int, arrival_rate: float, rng: random.Random) -> List[float]:
    """返回严格递增的到达时刻列表，长度为 num_requests。"""
    times = []
    t = 0.0
    for _ in range(num_requests):
        t += rng.expovariate(arrival_rate)
        times.append(t)
    return times
```

`generate_workload`（现有的 lognormal 版本，趁这次改动顺便重构一下）、`generate_workload_from_lengths`、`generate_bimodal_workload` 三个函数都调这一个 helper 拿到达时刻，只是各自决定"这个 request_id 对应的 output_len 怎么来"。这样三份 workload 的到达统计特性保证完全一致，差异只在长度分布上——这正是实验设计要控制的变量（Exp1/2/3 比较不同 scheduler 时，你不想到达过程本身也变成一个未受控的差异来源）。

#### 4.7.1 `load_length_pool`：把预处理好的 CSV 读成一个数字列表

```python
def load_length_pool(csv_path: str, column: str = "word_count") -> List[int]:
    """读 4.6 产出的 CSV，取某一列，返回长度列表（比如几万个 int）。
    这个函数只在实验脚本启动时调一次，把结果传给下面两个生成器复用，
    不要每次生成一个 request 都重新读一次文件。"""
```

#### 4.7.2 `generate_workload_from_lengths`：真实分布版本

```python
def generate_workload_from_lengths(
    num_requests: int,
    arrival_rate: float,
    length_pool: Sequence[int],
    seed: Optional[int] = None,
) -> List[Request]:
    rng = random.Random(seed)
    arrival_times = _poisson_arrival_times(num_requests, arrival_rate, rng)
    requests = []
    for request_id, arrival_time in enumerate(arrival_times):
        output_len = rng.choice(length_pool)   # 有放回随机抽样（bootstrap）：
                                                 # 把 length_pool 当成"真实 output length 的经验分布"，
                                                 # 每次独立抽一个值当这个 request 的真实长度
        requests.append(Request(request_id=request_id, arrival_time=arrival_time, output_len=output_len))
    return requests
```

`rng.choice` 而不是按顺序遍历 `length_pool`——顺序遍历会让 request 的长度序列和 `length_pool` 里对话原本的顺序绑定，等于泄露了"数据集里长回复和短回复的排列模式"，这是我们不想要的（到达顺序应该独立于长度，长度只是从分布里抽，不是从"某个固定序列"里取）。

生成完之后，`predicted_output_len`/`prediction_uncertainty` 都还是 `None`——这两个字段留给 `run_experiment.py` 在拿到 workload 之后、跑之前，统一调 `predictor.predict_length` 填上（原因见 2.3：同一份 workload 要支持在不同 `error_level` 下重跑，长度不能和预测误差在生成阶段就绑死）。

#### 4.7.3 `generate_bimodal_workload`：故意拉开长短比例的版本（给 Exp4 用）

```python
def generate_bimodal_workload(
    num_requests: int,
    arrival_rate: float,
    length_pool: Sequence[int],
    long_request_ratio: float,        # 例如 0.05 = 5% 的请求是"长请求"
    short_percentile: float = 50.0,   # 短请求池：<= 这个百分位的真实样本
    long_percentile: float = 95.0,    # 长请求池：>= 这个百分位的真实样本
    seed: Optional[int] = None,
) -> List[Request]:
    rng = random.Random(seed)
    short_cutoff = percentile(length_pool, short_percentile)   # 复用 metrics.percentile
    long_cutoff = percentile(length_pool, long_percentile)
    short_pool = [l for l in length_pool if l <= short_cutoff]
    long_pool = [l for l in length_pool if l >= long_cutoff]

    arrival_times = _poisson_arrival_times(num_requests, arrival_rate, rng)
    requests = []
    for request_id, arrival_time in enumerate(arrival_times):
        is_long = rng.random() < long_request_ratio   # 每个请求独立地、按比例决定长/短，
                                                         # 不是"先来一串短的再来一串长的"
        output_len = rng.choice(long_pool if is_long else short_pool)
        requests.append(Request(request_id=request_id, arrival_time=arrival_time, output_len=output_len))
    return requests
```

**为什么每个请求独立掷骰子决定长短，而不是先排好比例再打乱**：真实系统里长请求什么时候混进来是不可预知的，调度器要应对的正是"不知道下一个到达的是长是短"这件事——如果人为把顺序排好（比如固定每 20 个请求出现 1 个长的），等于给了调度器额外没有的先验信息，会让实验结果失真。用独立采样，长请求出现的位置本身也是随机的，更接近真实场景。

长/短的判定阈值（`short_percentile`/`long_percentile`）只用来决定"抽样池子"，不需要额外存进 `Request`——画 Figure 5 的时候，直接用每个 request 自己的 `output_len` 和这两个 cutoff 比较，就能事后分类出哪些点是"长请求"，不用改 `Request` 的字段。

原有的 `generate_workload`（lognormal）**不要删**——Sanity check（Exp0）和"数据没到位前先跑通框架"都还需要它，也留作 fallback：如果 HF 登录卡住，`run_experiment.py` 的 `--workload-source lognormal` 分支不受影响，三条路径接口一致（都是 `(num_requests, arrival_rate, ..., seed) -> List[Request]`），互不阻塞。

---

## 五、Phase 4：实验脚本

### 5.1 通用 runner `experiments/run_experiment.py`

替代 `run_fcfs_baseline.py` 的角色（那个脚本可以删，或保留作为最简单的 smoke test，不强求）。

```
--scheduler {fcfs, oracle_sjf, predicted_sjf, arrs}
--workload-source {lognormal, lmsys, bimodal}
--num-requests, --arrival-rate, --seed         # 通用
--mean-output-len                               # 仅 lognormal
--lmsys-csv                                      # 仅 lmsys/bimodal
--long-request-ratio                             # 仅 bimodal
--prediction-error   (float, 0.0~1.0)            # 仅 predicted_sjf / arrs；fcfs/oracle_sjf 忽略
--alpha --beta                                   # 仅 arrs
--max-batch-size --decode-time-per-step
--csv-out   (逐 request 明细，复用 metrics.write_csv)
```

内部流程：生成 workload → 如果 scheduler 需要预测，跑一遍 `predictor.predict_length` 写回每个 request → 构造对应 Scheduler → `Simulator(...).run()` → `compute_summary` → 打印 + 写 CSV。返回 `MetricsSummary`，这样其他脚本可以直接 `import` 这个 runner 的核心函数（比如 `run_once(config) -> MetricsSummary`）而不是 fork 子进程，扫参数的时候快很多。

**建议把"跑一次完整实验"包装成一个函数 `run_once(**kwargs) -> MetricsSummary`，CLI 的 `main()` 只是这个函数的一层壳**——因为 Exp1-4 全部是"同一件事跑很多次、换参数"，如果核心逻辑锁在 `if __name__ == "__main__"` 里，扫描脚本就只能 subprocess 调用，又慢又难传回结构化结果。

### 5.2 `exp1_overall_performance.py`

- workload: `lmsys`，3 档 arrival rate（2 / 5 / 10 RPS）
- 4 个 scheduler 全部跑（`predicted_sjf`/`arrs` 用固定一档 `prediction_error`，比如 0.25，代表"现实中等质量的预测器"）
- 输出汇总 CSV：`[scheduler, rps, avg_wait, avg_response, p50, p95, p99, throughput, fairness_jain]`（3 档 × 4 个 = 12 行）
- 对应 Figure 2（avg response vs RPS）、Figure 3（p95 vs RPS）、Figure 6（throughput vs RPS）

### 5.3 `exp2_prediction_robustness.py`（核心实验）

- workload 固定一份（同一个 seed 生成一次，所有 scheduler/error_level 复用，保证公平对比）
- `prediction_error` 扫 `[0.0, 0.10, 0.25, 0.50, 1.00]`
- scheduler：`predicted_sjf` 和 `arrs` 各扫一遍；`fcfs`/`oracle_sjf` 因为不受 `prediction_error` 影响，各跑一次当水平参照线就行，不用重复 5 次
- 输出：`[scheduler, error_level, avg_response, p95_response]`
- 对应 **Figure 4**：x=error_level，y=avg_response_time，四条线（oracle/fcfs 是横线，predicted_sjf 和 arrs 是随 error 上升的曲线）——这是整个项目最重要的一张图，核心结论是 ARRS 曲线应该更平

### 5.4 `exp3_congestion.py`

- 固定 `prediction_error=0.25`
- `arrival_rate` 扫 `[2, 5, 10, 15, 20]`
- 4 个 scheduler 全跑
- 输出：`[scheduler, rps, avg_response, p95_response, avg_wait, throughput]`
- 对应 Figure（可以并进 Figure 2/3，同一份数据不同 x 轴切法，看你们想不想单独列一张）

### 5.5 `exp4_starvation.py`

- workload: `generate_bimodal_workload`（大量短 + 少量长），**故意调小 `max_batch_size`**（比如 1~2）制造拥堵——这是本文档第二部分提到的关键点，不这么做 aging 项测不出效果
- scheduler：`predicted_sjf` vs `arrs`（`prediction_error` 固定一档，比如 0.25，代表两者站在同样的预测质量下比较 aging 的作用）
- 输出两份东西：
  1. 汇总：`[scheduler, max_waiting_time, p95_waiting_time]`
  2. 逐 request 明细（直接复用 `metrics.write_csv` 的 CSV，里面本来就有 `output_len` 和 `waiting_time`）——用来画 Figure 5：x=request 在长度分布里的排名或 arrival 顺序，y=waiting_time，把"长请求"的点标出来，直观看它是不是被晾在原地

---

## 六、Phase 5：画图 `experiments/plotting.py`

读取 Phase 4 产出的几份 CSV，用 matplotlib 出图：

1. **Figure 1** — LMSYS output length 分布直方图（直接读 `data/lmsys_output_lengths.csv`，不需要跑 simulator，可以最先画，用来验证数据管道）
2. **Figure 2** — avg response time vs RPS（4 条线）
3. **Figure 3** — p95 response time vs RPS（4 条线）
4. **Figure 4** — avg response time vs prediction error（⭐ 核心图，2 条曲线 + 2 条参照横线）
5. **Figure 5** — starvation 散点：waiting time vs request index，predicted_sjf vs arrs 对比（长请求点高亮）
6. **Figure 6** — throughput vs RPS

写图的时候会用 `dataviz` skill 定配色和样式，保证 6 张图风格统一，不用每张图重新决定配色方案。

---

## 七、Phase 6：测试

### 7.1 `tests/test_schedulers.py`

**Exp0 sanity check 直接写成这里的第一个测试**，不单独开实验脚本（这本质是回归测试，不是"实验"）：

```
R1: arrival=0, output_len=100
R2: arrival=0, output_len=10
R3: arrival=0, output_len=20
max_batch_size=1, decode_time_per_step=1

手算 FCFS（按 request_id tie-break）:
  R1: start=0,  finish=100 -> response=100
  R2: start=100,finish=110 -> response=110
  R3: start=110,finish=130 -> response=130
  avg_response = 113.33

手算 Oracle SJF（按 output_len 排序: R2(10) -> R3(20) -> R1(100)):
  R2: start=0, finish=10  -> response=10
  R3: start=10,finish=30  -> response=30
  R1: start=30,finish=130 -> response=130
  avg_response = 56.67

断言: oracle_avg < fcfs_avg  (56.67 < 113.33)
```

Predicted SJF 和 ARRS 的手算案例类似，给 `predicted_output_len` 直接手动赋值（不走 `predictor.py` 随机噪声，保证可复现），验证排序逻辑正确。

### 7.2 Starvation regression test

设计一个小规模、确定性（不用随机数，或者固定 seed）的场景，验证"ARRS 不会让长请求无限等待，Predicted SJF 会"：

```
max_batch_size = 1
1 个长请求: arrival=0, predicted_output_len=50 (真实也是50，error=0，排除预测噪声干扰)
200 个短请求: arrival=0.1, 0.2, 0.3, ... , predicted_output_len=1，持续到达

Predicted SJF: 只要短请求还在源源不断地到达，长请求的 score(=50) 永远输给短请求的 score(=1)，
  断言: 长请求的 waiting_time 在整个 200 个短请求跑完之前都 > 某个较大阈值（比如 > 150）

ARRS (alpha 取一个能在合理时间内翻盘的值，比如 alpha=1.0):
  长请求的 score 随等待时间线性下降: 50 - 1.0*waiting_time，
  等 waiting_time > ~50 左右就会反超大多数短请求的 score(1 - 1.0*它们各自很短的等待时间)，
  断言: 长请求的 waiting_time 有一个明显更低的上界（比如 < 60），且不随短请求数量趋势性增长
```

具体阈值等实现完再跑出来调，这里给的是设计意图和断言方向，不是最终数字。

---

## 八、风险与开放问题（写代码前建议再确认一遍）

1. **HF 登录卡住怎么办**：`workload.py` 的两条路径（lognormal / from_lengths）接口一致，`run_experiment.py` 用 `--workload-source` 切换，数据没到位不阻塞 Phase 1/2/4/6 的开发和测试，只阻塞"用真实 LMSYS 数据跑出的最终图"。
2. **alpha/beta 没有唯一正确值**：当成一个要在报告里讨论的 tradeoff，不是要调出"最优解"。
3. **`error_level` 和 `uncertainty` 的关系是我们自己定义的建模选择**（见 4.3 的说明），报告里要讲清楚这个假设，不要包装成"这是预测器的真实行为"。
4. **数据集条款**：commit 派生 CSV 这件事我给的是工程判断，不是法律判断，如果你们对 gated dataset 条款有顾虑，改成不 commit、每人各自跑一次预处理脚本即可，两种路径在代码上没有区别。

---

准备好之后，实现顺序建议：Phase 1 → Phase 2 →（写 Phase 6 的 Oracle/Predicted SJF 手算测试，边写边验证 Phase 1/2 没问题）→ Phase 3（并行：你去弄 HF 授权，我写脚本）→ Phase 4 → Phase 5。

---

## 九、借鉴 TIE 论文（Zheng et al., 2026, arXiv:2604.00499）的修改方案

背景：跑完 Phase 1-5 后，实测发现 Exp2 里预测误差 error_level 加到 1.0（乘性噪声上限）时，Predicted SJF 的表现离 FCFS 还很远，猜想应该继续退化才对。查了这篇同方向的论文（提出 TIE 调度器）看它怎么处理预测误差/不确定性，发现两件事：一是我们乘性噪声模型本身有数学上的"饱和"问题（`error_level` 大到一定程度后，两个请求预测值的相对大小比较跟 `error_level` 无关，继续调大没有意义）；二是论文里有一个我们可以直接借用、成本很低的设计——**风险权重 β 随系统拥堵程度自适应**，正好能解决我们在 Exp4 里发现的"固定 alpha 太猛会拖垮短请求"的问题。

### 9.1 采纳 / 不采纳一览

**采纳**：
1. `predictor.py` 改成加性噪声（这是我们自己发现的必要修复，跟论文本身无关，只是顺便一起改）
2. `ARRSScheduler` 的 `alpha` 根据当前拥堵程度自适应，借鉴论文 Eq.12 的 `β = clip(0.1·Lq/B, 0.1, 0.5)` 设计
3. Exp4 新增一个对比臂：`arrs-fixed` vs `arrs-adaptive`，验证自适应是否真的缓解了 convoy effect
4. 报告里引用论文的 **consistency / robustness** 术语（learning-augmented algorithms 框架）描述 Exp2 的 Figure 4，说明 ARRS 的设计目标不是我们自己发明的评价标准

**不采纳（超出这个项目的范围，仿真项目不需要跟到这个精细度）**：
1. 每个请求拟合一整条 log-t 分布（需要真实模型对同一个 prompt 生成 100 次估计分布参数）——我们没有真实 LLM 可以这样采样
2. CVaR 风险度量替代我们的 `β·uncertainty` 惩罚项——CVaR 建立在"有一整条分布"的前提上，我们只有点估计+噪声，没有分布，硬套没有意义
3. DeBERTa 编码器训练一个真实预测器——`predictor.py` 是误差注入模拟，不是训练出来的模型，不在同一个抽象层级
4. 周期性乘性衰减 `Score·γ^(tw/τ)` 替换我们现在的连续加性衰减——论文用周期性（每 30 秒）是因为他们有真实的后台预测线程开销要摊，我们是纯离散事件仿真，`schedule()` 本来就在每次 arrival/departure 时被调用，没有这个顾虑，继续用连续加性衰减更简单，没必要为了"贴论文"平添复杂度

### 9.2 `src/predictor.py`：加性噪声修复

```
predicted_length = max(1, true_length + N(0, (error_level * REFERENCE_SCALE)²))
uncertainty = error_level * REFERENCE_SCALE
```

`REFERENCE_SCALE` 用真实 LMSYS 长度分布的一个固定统计量（比如均值 ~120 词，或中位数 ~89 词，两者都行，取均值更能反映"典型噪声幅度"），**不再随每个请求自己的 `true_length` 缩放**——这是跟原乘性模型的关键区别。原理见上一次讨论：乘性噪声下 `error_level` 大到一定程度会被两侧同时约掉，继续调大不再增加随机性；加性噪声下，`error_level` 越大，噪声绝对量级越压过 `true_length` 本身的差异，预测值之间的相对大小会趋于跟真实长度无关，Predicted SJF 的平均表现应该趋近于"随机排序"——排队论上，跟任务大小无关的调度策略，平均等待时间跟 FCFS 是一样的（经典结论），这样才能在 Exp2 里真正观察到"退化到 FCFS 附近"。

`uncertainty` 也要跟着改成用 `REFERENCE_SCALE` 而不是 `predicted_length`，否则还是会有同样的尺度问题。

### 9.3 `src/schedulers/arrs.py`：alpha 自适应拥堵程度

```
pressure = len(self.waiting) / self.max_batch_size   # 借用论文 Eq.12 的 Lq/B，Scheduler 基类本来就有这两个属性
alpha_effective = base_alpha * clip(pressure, min_scale, max_scale)
```

`min_scale`/`max_scale` 的具体范围需要跑出来调（先猜 `[0.2, 1.0]`，参考论文 `β∈[0.1,0.5]` 的做法，量级不能照抄，因为我们的 `alpha` 语义跟他们的 `β` 不是同一个东西），不追求"调出最优值"，讲清楚为什么这样设计（空闲时温和、拥堵时才加重 aging）就够。

需要给 `ARRSScheduler` 加一个开关（比如构造参数 `adaptive: bool = False`），保留"fixed alpha"模式——不能删掉旧行为，因为 Exp1/2/3 已经用 fixed alpha 跑出一套结果，报告里还要能重现。

**副作用要写进文档**：这个改动会让 `alpha_effective` 依赖"当前还有多少人在排队"，也就不再是每个请求在到达时就能算死的静态值了——等于放弃了"用静态 key + 标准堆做到 O(log n)"这个可选的算法优化方向（见第三部分讨论过的图算法优化点）。这个 tradeoff 值得在报告里提一句：拿了"更好的经验效果"，让"更漂亮的复杂度证明"变得更麻烦（不是不能两者都要——论文用周期性重算是一种折中——但这三周的时间不打算两个都做）。

### 9.4 Exp4 加一个对比臂

现有：`predicted_sjf` vs `arrs`
改成：`predicted_sjf` vs `arrs-fixed` vs `arrs-adaptive`

预期结论："`arrs-fixed` 能救长请求但拖累短请求（Exp4 已经实测到这个现象）；`arrs-adaptive` 应该在保留大部分'救长请求'效果的同时，明显改善短请求的 avg_wait/p95_wait，不再出现 convoy effect。" 这直接是"为什么要做成自适应"这个故事的关键证据，比单纯说"论文这么做"有说服力。

### 9.5 Exp2 的表述调整（不改代码，改怎么讲）

论文的术语：**consistency**（预测准的时候表现多好）vs **robustness**（预测很差的时候表现多差），正好对应 Exp2 里 `error_level=0` 和 `error_level` 拉到很大这两个端点。报告里可以直接用这套术语描述 Figure 4，并引用这篇论文和它引用的 Purohit/Lykouris 的 learning-augmented algorithms 框架，说明 ARRS 的设计目标就是在这两者之间找平衡。

### 9.6 需要确认的开放问题

1. `alpha_effective` 的 `[min_scale, max_scale]` 范围，需要跑出来看效果再定
2. 最终报告要不要把 `arrs-fixed` 和 `arrs-adaptive` 都保留展示——建议保留，因为"自适应版本明显更好"本身就是一个值得展示的对比实验，不是纯粹的实现细节

### 9.7 修正：9.3/9.4 的"自适应 alpha"方案被证明行不通，已放弃

跑完参数网格搜索后发现：Exp4 是我们故意设计成**持续重度过载**的场景（不这样测不出 starvation），导致 `pressure=len(waiting)/max_batch_size` 几乎全程卡在 `max_scale` 这个上限上——**不管 `min_scale` 取多少，结果都跟直接固定 `alpha=base_alpha×max_scale` 完全一样**（实测 `adaptive[0.0,0.5]` 和 `adaptive[0.1,0.5]` 输出的五个指标逐位对齐，没有任何差异）。这不是参数没调对，是数学上的必然：在一个"从头到尾都很堵"的场景里，"根据拥堵程度调整"这件事没有信息量。

根源是映射错了：论文 Eq.12 的自适应 `β` 管的是"要不要更谨慎地避开看起来可能很长的请求"（对应我们的 `beta×uncertainty` 项），不是"要多用力救一个等了很久的长请求"（我们的 `alpha`，也就是 aging）。这两个是不同的机制，解决不同的问题，不能互换。**结论：放弃 9.3/9.4 的"自适应 alpha"，Exp4 保留 `predicted_sjf` vs `arrs`（固定 alpha）两条线**——这个决定推翻了 9.3/9.4/9.6 里"要做 arrs-fixed vs arrs-adaptive 三线对比"的计划,那部分不再执行。真正把"自适应"这个思路用对地方的方案见下面第十节（uncertainty 模型里 `k` 分档,以及 `beta` 本身）。

---

## 十、Uncertainty 建模的最终方案（每个参数都要能追溯到出处）

背景：9.2 节的"加性噪声"修复了乘性噪声的饱和问题，但留了一个隐藏 bug——`uncertainty` 被定义成"同一次实验里所有请求共享的一个常数"，代入 ARRS 打分公式后，`beta×uncertainty` 对每个请求都加了同一个数，两两比较时直接抵消，**`beta` 完全不影响调度结果**，虽然公式里有这一项，但从来没起作用。本节的方案同时修这个 bug，并且要求每一个参数的数值都能指出"这个数字是从哪来的"，不能有第二个"REFERENCE_SCALE=120 怎么来的"这种说不清楚的地方。

### 10.1 模型定义

```
u_i    ~ Uniform(0, 1)                     # 这个请求的"predictor 有多没把握"，跟 true_length 无关，每个请求独立抽
sigma_i = u_i × k × REFERENCE_SCALE         # 换算成绝对的词数误差范围
L_hat_i ~ Uniform(L_i − sigma_i, L_i + sigma_i)   # 在这个区间里均匀抽一个数当预测值
L_hat_i = clip(L_hat_i, 1, L_MAX)           # 夹回合法范围
uncertainty_i = sigma_i                     # 直接报给 scheduler，逐请求不同
```

`k` 是"预测质量档位"（下面 10.2 给出具体来源），`REFERENCE_SCALE`、`L_MAX` 是两个固定常数（不随请求变化）。代码在 `src/predictor.py`。

### 10.2 每个常数的出处

| 常数 | 值 | 出处 |
|---|---|---|
| `REFERENCE_SCALE` | **76**（词） | Choi et al. 2025（ELIS，arXiv:2505.09142）Table 2，"Fine-tuned BGE" 那一行：`RMSE=101.29 token`。101.29 × 0.75（词/token 的标准换算比例）= 75.97 ≈ **76**。这是在**同一个数据集**（LMSYS-Chat-1M）上，用一个真实训练出来的 predictor（只看 prompt，不看已生成内容）量出来的真实误差,不是我们猜的 |
| `QUALITY_TIERS["low"]` | 34.33/101.29 ≈ **0.339** | 同一张 Table 2，"Iterative predictor"（用了部分生成结果,比我们条件更宽松,这里只是借用这个数字定量级）的 `RMSE=34.33`，除以 REFERENCE_SCALE 对应的那一行 |
| `QUALITY_TIERS["realistic"]` | **1.0** | 定义上就是 1（REFERENCE_SCALE 本来就是从这一档算出来的） |
| `QUALITY_TIERS["high"]` | 224.98/101.29 ≈ **2.221** | 同一张 Table 2，"Pre-trained BGE"（没微调，`R²=-1.58`，比直接猜均值还差）的 `RMSE=224.98`，除以 REFERENCE_SCALE 对应的那一行 |
| `L_MAX` | **3072**（词） | 跟 `scripts/extract_lmsys_lengths.py` 的 `MAX_WORD_COUNT` 保持一致（4096 token 的常见 `max_generated_tokens` 配置换算成词），回答"这个请求真实能有多长"，跟 `REFERENCE_SCALE` 回答的是不同的问题（"predictor 一般会错多少"），两者不能换用 |
| `beta`（ARRS 打分里 `β×uncertainty` 的权重） | 0.2（现有实验的默认值） | **没有外部出处，是我们自己的调参选择**，报告里要如实说明，不要跟上面几个数字混为一谈 |

`k>2.221`（比如做 adversarial 压力测试用的 5、10、50）**没有 ELIS 的数据支持**，报告里要明确标注"这是我们为了探测极端场景而设的假设值，不对应任何已知真实 predictor"，不能跟 low/realistic/high 三档混着说成"都是有依据的"。

### 10.3 完整例子（可复现，seed=153，可以直接讲给老师）

场景：两个请求都用 `high` 档（`k=2.221`，模拟一个没微调过、很不靠谱的 predictor），`beta=0.2`，假设都刚到达（暂时不考虑 aging 项）。

```python
from src.predictor import predict_length, QUALITY_TIERS
import random
rng = random.Random(153)
predicted_A, unc_A = predict_length(300, QUALITY_TIERS["high"], rng)  # 请求 A：真实长度 300
predicted_B, unc_B = predict_length(150, QUALITY_TIERS["high"], rng)  # 请求 B：真实长度 150
```

跑出来的真实数字：

| | 真实长度 L | 预测值 L̂ | uncertainty σ | ARRS score = L̂+0.2σ |
|---|---|---|---|---|
| 请求 A | 300 | 157.69 | 167.69 | 157.69+33.54=**191.23** |
| 请求 B | 150 | 180.86 | 43.89 | 180.86+8.78=**189.64** |

**Predicted SJF** 只看预测值：`157.69 < 180.86` → 选 **A**。但 A 真实长度是 300，比 B 的 150 长一倍，Predicted SJF 选错了——而且它选错的原因不是"运气不好"，是它压根不知道自己应该对 A 的预测没那么有信心。

**ARRS**：`191.23 > 189.64` → 选 **B**。ARRS 因为看到 A 的 `uncertainty=167.69` 特别大（远超 B 的 43.89），加大了对 A 的风险惩罚，即使 A 的预测值本身看起来更短，ARRS 依然判断"这个预测不可信，先让 B 上"——这正是 `beta` 现在真正在起作用的证据（换算前的旧版本里，这一步不会发生,因为 uncertainty 对 A、B 是同一个数）。

这个例子任何人在自己电脑上用同一个 seed 都能重新跑出同样的数字，不是手算编出来的案例。

---

## 十一、TIEScheduler：log-normal + CVaR，正式验证"tail-awareness 能不能帮上平均延迟"

背景：第十节的对称 `uncertainty` 模型，数学上证明过 `beta×uncertainty` 帮不上平均延迟（`E[真实值|predicted,σ]` 跟 `σ` 无关）。尝试过的两个修补方案——人为加 `skew`（有效但没有真实数据支撑）、换成 log 空间对称噪声（数学论证是错的，实测反而更差）——都不理想。这一节是第三次尝试，也是第一次得到**干净、单调、没有 cherry-pick** 的正面结果。

### 11.1 跟第十节的关键区别：不再用一个标量 σ，而是拟合一条分布

第十节：每个请求只有 `(predicted_output_len, uncertainty)` 两个数,`uncertainty` 是"这个区间有多宽"，不携带方向信息。

第十一节：每个请求拟合一条 **log-normal 分布**（`X ~ LogNormal(μ, σ)`，`μ=log(L_pred)`），然后算这条分布的 **`E[X]`**（期望）和 **`CVaR_90[X]`**（最差 10% 情况下的条件期望）。log-normal 天生右偏（长尾在上不在下），所以 `CVaR_90` 天然比 `E[X]` 大得多——这个不对称性是分布形状带来的，不需要像 `skew` 那样额外声明一个方向性假设。

**为什么用 log-normal 不用论文（TIE, Zheng et al. 2026）的 log-t**：log-t 的 `E[X]`/`CVaR` 没有闭式解，论文里是用蒙特卡洛（1万样本/请求）算的，我们的模拟器要给几千个请求都算一遍，用蒙特卡洛太慢。log-normal 是 TIE 论文自己 Table 1 里报告的**第二拟合优度分布**（KS 通过率 60.3%，比 log-t 的 93.1% 差一些，但仍然过关），换来的好处是 `E[X]` 和 `CVaR_α[X]` 都有闭式公式：

```
E[X] = exp(μ + σ²/2)
CVaR_α[X] = E[X] × Φ(σ − Φ⁻¹(α)) / (1−α)      Φ = 标准正态 CDF
```

这两个公式已经用 200 万次蒙特卡洛采样验证过，跟闭式解精确匹配（见 `src/predictor.py` `lognormal_expectation`/`lognormal_cvar`）。

### 11.2 每个请求的 `(predicted, σ)` 怎么来——两档，分开标注出处

**95%-99% 的"正常"请求**：`predicted = true + ε`，`ε` 从一个 **95/5 的 Gaussian mixture** 里抽——不是单个 Gaussian，因为 ELIS 论文（Choi et al. 2025, Table 2, fine-tuned BGE）报的 `MAE=71.48` 和 `RMSE=101.29` 算出来的比值是 `101.29/71.48≈1.417`，而单个 Gaussian 的 MAE/RMSE 比值锁定在 `≈1.253`（RMSE 和 MAE 对 Gaussian 来说不是独立的两个自由度），匹配不上真实数据,说明真实误差比 Gaussian 更重尾。用 `scipy.optimize.fsolve` 解两个方程（`0.95σ1√(2/π)+0.05σ2√(2/π)=MAE`，`√(0.95σ1²+0.05σ2²)=RMSE`），解出 `σ1=78.77, σ2=295.52`——**这两个数字精确复现了 ELIS 的真实 MAE 和 RMSE**，不是凑的。`σ_normal=0.2`（log-normal 的 σ 参数,跟 log-t 论文里的量级同一个数量级)。

**1%-5% 的"tail"请求**：这个**没有 ELIS 或任何论文支撑，是我们明确构造出来的压力测试场景**——真实长度从整个真实分布的 **P99 以上** 抽（真实很长的请求），预测长度从 **P50 以下** 独立抽（看起来很短），两者故意不相关，制造"预测器严重低估"的场景。`σ_tail=0.8`（比 `σ_normal` 大很多，代表"这个预测的置信区间应该报得很宽"——这是我们的建模假设：**严重低估的场景往往对应预测器自己也该，但没有，报出高不确定性**，报告里要明确写成"合理的仿真假设"，不是"真实 predictor 的行为"）。

### 11.3 两个 Scheduler 怎么用这些数字

```python
# Predicted SJF：只看点估计，不看分布
score = predicted_output_len

# TIEScheduler（新增，src/schedulers/tie.py）：用整条分布
score = expected_length + beta * tail_risk      # E[X] + β·CVaR_90[X]
```

`TIEScheduler` 没有 aging 项——故意的，这一节只想单独验证"tail-awareness 本身能不能帮上平均延迟"，跟 aging/starvation（已经在 Exp4 验证过）分开，不要混在一起讲。

### 11.4 Experiment A / B 的结果（已跑完，5 个 seed 平均）

**Experiment A**（固定 `σ_tail=0.8`，扫 `tail_rate` 从 0% 到 5%）：

| tail_rate | Predicted SJF | TIE | 优势 |
|---|---|---|---|
| 0% | 295.52 | 295.52 | 0.00 |
| 1% | 326.06 | 323.66 | 2.40 |
| 2% | 362.90 | 358.27 | 4.63 |
| 3% | 410.43 | 400.94 | 9.49 |
| 4% | 457.91 | 441.35 | 16.56 |
| 5% | 497.01 | 476.52 | **20.49** |

![Figure A](figures/figA_tail_rate.png)

**Experiment B**（固定 `tail_rate=3%`，扫 `σ_tail` 从 0.2 到 1.0）：

| σ_tail | Predicted SJF | TIE | 优势 |
|---|---|---|---|
| 0.2 | 410.43 | 410.43 | 0.00 |
| 0.4 | 410.43 | 408.93 | 1.50 |
| 0.6 | 410.43 | 405.92 | 4.51 |
| 0.8 | 410.43 | 400.94 | 9.49 |
| 1.0 | 410.43 | 393.06 | **17.37** |

![Figure B](figures/figB_sigma_tail.png)

**两组都是单调、干净的结果，没有 cherry-pick**：`tail_rate=0`（没有污染）时两者完全相等（0.00），符合预期；优势随污染比例、随 σ_tail 都单调递增。Predicted SJF 在 Experiment B 里是一条水平线——符合预期，因为 `σ_tail` 只影响 TIE 怎么算分布,不影响 `predicted_output_len` 本身。

### 11.4.1 "表现不好"具体表现在哪——不是 tail 请求自己，是它连累的其他请求

拆开看 `tail_rate=3%` 这组数据（59个左右的 tail 请求 vs 剩下 1941 个正常请求，5 个 seed 平均）：

| | tail 请求自己的平均 response_time | 其他正常请求的平均 response_time |
|---|---|---|
| Predicted SJF | 69.8（很快，因为"看起来短"被插队优先跑） | **420.7**（被拖累） |
| TIE | 130.3（变慢，被正确地往后排） | **409.1**（净改善） |

![Figure: collateral damage](figures/fig_collateral_damage.png)

**这是 HOL blocking 的具体机制**：Predicted SJF 让被严重低估的长请求自己插队跑得很快,但它占用 batch 槽位的时间是按真实长度算的,这段时间把后面的正常请求都耽误了。TIE 因为看到这些请求的 `CVaR_90` 很高,故意让它们自己等更久,用它们自己的响应时间换来了大多数（97%）正常请求的净改善——这比只看整体平均数直观得多,能直接回答"预测器犯错的代价，最终是谁在承担"这个问题。

### 11.5 跟老师汇报时要讲清楚的边界

1. **"正常"档（95-99%）是有真实数据支撑的**（ELIS 的 MAE/RMSE），"tail"档（1-5%）**是我们明确构造的压力测试场景**，两者不能混着说成"都是真实的"
2. `tail_rate`、`σ_tail`、`σ_normal` 都是可以在报告里公开讨论的实验参数,不是"调出来凑数据"——两组敏感性实验（A、B）本身就是用来说明"这个结论在参数变化下是稳健的，不是撞对了一个特定数字才成立"
3. `TIEScheduler` 目前没有 aging 项——它只回答"tail-awareness 能不能帮上平均延迟"这一个问题，跟"aging 能不能防止 starvation"是两个不同的问题（后者是第九节 `ARRSScheduler` 想回答的，`ARRSScheduler` 已从代码库移除，见文档开头的更新说明），报告里建议分两节讲，不要合并成一个故事

## 十二、`decode_time_per_step` 的出处

所有实验脚本（`run_experiment.py`/`exp1`/`exp3`/`expA`/`expB`/`plot_four_schedulers_*`）用的 `decode_time_per_step=0.05`，含义是"仿真时间单位 = 秒"时，生成一个 token 要 **50ms**（即 20 tokens/秒的生成速度）。

**出处**：50ms/token 的量级参考自公开 LLM serving benchmark（vLLM、TensorRT-LLM、Anyscale LLMPerf 这类工具测出的 inter-token latency，ITL）里中等规模模型、有批处理场景下的典型范围：

- 7B 级别模型，单张 A100/H100：约 15~30ms/token
- 13B~34B 级别：约 30~60ms/token
- 70B 级别：约 50~100ms/token（取决于并行策略）

50ms/token 落在 13B~34B 这个区间，是一个有量级依据但**没有绑定到某一篇具体论文/某一次具体测试**的假设值，跟 `REFERENCE_SCALE` 这种直接引用 ELIS Table 2 数字的常数不是同一个可信度级别——如果要更精确，需要指定一个具体要对齐的模型规模/硬件，再换算成精确数字。

**为什么必须是"秒"这个单位、不能随便换**：`arrival_rate`（Poisson λ，单位 requests/秒）和 `throughput`（`src/metrics.py` 里算的 requests/秒）都是按"秒"定义的。只要 `decode_time_per_step` 也用秒做单位，`waiting_time`/`response_time`/`throughput` 就能在同一套时间轴上直接读数、互相换算，不需要额外的单位转换——这也是为什么这个值不能脱离"仿真时间=秒"这个约定单独调整。
