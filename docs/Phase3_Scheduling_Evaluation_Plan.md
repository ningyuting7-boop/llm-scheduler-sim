# 阶段三计划：在真实 vLLM + 真实模型上 benchmark TIE 调度

## 0. 目标

把阶段二训出来的 DeBERTa 预测器接进**官方 vLLM**（通过 `--scheduler-cls` 外挂我们自己的调度器类，**不 fork**，见第 3 节），用真实 GPU 服务真实请求，测量 TIE 调度相对 FCFS 的实际收益。

**不做仿真**——阶段一/二的仿真结果（[`Report.md`](Report.md)）用的是人为构造的不确定性，阶段三要的是真实系统上的端到端数字。

**核心问题**：当 (μ, σ) 来自真实训练的预测器、跑在真实 vLLM 上时，TIE 相对 FCFS 的延迟优势有多大？

⚠️ **前置认知**：阶段二测出 σ̂ 的测试集 R²≈0.05（见 [`Phase2_Execution_Log.md`](Phase2_Execution_Log.md)），几乎没有区分能力。TIE 的 CVaR 项完全由 σ̂ 驱动，所以**很可能测不出优势**。第 5 节的实验设计专门用来区分"方法不成立"和"我们的预测器不合格"——没有这个区分，阶段三的结论不可用。

---

## 1. TIE fork 提供了什么（`~/Documents/AI_inference/TIE`）

论文：Zheng et al., *Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions*, **ICML 2026**（OpenReview `I5IMkvVKd7`）。仓库是一份完整的 vLLM fork。

### 1.1 已经实现好、可以直接用的

| 组件 | 文件 | 说明 |
|---|---|---|
| TIE 等待队列 | `vllm/v1/core/sched/request_queue.py` → `UARequestQueue` | 最小堆 + 惰性删除 + 异步预测线程 + 乘性防饥饿衰减 |
| 打分 | `ua_predictor.py` → `_compute_score` | `E[X] + β·CVaR_0.9[X]`，**β 自适应**：`clip(0.1·L_q/B, 0.1, 0.5)` |
| 分布计算 | `ua_score_calculator.py` | log-t(ν=3.5) 蒙特卡洛（10,000 采样），**在 2048 处截断** |
| 启动脚本 | `start-server.sh` | `bash start-server.sh <policy> <CUDA_VISIBLE_DEVICES> <port> <model_path>` |
| 压测客户端 | `benchmarks/benchmark_serving.py` | vLLM 官方 serving benchmark（TTFT / TPOT / E2E / 吞吐） |

启动脚本里几个关键设置：`MAX_NUM_SEQS=32`（即自适应 β 公式里的 B，通过 `UA_GPU_BATCH_SIZE` 传给预测器）、`--gpu-memory-utilization 0.91`、`--max-model-len 8192`、`--no-enable-prefix-caching`。

### 1.2 实际**不能**用的（重要）

`create_request_queue()`（`request_queue.py:630`）只实现了三种：

```python
if policy == PRIORITY:  return PriorityRequestQueue()
elif policy == FCFS:    return FCFSRequestQueue()
elif policy == UA:      return UARequestQueue(tokenizer=tokenizer)
else:                   raise ValueError(...)     # SSJF / LTR / GMM_UA 全部在这里报错
```

`SchedulingPolicy` 枚举里有 `SSJF`、`LTR`、`GMM_UA`，`scheduler.py` 里也有对应的预测器加载代码，但**队列实现没有随匿名版本一起发布**。

**结论：我们能做的对比是 FCFS vs UA(TIE)**。想加 SSJF 这类对照臂，得自己实现队列——不建议，超出阶段三范围。

### 1.3 真实 vLLM 的 FCFS ≠ 仿真里的 FCFS（实验设计的前提）

官方 vLLM 默认策略就是 FCFS（`vllm/config/scheduler.py:112`，`policy: SchedulerPolicy = "fcfs"`），官方只实现 `fcfs` / `priority` 两种。但它的语义和阶段一仿真**不同**：

```python
self.waiting = create_request_queue(self.policy, tokenizer=tokenizer)   # scheduler.py:188
self.running: list[Request] = []                                        # scheduler.py:190
```

调度策略**只决定谁先进入 `running` batch**，不决定谁先做完。vLLM 是 continuous batching：`running` 里最多 `max_num_seqs` 条请求（`start-server.sh` 设 32）并发逐 token 解码，每个 step 各出一个 token。

| | 阶段一仿真 | 真实 vLLM |
|---|---|---|
| FCFS 语义 | 单服务台，先来者独占直到完成 | 先来者先**入场**，入场后与后来者**并发共享 GPU** |
| 队头阻塞 | 强 | 仅当 `running` 占满 32 条时才发生 |

**直接后果，必须写进实验设计**：如果负载没把 `max_num_seqs` 压饱和，`waiting` 队列长期为空，来一条进一条，**所有调度策略表现完全相同**。压测时必须先确认 `waiting` 队列非空（见 5.3）。同理，真机上 TIE 相对 FCFS 的 gap **预期会明显小于仿真结果**——这本身是要在报告里解释的发现，不是实验失败。

**抢占语义的差异**：KV cache 不足时 vLLM 会抢占（`scheduler.py:407`，LIFO 踢掉 `running` 最后一条），然后 `self.waiting.prepend_request(preempted_req)` 放回等待队列。两种队列对这个调用的处理不同：

- `FCFSRequestQueue.prepend_request` → `appendleft`，**插到队头**，优先恢复
- `UARequestQueue.prepend_request` → 直接调 `add_request`（`request_queue.py:561`），**按 TIE 分数重新入堆**

也就是说 TIE 下被抢占的请求不保证优先恢复，可能再次被推后。这是一个需要在结果分析时留意的机制性差异（尤其看 P99 和公平性指标时）。

---

## 2. checkpoint 兼容性（外挂方案下已基本消解）

我们阶段二的产物和 TIE 原版加载器有四处对不上：

| 项 | TIE 期望 | 我们的 | 处理 |
|---|---|---|---|
| 文件名 | `best_model.pth` | `best_model.pt` | 改名 |
| checkpoint 结构 | `torch.load(...)['model_state_dict']`（完整 dict，含 optimizer 等） | 裸 `state_dict` | 包一层，或改加载代码 |
| `normalize_stats.json` 键名 | `sigma_log_mean` / `sigma_log_std` | `log1p_sigma_mean` / `log1p_sigma_std` | 改键名（数值含义相同，都是 log1p 后再标准化） |
| 模型类结构 | `LogTPredictionModel`：`mu_feature_extractor` + `mu_predictor` 分开 | `LengthDistributionPredictor`：合并成一个 `mu_branch` | **参数名对不上，`load_state_dict` 会直接失败** |

**改用外挂方案后，这四项全部不再是障碍。** `vllm_tie/ua_predictor.py` 是**我们自己仓库里的文件**（改编自 TIE），加载逻辑由我们写，所以直接改成读我们的格式即可：

```python
from src.predictor_model import LengthDistributionPredictor, NormalizeStats

model = LengthDistributionPredictor.from_pretrained(DEBERTA_MODEL_NAME)
model.load_state_dict(torch.load("checkpoints/predictor_full/best_model.pt"))
stats = NormalizeStats.load("checkpoints/predictor_full/normalize_stats.json")
```

**不需要**改文件名、不需要包一层 `model_state_dict`、不需要改 json 键名，也不需要写 state_dict 映射脚本。上表保留在这里只是为了记录"如果走 fork 路线会遇到什么"——以及说明外挂方案额外省掉了多少事。

> 唯一仍需注意的：`ua_predictor.py` 里 `MODEL_PATH`、`encoder_path` 等位置在 TIE 的匿名版本里是 `"xxx"` 占位符，移植时要全局搜一遍填掉（任务 4）。

---

## 3. 部署策略：用官方 `--scheduler-cls` 外挂，不 fork、不编译

原计划打算复制 TIE 的 fork 再打补丁。**不需要**——官方 vLLM 自带可插拔调度器接口：

```python
# vllm/config/scheduler.py:129
scheduler_cls: str | type[object] = Field(default=None)
"""The scheduler class to use. "vllm.v1.core.sched.scheduler.Scheduler" is
the default scheduler. Can be a class directly or the path to a class of
form "mod.custom_class"."""
```

CLI 是 `--scheduler-cls mod.MyScheduler`（`vllm/engine/arg_utils.py:1092`），vLLM 用 `resolve_obj_by_qualname` 动态导入。TIE 之所以 fork，只是因为他们选择走 `--scheduling-policy` 那条路（要改枚举、改 `create_request_queue`），而同一份代码里本来就有这个更干净的外挂口。

### 3.1 只需要继承，不需要实现整个接口

`SchedulerInterface` 有 13 个抽象方法（`schedule`、`update_from_output`、KV cache、chunked prefill、抢占……），从零实现不现实。但不必——**继承官方 `Scheduler`，只替换 `self.waiting` 这一个成员**：

```python
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_tie.request_queue import UARequestQueue

class TIEScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)          # 官方逻辑原样跑完
        tokenizer = get_tokenizer(...)
        load_ua_predictor_model(device_id=...)
        self.waiting = UARequestQueue(tokenizer=tokenizer)   # 唯一的改动
```

对照 TIE fork 的 `Scheduler.__init__`（`scheduler.py:137-188`），他们的实质改动正好就是这三件事：取 tokenizer、加载预测器、换等待队列。**全部可以在 `super().__init__()` 之后的子类构造函数里完成，不碰官方源码一行。**

### 3.2 阶段三的代码结构

```
llm-scheduler-sim/
  vllm_tie/                     # 我们自己的包，pip 环境里 importable 即可
    __init__.py
    scheduler.py                # TIEScheduler(Scheduler)，~30 行
    request_queue.py            # UARequestQueue（改编自 TIE，文件头注明出处）
    ua_predictor.py             # 改编自 TIE，接我们阶段二的 checkpoint
    ua_score_calculator.py      # log-t 蒙特卡洛打分（来自 TIE）
```

启动：

```bash
# 基线：原封不动的官方 vLLM
vllm serve <qwen3-8b> --port 8000 ...

# TIE：只多一个参数
vllm serve <qwen3-8b> --port 8000 --scheduler-cls vllm_tie.scheduler.TIEScheduler ...
```

### 3.3 相比 fork 的好处

| | fork 方案 | `--scheduler-cls` 方案 |
|---|---|---|
| 安装 | 源码编译 CUDA kernel，数小时、易失败 | `pip install vllm==0.11.1` 官方 wheel，**不编译** |
| 原计划第 7 节的头号风险 | "vLLM fork 装不起来" | **消除** |
| FCFS 基线 | TIE 改过的 fork 里的 fcfs | 完全未修改的官方 vLLM，对照更干净 |
| 版权归属 | 整棵 vLLM 树混在我们仓库里 | 我们写的 / 改编自 TIE 的，边界清楚，报告里好写 |
| 代码量 | 整个 vLLM | 4 个文件 |

### 3.4 唯一的注意事项：接口不是公开 API

vLLM 自己在加载自定义调度器时会打印：

> `Using custom scheduler class %s. This scheduler interface is not public and compatibility may not be maintained.`

**应对：锁死版本**。TIE fork 的 `PKG-INFO` 显示 `Version: 0.11.1`，`requirements/cuda.txt` 要求 `torch==2.9.0`。我们就装 `vllm==0.11.1`，这样 TIE 那几个文件的 import（`RequestQueue`、`Request`、`SchedulingPolicy`）与我们的运行环境完全一致。

**任务 1 就是验证这一点**（纯 CPU，几十分钟）：`pip install vllm==0.11.1`，然后确认

1. `from vllm.v1.core.sched.scheduler import Scheduler` 能导入
2. `Scheduler.__init__` 里确实有 `self.waiting = create_request_queue(...)` 这个可替换点
3. `RequestQueue` 抽象基类、`Request` 类能从官方包正常 import（`UARequestQueue` 依赖它们）
4. `--scheduler-cls` 指向一个什么都不改的空子类，服务能正常起来

这四条过了，整个部署风险就基本清零。

---

## 4. 资源需求：单卡 A100，四臂全部同卡

**决定：用 1 张 A100，四个臂全部单卡跑**（`gpu:a100:1`，阶段一 `generate_labels.slurm` 已验证这个 gres 在本集群可用）。

TIE 原版 `start-server.sh` 保留一张卡给预测器，是为了避免干扰。我们不需要，原因有两条：

**(1) `FCFS+Predictor` 对照臂已经把干扰对称化了**（见 4.2）。②③ 两臂 GPU 负载完全相同，`③−②` 这个差值里不含预测器开销，单卡不影响这个核心对比的有效性。

**(2) A100 的显存余量足够**。关键在于 vLLM 的初始化顺序（`vllm/v1/engine/core.py`）：

```python
109:  num_gpu_blocks, ... = self._initialize_kv_caches(...)   # 先按 util 比例分配 KV cache
120:  Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
133:  self.scheduler = Scheduler(...)                          # 预测器在这里才加载
```

**KV cache 先分配、预测器后加载**——所以 DeBERTa 的显存**不被 `--gpu-memory-utilization` 统计**，它必须挤进剩下的 `(1 − util)` 那一块。算一下：

| 卡 | util=0.91 的剩余 | DeBERTa 需要 | 结论 |
|---|---|---|---|
| A100-80GB | 7.2 GB | ~1–3 GB | **充裕**，util 保持 0.91 |
| A100-40GB | 3.6 GB | ~1–3 GB | **偏紧**，建议降到 0.85（剩 6GB） |
| V100-32GB | 2.9 GB | ~1–3 GB | 勉强，且 fp16 另有风险 |

> ⚠️ **必须限制预测器 batch**。`UARequestQueue` 默认 `max_batch_size=128`（`request_queue.py:261`），突发负载下 batch_buffer 真会攒到 128。DeBERTa-v2 的 disentangled attention 本来就吃显存（阶段二训练时 batch 32 就 OOM 过），128×512 token 的 forward 会直接炸掉剩余显存。**移植时把 `max_batch_size` 降到 16–32**，并把预测器以 fp16 加载。这条写进任务 2。

A100 相比 V100 还消掉两个风险：原生支持 bf16（不用担心 Qwen3-8B 的 fp16 加载问题），以及**只要 1 张卡**——排队难度远低于 2 卡。

> **代价**：单卡下 `②③` 的绝对数值都含预测器开销，`③ vs ①` 这个"对外宣称的 TIE 收益"会偏保守。这要在报告里写明，并用 `②−①` 给出开销的具体数值。

### 4.1 "预测器开销会不会吃掉 TIE 的收益"

**先澄清一个常见误解**：预测**不在调度关键路径上**，和放哪张卡无关。`UARequestQueue.add_request`（`request_queue.py:517`）先用常数 `2048.0` 作为初始分立即入堆并返回，再把请求丢进 `_prediction_queue`；真正的推理在 daemon 线程 `_prediction_worker` 里批量执行，算完通过 `_push_updated_score` 惰性更新堆。**调度主循环从不阻塞等待预测。** 同卡/异卡改变的是干扰程度，不是同步性。

单卡（`UA_PREDICTOR_GPU=0`）的真实代价，按严重程度：

| # | 代价 | A100 上的量级 | 说明 |
|---|---|---|---|
| 1 | 显存挤压 | **A100-80GB 可忽略** | 见上表：util=0.91 仍剩 7.2GB。40GB 卡则需降到 0.85，代价是 KV cache block 减少 → 并发容量下降、更早抢占 |
| 2 | SM 争抢 | ~3–10% 占空比 | `optimal_batch_size=8` / `max_wait_time_ms=3.0`，低到达率下基本每来一条跑一次；DeBERTa-v3-base 单次 forward 在 A100 上约 3–8ms |
| 3 | GIL | 小 | 预测线程与调度器同进程，但 PyTorch CUDA 调用和 Rust fast tokenizer 都释放 GIL；且此项**同卡异卡都存在** |

**A100-80GB 单卡下这三项都不大**，这也是选 A100 而非凑 2 张 V100 的原因。但无论大小，②③ 对比都不受影响——干扰对两臂完全对称。

**因此，TTFT 在某些区间确实可能劣于 FCFS**——这和 1.3 的饱和度分析直接咬合：

| 负载 | TIE 排队收益 | 预测器开销 | 净效果 |
|---|---|---|---|
| `Waiting ≈ 0`（未饱和） | **0**（队列为空，调度无事可做） | 全额支付 | **TIE 更差** |
| `Waiting > 0`（饱和） | 数百 ms ~ 秒级 | ~10ms + KV cache 缩水 | 收益应占优 |

所以"低速率下 TIE 比 FCFS 差"是**可预期的正确结果**，报告里应当画出来并解释，而不是当成故障。

### 4.2 核心设计：`FCFS+Predictor` 对照臂（把开销和调度效果分离）

与其估算干扰有多大，不如**让基线付同样的成本**：

```python
class FCFSWithPredictorScheduler(Scheduler):
    """与 TIEScheduler 完全相同的 GPU 负载（加载 DeBERTa、照常预测、
    照常占显存和 SM），但等待队列仍是官方 FCFSRequestQueue，
    预测结果不参与排序。用于分离"预测器开销"与"调度决策收益"。"""
```

实现成本接近零（就是 `TIEScheduler` 换个队列类），但它把两件事彻底拆开：

- `TIE` vs `FCFS+Predictor` → **纯调度决策的效果**（GPU 负载完全对等）
- `FCFS+Predictor` vs `FCFS` → **纯预测器开销**

这比争取第二张卡更能把结论说清楚，而且**不用排 2 卡的队**。建议：即使拿到 2 卡也跑这一臂，它能直接量化"把预测器部署进生产要付多少成本"，这本身就是论文没给、而我们能给的数字。

阶段二的经历：单张 `gpu:1` 不指定型号要排到次日凌晨；`gpu-interactive` 分区 2 小时上限。**现在要 2 张卡、而且要跑多轮压测**，排队风险显著更高。

**应对**：
- 用 `sbatch`（批处理），不要交互式——阶段二已经吃过断线丢进度的亏
- 指定 `--gres=gpu:v100-sxm2:2`（阶段二验证过指定型号排队快很多）
- **一个 job 内跑完全部策略**：脚本里先起 FCFS server → 压测 → 停 → 起 TIE server → 压测 → 停。避免为每个臂单独排一次队
- V100 不支持 bf16 原生加速，需确认 Qwen3-8B 以 fp16 加载能否正常（任务 5 冒烟时验证）

**服务哪个模型**：必须是 **Qwen3-8B**。我们的预测器是在 Qwen3-8B 的输出长度分布上训练的，换成别的模型（比如 TIE 示例里的 Llama-3-8B）预测器就是无效的。

---

## 5. 实验设计

### 5.1 对照臂

| 优先级 | 臂 | 启动方式 | 作用 |
|---|---|---|---|
| **P0** | ② **FCFS+Predictor** | `--scheduler-cls ...FCFSWithPredictorScheduler` | **主对比的基准**，见 4.2 |
| **P0** | ⑤ **Predicted-SJF** | `--scheduler-cls ...TIEScheduler` + `TIE_BETA=0` | **隔离论文创新点**，见 5.1.1 |
| **P0** | ③ **TIE (真实预测器)** | `--scheduler-cls ...TIEScheduler` | **主对比** |
| P1 | ④ TIE-Oracle-σ | 同 ③，`TIE_MODE=oracle` | σ 质量天花板，见 5.2 |
| P1 | ① FCFS | `vllm serve <qwen3>`（零参数，官方默认） | 对外参照 + 量化开销 |

**先做 ②⑤③ 这三臂**（P0）。它们共用同一个二进制、同一个预测器加载路径、同样的 GPU 负载，只差队列类和一个 `TIE_BETA` 环境变量，**边际成本接近零**。

**①④ 随后补**（P1）：① 是零代码（原版 `vllm serve`）；④ 与 ③ 同一个二进制，只改 `TIE_MODE=oracle`。

五臂**除了调度器全部一致**：同一个模型、同一套 `--max-num-seqs 32 --max-model-len 8192 --no-enable-prefix-caching`，同一份 workload，同一组到达率。

> ⚠️ `--gpu-memory-utilization` 必须**五臂取同一个值**。**不能只降带预测器的臂**——否则 KV cache 容量不同，对比无效。以"带预测器的臂能稳定跑起来的最大值"为准（`benchmark_serving.slurm` 默认 0.88）。

四条正交的读法：

| 对比 | 测出什么 |
|---|---|
| **③ vs ⑤** | **CVaR 项的净贡献**——即论文相对普通 SJF 的创新点 |
| **⑤ vs ②** | SJF 排序本身的收益（GPU 负载对等） |
| ② vs ① | 预测器的部署开销 |
| ④ vs ③ | σ 预测质量的天花板 |

### 5.1.1 为什么必须有 ⑤：CVaR 项在本 workload 上可能几乎不改变排序

**这是一个在消耗任何 GPU 时间之前就测出来的结果**，用 `data/benchmark_oracle_labels.csv`（阶段一用 20 次真实采样拟合出的 σ，即 oracle 质量）直接计算：

```
CVaR/E[X] 比值:  p10=1.11   median=1.30   p90=1.92   max=8.31
σ 分位:         p50=0.103  p90=0.257    p99=0.704
CVaR/E[X] > 2.0 的 prompt: 154 / 1817 = 8.5%
```

| | Spearman(TIE 排序, 纯 E[X] 排序) | top-32 队头重叠 | top-8 队头重叠 |
|---|---|---|---|
| β=0.1（下限，队列不深时的常态） | **0.99957** | 96.9% | 75.0% |
| β=0.3 | 0.99791 | 93.8% | 62.5% |
| β=0.5（上限） | **0.99625** | 93.8% | 62.5% |

**即使用完美的 σ，TIE 的排序也几乎等于按 E[X] 排序**：top-32（正好是 `max_num_seqs`，也就是一批要调度的量）只有 1–2 个位置不同。原因是本 workload 的 σ 太小——Qwen3-8B 在官方采样参数下，多数 prompt 的输出长度相当稳定。

**后果**：`③ vs ②` 大概率会显示 TIE 有显著收益，**但那个收益来自 SJF 排序，不是来自论文的不确定性感知**。没有 ⑤ 这一臂就无法区分这两者，而区分它们正是复现这篇论文的意义所在。

> 这条同时也是一个**独立于 GPU 实验的、可直接写进报告的发现**：在我们的 workload 上，TIE 打分的排序与按期望长度排序的一致性达到 Spearman ≥ 0.996，因此 CVaR 项能贡献的上限本身就很小——**与预测器质量无关**。若 GPU 实验测出 `③ ≈ ⑤`，这个计算就是它的机制解释；若测出 `③ > ⑤`，则说明那 8.5% 的高不确定性 prompt 在饱和时的影响被放大了，同样是有价值的结果。
>
> 复现命令见 `scripts/rank_agreement_check.py`。

### 5.2 TIE-Oracle-σ：让结果可解释的关键

σ̂ 的 R² 只有 0.05，TIE 很可能没优势。**没有这一臂，我们无法判断该归咎于方法还是归咎于预测器**，阶段三就白做了。

**做法**：在 `vllm_tie/ua_predictor.py` 里加一个"查表模式"（`UA_MODE=oracle`）——不跑 DeBERTa 推理，而是从预计算的 CSV 里按 prompt 查出 (μ, σ)。用阶段二标签里那份**由 20 次真实采样拟合出来的 (μ, σ)**（即 oracle 质量的分布参数）。因为代码在我们自己的包里，这只是加个分支，不涉及改任何第三方源码。

**前提**：压测用的 prompt 必须是我们有 oracle 标签的。所以 **workload 用阶段二的 test split**（1,817 个去重 prompt）——这些 prompt 预测器训练时确实没见过（按 prompt 分组切分保证），同时我们手上有它们的 20 次采样和拟合标签。

**判读时一律以 `FCFS+Predictor` 为基准**（而非裸 FCFS），这样比较不含预测器开销：

| 观察 | 结论 |
|---|---|
| Oracle-σ 明显优于 FCFS+Predictor，但真实预测器的 TIE 没有 | **方法成立，瓶颈是我们的 σ 预测器**——路径清晰（加数据/改方法） |
| Oracle-σ 也没优势 | **即使有完美不确定性信息，CVaR 机制在这个 workload 上也无效**——关于方法本身的更强结论 |
| 两者都有优势 | σ̂ 虽然 R² 低，仍保留了足够的粗粒度信息 |
| 两者相对 FCFS+Predictor 有优势，但相对裸 FCFS 没有 | **调度有效，但被预测器开销吃掉了**——对实际部署很有价值的结论 |

> 关于阶段一记录的"阶段三应取 LMSYS index ≥ 10,000"约束：用 test split 在**防泄漏的实质意义上是等价的**（预测器从未见过这些 prompt），而且换来了 Oracle-σ 臂。若审阅时质疑，再用 index ≥ 10,000 重做（代价是 oracle 臂需要新的 Qwen3 采样）。

### 5.3 压测配置

用 vLLM 官方的 `benchmarks/benchmark_serving.py`（官方仓库自带，不需要 fork），关键参数：

- `--request-rate`：**扫多个到达率**。先用低速率测出饱和点，然后在饱和点两侧取点（阶段一的教训：全部落在过载区会导致指标饱和、看不出差异）
- `--num-prompts`：1,817（test split 全量）或其子集
- 自定义数据集：需要把 test split 的 prompt 导成 benchmark 脚本能吃的格式（`benchmark_serving.py` 支持 ShareGPT / sonnet / 自定义 jsonl，要确认具体格式）

#### 必须先确认队列真的在排队

见 1.3：**`waiting` 队列为空时，四个臂的行为完全相同**。所以正式压测前要做一次"饱和度标定"：

1. 从低速率开始逐步升高 `--request-rate`
2. 在 ② `FCFS+Predictor` 臂上观察 vLLM 日志里的 `Waiting: N reqs`（vLLM 每隔几秒打印一次运行/等待计数）
3. 选择**能让 `Waiting` 持续 > 0** 的速率作为实验区间的下界

> 用 ② 而不是 ① 来标定，是因为 ② 的显存/算力占用与主对比的两臂一致，标出来的饱和点才对得上。A100-80GB 上 KV cache 余量很大（可容纳的并发数远超 `max_num_seqs=32`），**饱和更可能由 `max_num_seqs` 而非显存触发**——这是好事，意味着瓶颈干净、可控。

> 如果在某个速率下 `Waiting` 长期为 0，那个数据点上"三臂无差异"是**预期中的正确结果**，不是 bug——报告里应当明确画出这一段，它本身说明了"调度只在饱和时才起作用"。

**指标**（`benchmark_serving.py` 自带）：TTFT（首 token 延迟）、TPOT（每 token 延迟）、端到端延迟的均值/中位数/P99、请求吞吐、token 吞吐。

> 注意：这套指标和仿真阶段用的（等待时间、Jain 公平性）**不完全对应**。真实系统里"等待时间"≈TTFT。公平性指标 benchmark 脚本不直接给，需要从逐请求的原始结果里自己算——`benchmark_serving.py` 支持 `--save-result` 导出每条请求的明细。

**额外要记录的**：抢占次数。1.3 提到 FCFS 和 TIE 对被抢占请求的处理语义不同（插队头 vs 按分数重排），如果压测中发生了抢占，这会直接影响 P99 和公平性的对比解释。vLLM 日志里有抢占计数，压测时一并抓下来。

---

## 6. 任务清单

| # | 任务 | 状态 | 需要 GPU | 产物 |
|---|---|---|---|---|
| 1 | `vllm_tie/score_calculator.py`：log-t 截断 E[X] / CVaR（照论文重写，非复制） | ✅ | 否 | 已测（`tests/test_vllm_tie.py`） |
| 2 | `vllm_tie/predictor.py`：接阶段二 checkpoint、自适应 β、chat template 剥离、oracle 查表 | ✅ | 否 | torch 惰性导入，核心逻辑可离线测 |
| 3 | `vllm_tie/request_queue.py`：`TIERequestQueue`（改编自 TIE）+ `FCFSWithPredictionQueue` | ✅ | 否 | worker 抽成共享基类 |
| 4 | `vllm_tie/scheduler.py`：`TIEScheduler` / `FCFSWithPredictorScheduler` | ✅ | 否 | 各 ~10 行，只换 `self.waiting` |
| 5 | `scripts/export_benchmark_dataset.py`：test split → jsonl + oracle CSV | ✅ | 否 | 1,817 prompt，**oracle 标签 100% 覆盖** |
| 6 | `scripts/rank_agreement_check.py`：CVaR 项到底改不改排序 | ✅ | 否 | **见 5.1.1 的发现** |
| 7 | `hpc/benchmark_serving.slurm`：`gpu:a100:1`，一 job 跑完所有臂 | ✅ | 否 | `ARMS` 可覆盖 |
| 8 | 装 `vllm==0.11.1`，跑通 3.4 的四项验证 | ⬜ | 否* | **下一步** |
| **P0 到此为止 ↑，下面进 GPU** | | | | |
| 9 | 冒烟：起 ② 确认服务正常；起 ③ 确认预测器加载、`popped_before_prediction` 不是全部 | ⬜ | 是 | 2 h |
| 10 | 饱和度标定：扫 request-rate 找 `Waiting > 0` 区间；定死五臂统一的 `--gpu-memory-utilization` | ⬜ | 是 | 2 h |
| 11 | **主实验：② / ⑤ / ③** × 多个 request-rate | ⬜ | 是 | 半天 |
| 12 | **中期分析**：`③−⑤`（CVaR 净贡献）和 `⑤−②`（SJF 收益）各是多少 | ⬜ | 否 | 半天 |
| **P1 补充臂 ↓** | | | | |
| 13 | 补跑 ① 和 ④（都是零新代码，改 `ARMS` 即可） | ⬜ | 是 | 半天 |
| 14 | 结果分析 + 画图 + 写报告章节 | ⬜ | 否 | 1 天 |

\* 任务 8 的前三项（纯 import 检查）在登录节点就能做；第四项（起服务）需要 GPU，可并入任务 9。

**为什么在 12 设中期分析点**：三臂的差值直接决定后续。

- `⑤ − ②` 显著、`③ − ⑤` 不显著 → **SJF 有效但 CVaR 项无贡献**，与 5.1.1 的排序计算吻合，这是一个完整的结论，④ 只用来确认"换成完美 σ 也一样"
- `③ − ⑤` 显著 → 不确定性感知确实起作用，④ 升级为必做，用来量化 σ 质量的天花板
- `⑤ ≈ ②` → 连 SJF 都没收益，先回头查饱和度和 `popped_before_prediction`，很可能是队列根本没排上队

> 相比改为外挂方案之前，原"在 HPC 上编译安装 vLLM fork（0.5–1 天，高失败风险）"被 `pip install` 取代；原"转 checkpoint 格式去迁就 TIE 的加载器"也不再需要——加载代码现在是我们自己的。

---

## 7. 风险清单

| 风险 | 影响 | 应对 |
|---|---|---|
| ~~vLLM fork 装不起来~~ | — | **已消除**：改用 `--scheduler-cls` 外挂 + 官方 wheel（第 3 节） |
| **`SchedulerInterface` 不是公开 API，版本漂移** | 子类挂不上去 | 锁死 `vllm==0.11.1`（与 TIE fork 同版本）；任务 1 的四项验证提前证伪 |
| **负载没压到饱和，四臂无差异** | 实验看不出东西 | 任务 9 饱和度标定，以 `Waiting > 0` 为准（1.3 / 5.3） |
| **预测器开销吃掉 TIE 收益** | TTFT 可能劣于裸 FCFS | 主对比是 `③ vs ②`（GPU 负载对等，不含此开销）；`② vs ①` 单独量化开销；四臂统一 `--gpu-memory-utilization` |
| **A100 排不到队** | 拖延 | 只要 1 张卡（4.1），比 2 卡容易得多；用 sbatch、一个 job 跑完所有臂；阶段一 `generate_labels.slurm` 已验证 `gpu:a100:1` 可用 |
| **预测器 batch 撑爆剩余显存** | 服务 OOM 崩溃 | `max_batch_size` 从默认 128 降到 16–32、预测器 fp16 加载（任务 2/4）；冒烟时用突发流量压一次 |
| **σ̂ 无信息量导致 TIE 无优势** | 结论可能是负面的 | Oracle-σ 臂（5.2）保证结论可解释；负面结果本身也是有价值的发现 |
| 移植 TIE 代码时漏填 `"xxx"` 占位符 | 运行时报错 | 任务 4 里全局搜 `"xxx"` 一次性处理 |
| ~~V100 不支持 bf16~~ | — | **已消除**：改用 A100，原生支持 bf16 |
| 抢占语义差异干扰 P99/公平性解读 | 结论归因错误 | 记录抢占计数（5.3）；若抢占频繁，在报告中区分"调度效果"与"抢占策略副作用"（1.3） |
| 论文自己的 σ R² 未知 | 无法判断我们是"复现失败"还是"成功复现一个已知难点" | **去 OpenReview 下载论文正文**（`I5IMkvVKd7`），查预测器指标表——这件事现在就能做，且能同时验证阶段二那个"每 prompt 采样 100 次"的未证实前提 |

---

## 8. 立刻可以做的两件事（都不需要 GPU）

1. **任务 1**：`pip install vllm==0.11.1`，跑 3.4 的前三项验证（纯 import 检查，本地就能做）。这是现在最大的技术不确定性，几十分钟就能清掉。
2. **下载论文正文**（OpenReview `I5IMkvVKd7`），核对两件事：他们报告的预测器 μ/σ 指标、以及每个 prompt 的采样次数。这直接决定阶段二结论怎么写。

---

## 附：本次修订（外挂方案）相对原计划的变化

| 项 | 原计划 | 现在 |
|---|---|---|
| 部署 | 复制 TIE 的 vLLM fork，源码编译或覆盖补丁 | `pip install vllm==0.11.1` + `--scheduler-cls` 外挂自己的类 |
| 代码量 | 整棵 vLLM 树进我们仓库 | `vllm_tie/` 四个文件 |
| checkpoint | 改名 `.pth`、包 `model_state_dict`、改 json 键名去迁就 TIE 的加载器 | 加载代码是我们自己的，直接读我们的格式 |
| 对照臂 | 3 臂，并列 | 4 臂，**分两批**：P0 做 `② vs ③`（主结论），P1 补 `①④` |
| GPU | 2 张 V100（TIE 臂需独立预测器卡） | **1 张 A100**，四臂同卡 |
| 头号风险 | "fork 装不起来" | 已消除，转为"负载是否压到饱和" |
