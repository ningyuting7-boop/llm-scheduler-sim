# AI Infra 岗面试手册

> 和 [`Interview_Deep_Dive.md`](Interview_Deep_Dive.md) 的区别：那份按项目阶段组织，重点在实验设计和研究结论；这份按 **infra 面试官关心的能力**组织，重点在 **vLLM 内部机制、GPU 性能分析、排队论、系统集成、生产调试**。
>
> **predictor 放在第 9 节，不是因为它不重要，而是因为它不该是开场。** 先讲系统，被问到再展开模型——顺序不同，对方的追问方向完全不同。
>
> 第 9 节本身是完整的，而且是按 **infra 角度**组织的：预测器是**服务路径上的一个模型**，涉及延迟预算、批处理、fp16、显存争抢、train/serve 一致性。这些和训练技巧一样是 infra 内容。
>
> 每个回答都给了**数字怎么算的**和**为什么要算这个**。追问部分是预演。

---

## 0. 开场

### 60 秒版

> 我做了一个 LLM 推理调度的项目：把一个基于输出长度预测的调度器挂进 vLLM，在 A100 上做端到端压测，搞清楚调度到底能改变什么、不能改变什么。
>
> **核心结论是：在并发被 `max_num_seqs` 约束的配置下，调度不能提高吞吐，只能重新分配等待。** 我测到中位排队时间降了 7.4 倍（24.4 秒 → 3.3 秒），但平均值几乎不动，因为总排队时间守恒。要真的提高吞吐，得让显存成为约束，让短请求能多塞几条进去。
>
> 过程中还发现解码阶段是带宽瓶颈：每个 step 要把 16.4 GB 权重完整读一遍，KV 只占 9%。所以 batch 32 时 GPU 在 75% 带宽利用率下只产出 2237 tok/s——**大约 7 倍的吞吐留在桌上没拿**。

**注意这段里一次都没提 DeBERTa。** 对方感兴趣会追问"预测哪来的"，那时再讲。

### 规模

| | |
|---|---|
| 服务模型 | Qwen3-8B bf16，单卡 A100-40GB |
| 引擎 | 官方 vLLM 0.11.1（PyPI wheel，未 fork） |
| 实验 | 5 个调度臂 × 6 个到达率 × 1000 条 = **30 轮，30,000 请求，0 失败** |
| Workload | LMSYS-Chat-1M，平均输入 102 token、输出 249 token |

---

## 1. vLLM 内部机制

### Q1.1 vLLM 的 scheduler 是怎么工作的？

```python
# vllm/v1/core/sched/scheduler.py
self.waiting = create_request_queue(self.policy)   # :188  等待队列
self.running: list[Request] = []                   # :190  运行批次
```

**关键点：调度策略只决定"谁先进 running batch"，不决定"谁先做完"。**

vLLM 用 continuous batching——`running` 里最多 `max_num_seqs` 条请求并发逐 token 解码，每个 step 各产出一个 token。请求一旦进入批次就不会被顺序影响，只会因为显存不足被抢占。

**为什么这点重要**：它直接决定了调度能影响什么。如果负载没把 `running` 压满，`waiting` 长期为空，**所有调度策略行为完全相同**。我实测过：rate 2 req/s 时峰值并发只有 19（上限 32），五个臂的 p50 TTFT 全在 26–32 ms，差异在噪声里。

<details><summary><b>追问：那和教科书里的 FCFS 有什么不同？</b></summary>

教科书 FCFS 是单服务台、先来者独占直到完成，队头阻塞很强。vLLM 的 FCFS 是**先来者先入场，入场后与后来者并发共享 GPU**，队头阻塞只在 32 个槽位占满时才发生。

所以仿真器上测出来的调度收益会**系统性高于真机**——我阶段一的仿真就是这样，真机上 gap 小得多。
</details>

<details><summary><b>追问：vLLM 官方支持哪些调度策略？</b></summary>

只有两种：

```python
# vllm/config/scheduler.py:23
SchedulerPolicy = Literal["fcfs", "priority"]
```

`priority` 是按外部传入的 priority 字段排，不是按预测长度。所以任何基于预测的调度都得自己实现。
</details>

### Q1.2 KV cache 是怎么管理的？你测到多少？

分页式（PagedAttention），按 block 分配，避免外部碎片。

vLLM 启动时会打印实际容量，我这边是：

```
GPU KV cache size: 394,544 tokens
Maximum concurrency for 8,192 tokens per request: 48.16x
```

**怎么验算**：Qwen3-8B 有 36 层、8 个 KV head（GQA）、head_dim 128、bf16：

$$2\,(K,V) \times 36 \times 8 \times 128 \times 2\text{B} = 147{,}456\text{ B} = \mathbf{144\ KB/token}$$

我的平均序列长度是 102(输入) + 249(输出) = **351 token**，所以：

$$\frac{394{,}544}{351} \approx \mathbf{1{,}124\ 条并发}$$

**但我把 `max_num_seqs` 设成了 32。实测 KV 利用率只有 2.0%（峰值 2.6%）。**

**为什么要算这个**：它决定了并发到底被什么卡住。vLLM 自己那行 `48.16x` 说的是"每条用满 8192 上下文时能装 48 条"——**48 > 32，所以无论请求多长，显存都轮不到成为约束**。这一条推翻了后面一连串结论（见 Q3.2）。

<details><summary><b>追问：你怎么确认没有发生抢占？</b></summary>

抓 server 日志：

```bash
grep -ci "preempt" results/bench_*/server_*.log   # 五个臂全是 0
```

和 KV 2% 的利用率互相印证——显存压力根本不存在。
</details>

### Q1.3 什么时候会发生抢占？语义是什么？

KV cache 不够给 running 的请求分配下一个 token 时：

```python
# scheduler.py:407
preempted_req = self.running.pop()          # LIFO：踢掉最后进入的
...
self.waiting.prepend_request(preempted_req) # 放回等待队列
```

**`prepend_request` 的语义因队列而异，这点容易被忽略：**

| 队列 | `prepend_request` 行为 | 后果 |
|---|---|---|
| `FCFSRequestQueue` | `appendleft` —— **插到队头** | 被抢占的请求优先恢复 |
| 我的 `TIERequestQueue` | 调 `add_request` —— **按分数重新入堆** | 被抢占的长请求可能**再次被推后** |

**为什么要注意**：这是一个会污染尾延迟归因的机制性差异。如果压测中抢占频繁，p99 的差异可能来自抢占策略而不是调度策略。所以我在压测脚本里专门统计抢占次数——确认是 0 之后，才能说 p99 的差异纯粹来自调度。

### Q1.4 chunked prefill 有什么影响？

日志里是 `Chunked prefill is enabled with max_num_batched_tokens=2048`。

prefill 和 decode **共享每个 step 的 token 预算**。一条长 prompt 的 prefill 会被切成多个 chunk，分散在多个 step 里，和 decode 交织执行。

**影响**：它让长 prompt 不会一次性阻塞整个批次（这是好事），但也意味着 **prefill 会挤占 decode 的吞吐**。我的 workload 平均输入 102 token，prefill 占比很小，所以这项不是瓶颈。

---

## 2. GPU 性能分析（roofline）

### Q2.1 你的服务吞吐是多少？瓶颈在哪？

实测天花板：**8.8–9.0 req/s，2,237 tok/s**。

**瓶颈是显存带宽，不是算力。** 推导：

解码阶段每个 step，GPU 必须把**整个模型权重读一遍**——与 batch 大小无关。

| 量 | 值 |
|---|---|
| Qwen3-8B bf16 权重 | 16.4 GB |
| A100-40GB HBM 带宽 | 1,555 GB/s |
| **理论最短 step 时间** | $16.4 / 1555 = \mathbf{10.5\ ms}$ |
| 实测 TPOT | **14 ms** |
| **带宽利用率** | $10.5/14 = \mathbf{75\%}$ |

**验算吞吐**（这步是关键，它证明模型没算错）：

$$\frac{\text{batch } 32}{14\text{ ms}} = 2{,}286 \text{ tok/s} \quad\text{vs}\quad \text{实测 } 2{,}237 \text{ tok/s} \qquad \textbf{误差 2\%}$$

**为什么要做这个计算**：不做的话你只知道"吞吐是 2237"，不知道这个数是好是坏、离上限多远、该往哪优化。做完之后你知道：GPU 在 75% 带宽利用率下运行，**但每读一次 16.4 GB 权重只产出 32 个 token**——SM 大部分时间在等内存。

<details><summary><b>追问：那 KV cache 的读取占多少？</b></summary>

每个 step 读取的总量：

| 读什么 | 大小 | 占比 |
|---|---|---|
| 模型权重（**与 batch 无关**） | 16.40 GB | **91%** |
| 32 条请求的 KV（351 token × 144 KB × 32） | 1.62 GB | 9% |
| 合计 | 18.02 GB | |

**权重占 91%**，这就是为什么增大 batch 几乎是"免费"的吞吐。
</details>

<details><summary><b>追问：那怎么提高吞吐？能提多少？</b></summary>

开大 `max_num_seqs`。step 时间 = (权重 + batch×KV) / 带宽：

| batch | 每 step 读取 | step 时间 | token 吞吐 | req/s |
|---|---|---|---|---|
| **32**（当前） | 18 GB | 14 ms | 2,237 | **8.8** |
| 128 | 23 GB | ~18 ms | ~7,100 | ~28 |
| 512 | 42 GB | ~33 ms | ~15,500 | **~62** |

**batch 开到 512，吞吐涨约 7 倍，而单条请求的 TPOT 只从 14 ms 涨到 33 ms。** 这就是 continuous batching 的核心收益，而我的配置完全没用上。

代价是 TPOT 翻倍——这是个明确的吞吐/延迟取舍，取决于 SLO。
</details>

<details><summary><b>追问：你直接测 GPU 利用率了吗？</b></summary>

**没有，这是个疏漏。** slurm 脚本只在开头跑了一次 `nvidia-smi`，没有全程采样。上面的 roofline 是推算，但用实测 TPOT 验证到了 2% 误差。

要补的话加一行后台采样：
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 5 > gpu.csv &
```

**而且要注意 `utilization.gpu` 常被误读**——它是"有 kernel 在跑的时间占比"，不是算力饱和度。解码时它可能显示 95%+ 而 SM 其实在等内存。`utilization.memory` 才是带宽占用。
</details>

### Q2.2 为什么 prefill 和 decode 的性能特征不同？

- **prefill**：一次处理整个 prompt 的所有 token，矩阵乘法是 $L \times d \times d$，**算力密集**（compute-bound）
- **decode**：每步只处理 1 个 token，矩阵乘法退化成矩阵×向量，**权重读取主导**（memory-bound）

所以 decode 的 step 时间几乎只取决于"读多少字节"，这正是上面 roofline 成立的前提。

这也解释了为什么 continuous batching 对 decode 收益巨大：**batch 翻倍，读取量几乎不变，产出翻倍。**

---

## 3. 排队论与容量规划

### Q3.1 负载和延迟的关系你怎么刻画？

用利用率 $\rho = \lambda / \mu$（发起速率 / 实际吞吐）：

| 发起 | 实际吞吐 | ρ | p50 TTFT | p99 TTFT |
|---|---|---|---|---|
| 2 | 1.99 | 1.00 | 32 ms | 77 ms |
| 4 | 3.96 | 1.01 | 31 ms | 78 ms |
| 6 | 5.90 | 1.02 | 33 ms | 264 ms |
| 8 | 7.81 | 1.02 | 50 ms | 1,408 ms |
| **16** | 8.74 | **1.83** | **24,382 ms** | 48,081 ms |
| 32 | 8.80 | 3.64 | 39,419 ms | 78,314 ms |

**ρ 跨过 1 的时候，TTFT 从毫秒级跳到秒级。** p90 从 rate 8 的 832 ms 跳到 rate 16 的 44,025 ms——50 倍。

**这不是异常，是队列从稳定变发散。** ρ>1 时队列线性增长，第 $i$ 条请求的等待：

$$W(i) = i\left(\frac{1}{\mu} - \frac{1}{\lambda}\right)$$

验算（FCFS 臂）：

| 负载 | 分位 | 公式预测 | 实测 | 误差 |
|---|---|---|---|---|
| rate 16 | p50 | 24,678 ms | 22,157 ms | +11% |
| | p90 | 44,421 ms | 41,475 ms | +7% |
| | p99 | 48,863 ms | 45,495 ms | +7% |
| rate 32 | p90 | 71,764 ms | 68,590 ms | +5% |
| | p99 | 78,940 ms | 75,672 ms | +4% |

**误差 4–11%。**

**为什么要算这个**：它把"p90 怎么突然变成 40 秒"从一个 bug 疑虑变成一个可预测的量。而且它告诉你**过载时 FCFS 的 TTFT 纯粹是到达序号的函数**——和请求本身无关，所以那个区间测的是"你第几个到"，不是"调度器好不好"。

### Q3.2 为什么你的吞吐五个臂完全一样？

**因为并发被配置卡死，不是被显存。** 完整的因果链，每一环都有实测：

```
max_num_seqs=32 钉死并发
  → 各臂 batch 都跑满 ~31.8/32（实测 31.60–31.89，跨臂差异 <1%）
  → KV 用量都是 2%，抢占 0 次
  → GPU 做同样的工作、同样的批大小
  → step 时间相同 → token 吞吐一致（2,132–2,237 tok/s，极差 4.2%）
  → 总 token 工作量固定 → duration 固定
  → 总排队时间守恒
```

**总排队时间 = 队列长度曲线下的面积**，只取决于到达过程和服务速率，**与服务顺序无关**。所以调度只能**重新分配**等待，不能减少。

实测吞吐对比：

| rate | 五臂极差 | 差异来自 |
|---|---|---|
| 2 / 4 / 6 / 8 | **0.0–0.1%** | 未饱和，吞吐=发起速率 |
| 16 / 32 | 3.9–4.2% | **全部来自预测器开销**，不是调度 |

拆开看：`FCFS+Predictor vs FCFS` 是 −2.2%，`SJF vs FCFS+Predictor` 是 −1.8%～+0.2%。

<details><summary><b>追问：那顺序最多能影响吞吐多少？</b></summary>

能算出上界。step 时间由读取量决定，权重占 91% 且与 batch 内容无关，**顺序只能影响那 9% 的 KV 部分**。

如果 SJF 让批内平均序列长度减半，KV 从 1.62 GB 降到 0.81 GB，step 时间减少 **4.5%**。

**这就是这个配置下的理论天花板：约 4.5%。** 实测是 −2%（预测器开销盖过了它）。
</details>

### Q3.3 那调度到底改变了什么？

**分布，不是总量。** rate 16（ρ=1.83），相对负载对等的基准：

| 臂 | p50 TTFT | p90 | p99 | **mean** |
|---|---|---|---|---|
| FCFS + Predictor | 24,382 ms | 44,025 | 48,081 | 24,514 |
| Predicted-SJF | **3,303 ms（−86.5%）** | +65.7% | +103.6% | **+4.3%** |

**中位排队时间降低 7.4 倍，平均值几乎不动。** 省下的 21 秒被搬到了尾部。

> **方法论教训**：我的分析脚本第一版只打印 mean 和 p99，**把 7.4 倍的改善报告成了"无效果"（+4.3%）**。补上 p50 才发现主效应。在守恒的系统里，mean 是唯一不会动的那个统计量。

### Q3.4 这个收益在所有负载下都成立吗？

不。我推出一条规律并验证了：

过载时，一个到达窗口内最多只有 $\mu/\lambda = 1/\rho$ 比例的请求能"即到即走"。SJF 的收益正来自让这批短请求插队，其余的无论怎么排都得在积压里等。所以：

> **SJF 只能改善低于 $p^\* = 100/\rho$ 的分位数，高于它的必然恶化。**

| 负载 | ρ | $p^\*$ | p50 | p90 | p99 |
|---|---|---|---|---|---|
| rate 8 | 1.02 | 98 | −2.9% ✓ | −24.5% ✓ | +111.1% ✓ |
| rate 16 | 1.83 | 55 | −86.5% ✓ | +65.7% ✓ | +103.6% ✓ |
| rate 32 | 3.64 | 27 | −3.6% ✗ | +15.5% ✓ | +28.0% ✓ |

**9 个预测中 8 个成立**（唯一不符的那格预测恶化、实测持平，在噪声量级）。

**这条规律的实用价值**：它告诉你 SJF 类调度有一个**最优负载窗口**。ρ=1.83 时效果最戏剧化（$p^\*≈55$ 正好落在中位数）；ρ=3.64 时只有最快的 27% 能受益，中位数已经在积压里了。

**不是越过载越好。**

<details><summary><b>追问：那实际部署该怎么用这条规律？</b></summary>

反过来用：**给定 SLO，算出能跑到多高的 ρ**。

比如 SLO 是"p90 TTFT < 1 秒"：
- 看表，rate 8（ρ=1.02）时 p90 是 832 ms，刚好满足
- rate 16 时 p90 是 44 秒，远超

所以这个配置下的可用容量上限是 **~8 req/s**，而不是吞吐天花板的 8.8。**SLO 决定容量，不是吞吐决定容量。**

而且 $p^\*=100/\rho$ 告诉你：如果 SLO 关心的是 p90，那只有 ρ < 1.11 时 SJF 才帮得上忙（$100/1.11 = 90$）。
</details>

---

## 4. 系统集成

### Q4.1 你怎么把自己的调度器接进 vLLM 的？

**没有 fork。** vLLM 官方自带可插拔调度器：

```python
# vllm/config/scheduler.py:129
scheduler_cls: str | type[object] = Field(default=None)
```

命令行 `--scheduler-cls mod.MyScheduler`，vLLM 用 `resolve_obj_by_qualname` 动态导入。

`SchedulerInterface` 有 13 个抽象方法（`schedule`、`update_from_output`、KV cache、chunked prefill、抢占……），从零实现不现实。**但不需要——继承官方 `Scheduler`，只替换 `self.waiting` 一个成员**：

```python
class TIEScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)      # 官方逻辑原样跑完
        self.waiting = TIERequestQueue(...)    # 唯一的改动
```

**为什么这样做**：参考实现是改 vLLM 源码再整树发布。照搬要从源码编译 CUDA kernel，在共享集群上耗时长、对版本敏感、容易失败。外挂方案用 `pip install vllm==0.11.1` 官方预编译 wheel，**不编译**。

**代价**：vLLM 自己警告 `SchedulerInterface` 不是公开 API（加载时会打印兼容性提示），所以版本必须锁死。

<details><summary><b>追问：你怎么确认这个方案可行，而不是跑到一半才发现接口不对？</b></summary>

写了一个纯 CPU 的关口检查脚本（`scripts/verify_vllm_integration.py`），18 项，在花任何 GPU 时间之前跑：

1. vLLM 版本 == 0.11.1
2. `Scheduler` / `RequestQueue` / `Request` 能从官方包 import
3. **`Scheduler.__init__` 里确实有 `self.waiting = ...`**（我们要覆盖的那个点）
4. 我的两个队列类没有未实现的 `RequestQueue` 抽象方法
5. 我的调度器类满足 `SchedulerInterface` 且 `__abstractmethods__` 为空
6. `resolve_obj_by_qualname("vllm_tie.scheduler.TIEScheduler")` 能解析

**第 4、5 条尤其重要**：未实现的抽象方法只会在 serve 时才暴露，那时已经跑到压测一半了。
</details>

<details><summary><b>追问：预测器加载在哪一步？有什么约束？</b></summary>

这是个容易踩的坑。看 vLLM 的引擎初始化顺序：

```python
# vllm/v1/engine/core.py
109:  num_gpu_blocks, ... = self._initialize_kv_caches(...)   # 先按 util 分配 KV cache
120:  Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
133:  self.scheduler = Scheduler(...)                          # 预测器在这里才加载
```

**KV cache 先分配、调度器后构造。** 所以我的 DeBERTa **不被 `--gpu-memory-utilization` 统计**，只能挤进剩下的 `(1 − util)`。

A100-40GB、util=0.88 → 剩 4.9 GB。因此必须：
- 预测器 **fp16 加载**（权重减半）
- **`max_batch_size` 从参考实现的 128 降到 32**（128 条 × 512 token 的 DeBERTa forward 会炸掉剩余显存）
</details>

### Q4.2 预测在调度的关键路径上吗？

**不在。** 这是个常见误解。

```python
def add_request(self, request):
    initial_score = 2048.0                           # 占位分，立即入堆
    heapq.heappush(self._heap, (initial_score, ...))
    self._prediction_queue.put(request)              # 丢给后台线程，立即返回
```

后台 daemon 线程攒批（8 条或 3 ms 触发）→ DeBERTa forward → 蒙特卡洛算 E[X]/CVaR → 通过版本号惰性更新堆。**调度主循环从不阻塞等待预测。**

实测预测批延迟 **44.8–48.3 ms/批**。

<details><summary><b>追问：那预测晚到会怎样？你怎么知道它没晚到？</b></summary>

**这是最危险的静默失败模式**：如果预测总是在请求被调度之后才到，队列实质在跑 FCFS，但**延迟数据看起来完全正常**，你会以为测到了调度效果。

所以我在队列里加了计数器 `popped_before_prediction`——被 pop 时还没拿到分数的请求数。分阶段统计：

| popped 区间 | 未打分占比 | 队深 |
|---|---|---|
| 0–2,115（rate 2/4/6） | **100%** | 0 |
| 3,727–4,243 | 38% | 188 |
| **4,243–5,836（rate 16/32）** | **0–6%** | 170–708 |

低负载 100% 未打分**不是问题**——队列是空的，根本没有调度决策可做。队列一深，几乎所有请求都及时拿到了分数。

冒烟测试脚本里把这个比例 ≥90% 设成 FAIL、≥40% 设成 WARN。
</details>

---

## 5. 并发数据结构

### Q5.1 等待队列用什么数据结构？复杂度？

**最小堆 + 惰性删除（版本号）**：

```
堆元素 = (effective_score, arrival_time, version, req_id, request)
```

| 操作 | 复杂度 | 实现 |
|---|---|---|
| `peek` | **O(1) 摊还** | 从堆顶丢弃过期项直到遇到有效项 |
| `push` | O(log n) | 标准 heappush |
| `pop` | O(log n) 摊还 | 同 peek |
| **`update`** | **O(log n)** | 压入新版本号的新项，旧项自动失效 |
| **`remove`** | **O(1)** | 只删版本号记录，堆里的项留给后续 pop/peek 清理 |

**为什么需要惰性删除**：打分是异步的，分数会后到。如果用"找到旧元素并修改"的方式更新，需要 O(n) 查找或维护额外的索引结构（还要和堆同步）。**版本号把"更新"退化成一次 push**：

```python
def _push_updated_score(self, req_id, new_base):
    version = self._versions[req_id] + 1    # 旧版本的堆项从此作废
    self._versions[req_id] = version
    heapq.heappush(self._heap, (effective, arrival_time, version, req_id, request))
```

pop/peek 时比对版本号，不匹配就丢弃。

<details><summary><b>追问：堆里会不会无限堆积垃圾？</b></summary>

不会，有两个清理路径：

1. **pop/peek 时顺手丢弃**堆顶的过期项
2. **防饥饿重建**：每 5 秒全量重建堆，所有版本号 +1，顺带把惰性删除积累的过期项一次清空

重建是 O(n)，但**跑在后台线程上，不在每请求的调度路径上**。
</details>

<details><summary><b>追问：为什么堆元素里要带 arrival_time？</b></summary>

作为第二排序键，保证**分数相同时退化为 FCFS**。

这在刚启动时很关键——所有请求都还是占位分 2048，如果不带 arrival_time，堆的出队顺序就是任意的（Python 的 heapq 会去比较第三个元素）。带上之后，未打分的请求之间是严格 FCFS，这也是正确的默认行为。
</details>

### Q5.2 防饥饿是怎么做的？

**乘性时间衰减**：

$$\text{effective} = \text{base} \times \gamma^{t_w/\tau}, \qquad \gamma = 0.9,\ \tau = 30\text{s}$$

分数越低越先调度，所以等得越久分数越低、越往队头走。

| 等待时间 | 分数系数 |
|---|---|
| 30 s | ×0.90 |
| 60 s | ×0.81 |
| 300 s | ×0.35 |

**和减性老化的区别**：减性是 `score -= α × 等待时间`（α 单位 tokens/s），对所有请求一视同仁；乘性是**按比例**，分数 2000 的等 30 秒减 200，分数 100 的只减 10——**长请求获得的绝对补偿更大**。

这会影响公平性的解读，两种做法在报告里要说清楚用的是哪个。

### Q5.3 线程安全怎么保证？

一把 `threading.RLock` 保护堆和三个 dict（`_versions` / `_base_scores` / `_request_info`）。

**后台预测线程和调度主线程的交互点只有两个**：
1. 线程通过 `queue.Queue` 拿任务（本身线程安全）
2. 算完后拿锁更新堆

**GIL 的影响有限**：PyTorch 的 CUDA 调用和 Rust 实现的 fast tokenizer 都会释放 GIL，所以预测线程不会长时间霸占解释器。

---

## 6. 实验设计与混杂控制

### Q6.1 你怎么确保测的是调度而不是别的？

**五个对照臂，每个相对相邻臂只改变一个变量：**

| 臂 | 相对前一臂改了什么 |
|---|---|
| ① FCFS | — |
| ② FCFS + Predictor | **加载并运行预测器，但丢弃结果** |
| ③ Predicted-SJF | **队列按 E[X] 排序** |
| ④ TIE | **加上 CVaR 风险项** |
| ⑤ TIE-oracle | **参数换成真实拟合值** |

| 对比 | 测出什么 |
|---|---|
| ② − ① | **预测器的部署开销** |
| ③ − ② | **调度排序的收益**（GPU 负载完全对等） |
| ④ − ③ | 风险项的净贡献 |
| ⑤ − ④ | 预测质量的天花板 |

**② 是关键的一臂**，而且参考实现没有。它让 GPU 负载和 ③④⑤ 完全对等——都加载 DeBERTa、都跑预测、都占同样显存，**唯一差别是队列用不用结果**。这样 `③ − ②` 就是纯调度收益，**单卡部署不引入任何偏差**，省掉了争取第二张 GPU 的排队风险。

**其他全部对齐**：同一模型、同一套 `--max-num-seqs 32 --max-model-len 8192 --no-enable-prefix-caching --gpu-memory-utilization`、同一份 workload、**同一个 `--seed`**（压测客户端会按 seed shuffle prompt 顺序，各臂必须一致）。

<details><summary><b>追问：为什么 --gpu-memory-utilization 必须所有臂一样？</b></summary>

因为它决定 KV cache 容量，而 KV 容量影响并发能力和抢占频率。

**如果只给带预测器的臂调低（为了腾显存），那几个臂的 KV cache 就比基准小，对比直接失效**——你分不清延迟差异是来自调度还是来自 KV 容量。

所以取值要以"带预测器的臂能稳定跑起来的最大值"为准，然后**所有臂统一用这个值**。
</details>

### Q6.2 实验前你做了什么准备？

两件，都在花 GPU 时间之前：

**1. 关口检查**（Q4.1）——18 项纯 CPU 验证，确认集成方案可行。

**2. 离线排序分析**——用标签直接算"新策略和基线排出来的队有多大差别"：

```
β=0.1:  Spearman 0.99957,  队头 top-32 重叠 96.9%
β=0.5:  Spearman 0.99625,  队头 top-32 重叠 93.8%
```

top-32 正好是一批要调度的量，只有 1–2 个位置不同。**这十分钟的计算告诉我：必须加一个中间臂（Predicted-SJF），否则测出来有差异也分不清来源。**

**真机结果和这个离线预测完全一致。**

### Q6.3 怎么知道负载真的压饱和了？

三个独立信号：

1. **`Peak concurrent requests`**：rate 2 时只有 19（上限 32）→ 未饱和
2. **p99 TTFT 的量级**：rate 2/4/6 都在 77–264 ms → 没排队；rate 8 跳到 1,408 ms → 开始排队
3. **vLLM 日志的 `Waiting: N reqs`**：饱和期均值 237–248

**低负载下五臂数字相同是对照组，不是空结果**——它证明了"队列空时调度无效"，是要画进报告的边界条件。

---

## 7. 生产调试（行为面试弹药）

### Q7.1 讲一个你排查过的难缠问题

**代理让健康检查永远不通。**

压测脚本等服务启动，轮询 120 次全部失败，10 分钟后放弃。**但服务一直是好的**——日志里 `Application startup complete` 和调度器的心跳都在。

**根因**：集群设了 `http_proxy=10.99.0.130:3128` 但没设 `no_proxy`。curl 把 `localhost:8000` 也往代理上送，而代理机到不了计算节点的 loopback。

**我自己的两个错误放大了它**：

```bash
if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
```

1. `2>&1 >/dev/null` 把 curl 的报错全吞了——"代理超时"/"curl 不存在"/"服务真没起来"三种完全不同的情况，表现一模一样
2. 依赖了计算节点上不保证存在的 curl

**更要紧的是**：`vllm/benchmarks/serve.py:532` 用 `aiohttp.ClientSession(trust_env=True)`，**压测客户端也会走代理**。只修健康检查的话，压测请求会在另一个地方失败，更难定位。

**修法**：导出 `no_proxy`、探活换成 Python + 空 `ProxyHandler`（不依赖 curl）、失败时打印完整诊断（启动是否完成、代理变量、端口监听、路由列表、原始报错）。

### Q7.2 讲一个最危险的 bug

**僵尸服务冒充我们的服务。**

```
OSError: [Errno 98] Address already in use
```

端口 8000 被占，我们的 server 一启动就死。**但健康检查过了**——因为占端口的那个服务回了 200。

脚本于是报告 "server up"，压测客户端打到一个 engine 已死的进程上，每个速率空等 600 秒。

**根本错误在检查顺序**：

```python
for _ in range(180):
    if health_ok(): return 0          # 僵尸答应了
    if not process_alive(): ...       # 永远走不到这里
```

**先查健康再查进程存活。**

**为什么说它比崩溃危险**：即使没有端口冲突，只要我们的服务因任何原因启动失败、而节点上恰好有别的服务在那个端口，**实验就会静默地跑在错误的目标上，而数据看起来完全正常**。崩溃至少会告诉你出事了。

**修法**：端口改成 `8000 + job_id % 1000`、启动前预检端口、**先查进程存活再查健康**、清理时按命令行 `pkill` 兜底（`vllm serve` 的 engine 是子进程，杀父进程会留下孤儿——这个僵尸就是上一次冒烟测试自己制造的）。

### Q7.3 还有没有别的？

**默认配置悄悄删掉了数据。**

`vllm bench serve --save-result` 看起来保存了结果，但：

```python
# vllm/benchmarks/serve.py:1489
if not args.save_detailed:
    for field in ["input_lens", "output_lens", "ttfts", "itls", ...]:
        del result_json[field]
```

**删除发生在写文件之前。** 所以不加 `--save-detailed` 就只有聚合分位数，拿不到逐请求数组——导致我算不了需要逐请求数据的指标。

**教训**：用别人的工具收集数据时，要确认它**实际写进文件的是什么**，而不是它看起来收集了什么。

---

## 8. 开放设计题

### Q8.1 如果让你优化这个服务，你会先做什么？

按预期收益排：

| 优先级 | 动作 | 预期收益 | 代价 |
|---|---|---|---|
| **1** | **`max_num_seqs` 32 → 512** | 吞吐 **~7×**（2,237 → ~15,500 tok/s） | TPOT 14 → 33 ms |
| 2 | 开 prefix caching | 取决于 workload 的前缀重复率 | 显存 |
| 3 | 量化（FP8/INT8） | 权重减半 → step 时间减半 → 吞吐翻倍 | 精度 |
| 4 | 调度优化（本项目） | **吞吐 0**，p50 TTFT −86% | 预测器 −2% 吞吐 |

**把调度排在最后是有意的**：我自己的实验证明了它不改变吞吐。**在 batch 只开到容量 2.8% 的情况下谈调度优化，是在优化一个不是瓶颈的东西。**

<details><summary><b>追问：那什么情况下调度才值得做？</b></summary>

两个条件：

1. **已经饱和**——ρ 接近或超过 1，队列真的有积压
2. **SLO 关心的分位数低于 $p^\* = 100/\rho$**——否则调度反而让它更差

另外，如果并发是**显存约束**而不是配置约束，调度还能通过"短请求占更少 KV → 装下更多条"提高吞吐。我的配置下这条路被堵死了（KV 只用 2%），所以测不出来。
</details>

### Q8.2 怎么给这个服务定容量？

**不是看吞吐天花板，是看 SLO。**

吞吐天花板是 8.8 req/s，但：

| SLO | 可用容量 |
|---|---|
| p90 TTFT < 1 s | **~8 req/s**（rate 8 时 p90 = 832 ms） |
| p90 TTFT < 100 ms | **~6 req/s**（rate 6 时 p90 = 53 ms，rate 8 已经 832 ms） |
| 无延迟要求（批处理） | 8.8 req/s |

**ρ 接近 1 时延迟急剧恶化**，所以生产上通常留 20–30% 余量，实际跑在 ρ≈0.7–0.8。

而且 $p^\*=100/\rho$ 告诉你：**如果 SLO 关心 p90，那只有 ρ < 1.11 时 SJF 才帮得上忙。**

---

## 9. 预测器：服务路径上的一个模型

**不要用这节开场**，但被问到就完整讲。这节按 infra 角度组织——训练技巧在 [`Interview_Deep_Dive.md`](Interview_Deep_Dive.md) 第 4 节。

### Q9.1 这个模型在服务路径上的成本是多少？

**DeBERTa-v3-base，184M 参数，和 Qwen3-8B 共享同一张 A100-40GB。**

| 项 | 值 | 怎么测的 |
|---|---|---|
| 预测批延迟 | **44.8–48.3 ms** | 队列的 `avg_batch_predict` 计数器 |
| TPOT 开销 | **+2~5%** | `FCFS+Predictor` 相对裸 `FCFS` |
| 吞吐开销 | **−2.2%**（高负载） | 同上 |
| 显存 | fp16 权重 ~0.37 GB + 激活 | 挤在 `(1 − util)` 的 4.9 GB 里 |

**这是论文没报告的数字**，而它正是我加 `FCFS+Predictor` 对照臂的目的（Q6.1）：**没有这一臂，你分不清延迟变化是调度带来的还是预测器开销带来的。**

**三个让它能共卡的工程决定：**

1. **fp16 加载**——权重减半。因为 vLLM 在构造 scheduler **之前**就分配完了 KV cache（Q4.1 追问），预测器只能用剩余显存
2. **`max_batch_size` 从参考实现的 128 降到 32**——128 条 × 512 token 的 DeBERTa forward 会炸掉那 4.9 GB
3. **异步、不在关键路径上**（Q4.2）——请求先拿占位分入堆，后台线程算完再惰性更新

<details><summary><b>追问：45 ms 的预测延迟会不会太慢？</b></summary>

不会，因为**它不在关键路径上**。请求入队时拿常数占位分立即返回，预测在 daemon 线程里跑。

真正要确认的是**预测能不能跟上到达率**。攒批参数是"8 条或 3 ms 触发"，所以稳态下每批约 8–32 条、45 ms 一批 → **约 180–700 预测/秒**，而我的最高到达率是 32 req/s。余量充足。

实测佐证：高负载下 `popped_before_prediction` 只有 0–6%（Q4.2 追问）。
</details>

<details><summary><b>追问：为什么不把预测器放到另一张卡上？</b></summary>

参考实现就是这么做的（`start-server.sh` 里 `TP_SIZE = GPU数 − 1`，留最后一张给预测器）。

我没这么做，因为 **`FCFS+Predictor` 对照臂已经把干扰对称化了**——两个臂都加载预测器、都跑预测，`调度臂 − 对照臂` 这个差值里不含预测器开销。单卡不影响核心对比的有效性，而且省掉了申请 2 张卡的排队风险。

代价是：`调度臂 vs 裸 FCFS` 这个"对外宣称的收益"会偏保守。所以我两个基准都报。
</details>

### Q9.2 训练和服务的输入一致吗？怎么保证？

**这是个真实的 train/serve skew，而且不处理不会报错。**

训练时用的是 LMSYS 的**裸 prompt 文本**。但服务时，压测客户端会套 Qwen3 的 chat template：

```python
tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                              add_generation_prompt=True, tokenize=False)
```

所以调度器拿到 `request.prompt_token_ids` 解码出来的是**模板化后的文本**：

```
[system
{system}
]user
{prompt}
assistant
[<think>

</think>

]
```

直接喂给 DeBERTa 就是在给模板样板文字打分，**而且 forward 不会失败，你拿到的是一个看起来正常的数字**。

**处理**：在预测器里把模板剥掉，锚定在**第一个 `user` 标记之后**和**最后一个 `assistant` 标记之前**。

<details><summary><b>追问：为什么是"最后一个" assistant 标记？</b></summary>

因为 prompt 本身可能包含 "assistant" 这个词——LMSYS 里有不少讨论聊天记录的 prompt。

锚定第一个 `
assistant` 会把这种 prompt 截断：

```
user
Rewrite this:
assistant
said hello
please make it formal
assistant

         ^^^^^^^^^^^^^^^ 锚定第一个就只剩这里
```

参考实现用的是正则 first-match，有这个问题。我写了单元测试覆盖这个 case。
</details>

<details><summary><b>追问：为什么不干脆在压测时关掉 chat template？</b></summary>

因为那会破坏**生成条件的一致性**。我的标签是用 chat template 采样 Qwen3-8B 得到的，如果服务时不套模板，输出长度分布就和标签描述的不是同一件事了。

**所以正确的做法是：服务时套模板（和打标签一致），预测器内部剥模板（和训练一致）。** 两边都对齐，而不是改一边去迁就另一边。
</details>

### Q9.3 训练时显存不够怎么解决的？

**OOM 了两次，第二次才是重点。**

**第一次**：`padding="max_length"`，每条 prompt 补到 512，而**中位 prompt 只有 25 个 token**。注意力是 $O(\text{seq}^2)$，代价是 $(512/25)^2 \approx 420$ 倍。

算一下 DeBERTa-v3-base（12 层、12 头、batch 32、seq 512、fp32）：

$$32 \times 12 \times 512^2 \times 4\text{B} = 403\text{ MB/层}$$

而 **DeBERTa 的 disentangled attention 要算三项**（content-to-content、content-to-position、position-to-content），每层保留 4–5 个这样的张量用于反向：

$$403\text{ MB} \times 5 \times 12\text{ 层} \approx \mathbf{24\ GB}$$

加上权重 0.74 + 梯度 0.74 + AdamW 两个动量 1.5 + 隐状态 ≈ **29–30 GB**。报错栈停在 `disentangled_attention_bias` → `p2c_att = torch.gather(...)`，正是这些大张量生成的位置。

**改成动态 padding 后又 OOM 了，报错一模一样。** 这才是重点：

> **动态 padding 降低的是平均显存，不是最坏情况。** 一个 batch 里只要有一条 512 token 的 prompt，整批还是要补到 512。而 **OOM 是由最坏情况决定的事件**。
>
> 更糟的是 sampler 的 seed 是固定的，所以**同一个倒霉 batch 在同一步确定性地复现**。

**最终解法**：`--batch-size 8 --grad-accum-steps 4`，等效 batch 仍是 32，但显存里一次只有 8 条，注意力张量降到 ~3.6 GB。

**为什么结果不变**，两个前提：

```python
(loss / grad_accum_steps).backward()        # 否则累积出的是 4 倍的和
```

而且模型用的是 **LayerNorm 不是 BatchNorm**——LayerNorm 对每个样本独立归一化，所以拆成 4×8 再累积梯度**数学上完全等价**。BatchNorm 就不等价了。

还有一个容易写错的点：**梯度裁剪必须在累积完成后做**，不能每个 micro-batch 各裁一次。

<details><summary><b>追问：为什么不用 AMP 或 gradient checkpointing？</b></summary>

都能解决，但引入新变量（AMP 的数值稳定性、checkpointing 的重算开销）。

**梯度累积是数学等价的，不改变任何训练动态**——我当时正在排查 σ 学不出来的原因，**最不需要的就是再引入一个可能影响结果的变量**。

另外我把 tokenize 从 `__getitem__` 挪到 `collate_fn` 时，**验证了改动前后 loss 值逐位相同**。
</details>

<details><summary><b>追问：为什么不用 length-grouped batching？</b></summary>

它更优——把长度相近的分到一批能同时降低平均和最坏情况。

我没用是因为它**破坏随机性**：同一批里的样本变得相关，对小数据集（9,081 条唯一 prompt）的泛化有影响。梯度累积没这个副作用。
</details>

### Q9.4 结果怎么样？σ 学不出来你怎么排查的？

| 目标 | test R² |
|---|---|
| μ（期望长度） | **0.7668** |
| σ（不确定性） | **0.0501** |

**σ 基本学不出来。我系统性排除了五个解释：**

| 假设 | 怎么测的 | 结果 |
|---|---|---|
| 标签是随机噪声 | **分半信度 + Spearman-Brown 校正** | ❌ σ 的 $r^2$=0.6031±0.0285，校正后 **0.8740**。天花板远高于 0.05 |
| 标签因采样不足被压缩 | **子采样重拟合**（20 个样本抽 5/10/15 重新拟合） | ❌ $\sigma(n)=\sigma_\infty-c/n$ 拟合到四位小数，**n=20 已达渐近值 96.5%**，加到 100 次只涨 2.9%；而且 `Spearman(σ₁₀,σ₂₀)=0.9371`，排序几乎不变 |
| 损失权重失衡 | σ 权重 1.0 / 2.0 / 3.0→4.0 + 梯度裁剪 | ❌ 无改善，2.0 反而更差（R²=0.014） |
| 过拟合 | 第 2 epoch 冻结 encoder，lr→5e-5 | ❌ 机制上生效（可训参数降到 1.38M，val loss 1.3021 vs 1.3602），σ 不变 |
| 多任务干扰 | `mu_weight=0` 单训 σ 头 | ❌ R²=0.0316 |

**剩下最可能的解释：信号本身不在 prompt 文本里。** 输出长度的方差很大一部分来自**解码的随机性**——同一条 prompt 采样 20 次得到 20 个不同长度，这部分方差无论什么模型都无法从文本预测。μ 可学，σ 可能本质上就是个噪声量。

**这个结论对调度是有用的**，而不只是一个失败：我用 oracle 参数单独跑了一臂（Q6.1 的 ⑤），结果和预测参数一样，**说明调度的瓶颈不在预测精度**。

<details><summary><b>追问：分半信度具体怎么做的？</b></summary>

把每条 prompt 的 20 个样本随机劈成两半各 10 个，分别拟合 σ，然后算两边的相关。重复多次取均值和标准差。

$r^2 = 0.6031$ 是**半长度**（10 个样本）的信度，用 Spearman-Brown 公式校正到全长度（20 个）：

$$r_{\text{full}} = \frac{2r}{1+r} \;\Rightarrow\; 0.8740$$

**但我后来发现自己对这个测试的解读有漏洞**：信度测的是**一致性**，不是**准确性**。如果两半都因为抽不到尾部而偏低，它们会一致地偏低，信度照样高。

所以它只排除了"随机噪声"这一种失败模式，**没排除"系统性压缩"**——那需要子采样重拟合才能测，就是上表第二行。
</details>

<details><summary><b>追问：训练数据怎么切分的？</b></summary>

**按唯一 prompt 文本分组切分，不是按行。**

数据集 9,998 条里有 **917 条重复 prompt**（唯一 9,081 条，重复率 9.2%）。按行切分会让同一条 prompt 同时落进 train 和 test——典型的泄漏。

参考实现用的是 `train_test_split(df, test_size=0.4)`，**按行切且不去重**。而且它的 `RANDOM_SEED = random.randint(0, 10000)` 每次运行都不同，单次结果无法复现。
</details>

### Q9.5 标签是怎么来的？

**阶段一自建的管线**（参考实现只发布了训练代码，采样和拟合没有）：

```
LMSYS-Chat-1M prompt[0:10000]
  → Qwen3-8B 每条采样 20 次（官方非思考模式参数）
  → MLE 拟合 log-t(μ, σ, ν=3.5)
  → 9,998 条 (prompt, μ, σ)
```

**这里有个值得讲的 bug**——退化检测写了但从来没生效过：

```python
if np.std(np.log(lengths)) == 0.0:     # 永远不成立
    return degenerate_fit(...)
```

`np.std` 对一组**数值完全相同**的浮点数返回的是 **~4.44e-16，不是精确的 0.0**（浮点累加误差）。所以这个分支一次都没进去，**9,998 条里 1,228 条（12.3%）走错了路径**。

排查时还发现第二个独立问题：另有 **214 条（2.1%）**采样确实有分散度，但拟合出的 σ 接近 0（最糟的一条 `raw_std=0.2091 → 7.59e-06`）。这是**重尾分布 MLE 的病态**——t 分布可以用极小的 σ 配合极端 t 值来"解释"离群点。

**三部分修复**：阈值改成 `< 1e-8`、给 L-BFGS-B 加 `log(σ) ≥ log(0.01)` 的下界、事后用矩估计 $\sigma \approx s\sqrt{(\nu-2)/\nu}$ 兜底。重新拟合改变了 **1,561 条**。

同样的 `== 0.0` 还出现在另一个脚本里，一并修了。

**为什么这个 bug 值得讲**：σ=0 意味着 CVaR = E[X]，**风险项完全退化**。不修的话 12% 的标签是错的，而且模型会学到"很多 prompt 的不确定性是 0"这个假事实。而它**不会报错、不会崩溃**，只会静默地产出错的标签。

## 10. 数字速查

| 类别 | | |
|---|---|---|
| **硬件** | A100-**40GB**，HBM 1,555 GB/s | |
| | Qwen3-8B bf16 权重 | 16.4 GB |
| **配置** | `max_num_seqs` | **32**（官方默认 128） |
| | `max_model_len` / `max_num_batched_tokens` | 8192 / 2048 |
| | `gpu-memory-utilization` | 0.88 |
| **KV cache** | 容量 | **394,544 token** |
| | 平均序列长度（输入 102 + 输出 249） | 351 token |
| | 理论可容纳并发 | **1,124 条** |
| | **实测利用率** | **2.0%**（峰值 2.6%） |
| | 每 token KV（36 层 × 2 × 8 头 × 128 × 2B） | **144 KB** |
| **吞吐** | 天花板 | **8.8–9.0 req/s，2,237 tok/s** |
| | TPOT | **14 ms** |
| | roofline 下界（16.4 GB ÷ 1555 GB/s） | **10.5 ms** |
| | **带宽利用率** | **75%** |
| | 验算误差（32÷14ms vs 实测） | **2%** |
| | 权重占每 step 读取量 | **91%** |
| **批次** | Running 占用 | 31.60–31.89 / 32 |
| | 饱和期 Waiting 均值 | 237–248 |
| | **抢占次数** | **0**（全部五臂） |
| **结果** | rate 16 p50 TTFT | **24,382 → 3,303 ms（−86.5%）** |
| | 同条件 mean TTFT | +4.3%（**守恒**） |
| | 同条件 p99 TTFT | +103.6% |
| | 预测器开销 | TPOT **+2~5%**，吞吐 **−2%** |
| | 预测批延迟 | 44.8–48.3 ms |
| | 顺序对吞吐的理论上界 | **4.5%** |
| **排队** | ρ（rate 8 / 16 / 32） | 1.02 / 1.83 / 3.64 |
| | $W(i)$ 公式误差 | **4–11%** |
| | $p^\*=100/\rho$ 命中 | **8 / 9** |
| **规模** | 压测 | 30 轮，30,000 请求，**0 失败** |
| **预测器** | DeBERTa-v3-base | 184M 参数，fp16 |
| | 预测批延迟 | 44.8–48.3 ms |
| | `max_batch_size`（参考实现 128） | **32** |
| | μ / σ 的 test R² | **0.7668 / 0.0501** |
| | σ 标签信度（Spearman-Brown） | **0.8740** |
| | 采样次数的影响（20 → 100） | **+2.9%**（n=20 已达渐近值 96.5%） |
| **标签** | 规模 | 9,998 条（唯一 prompt 9,081，重复 9.2%） |
| | 退化 σ bug 影响 | **1,228 条（12.3%）** + 214 条 MLE 病态 |
| | 重新拟合改变 | **1,561 条** |

---

## 11. 不许硬编的清单

1. **没有直接采样 GPU 利用率**。roofline 是推算，但用实测 TPOT 验证到 2% 误差。要补就加 `nvidia-smi -l 5` 后台采样。
2. **rate 8（ρ=1.02）到 rate 16（ρ=1.83）之间没采样**，而那是生产最常见的区间。
3. **`max_num_seqs=512` 的实验没做**。吞吐 7× 是 roofline 外推，不是实测。
4. **没有逐请求数据**（`--save-detailed` 当时没加），所以算不了需要逐请求数组的指标。
5. **σ 为什么学不出来，最后那个解释没验证**。"信号不在 prompt 文本里"是排除法之后剩下的，不是直接测出来的。要验证需要新的采样实验（比如固定 seed 贪心解码看 σ 是否塌缩）。

被问到就直说，然后讲会怎么补。**编一个数字比承认没测更糟。**

---

## 附：相关文档

| 文件 | 内容 |
|---|---|
| [`Interview_Deep_Dive.md`](Interview_Deep_Dive.md) | 按项目阶段组织，含 ML 细节和研究结论 |
| [`Phase3_Results.md`](Phase3_Results.md) | 30 轮完整数据和机制分析 |
| [`Phase3_Scheduling_Evaluation_Plan.md`](Phase3_Scheduling_Evaluation_Plan.md) | 五臂设计的推理过程 |
