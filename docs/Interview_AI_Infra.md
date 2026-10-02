# AI Infra 岗面试手册

> 和 [`Interview_Deep_Dive.md`](Interview_Deep_Dive.md) 的区别：那份按项目阶段组织，重点在实验设计和研究结论；这份按 **infra 面试官关心的能力**组织，重点在 **vLLM 内部机制、GPU 性能分析、排队论、系统集成、生产调试**。
>
> **predictor 的训练部分放在最后，一笔带过。** 对 infra 岗来说那是最弱的一环——结果本身不好看（σ R²=0.05），而且训模型是 table stakes。真正值钱的是你能从源码和第一性原理解释系统行为。
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

## 9. 预测器（一笔带过）

被问到再讲，**不要主动展开**。

> 用 DeBERTa-v3-base 从 prompt 文本预测输出长度分布的参数。μ 的 test R² 是 0.77，σ 是 0.05——**σ 基本学不出来**。
>
> 我做了一系列消融排除解释：分半信度证明不是标签噪声（天花板 0.87），子采样重拟合证明不是采样次数不够（20 次已达渐近值的 96.5%），损失加权、编码器冻结、单任务训练都试过。**剩下最可能的解释是信号本身不在 prompt 文本里**——输出长度的方差很大一部分来自解码的随机性，而不是 prompt 决定的。
>
> 对调度来说这个结论反而是有用的：我用 oracle 参数单独跑了一臂，结果和预测参数一样，**说明瓶颈不在预测精度**。

**如果对方追问 ML 细节**（架构、训练技巧、OOM 排查），转 [`Interview_Deep_Dive.md`](Interview_Deep_Dive.md) 第 4 节。

---

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

---

## 11. 不许硬编的清单

1. **没有直接采样 GPU 利用率**。roofline 是推算，但用实测 TPOT 验证到 2% 误差。要补就加 `nvidia-smi -l 5` 后台采样。
2. **rate 8（ρ=1.02）到 rate 16（ρ=1.83）之间没采样**，而那是生产最常见的区间。
3. **`max_num_seqs=512` 的实验没做**。吞吐 7× 是 roofline 外推，不是实测。
4. **没有逐请求数据**（`--save-detailed` 当时没加），所以算不了需要逐请求数组的指标。

被问到就直说，然后讲会怎么补。**编一个数字比承认没测更糟。**

---

## 附：相关文档

| 文件 | 内容 |
|---|---|
| [`Interview_Deep_Dive.md`](Interview_Deep_Dive.md) | 按项目阶段组织，含 ML 细节和研究结论 |
| [`Phase3_Results.md`](Phase3_Results.md) | 30 轮完整数据和机制分析 |
| [`Phase3_Scheduling_Evaluation_Plan.md`](Phase3_Scheduling_Evaluation_Plan.md) | 五臂设计的推理过程 |
