# AI Infra 岗面试手册

> **怎么用这份文档**
>
> 这是项目的完整技术底稿，按 **infra 面试官关心的能力**组织，不按项目阶段。目标是把所有能被追问到的东西都写进来，你按需要挑着学。
>
> - **第 2 节**背熟（开场）
> - **第 3–8 节**是核心技术面，每个数字都给了**怎么算的**和**为什么要算**
> - **第 9–11 节**是实验方法论，研究型岗位会深挖
> - **第 12 节**是行为面试弹药
> - **第 13–14 节**是模型和数据管线
> - **第 17 节**是**不许硬编**的清单，守住边界比多答一道题重要
>
> 按项目阶段组织的版本见 [`Interview_Deep_Dive.md`](Interview_Deep_Dive.md)。完整实验数据见 [`Phase3_Results.md`](Phase3_Results.md)。

---

## 1. 项目全貌

```
阶段一  标签生成
  LMSYS-Chat-1M prompt[0:10000]
    → Qwen3-8B 每条采样 20 次（temp 0.7 / top_p 0.8 / top_k 20，非思考模式）
    → MLE 拟合 log-t(μ, σ, ν=3.5)，max_tokens=2048
    → 9,998 条 (prompt, μ, σ)

阶段二  预测器
  DeBERTa-v3-base（184M）
    → 多池化 [CLS, masked mean, masked max] = 768×3 = 2304
    → 两条独立 MLP 分支
    → (μ̂, σ̂)
  结果：μ test R² = 0.7668，σ = 0.0501

阶段三  真机基准
  官方 vLLM 0.11.1 + --scheduler-cls 外挂（不 fork）
    → 5 个调度臂 × 6 个到达率 × 1000 条请求
    → 30 轮，30,000 请求，0 失败
    → 单卡 A100-40GB，Qwen3-8B bf16
```

### 1.1 调度规则

$$\text{score} = \mathbb{E}[X] + \beta \cdot \text{CVaR}_{0.9}[X], \qquad \beta = \text{clip}\!\left(\frac{0.1\,L_q}{B},\ 0.1,\ 0.5\right)$$

- $X$ ~ log-t(μ, σ, ν=3.5)，**两项都在 2048 token 处截断**
- $L_q$ = 当前等待队列长度，$B$ = `max_num_seqs`
- **分数越低越先调度**
- 直觉：按预期长度排序（SJF），再对"可能很长"的请求加一个风险罚

### 1.2 五个对照臂

| 臂 | 实现 | 相对前一臂改变了什么 |
|---|---|---|
| ① FCFS | 原版 vLLM，零参数 | — |
| ② FCFS + Predictor | 自定义 scheduler，保留 FCFS 队列 | **加载并运行预测器，但丢弃结果** |
| ③ Predicted-SJF | `TIE_BETA=0` | **队列按 E[X] 排序** |
| ④ TIE | 完整打分 | **加上 β·CVaR 风险项** |
| ⑤ TIE-oracle | `TIE_MODE=oracle` | **参数换成阶段一的真实拟合值** |

---

## 2. 开场陈述

### 60 秒版（infra 岗）

> 我做了一个 LLM 推理调度的项目：把一个基于输出长度预测的调度器挂进 vLLM，在 A100 上做端到端压测，搞清楚调度到底能改变什么、不能改变什么。
>
> **核心结论是：在并发被 `max_num_seqs` 约束的配置下，调度不能提高吞吐，只能重新分配等待。** 我测到中位排队时间降了 7.4 倍（24.4 秒 → 3.3 秒），但平均值几乎不动，因为总排队时间守恒。要真的提高吞吐，得让显存成为约束，让短请求能多塞几条进去。
>
> 过程中还发现解码阶段是带宽瓶颈：每个 step 要把 16.4 GB 权重完整读一遍，KV 只占 9%。所以 batch 32 时 GPU 在 75% 带宽利用率下只产出 2237 tok/s——**大约 7 倍的吞吐留在桌上没拿**。

**这段里一次都没提 DeBERTa。** 先给系统结论，对方感兴趣会追问预测怎么来的——那时第 13 节展开。

### 2 分钟版（加这三段）

**实验怎么控制的**：五个对照臂，每个相对相邻臂只改变一个变量。关键是第二臂——它加载并运行预测器但丢弃结果，**让 GPU 负载和调度臂完全对等**，这样"调度的收益"和"预测器的开销"就分开了。

**为什么敢说吞吐守恒**：并发被配置卡死在 32，而 KV cache 实测只用了 2%、抢占 0 次、各臂 batch 占用率 31.6–31.9/32（差异 <1%）。GPU 做的是同样的工作量，所以总排队时间等于队列长度曲线下的面积，和顺序无关。

**有个在跑实验前就算出来的结论**：我用标签直接算了"新策略和基线排出来的队有多大差别"——Spearman 0.9963、队头 top-32 重叠 94%。这十分钟的计算让我在花 GPU 时间之前就知道必须加一个中间对照臂。真机结果和这个离线预测完全一致。

---

## 3. vLLM 内部机制

### Q3.1 vLLM 的 scheduler 是怎么工作的？

```python
# vllm/v1/core/sched/scheduler.py
self.waiting = create_request_queue(self.policy)   # :188  等待队列
self.running: list[Request] = []                   # :190  运行批次
```

**关键点：调度策略只决定"谁先进 running batch"，不决定"谁先做完"。**

vLLM 用 continuous batching——`running` 里最多 `max_num_seqs` 条请求并发逐 token 解码，每个 step 各产出一个 token。请求一旦进入批次就不再受调度顺序影响，只会因为显存不足被抢占。

**为什么这点重要**：它直接决定调度能影响什么。如果负载没把 `running` 压满，`waiting` 长期为空，**所有策略行为完全相同**。实测：rate 2 req/s 时峰值并发只有 19（上限 32），五个臂的 p50 TTFT 全在 26–32 ms。

<details><summary><b>追问：那和教科书里的 FCFS 有什么不同？</b></summary>

教科书 FCFS 是单服务台、先来者独占直到完成，队头阻塞很强。vLLM 的 FCFS 是**先来者先入场，入场后与后来者并发共享 GPU**，队头阻塞只在 32 个槽位占满时才发生。

所以仿真器上测出来的调度收益会**系统性高于真机**——我自己先做过仿真，真机上 gap 小得多。这个差异本身值得在报告里解释。
</details>

<details><summary><b>追问：vLLM 官方支持哪些调度策略？</b></summary>

只有两种：

```python
# vllm/config/scheduler.py:23
SchedulerPolicy = Literal["fcfs", "priority"]
```

`priority` 是按外部传入的 priority 字段排，不是按预测长度。所以任何基于预测的调度都得自己实现，这也是为什么我需要 `--scheduler-cls`（第 6 节）。
</details>

<details><summary><b>追问：一个请求从到达到完成，经过哪些状态？</b></summary>

```
到达 → add_request() → waiting 队列
     → schedule() 把它 pop 出来放进 running
     → 每个 step 产出一个 token
     → 遇到 EOS 或 max_tokens → finished
     （中途可能因 KV 不足被抢占，退回 waiting）
```

**TTFT = 在 waiting 里待的时间 + 第一次 prefill/decode 的时间**，在饱和时几乎完全由排队时间主导。这就是为什么调度只影响 TTFT，不影响 TPOT。
</details>

### Q3.2 KV cache 是怎么管理的？你测到多少？

**PagedAttention**：KV cache 按固定大小的 block 分配（类似虚拟内存分页），避免为每条请求预留 `max_model_len` 的连续空间造成的内部碎片。

vLLM 启动时打印实际容量，我这边是：

```
GPU KV cache size: 394,544 tokens
Maximum concurrency for 8,192 tokens per request: 48.16x
```

**怎么验算**：Qwen3-8B 有 36 层、**8 个 KV head（GQA）**、head_dim 128、bf16：

$$2\,(K,V) \times 36 \times 8 \times 128 \times 2\text{B} = 147{,}456\text{ B} = \mathbf{144\ KB/token}$$

我的平均序列长度 = 102（输入）+ 249（输出）= **351 token**：

$$\frac{394{,}544}{351} \approx \mathbf{1{,}124\ 条并发}$$

**但 `max_num_seqs` 设成了 32，实测 KV 利用率只有 2.0%（峰值 2.6%）。**

**为什么要算这个**：它决定并发到底被什么卡住。vLLM 自报的 `48.16x` 说的是"每条用满 8192 上下文时能装 48 条"——**48 > 32，所以无论请求多长，显存都轮不到成为约束**。这一条推翻了后面一连串结论（Q5.3）。

<details><summary><b>追问：GQA 对 KV cache 的影响有多大？</b></summary>

Qwen3-8B 有 32 个 Q head 但只有 8 个 KV head（分组比 4:1）。如果是 MHA（32 个 KV head），每 token 的 KV 会是：

$$2 \times 36 \times 32 \times 128 \times 2\text{B} = 576\text{ KB/token}$$

**是现在的 4 倍**，KV 容量从 394k token 掉到 98k token，能装的并发从 1,124 降到 281。

**GQA 是 KV cache 容量的 4 倍杠杆**，这也是为什么现代模型几乎都用它。
</details>

<details><summary><b>追问：为什么要分页？不分页会怎样？</b></summary>

不分页就得为每条请求预留 `max_model_len`（8192 token × 144 KB = **1.15 GB/条**）的连续空间。32 条就是 37 GB——**整张 40GB 卡都不够**。

而实际平均序列只有 351 token，预留的 96% 都浪费了。分页把分配粒度降到 block（通常 16 token），按需增长，内部碎片最多一个 block。
</details>

<details><summary><b>追问：你怎么确认没有发生抢占？</b></summary>

抓 server 日志：

```bash
grep -ci "preempt" results/bench_*/server_*.log   # 五个臂全是 0
```

和 KV 2% 的利用率互相印证——显存压力根本不存在。这是后面"吞吐守恒"论证的一环。
</details>

### Q3.3 什么时候会发生抢占？语义是什么？

KV cache 不够给 running 的请求分配下一个 token 时：

```python
# scheduler.py:407
preempted_req = self.running.pop()          # LIFO：踢掉最后进入的
...
self.waiting.prepend_request(preempted_req) # 放回等待队列
```

**`prepend_request` 的语义因队列而异，这点容易被忽略：**

| 队列 | 行为 | 后果 |
|---|---|---|
| `FCFSRequestQueue` | `appendleft` —— **插到队头** | 被抢占的请求优先恢复 |
| 我的 `TIERequestQueue` | 调 `add_request` —— **按分数重新入堆** | 被抢占的长请求可能**再次被推后** |

**为什么要注意**：这是一个会污染尾延迟归因的机制性差异。如果压测中抢占频繁，p99 的差异可能来自抢占策略而不是调度策略。所以我在压测脚本里专门统计抢占次数——**确认是 0 之后，才能说 p99 的差异纯粹来自调度**。

<details><summary><b>追问：抢占的代价是什么？</b></summary>

被抢占的请求会**丢掉已经算好的 KV cache**，恢复时要重新 prefill 整个已生成的序列。

所以抢占是**纯浪费的重算**。一条已经生成了 500 token 的请求被抢占，恢复时要重算 500+102 个 token 的 prefill。

这也是为什么"调度能不能减少抢占"是一条真实的吞吐提升路径——**在显存紧张的配置下**。我的配置下 KV 用 2%，这条路不存在。
</details>

### Q3.4 chunked prefill 有什么影响？

日志：`Chunked prefill is enabled with max_num_batched_tokens=2048`。

prefill 和 decode **共享每个 step 的 token 预算**。一条长 prompt 的 prefill 被切成多个 chunk，分散在多个 step 里，和 decode 交织执行。

**好处**：长 prompt 不会一次性阻塞整个批次（没有 chunked prefill 时，一条 8192 token 的 prompt 会独占一个 step，让所有 decode 停顿）。

**代价**：prefill 会挤占 decode 的 token 预算。我的 workload 平均输入 102 token，prefill 占比很小，不是瓶颈。

### Q3.5 prefill 和 decode 的性能特征为什么不同？

| | prefill | decode |
|---|---|---|
| 一次处理 | 整个 prompt 的 $L$ 个 token | **1 个** token |
| 矩阵运算 | $L \times d \times d$ 矩阵乘 | $1 \times d \times d$ **矩阵×向量** |
| 算术强度 | 高 | **极低** |
| 瓶颈 | **算力**（compute-bound） | **显存带宽**（memory-bound） |

**decode 的 step 时间几乎只取决于"读多少字节"**，这正是第 4 节 roofline 成立的前提。

也解释了为什么 continuous batching 对 decode 收益巨大：**batch 翻倍，权重读取量不变，产出翻倍。**

---

## 4. GPU 性能分析（roofline）

### Q4.1 你的服务吞吐是多少？瓶颈在哪？

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

**验算吞吐**（关键一步，它证明模型没算错）：

$$\frac{\text{batch } 32}{14\text{ ms}} = 2{,}286 \text{ tok/s} \quad\text{vs}\quad \text{实测 } 2{,}237 \text{ tok/s} \qquad \textbf{误差 2\%}$$

**为什么要做这个计算**：不做的话你只知道"吞吐是 2237"，不知道这个数是好是坏、离上限多远、该往哪优化。做完之后你知道：GPU 在 75% 带宽利用率下运行，**但每读一次 16.4 GB 权重只产出 32 个 token**——SM 大部分时间在等内存。

### Q4.2 KV cache 的读取占多少？

每个 step 读取的总量：

| 读什么 | 大小 | 占比 |
|---|---|---|
| 模型权重（**与 batch 无关**） | 16.40 GB | **91%** |
| 32 条请求的 KV（351 token × 144 KB × 32） | 1.62 GB | 9% |
| 合计 | 18.02 GB | |

**权重占 91%**，这就是为什么增大 batch 几乎是"免费"的吞吐。

**这个比例还有第二个用途**：它给出了**调度顺序能影响吞吐的理论上界**。顺序只能改变批内请求的长度分布，也就是那 9%。即使 SJF 让批内平均序列长度减半（KV 1.62 → 0.81 GB），step 时间也只减少 **4.5%**。

实测是 −2%（预测器开销盖过了它）。**所以"调度提升吞吐"在这个配置下的天花板是 4.5%。**

### Q4.3 怎么提高吞吐？能提多少？

开大 `max_num_seqs`。step 时间 = (权重 + batch × 每条 KV) / 带宽：

| batch | 每 step 读取 | step 时间 | token 吞吐 | req/s | TPOT |
|---|---|---|---|---|---|
| **32**（当前） | 18 GB | 14 ms | 2,237 | **8.8** | 14 ms |
| 128 | 23 GB | ~18 ms | ~7,100 | ~28 | 18 ms |
| 512 | 42 GB | ~33 ms | ~15,500 | **~62** | 33 ms |

**batch 开到 512，吞吐涨约 7 倍，而单条请求的 TPOT 只从 14 ms 涨到 33 ms。**

这是个明确的**吞吐/延迟取舍**，取决于 SLO。注意 TPOT 翻倍意味着一条 250 token 的回复从 3.5 秒变成 8.25 秒——流式输出的体感会变差。

<details><summary><b>追问：为什么不无限开大 batch？</b></summary>

三个约束：

1. **KV cache 容量**：batch 512 × 351 token × 144 KB = 25.9 GB，已占 KV 容量的 46%。再大就会触发抢占
2. **TPOT 随 batch 线性增长**（因为 KV 读取量线性增长），SLO 会先顶不住
3. **prefill 的突发**：batch 大了，同时到达的新请求 prefill 会挤占 token 预算

实际上 batch 和 SLO 是个联立约束，要扫一遍测。
</details>

<details><summary><b>追问：你直接测 GPU 利用率了吗？</b></summary>

**没有，这是个疏漏。** slurm 脚本只在开头跑了一次 `nvidia-smi`，没有全程采样。上面的 roofline 是推算，但用实测 TPOT 验证到了 2% 误差。

要补的话加一行后台采样：
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 5 > gpu.csv &
```

**而且 `utilization.gpu` 常被误读**——它是"有 kernel 在跑的时间占比"，不是算力饱和度。解码时它可能显示 95%+ 而 SM 其实在等内存。`utilization.memory` 才是带宽占用，这个才能验证我的 75%。
</details>

<details><summary><b>追问：为什么用 bf16 不用 fp16？</b></summary>

A100 原生支持 bf16，而且 bf16 的**指数位和 fp32 一样宽**（8 位），动态范围大，训练和推理都不容易溢出。fp16 只有 5 位指数，需要 loss scaling 之类的技巧。

这也是我选 A100 而不是 V100 的原因之一——V100（Volta）**不支持 bf16**，跑 Qwen3-8B 要转 fp16，有数值风险。
</details>

---

## 5. 排队论与容量规划

### Q5.1 负载和延迟的关系你怎么刻画？

用利用率 $\rho = \lambda/\mu$（发起速率 / 实际吞吐）：

| 发起 | 实际吞吐 | ρ | p50 TTFT | p90 TTFT | p99 TTFT |
|---|---|---|---|---|---|
| 2 | 1.99 | 1.00 | 32 ms | 52 ms | 77 ms |
| 4 | 3.96 | 1.01 | 31 ms | 47 ms | 78 ms |
| 6 | 5.90 | 1.02 | 33 ms | 53 ms | 264 ms |
| 8 | 7.81 | 1.02 | 50 ms | 832 ms | 1,408 ms |
| **16** | 8.74 | **1.83** | **24,382 ms** | 44,025 ms | 48,081 ms |
| 32 | 8.80 | 3.64 | 39,419 ms | 71,378 ms | 78,314 ms |

**ρ 跨过 1 的时候，TTFT 从毫秒级跳到秒级。** p90 从 rate 8 的 832 ms 跳到 rate 16 的 44,025 ms——**53 倍**。

### Q5.2 那个跳变是 bug 吗？

**不是，是队列从稳定变发散。** ρ>1 时队列线性增长，第 $i$ 条请求的等待时间：

$$W(i) = i\left(\frac{1}{\mu} - \frac{1}{\lambda}\right)$$

直觉：第 $i$ 条在 $i/\lambda$ 时刻到达，但要等到 $i/\mu$ 时刻才轮到它。

**验算（FCFS 臂）**：

| 负载 | 分位 | 公式预测 | 实测 | 误差 |
|---|---|---|---|---|
| rate 16 | p50 | 24,678 ms | 22,157 ms | +11% |
| | p90 | 44,421 ms | 41,475 ms | +7% |
| | p99 | 48,863 ms | 45,495 ms | +7% |
| rate 32 | p50 | 39,869 ms | 37,129 ms | +7% |
| | p90 | 71,764 ms | 68,590 ms | +5% |
| | p99 | 78,940 ms | 75,672 ms | +4% |

**误差 4–11%**（系统性偏高，因为公式忽略了到达窗口内已经被服务掉的那部分）。

**为什么要算这个**：它把"p90 怎么突然变成 40 秒"从一个 bug 疑虑变成一个可预测的量。而且它说明**过载时 FCFS 的 TTFT 纯粹是到达序号的函数**——和请求内容无关，所以那个区间测的是"你第几个到"，不是"调度器好不好"。

### Q5.3 为什么你的吞吐五个臂完全一样？

**因为并发被配置卡死，不是被显存。** 完整因果链，每一环都有实测：

```
max_num_seqs=32 钉死并发
  → 各臂 batch 都跑满 ~31.8/32（实测 31.60–31.89，跨臂差异 <1%）
  → KV 用量都是 2%，抢占 0 次
  → GPU 做同样的工作、同样的批大小
  → step 时间相同 → token 吞吐一致（2,132–2,237 tok/s，极差 4.2%）
  → 总 token 工作量固定 → duration 固定
  → 总排队时间守恒
```

**总排队时间 = 队列长度曲线下的面积**，只取决于到达过程和服务速率，**与服务顺序无关**。调度只能**重新分配**等待，不能减少。

实测吞吐对比：

| rate | 五臂极差 | 差异来自 |
|---|---|---|
| 2 / 4 / 6 / 8 | **0.0–0.1%** | 未饱和，吞吐=发起速率 |
| 16 / 32 | 3.9–4.2% | **全部来自预测器开销**，不是调度 |

拆开看：`FCFS+Predictor vs FCFS` 是 **−2.2%**；`SJF vs FCFS+Predictor` 是 −1.8%～+0.2%（噪声）。

<details><summary><b>追问：这和 Little's Law 什么关系？</b></summary>

Little's Law：$L = \lambda W$（系统内平均请求数 = 到达率 × 平均停留时间）。

在饱和的稳态下，$L$ 被 `max_num_seqs` 和队列深度决定，$\lambda$ 是外部给定的，所以 $W$（平均停留时间）也被决定了——**调度改不了它**。

调度能改的是 $W$ 的**分布**：谁等得久、谁等得短。这正是我观察到的 p50 降 7.4 倍而 mean 不动。
</details>

<details><summary><b>追问：什么情况下调度能真的提高吞吐？</b></summary>

两条路，都要求显存成为约束：

1. **提高并发**：短请求占用更少 KV → 同时装下更多条 → $\text{吞吐} = \text{并发}/\text{step时间}$ 上升
2. **减少抢占**：长请求撑满 KV 触发抢占 → 重算 prefill 是纯浪费 → 少抢占就少浪费

**我的配置两条都不成立**：KV 用 2%，抢占 0 次。所以要验证需要把 `max_num_seqs` 开到 512 让显存说话——这是我列的下一步实验。
</details>

### Q5.4 那调度到底改变了什么？

**分布，不是总量。** rate 16（ρ=1.83），相对负载对等的基准：

| 臂 | p50 TTFT | p90 | p99 | **mean** |
|---|---|---|---|---|
| FCFS + Predictor | 24,382 ms | 44,025 | 48,081 | 24,514 |
| Predicted-SJF | **3,303 ms（−86.5%）** | +65.7% | +103.6% | **+4.3%** |
| TIE | 3,759 ms（−84.6%） | +59.4% | +86.0% | +1.6% |

**中位排队时间降低 7.4 倍，平均值几乎不动。** 省下的 21 秒被搬到了尾部。

> **方法论教训**：我的分析脚本第一版只打印 mean 和 p99，**把 7.4 倍的改善报告成了"无效果"（+4.3%）**。补上 p50 才发现主效应。
>
> **在守恒的系统里，mean 是唯一不会动的那个统计量**——只看 mean 等于什么都没测。

### Q5.5 这个收益在所有负载下都成立吗？

不。我推出一条规律并验证了。

过载时，一个到达窗口内最多只有 $\mu/\lambda = 1/\rho$ 比例的请求能"即到即走"。SJF 的收益正来自让这批短请求插队，其余的无论怎么排都得在积压里等。所以：

> **SJF 只能改善低于 $p^\* = 100/\rho$ 的分位数，高于它的必然恶化。**

| 负载 | ρ | $p^\*$ | p50 | p90 | p99 |
|---|---|---|---|---|---|
| rate 8 | 1.02 | 98 | −2.9% ✓ | −24.5% ✓ | +111.1% ✓ |
| rate 16 | 1.83 | 55 | −86.5% ✓ | +65.7% ✓ | +103.6% ✓ |
| rate 32 | 3.64 | 27 | −3.6% ✗ | +15.5% ✓ | +28.0% ✓ |

**9 个预测中 8 个成立**（唯一不符的那格预测恶化、实测持平 −3.6%，在噪声量级）。

**实用价值**：SJF 类调度有一个**最优负载窗口**。ρ=1.83 时效果最戏剧化（$p^\*≈55$ 正好落在中位数）；ρ=3.64 时只有最快的 27% 能受益，中位数已经在积压里。

**不是越过载越好。**

### Q5.6 怎么给这个服务定容量？

**不是看吞吐天花板，是看 SLO。**

吞吐天花板 8.8 req/s，但：

| SLO | 可用容量 | 依据 |
|---|---|---|
| p90 TTFT < 100 ms | **~6 req/s** | rate 6 的 p90 = 53 ms；rate 8 已经 832 ms |
| p90 TTFT < 1 s | **~8 req/s** | rate 8 的 p90 = 832 ms |
| 无延迟要求（离线批处理） | 8.8 req/s | 吞吐上限 |

**ρ 接近 1 时延迟急剧恶化**，所以生产上通常留 20–30% 余量，实际跑在 ρ≈0.7–0.8。

而 $p^\*=100/\rho$ 补充了一条：**如果 SLO 关心 p90，只有 ρ < 1.11 时 SJF 才帮得上忙**（$100/1.11 = 90$）。超过这个点，SJF 会让 p90 更差。

---

## 6. 系统集成

### Q6.1 你怎么把自己的调度器接进 vLLM 的？

**没有 fork。** vLLM 官方自带可插拔调度器：

```python
# vllm/config/scheduler.py:129
scheduler_cls: str | type[object] = Field(default=None)
"""The scheduler class to use. "vllm.v1.core.sched.scheduler.Scheduler" is
the default scheduler. Can be a class directly or the path to a class of
form "mod.custom_class"."""
```

命令行 `--scheduler-cls mod.MyScheduler`（`arg_utils.py:1092`），vLLM 用 `resolve_obj_by_qualname` 动态导入。

`SchedulerInterface` 有 13 个抽象方法（`schedule`、`update_from_output`、`get_grammar_bitmask`、KV cache、抢占……），从零实现不现实。**但不需要——继承官方 `Scheduler`，只替换 `self.waiting` 一个成员**：

```python
class TIEScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)      # 官方逻辑原样跑完
        self.waiting = TIERequestQueue(...)    # 唯一的改动
```

**为什么这样做**：参考实现是改 vLLM 源码再整树发布。照搬要从源码编译 CUDA kernel，在共享集群上耗时长、对 CUDA/torch 版本敏感、容易卡住。外挂方案用 `pip install vllm==0.11.1` 官方预编译 wheel，**不编译**。

**代价**：vLLM 自己在加载时会警告

> `Using custom scheduler class %s. This scheduler interface is not public and compatibility may not be maintained.`

所以版本必须锁死。`requirements` 里写死 `vllm==0.11.1`，不写 `>=`。

<details><summary><b>追问：你怎么确认这个方案可行，而不是跑到一半才发现接口不对？</b></summary>

写了一个纯 CPU 的关口检查脚本（`scripts/verify_vllm_integration.py`），**18 项，在花任何 GPU 时间之前跑**：

1. vLLM 版本 == 0.11.1
2. `Scheduler` / `RequestQueue` / `FCFSRequestQueue` / `Request` 能从官方包 import
3. **`Scheduler.__init__` 里确实有 `self.waiting = ...`**（我们要覆盖的那个点）——用 `inspect.getsource` 检查
4. 我的两个队列类没有未实现的 `RequestQueue` 抽象方法
5. 我的调度器类满足 `SchedulerInterface` 且 `__abstractmethods__` 为空
6. `resolve_obj_by_qualname("vllm_tie.scheduler.TIEScheduler")` 能解析到正确的类
7. checkpoint 文件存在

**第 4、5 条尤其重要**：未实现的抽象方法只会在 serve 时暴露，那时已经跑到压测一半了。

这个脚本在我的环境上一次跑过全部 18 项，整个部署风险当场清零。
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
- 预测器 **fp16 加载**（权重 0.74 → 0.37 GB）
- **`max_batch_size` 从参考实现的 128 降到 32**（128 条 × 512 token 的 DeBERTa forward 会炸掉那 4.9 GB）

**这个顺序是读源码读出来的，不是文档写的。**
</details>

### Q6.2 预测在调度的关键路径上吗？

**不在。** 这是个常见误解。

```python
def add_request(self, request):
    initial_score = 2048.0                           # 占位分，立即入堆
    heapq.heappush(self._heap, (initial_score, ...))
    self._prediction_queue.put(request)              # 丢给后台线程，立即返回
```

后台 daemon 线程攒批（**8 条或 3 ms 触发**）→ DeBERTa forward → 蒙特卡洛算 E[X]/CVaR → 通过版本号惰性更新堆。**调度主循环从不阻塞等待预测。**

实测预测批延迟 **44.8–48.3 ms/批**。

<details><summary><b>追问：45 ms 会不会太慢？</b></summary>

不会，因为它不在关键路径上。真正要确认的是**预测能不能跟上到达率**。

攒批参数"8 条或 3 ms"意味着稳态下每批 8–32 条、45 ms 一批 → **约 180–700 预测/秒**，而我的最高到达率是 32 req/s。余量充足。
</details>

<details><summary><b>追问：那预测晚到会怎样？你怎么知道它没晚到？</b></summary>

**这是最危险的静默失败模式**：如果预测总是在请求被调度之后才到，队列实质在跑 FCFS，但**延迟数据看起来完全正常**，你会以为测到了调度效果。

所以我在队列里加了计数器 `popped_before_prediction`——被 pop 时还没拿到分数的请求数。分阶段统计：

| popped 区间 | 未打分占比 | 队深 |
|---|---|---|
| 0–2,115（rate 2/4/6） | **100%** | 0 |
| 2,115–3,727（rate 6/8） | 97% → 63% | 0–5 |
| 3,727–4,243 | 38% | 188 |
| **4,243–5,836（rate 16/32）** | **0–6%** | 170–708 |

低负载 100% 未打分**不是问题**——队列是空的，根本没有调度决策可做。队列一深，几乎所有请求都及时拿到了分数。

冒烟测试脚本里把这个比例 **≥90% 设成 FAIL、≥40% 设成 WARN**。
</details>

### Q6.3 怎么验证部署是对的？

写了一个单臂冒烟脚本（`scripts/smoke_test.sh`），用**真实的压测客户端**打一波短突发，然后判定三件事：

1. **预测器装得下吗**——vLLM 先分配 KV cache，DeBERTa 只能用剩余显存，装不下会在加载时 OOM 而不是启动时
2. **打分有意义吗**——`popped_before_prediction` 不能接近 100%
3. **队列有深度吗**——未饱和时所有臂行为相同，测不出东西

脚本用 `vllm bench serve` 而不是手写 curl，**这样冒烟走的是和正式压测完全相同的代码路径**。

---

## 7. 并发数据结构

### Q7.1 等待队列用什么数据结构？复杂度？

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

**为什么需要惰性删除**：打分是异步的，分数会后到。如果用"找到旧元素并修改"的方式更新，需要 O(n) 查找或维护额外的位置索引（还要和堆的每次 sift 同步，很容易写错）。

**版本号把"更新"退化成一次 push**：

```python
def _push_updated_score(self, req_id, new_base):
    if req_id not in self._versions:
        return                                   # 已经被 pop 或 remove 了
    version = self._versions[req_id] + 1         # 旧版本的堆项从此作废
    self._versions[req_id] = version
    heapq.heappush(self._heap, (effective, arrival_time, version, req_id, request))
```

pop/peek 时比对版本号，不匹配就丢弃。

辅助结构（都用同一把锁保护）：

| 结构 | 作用 |
|---|---|
| `_versions: dict[str, int]` | 当前有效的版本号，**同时充当"是否还在队列里"的判定** |
| `_base_scores: dict[str, float]` | 未衰减的原始分数（防饥饿重建时要用） |
| `_request_info: dict[str, (float, Request)]` | O(1) 拿到 arrival_time 和对象，避免遍历堆 |

<details><summary><b>追问：堆里会不会无限堆积垃圾？</b></summary>

不会，有两个清理路径：

1. **pop/peek 时顺手丢弃**堆顶的过期项
2. **防饥饿重建**：每 5 秒全量重建堆，所有版本号 +1，顺带把惰性删除积累的过期项一次清空

重建是 O(n)，但**跑在后台线程上，不在每请求的调度路径上**。

最坏情况下堆的大小是 O(更新次数)，但每条请求平均只更新一次（预测落地），加上每 5 秒一次重建，实际大小接近 O(n)。
</details>

<details><summary><b>追问：为什么堆元素里要带 arrival_time？</b></summary>

两个作用：

1. **作为第二排序键，保证分数相同时退化为 FCFS**。这在刚启动时很关键——所有请求都还是占位分 2048，如果不带 arrival_time，堆会去比较第三个元素（version），出队顺序就是任意的
2. **防饥饿衰减要用它算等待时长**，放在元组里省一次字典查找

</details>

<details><summary><b>追问：为什么初始分数是 2048？</b></summary>

它等于 `max_tokens` 上限，也就是**一条请求可能的最长输出**。

**语义是悲观假设**：在知道真实分数之前，假设它是最长的，排到队尾。这样未打分的请求不会抢占已打分的短请求。

如果反过来给 0，未打分的请求会全部插到队头，**调度实质变成"谁刚到谁先走"（LIFO）**，比 FCFS 还糟。
</details>

### Q7.2 防饥饿是怎么做的？

**乘性时间衰减**：

$$\text{effective} = \text{base} \times \gamma^{t_w/\tau}, \qquad \gamma = 0.9,\ \tau = 30\text{s}$$

分数越低越先调度，所以等得越久分数越低、越往队头走。

| 等待时间 | 分数系数 |
|---|---|
| 30 s | ×0.90 |
| 60 s | ×0.81 |
| 120 s | ×0.66 |
| 300 s | ×0.35 |

生效在两处：预测结果落地时（`_push_updated_score`）、以及后台线程每 **5 秒**的全量重建。

**和减性老化的区别**：

| | 公式 | 性质 |
|---|---|---|
| 乘性（本项目） | $s \times \gamma^{t_w/\tau}$ | **按比例**——分数 2000 的等 30 秒减 200，分数 100 的只减 10 |
| 减性 | $s - \alpha t_w$（α 单位 tokens/s） | **一视同仁**——所有请求等 30 秒都减同样多 |

**乘性给长请求的绝对补偿更大**，这会影响公平性的解读，报告里要写明用的是哪个。

<details><summary><b>追问：这个衰减够不够强？</b></summary>

算一下：要把 48 倍的分数差距（我的 E[X] 跨度）抹平，需要

$$0.9^{\Delta t/30} = \frac{1}{48} \;\Rightarrow\; \Delta t = 30 \cdot \frac{\ln 48}{\ln(1/0.9)} \approx 1{,}104\text{ s}$$

**要等待时长相差 18 分钟才能翻转。** 我实测的最长等待是 80 秒左右，衰减系数 0.75——**远不足以改变排序**。

所以在我的实验里，防饥饿机制基本没起作用。这也意味着**长请求确实被饿着了**（p99 TTFT 涨了一倍），如果生产上要用，τ 得调小很多。
</details>

### Q7.3 线程安全怎么保证？

一把 `threading.RLock` 保护堆和三个 dict。

**后台预测线程和调度主线程的交互点只有两个**：
1. 线程通过 `queue.Queue` 拿任务（本身线程安全）
2. 算完后拿锁批量更新堆

**用 RLock 不用 Lock** 的原因：`_push_updated_score` 会被已经持锁的 `_on_scores` 调用，可重入锁避免自死锁。

**GIL 的影响有限**：PyTorch 的 CUDA 调用和 Rust 实现的 fast tokenizer 都会释放 GIL，所以预测线程不会长时间霸占解释器。但**蒙特卡洛打分是纯 numpy**，那部分会持有 GIL——不过 10,000 次采样只要零点几毫秒。

<details><summary><b>追问：锁的粒度会不会成为瓶颈？</b></summary>

不会。调度路径上的操作（pop/peek）是 O(log n) 的堆操作，持锁时间在微秒量级，而 step 时间是 14 **毫秒**——差四个数量级。

后台线程持锁的时间也很短：DeBERTa forward 和蒙特卡洛**都在锁外**，只有最后批量更新堆时才拿锁。
</details>

---

## 8. 打分计算

### Q8.1 分数具体怎么算的？

$$\text{score} = \mathbb{E}[X] + \beta \cdot \text{CVaR}_{0.9}[X], \qquad X \sim \text{log-}t(\mu, \sigma, \nu=3.5)$$

**log-t 是什么**：$\ln X$ 服从 $t$ 分布（而不是正态）。相比 log-normal，$t$ 分布的尾更厚，更贴合输出长度的实际分布。$\nu=3.5$ 固定，不学习——这样 μ、σ 两个参数可解释、可跨 prompt 比较。

**两项都用蒙特卡洛算**（10,000 次采样），因为截断后没有闭式解：

```python
t_samples = stats.t.rvs(df=3.5, size=10000, random_state=_RNG)
x = np.exp(mu + sigma * t_samples)

# E[X]：截断均值
mean = np.mean(np.minimum(x, 2028.0))

# CVaR_0.9：最差 10% 的均值
x_c = np.minimum(x, 2048.0)
var = np.percentile(x_c, 90)
cvar = np.mean(x_c[x_c > var])
```

### Q8.2 为什么要截断？

因为 vLLM 设了 `max_tokens`，**一条本该生成 10,000 token 的请求，实际也只会占用 GPU 2048 个 token 的解码时间**。

用未截断的期望给它打分会**高估它的阻塞程度**，把它排得过于靠后。

**截断是把"统计量"翻译成"系统代价"的那一步。**

<details><summary><b>追问：截断有什么副作用？</b></summary>

σ 很大时，E[X] 和 CVaR 会**一起向上限饱和**，比值 CVaR/E[X] 不再增长。

我实测过：把 σ 整体放大 2×～12×，排序一致性（和纯 E[X] 排序比）始终 ≥0.99，正是因为截断压平了差异。

**这是 CVaR 项失效的一个结构性原因**（第 10 节）。
</details>

<details><summary><b>追问：为什么 E[X] 截断在 2028 而 CVaR 在 2048？</b></summary>

参考实现里就是这两个数（`MAX_GENERATED_TOKENS = 2028.0` / `CVAR_MAX_GENERATED_TOKENS = 2048.0`），我照搬了以保持可比性。

20 token 的差别在实践中可忽略。我猜是他们给 EOS/特殊 token 留的余量，但**代码里没写原因，我不编解释**。
</details>

### Q8.3 β 为什么是自适应的？

$$\beta = \text{clip}\!\left(\frac{0.1\,L_q}{B},\ 0.1,\ 0.5\right)$$

$L_q$ 是当前队列深度，$B$ 是 `max_num_seqs`。

**语义**：$L_q/B$ 衡量**系统压力**——队列相对于正在服务的批量有多深。

- 队列浅 → β 取下限 0.1 → 分数几乎就是 E[X]，等价于 SJF
- 队列深（$L_q = 5B$，即 160）→ β 顶到 0.5 → 尾部风险权重最大

**直觉**：队列不深的时候，调度错了也没多大代价（反正很快轮到）；队列深的时候，把一条可能很长的请求放前面会阻塞很多人，这时候才值得为风险付出代价。

<details><summary><b>追问：为什么 B 必须等于 max_num_seqs？</b></summary>

因为 $L_q/B$ 的含义是"队列深度相对于服务批量"。如果 B 和实际 batch 脱钩，β 的含义就漂了。

举例：B 固定 32 而实际 `max_num_seqs=128`，那么队列到 160（只有批量的 1.25 倍）β 就顶到 0.5，**防风险过于激进**。

**我的实现把 B 从 `self.max_num_running_reqs` 实时读取**：

```python
predictor.set_gpu_batch_size(self.max_num_running_reqs)
```

参考实现是靠启动脚本导出两个一致的环境变量（`UA_GPU_BATCH_SIZE=$MAX_NUM_SEQS`）——**任何绕过那个脚本的部署方式都会静默失配，而且不报错**。
</details>

<details><summary><b>追问：这带来什么泛化性问题？</b></summary>

因为 B 绑定在 `max_num_seqs` 上，**它不是一个自由超参数**。

如果生产上用 `max_num_seqs=256`，那 $\beta = \text{clip}(0.1 L_q/256, \ldots)$ 需要队列深到 **1280** 才顶到上限——**防风险力度被稀释 8 倍**，而论文只在 B=32 下测过，这个区间没有数据。

这是我读代码读出来的一个真实局限。
</details>

### Q8.4 打分的随机性怎么处理？

蒙特卡洛用 10,000 次采样，**同一组 (μ, σ) 每次算出的分数会有微小差异**。

参考实现用的是**全局 numpy RNG**，意味着同一条 prompt 在不同时刻打分结果不同，**整个实验无法复现**。

我改成独立的 seeded 生成器：

```python
_seed_env = os.environ.get("TIE_SCORE_SEED")
_RNG = np.random.default_rng(int(_seed_env) if _seed_env else None)
```

压测时设 `TIE_SCORE_SEED=0`，**同样的输入必然得到同样的分数**。这让"离线排序分析"和"真机结果"可以严格对照（Q9.3）。

<details><summary><b>追问：10,000 次采样的开销多大？</b></summary>

`stats.t.rvs(size=10000)` 加上 `np.exp` 和分位数计算，大约 **0.3–0.5 ms**。

一批 32 条请求就是 ~15 ms，相比 DeBERTa forward 的 45 ms 是次要项。而且全部在后台线程上。

**如果要优化**：可以预先生成一组固定的 t 分位点做确定性积分（Gauss-Hermite 之类），零方差且更快。我没做，因为要保持和参考实现的数值可比性。
</details>

---

## 9. 实验设计

### Q9.1 你怎么确保测的是调度而不是别的？

**五个对照臂，每个相对相邻臂只改变一个变量。**

| 对比 | 唯一变量 | 测出什么 |
|---|---|---|
| ② − ① | 跑不跑预测器（调度逻辑相同） | **预测器的部署开销** |
| ③ − ② | 调度用不用预测结果（GPU 负载相同） | **调度排序的收益** |
| ④ − ③ | 有没有 CVaR 项 | 风险项的净贡献 |
| ⑤ − ④ | σ 用预测值还是真实值 | 预测质量的天花板 |

**② 是关键的一臂，参考实现没有。** 它加载并运行预测器但**丢弃结果**，让 GPU 负载和 ③④⑤ 完全对等——都加载 DeBERTa、都跑预测、都占同样显存，**唯一差别是队列用不用结果**。

这样 `③ − ②` 就是纯调度收益，**单卡部署不引入任何偏差**，省掉了申请第二张 GPU 的排队风险。

实现成本接近零：

```python
class FCFSWithPredictionQueue(_PredictionWorker, FCFSRequestQueue):
    def add_request(self, request):
        super().add_request(request)   # vLLM 官方 FCFS 行为
        self._submit(request)          # 照样送去预测，结果扔掉
```

**其他全部对齐**：同一模型、同一套 `--max-num-seqs 32 --max-model-len 8192 --no-enable-prefix-caching --gpu-memory-utilization 0.88`、同一份 workload、**同一个 `--seed`**。

<details><summary><b>追问：为什么 --gpu-memory-utilization 必须所有臂一样？</b></summary>

因为它决定 KV cache 容量，而 KV 容量影响并发能力和抢占频率。

**如果只给带预测器的臂调低（为了腾显存），那几个臂的 KV cache 就比基准小，对比直接失效**——你分不清延迟差异来自调度还是来自 KV 容量。

取值要以"带预测器的臂能稳定跑起来的最大值"为准，然后**所有臂统一用这个值**。
</details>

<details><summary><b>追问：为什么 seed 也要统一？</b></summary>

`vllm bench serve --dataset-name custom` 会用 `--seed` 打乱 prompt 顺序：

```python
random.seed(self.random_seed)
random.shuffle(self.data)
```

如果各臂 seed 不同，**它们跑的就不是同一个 workload**——到达顺序不同，长短请求的分布也不同。延迟差异里就混进了 workload 差异。

默认就是 0，但我显式钉死了，因为这是对比的**正确性要求**，不是便利性。
</details>

### Q9.2 实验前你做了什么准备？

两件，都在花 GPU 时间之前。

**一、关口检查**（Q6.1 追问）——18 项纯 CPU 验证，确认集成方案可行。

**二、离线排序分析**——用标签直接算"新策略和基线排出来的队有多大差别"：

```
β=0.1:  Spearman 0.99957,  队头 top-32 重叠 96.9%
β=0.3:  Spearman 0.99791,  队头 top-32 重叠 93.8%
β=0.5:  Spearman 0.99625,  队头 top-32 重叠 93.8%
```

top-32 正好是一批要调度的量，只有 1–2 个位置不同。

**这十分钟的计算告诉我：必须加一个中间臂（Predicted-SJF），否则测出来有差异也分不清来源**——到底是 SJF 排序的功劳还是 CVaR 项的功劳。

**真机结果和这个离线预测完全一致**（第 10 节）。

<details><summary><b>追问：为什么要同时看 Spearman 和 top-k 重叠？</b></summary>

它们回答不同的问题：

- **Spearman** 看全局排序一致性。但它可能很高的同时队头被打乱
- **top-k 重叠**看**队头**——而调度只看队头，k=32 正好是一批要调度的量

全局相关高 ≠ 调度行为相同。两个一起看才有说服力。

实测 top-8 重叠只有 62.5–75%，说明最队头确实有差异，只是规模不足以改变整体结果。
</details>

### Q9.3 怎么知道负载真的压饱和了？

三个独立信号：

| 信号 | 怎么看 | 实测 |
|---|---|---|
| **峰值并发** | benchmark 结果里的 `max_concurrent_requests` | rate 2 时只有 19（上限 32）→ 未饱和 |
| **p99 TTFT 量级** | 没排队就是毫秒级 | rate 2/4/6 是 77–264 ms；rate 8 跳到 1,408 ms |
| **vLLM 的 `Waiting: N reqs`** | 服务端日志 | 饱和期均值 237–248 |

**低负载下五臂数字相同是对照组，不是空结果**——它证明了"队列空时调度无效"，是要画进报告的边界条件。

<details><summary><b>追问：压测跑多长才够？</b></summary>

我第一次冒烟只压了 60 条、9.4 秒，结果 `popped_before_prediction` 是 54%——看起来很糟。

但分析后发现：那 54% 正好是**前 32 条**，也就是 running batch 还没满、一到就被调度走的那批。**它们根本没有调度决策可做。**

启动瞬态在 60 条里占一半，在正式的 1000 条里只占 3%。**所以压测规模要大到让稳态主导瞬态。**

正式实验用 1000 条/轮，rate 16 时持续 114 秒。
</details>

### Q9.4 完整结果

TTFT / TPOT 单位 ms。

| 负载 | ρ | 调度器 | p50 TTFT | mean TTFT | p90 TTFT | p99 TTFT | mean TPOT | req/s | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| **2** | 1.00 | FCFS | 26 | 29 | 42 | 67 | 11.80 | 1.99 | 495 |
| | | FCFS+Pred | 32 | 35 | 52 | 77 | 12.01 | 1.99 | 496 |
| | | Pred-SJF | 30 | 34 | 49 | 80 | 11.96 | 1.99 | 496 |
| | | TIE | 30 | 33 | 47 | 75 | 11.97 | 1.99 | 495 |
| | | TIE-oracle | 28 | 31 | 43 | 72 | 11.91 | 1.99 | 495 |
| **4** | 1.01 | FCFS | 27 | 30 | 44 | 71 | 12.37 | 3.96 | 982 |
| | | FCFS+Pred | 31 | 34 | 47 | 78 | 12.66 | 3.96 | 982 |
| | | Pred-SJF | 30 | 33 | 47 | 74 | 12.60 | 3.96 | 983 |
| | | TIE | 31 | 34 | 48 | 79 | 12.63 | 3.96 | 983 |
| | | TIE-oracle | 28 | 31 | 45 | 70 | 12.52 | 3.96 | 983 |
| **6** | 1.02 | FCFS | 29 | 35 | 47 | 113 | 13.06 | 5.90 | 1,464 |
| | | FCFS+Pred | 33 | 42 | 53 | 264 | 13.49 | 5.90 | 1,463 |
| | | Pred-SJF | 32 | 42 | 53 | 220 | 13.41 | 5.90 | 1,464 |
| | | TIE | 34 | 45 | 54 | 284 | 13.54 | 5.90 | 1,463 |
| | | TIE-oracle | 33 | 43 | 55 | 309 | 13.48 | 5.89 | 1,463 |
| **8** | 1.02 | FCFS | 34 | 169 | 569 | 1,140 | 13.67 | 7.81 | 1,941 |
| | | FCFS+Pred | 50 | 265 | 832 | 1,408 | 14.30 | 7.81 | 1,940 |
| | | **Pred-SJF** | 49 | 262 | **628** | 2,973 | 14.28 | 7.81 | 1,939 |
| | | **TIE** | 47 | 249 | **582** | 2,792 | 14.22 | 7.81 | 1,941 |
| | | TIE-oracle | 46 | 229 | 574 | 2,779 | 14.15 | 7.81 | 1,941 |
| **16** | 1.83 | FCFS | 22,157 | 22,762 | 41,475 | 45,495 | 14.01 | 8.94 | 2,222 |
| | | FCFS+Pred | 24,382 | 24,514 | 44,025 | 48,081 | 14.36 | 8.74 | 2,170 |
| | | **Pred-SJF** | **3,303** | 25,559 | 72,928 | 97,905 | 14.61 | 8.58 | 2,132 |
| | | **TIE** | **3,759** | 24,914 | 70,155 | 89,413 | 14.47 | 8.66 | 2,153 |
| | | TIE-oracle | **3,506** | 24,662 | 71,280 | 92,129 | 14.57 | 8.60 | 2,139 |
| **32** | 3.64 | FCFS | 37,129 | 37,588 | 68,590 | 75,672 | 13.99 | 9.01 | 2,237 |
| | | FCFS+Pred | 39,419 | 39,587 | 71,378 | 78,314 | 14.34 | 8.80 | 2,185 |
| | | Pred-SJF | 37,996 | 39,132 | 82,431 | 100,244 | 14.36 | 8.82 | 2,188 |
| | | TIE | 38,706 | 38,886 | 81,150 | 96,794 | 14.32 | 8.84 | 2,194 |
| | | TIE-oracle | 39,195 | 39,330 | 82,922 | 99,431 | 14.55 | 8.66 | 2,152 |

---

## 10. 结果与机制

### Q10.1 CVaR 项有效果吗？

**没有可测量的效果。** `TIE vs Predicted-SJF`（唯一变量：有无 β·CVaR 项）：

| rate | p50 | mean | p90 | p99 |
|---|---|---|---|---|
| 2 | −1.2% | −3.0% | −4.2% | −6.2% |
| 4 | +2.0% | +2.3% | +3.4% | +7.9% |
| 6 | +4.4% | +7.8% | +2.0% | +29.0% |
| 8 | −4.5% | −5.1% | −7.4% | −6.1% |
| 16 | **+13.8%** | −2.5% | −3.8% | −8.7% |
| 32 | +1.9% | −0.6% | −1.6% | −3.4% |

**没有一致方向**，幅度在噪声量级，rate 16 的 p50 甚至更差。

### Q10.2 是不是因为你的预测器不行？

**不是。** `TIE-oracle vs TIE`（唯一变量：σ 用预测值还是阶段一的真实拟合值）：p50 在 **−7.8% 到 +1.3%** 之间，同样是噪声量级。

**即使把 R²=0.05 的 σ̂ 换成 20 次真实采样拟合出的 σ，结果也不变。** 这正是引入 oracle 臂的目的。

### Q10.3 那是为什么？

**两层解释。**

**第一层：量级。** 打分可改写成

$$\text{score} = \mathbb{E}[X]\,(1+\beta r), \qquad r = \frac{\text{CVaR}_{0.9}}{\mathbb{E}[X]}$$

排序由两个因子的**相对离散度**决定（p10→p90）：

| 因子 | 跨度 |
|---|---|
| $\mathbb{E}[X]$ | **48.5×** |
| $(1+\beta r)$，β=0.1 | 1.07× |
| $(1+\beta r)$，β=0.5 | 1.26× |

**E[X] 的话语权是 CVaR 项的 40 倍以上。**

由此可导出**换序上界**：TIE 要交换 $\mathbb{E}_i<\mathbb{E}_j$ 的两条请求，需要

$$\frac{\mathbb{E}_j}{\mathbb{E}_i} < \frac{1+\beta r_{\max}}{1+\beta r_{\min}} \approx \mathbf{3.3\times}$$

**只能交换预期长度相差 3.3 倍以内的请求**，而 workload 的 E[X] 跨度是 48 倍。

**第二层（更根本）：冗余。** 对数正态族的期望**本身就是 σ 的增函数**（log-normal 下即 $\mathbb{E}[X]=e^{\mu+\sigma^2/2}$）。所以：

> **按 E[X] 排序的 SJF 本身就是一阶风险感知的。** σ 大的请求，其期望长度已被推高，SJF 自动把它推后。CVaR 项只提供二阶差异。

**验证**：固定 μ、令 σ 双峰（一半 0.05、一半 1.5），Spearman = **1.0000**——完全一致。因为 μ 固定时按 E[X] 排序已等价于按 σ 排序。

### Q10.4 那什么 workload 下 TIE 才会赢？

只有当**预期短但风险高**与**预期长但确定**两类请求共存时，均值和 CVaR 才会给出不同排序。决定变量是 $\mathrm{corr}(\mu,\sigma)$：

| Workload | corr(μ,σ) | Spearman | top-32 重叠 |
|---|---|---|---|
| **本实验实测** | **+0.04** | **0.9963** | **94%** |
| μ,σ 独立 | +0.02 | 0.9938 | 88% |
| 强反向 | −0.94 | 0.8799 | **53%** |
| 近乎完全反向 | −0.97 | 0.8109 | **41%** |
| 极端构造 | −1.00 | **−0.6338** | **0%** |

对应的真实场景：

| 类型 | μ | σ | 例子 |
|---|---|---|---|
| 长但可预测 | 高 | 低 | 定长翻译、"写 500 字…"、结构化生成、代码补全 |
| 短但会爆 | 低 | 高 | 歧义提问、时而拒答时而长篇、会触发 reasoning 的问题 |

**生产环境里这种组合并不罕见**——翻译 API（长、稳定）和聊天端点（中位短、重尾）跑在同一个服务上。而 LMSYS 纯聊天任务类型同质，`corr(μ,σ)=+0.04`，恰好落在 TIE 最无效的区域。

**注意：σ 整体放大无效**（2×～12× 都试过，Spearman 始终 ≥0.99），因为 2048 截断让 E[X] 和 CVaR 一起向上限饱和。

### Q10.5 你怎么和论文的结果对照？

**论文的 100 RPS 是发起速率，不是实际吞吐。**

用它自己的 Table 3 反推：FCFS 跑完 3,000 条需 229.37 s → **实际吞吐 13.1 req/s**。发起 100 RPS 即**过载 7.6 倍**。我 rate 32 时实际 9.01 req/s，过载 3.6 倍——**同一区间，只是标注方式不同**。

**它的 "Per-token Latency" 是归一化延迟**：

$$\text{PTL}_i = \frac{E2E_i}{N_i}$$

用其自身数据交叉验证（反推输出长度一致）：

| 测试集/模型 | Avg TTFT | Avg PTL | 反推 $N$ |
|---|---|---|---|
| ShareGPT 8B | 161.56 s | 1.66 s/tok | ≈ 97 |
| ShareGPT 8B (P90) | 316.07 s | 3.27 s/tok | ≈ 97 ✓ |
| Alpaca 8B | 83.65 s | 1.45 s/tok | ≈ 58 |
| Alpaca 70B | 235.64 s | 4.52 s/tok | ≈ 52 ✓ |

该指标被 TTFT 完全主导（我的 TPOT 只有 0.014 s/token，相差 100–600 倍）。

**⚠️ 这个指标在结构上偏袒 SJF 类策略。** 分母是输出长度，让短请求等待会被**放大**惩罚，让长请求等待则被**稀释**。极端例子——两条请求，10 token 和 1000 token：

| 策略 | 短请求等待 | 长请求等待 | 短的 PTL | 长的 PTL | **平均 PTL** |
|---|---|---|---|---|---|
| FCFS | 100 s | 100 s | 10.0 | 0.1 | **5.05** |
| SJF | 0 s | 200 s | 0.0 | 0.2 | **0.10** |

**总等待完全相同（200 s），归一化延迟差 50 倍。**

所以论文报告的收益可拆为三部分：

| 来源 | 是否真实减少了总等待 |
|---|---|
| 归一化指标的结构性偏袒 | ❌ 纯指标效应 |
| 等待在请求间重新分配 | ❌ 守恒，仅分布改变 |
| 吞吐提升（Table 3 的 1.42×） | ✅ 真实，但需显存成为约束 |

<details><summary><b>追问：论文的吞吐提升你复现出来了吗？</b></summary>

**没有，而且我认为在我的配置下结构性不可能**（Q5.3）。

我去读了作者公布的代码想找答案，结果**公布的配置排除了我能识别的唯一机制**：

- `start-server.sh` 里 `MAX_NUM_SEQS=32` 对**所有**策略生效，包括 FCFS
- 按他们自己的 `--gpu-memory-utilization 0.91` 和 `--max-model-len 8192` 算，8B（TP=1）KV 容量约 389k token、70B（TP=4）约 437k token，而 32 条请求**全部用满 8192 上下文**也只要 262k。**两种规模都装得下**，并发同样是配置约束

另外发现一个方法学问题：脚本里 `fcfs` 落到 `else` 分支拿到**全部** GPU，而 `ua`/`ssjf`/`ltr` 的 `TP_SIZE = GPU数 − 1`（留一张给预测器）。同一设备列表下 FCFS 会多一倍模型并行度——方向对 TIE 不利，但脚本不自动配平，实际调用方式没公布。

**压测 harness 根本没在仓库里**（README 只列了训练代码和四个调度文件），所以无法进一步排查。**我没有确证，不编解释。**
</details>

---

## 11. 与参考实现的对比

读过作者公布的 `train/model_train.py` 和四个调度文件。差异分三类。

### 11.1 我修掉的方法学问题

| 项 | 参考实现 | 我的实现 | 后果 |
|---|---|---|---|
| **数据切分** | `train_test_split(df, test_size=0.4)`，**按行切、不去重** | **按唯一 prompt 文本分组切分** | 我的数据 9,998 条里有 **917 条重复 prompt**（9.2%），按行切会泄漏 |
| **随机种子** | `RANDOM_SEED = random.randint(0, 10000)`，**每次运行都不同** | 固定 | 他们单次结果无法复现，跑两次也无法比较 |
| **打分随机性** | 蒙特卡洛用全局 numpy RNG | 独立 seeded 生成器（`TIE_SCORE_SEED`） | 同一 prompt 每次打分不同，实验不可复现 |
| **β 公式的 B** | 独立环境变量 `UA_GPU_BATCH_SIZE`，靠启动脚本导出 | 从 `self.max_num_running_reqs` **实时读取** | 绕过启动脚本就静默失配，不报错 |
| **prompt 提取** | 正则 first-match 锚定第一个 `assistant` | 锚定**最后一个** | prompt 本身含 "assistant" 时会被截断（LMSYS 里有讨论聊天记录的 prompt） |

### 11.2 独立收敛到同一做法

他们的**验证损失**用的是 `loss_mu + loss_sigma`（不加权），而训练用的是 3→4 的动态 σ 权重。

我在自己的代码里也发现并修了这个问题（引入 `MONITORING_SIGMA_WEIGHT = 1.0`）——**如果监控指标跟着训练权重变，你分不清模型变好了还是权重变了**。

这条说出来比"我修了个 bug"有分量，因为它说明判断和原作者一致。

### 11.3 他们有而我没采用的

`compute_sample_weights()` 用绝对阈值上采样尾部：

```python
weights[df['logt_mu'] > 5.5] = 1.5
weights[df['logt_sigma'] > 1.0] = 1.5
weights[(df['logt_mu'] > 6.0) | (df['logt_sigma'] > 1.3)] = 2.0
```

**我没用，因为这些阈值是针对他们的标签分布硬编码的**。套到我的标签上：

| 阈值 | 他们给的权重 | 我的数据命中 |
|---|---|---|
| `logt_mu > 5.5` | 1.5 | **51.3%**（5,130 条）✓ 合理 |
| `logt_sigma > 1.0` | 1.5 | **0.36%**（36 条）✗ |
| `mu > 6.0 或 sigma > 1.3` | 2.0 | 40.4%（几乎全来自 μ） |

**μ 的阈值合理，σ 的几乎不命中。** 没人会为 36 个样本写一条加权规则——**所以他们的 σ 分布比我的宽**。

<details><summary><b>追问：他们 σ 为什么更宽？是采样次数吗？</b></summary>

**我提了这个假设，然后用子采样重拟合推翻了它**（见 Q13.5）。

20 次采样已达渐近 σ 的 96.5%，加到 100 次只涨 2.9%。**不是采样次数。**

剩下的候选是模型不同（他们示例用 Llama-3-8B）、采样参数不同、或者**我显式关掉了 Qwen3 的 thinking mode**——思考链长度本身高度不可预测，开着它 σ 会宽得多。**我倾向于最后一个，但没验证。**
</details>

### 11.4 其他对齐的部分

超参我是对齐的：`MAX_LENGTH=512`、LR 2e-5 → 5e-5、σ 权重 3→4 线性、`clip_grad_norm_(1.0)`、μ 用 z-score / σ 用 log1p+z-score、多池化架构、双分支。

**编码器冻结时机不同**——他们 20 epoch 里第 12 个冻结，我在第 2 个（我的数据量小得多，更早过拟合）。

---

## 12. 生产调试

### Q12.1 讲一个你排查过的难缠问题

**代理让健康检查永远不通。**

压测脚本等服务启动，轮询 120 次全部失败，10 分钟后放弃。**但服务一直是好的**——日志里 `Application startup complete` 和调度器的心跳都在。

**根因**：集群设了 `http_proxy=10.99.0.130:3128` 但没设 `no_proxy`。curl 把 `localhost:8000` 也往代理上送，而代理机到不了计算节点的 loopback。

**我自己的两个错误放大了它**：

```bash
if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
```

1. `2>&1 >/dev/null` 把 curl 的报错全吞了——"代理超时"/"curl 不存在"/"服务真没起来"三种完全不同的情况，**表现一模一样**
2. 依赖了计算节点上不保证存在的 curl

**更要紧的是**：`vllm/benchmarks/serve.py:532` 用 `aiohttp.ClientSession(trust_env=True)`，**压测客户端也会走代理**。只修健康检查的话，60 条压测请求会在另一个地方失败，更难定位。

**修法**：导出 `no_proxy`、探活换成 Python + 空 `ProxyHandler`（不依赖 curl）、失败时打印完整诊断（启动是否完成、代理变量、端口监听、`/health` 路由是否注册、原始报错）。

### Q12.2 讲一个最危险的 bug

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

**修法**：端口改成 `8000 + job_id % 1000`、启动前预检端口、**先查进程存活再查健康**、清理时按命令行 `pkill` 兜底——`vllm serve` 的 engine 是**子进程**，杀父进程会留下孤儿继续占着端口和显存（这个僵尸就是上一次冒烟测试自己制造的）。

### Q12.3 还有别的吗？

**默认配置悄悄删掉了数据。**

`vllm bench serve --save-result` 看起来保存了结果，但：

```python
# vllm/benchmarks/serve.py:1489
if not args.save_detailed:
    for field in ["input_lens", "output_lens", "ttfts", "itls", ...]:
        del result_json[field]
```

**删除发生在写文件之前。** 所以不加 `--save-detailed` 就只有聚合分位数，拿不到逐请求数组——导致我算不了需要逐请求数据的指标（比如论文口径的 PTL）。

**教训**：用别人的工具收集数据时，要确认它**实际写进文件的是什么**，而不是它看起来收集了什么。

### Q12.4 集群相关的坑

| 问题 | 现象 | 解法 |
|---|---|---|
| **SSH 断线丢进度** | 交互式 `srun` 跑到第 8 epoch，笔记本合盖，**整个分配被撕掉** | 用 `sbatch`。`nohup`/`disown` 没用——是 SLURM 分配本身被回收，不是进程被挂断 |
| **登录节点 OOM** | `import torch` 直接 `Killed` | 登录节点有内存限制。检查文件用 `grep` 不要 import |
| **GPU 排不上队** | 不指定型号的 `--gres=gpu:1` 排到次日 | **指定具体型号**（`gpu:v100-sxm2:1` / `gpu:a100:1`）反而快——反直觉但实测有效，推测是该型号竞争较少 |
| **时限太长排不上** | `--time=08:00:00` 回填机会小 | 按实测重算需求。我把 8h 降到 4h（实际用 2.4h），并在脚本里**开跑就打印耗时预估** |
| **交互式分区有上限** | `gpu-interactive` 拒绝 >2h 的请求 | 长任务必须 `sbatch` |

---

## 13. 预测器：服务路径上的一个模型

**不要用这节开场**，但被问到就完整讲。按 infra 角度组织——纯 ML 细节在 [`Interview_Deep_Dive.md`](Interview_Deep_Dive.md) 第 4 节。

### Q13.1 这个模型在服务路径上的成本是多少？

**DeBERTa-v3-base，184M 参数，和 Qwen3-8B 共享同一张 A100-40GB。**

| 项 | 值 | 怎么测的 |
|---|---|---|
| 预测批延迟 | **44.8–48.3 ms** | 队列的 `avg_batch_predict` 计数器 |
| TPOT 开销 | **+2~5%** | `FCFS+Predictor` 相对裸 `FCFS` |
| 吞吐开销 | **−2.2%**（高负载） | 同上 |
| 显存 | fp16 权重 ~0.37 GB + 激活 | 挤在 `(1 − util)` 的 4.9 GB 里 |

**这是论文没报告的数字**，而它正是我加 `FCFS+Predictor` 对照臂的目的：**没有这一臂，你分不清延迟变化是调度带来的还是预测器开销带来的。**

**三个让它能共卡的工程决定**：

1. **fp16 加载**——权重减半。因为 vLLM 在构造 scheduler **之前**就分配完了 KV cache
2. **`max_batch_size` 从参考实现的 128 降到 32**——128 条 × 512 token 的 DeBERTa forward 会炸掉那 4.9 GB
3. **异步、不在关键路径上**——请求先拿占位分入堆，后台线程算完再惰性更新

<details><summary><b>追问：为什么不把预测器放到另一张卡上？</b></summary>

参考实现就是这么做的（`TP_SIZE = GPU数 − 1`，留最后一张给预测器）。

我没这么做，因为 **`FCFS+Predictor` 对照臂已经把干扰对称化了**——两个臂都加载预测器、都跑预测，`调度臂 − 对照臂` 这个差值里不含预测器开销。单卡不影响核心对比的有效性，而且省掉了申请 2 张卡的排队风险。

代价是：`调度臂 vs 裸 FCFS` 这个"对外宣称的收益"会偏保守。所以我**两个基准都报**。
</details>

### Q13.2 训练和服务的输入一致吗？

**这是个真实的 train/serve skew，而且不处理不会报错。**

训练时用的是 LMSYS 的**裸 prompt 文本**。但服务时，压测客户端会套 Qwen3 的 chat template：

```python
tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                              add_generation_prompt=True, tokenize=False)
```

所以调度器拿到 `request.prompt_token_ids` 解码出来的是**模板化后的文本**：

```
[system\n{system}\n]user\n{prompt}\nassistant\n[<think>\n\n</think>\n\n]
```

直接喂给 DeBERTa 就是在给模板样板文字打分，**而且 forward 不会失败，你拿到的是一个看起来正常的数字**。

**处理**：在预测器里把模板剥掉，锚定在**第一个 `user` 标记之后**和**最后一个 `assistant` 标记之前**，并写了单元测试覆盖边界情况。

<details><summary><b>追问：为什么是"最后一个" assistant 标记？</b></summary>

因为 prompt 本身可能包含 "assistant"——LMSYS 里有不少讨论聊天记录的 prompt。

锚定第一个会把这种 prompt 截断：

```
user\nRewrite this:\nassistant\nsaid hello\nplease make it formal\nassistant\n
         ^^^^^^^^^^^^^^^ 锚定第一个就只剩这里
```

参考实现用的是正则 first-match，有这个问题。
</details>

<details><summary><b>追问：为什么不干脆在压测时关掉 chat template？</b></summary>

因为那会破坏**生成条件的一致性**。我的标签是用 chat template 采样 Qwen3-8B 得到的，如果服务时不套模板，输出长度分布就和标签描述的不是同一件事了。

**正确的做法是：服务时套模板（和打标签一致），预测器内部剥模板（和训练一致）。** 两边都对齐，而不是改一边去迁就另一边。
</details>

### Q13.3 模型结构是什么？为什么这么设计？

```
prompt text
  → DeBERTa-v3-base encoder（12 层，hidden 768）
  → 多池化：[CLS, masked mean, masked max] 拼接 = 768×3 = 2304
  → 两条独立分支（μ 和 σ），各自：
       (Linear → LayerNorm → GELU → Dropout 0.2) × 2，hidden 256
       → Linear(256→128) → GELU → Dropout 0.1 → Linear(128→1)
  → (μ̂, σ̂)
```

**为什么多池化不只用 CLS**：CLS 表示整体语义倾向，但会丢失局部最强信号。预测长度时，prompt 里一个短语（"详细解释"、"列出所有"）可能强烈暗示长回答——**max-pooling 保留这种信息更好**。

**为什么两条独立分支不共享主干**：μ 和 σ 衡量的是不同的东西——"多长" vs "我有多不确定"。共享末端层会让两者梯度互相干扰。

这个设计被单任务消融验证了：`mu_weight=0` 只训 σ，R² 依然是 0.03，**说明不是多任务干扰**。

**归一化**：μ 做 z-score；σ 先 `log1p` 再 z-score（σ 分布右偏，log1p 压缩尺度且避免 log(0)）。**统计量只在 train split 上拟合**，val/test 和推理复用——在别的 split 上重新拟合会静默破坏训练/推理的对应关系。

### Q13.4 训练时显存不够怎么解决的？

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

而且模型用的是 **LayerNorm 不是 BatchNorm**——LayerNorm 对每个样本独立归一化，所以拆成 4×8 再累积梯度**数学上完全等价**。BatchNorm 的统计量依赖批内其他样本，拆开就不等价了。

还有一个容易写错的点：**梯度裁剪必须在累积完成后做**，不能每个 micro-batch 各裁一次——那会把 4 个 1/4 缩放的梯度分别裁剪，和裁剪一个完整梯度结果不同。

<details><summary><b>追问：为什么不用 AMP 或 gradient checkpointing？</b></summary>

都能解决，但引入新变量（AMP 的数值稳定性、checkpointing 的重算开销）。

**梯度累积是数学等价的，不改变任何训练动态**——我当时正在排查 σ 学不出来的原因，**最不需要的就是再引入一个可能影响结果的变量**。

另外我把 tokenize 从 `__getitem__` 挪到 `collate_fn` 时，**验证了改动前后 loss 值逐位相同**。
</details>

<details><summary><b>追问：为什么不用 length-grouped batching？</b></summary>

它更优——把长度相近的分到一批能同时降低平均和最坏情况。

我没用是因为它**破坏随机性**：同一批里的样本变得相关，对小数据集（9,081 条唯一 prompt）的泛化有影响。梯度累积没这个副作用。
</details>

### Q13.5 结果怎么样？σ 学不出来你怎么排查的？

| 目标 | test R² |
|---|---|
| μ（期望长度） | **0.7668** |
| σ（不确定性） | **0.0501** |

**σ 基本学不出来。我系统性排除了五个解释：**

| 假设 | 怎么测的 | 结果 |
|---|---|---|
| 标签是随机噪声 | **分半信度 + Spearman-Brown 校正** | ❌ σ 的 $r^2$=0.6031±0.0285，校正后 **0.8740**。天花板远高于 0.05 |
| 标签因采样不足被压缩 | **子采样重拟合**（20 个样本抽 5/10/15 重新拟合） | ❌ $\sigma(n)=\sigma_\infty-c/n$ 拟合到四位小数，**n=20 已达渐近值 96.5%**，加到 100 次只涨 2.9%；`Spearman(σ₁₀,σ₂₀)=0.9371` |
| 损失权重失衡 | σ 权重 1.0 / 2.0 / 3.0→4.0 + 梯度裁剪 | ❌ 无改善，2.0 反而更差（R²=0.014） |
| 过拟合 | 第 2 epoch 冻结 encoder，lr→5e-5 | ❌ 机制上生效（可训参数降到 1.38M，val loss 1.3021 vs 1.3602），σ 不变 |
| 多任务干扰 | `mu_weight=0` 单训 σ 头 | ❌ R²=0.0316 |

**剩下最可能的解释：信号本身不在 prompt 文本里。** 输出长度的方差很大一部分来自**解码的随机性**——同一条 prompt 采样 20 次得到 20 个不同长度，这部分方差无论什么模型都无法从文本预测。μ 可学，σ 可能本质上就是个噪声量。

**这个结论对调度是有用的**，而不只是一个失败：我用 oracle 参数单独跑了一臂，结果和预测参数一样，**说明调度的瓶颈不在预测精度**。

<details><summary><b>追问：分半信度具体怎么做的？</b></summary>

把每条 prompt 的 20 个样本随机劈成两半各 10 个，分别拟合 σ，然后算两边的相关。重复多次取均值和标准差。

$r^2 = 0.6031$ 是**半长度**（10 个样本）的信度，用 Spearman-Brown 公式校正到全长度：

$$r_{\text{full}} = \frac{2r}{1+r} \;\Rightarrow\; 0.8740$$

**但我后来发现自己对这个测试的解读有漏洞**：信度测的是**一致性**，不是**准确性**。如果两半都因为抽不到尾部而偏低，它们会一致地偏低，信度照样高。

所以它只排除了"随机噪声"这一种失败模式，**没排除"系统性压缩"**——那需要子采样重拟合才能测。
</details>

<details><summary><b>追问：子采样重拟合的结果是什么？</b></summary>

9,998 条全量，每条从 20 个样本随机抽 5/10/15 个重新拟合：

| n | σ 均值 | 相对 n=20 | Spearman(σₙ, σ₂₀) |
|---|---|---|---|
| 5 | 0.1166 | −10.8% | 0.8189 |
| 10 | 0.1262 | −3.4% | **0.9371** |
| 15 | 0.1290 | −1.3% | 0.9744 |
| 20 | 0.1307 | — | — |

用 $\sigma(n)=\sigma_\infty-c/n$ 拟合，**预测值和实测值小数点后四位吻合**：$\sigma_\infty=0.1354$, $c=0.0941$。

> **n=20 已达渐近值的 96.5%；20 → 100 只能让 σ 均值涨 2.9%。**

压缩效应只在极窄一段成立：低半 −1.6%、p90–p99 −3.1%、**top 1% −11.7%**（top 10% 内部 Spearman 降到 0.7788）。但按同样 $1/n$ 规律外推，即使 top 1% 也只挽回约 9%。

**假设被我自己的数据推翻。**
</details>

---

## 14. 标签管线

### Q14.1 标签是怎么来的？

**阶段一自建的管线**（参考实现只发布了训练代码，采样和拟合没有）：

```
LMSYS-Chat-1M prompt[0:10000]
  → Qwen3-8B 每条采样 20 次
     temperature 0.7 / top_p 0.8 / top_k 20 / min_p 0
     enable_thinking=False，max_tokens=2048
  → MLE 拟合 log-t(μ, σ, ν=3.5)
  → 9,998 条 (prompt, μ, σ)
```

采样参数用的是 **Qwen3 model card 推荐的非思考模式设置**——论文和仓库都没说用什么，而**拟合出的 (μ, σ) 是条件于解码配置的**，所以必须明确记录。

### Q14.2 为什么选 log-t 而不是 log-normal？

输出长度右偏，对数后仍有重尾。$t$ 分布（ν=3.5）比正态尾更厚，KS 检验通过率更高。

**ν 固定不学习**，这样 μ、σ 两个参数可解释、可跨 prompt 比较。我单独扫过 ν（`scripts/sweep_nu.py`）确认 3.5 是合理选择。

### Q14.3 讲一个这里的 bug

**退化检测写了但从来没生效过。**

```python
if np.std(np.log(lengths)) == 0.0:     # 永远不成立
    return degenerate_fit(...)
```

`np.std` 对一组**数值完全相同**的浮点数返回的是 **~4.44e-16，不是精确的 0.0**（浮点累加误差）。所以这个分支一次都没进去，**9,998 条里 1,228 条（12.3%）走错了路径**。

排查时还发现第二个独立问题：另有 **214 条（2.1%）**采样确实有分散度，但拟合出的 σ 接近 0（最糟的一条 `raw_std=0.2091 → 7.59e-06`）。这是**重尾分布 MLE 的病态**——$t$ 分布可以用极小的 σ 配合极端 $t$ 值来"解释"离群点。

**三部分修复**：

```python
_DEGENERATE_STD_THRESHOLD = 1e-8          # 1. 阈值，不用 == 0
_MIN_LOG_SIGMA = math.log(0.01)           # 2. 给 L-BFGS-B 加下界
_MIN_FIT_TO_RAW_STD_RATIO = 0.1           # 3. 事后兜底

if nu > 2 and sigma_hat < sample_std * _MIN_FIT_TO_RAW_STD_RATIO:
    sigma_hat = sample_std * math.sqrt((nu - 2) / nu)   # 矩估计
```

重新拟合改变了 **1,561 条**。同样的 `== 0.0` 还出现在另一个脚本里，一并修了。

**为什么这个 bug 值得讲**：σ=0 意味着 CVaR = E[X]，**风险项完全退化**。不修的话 12% 的标签是错的，而且模型会学到"很多 prompt 的不确定性是 0"这个假事实。

而它**不会报错、不会崩溃**，只会静默产出错的标签——和 Q12.2 那个僵尸服务是同一类问题。

---

## 15. 开放设计题

### Q15.1 如果让你优化这个服务，先做什么？

按预期收益排：

| 优先级 | 动作 | 预期收益 | 代价 |
|---|---|---|---|
| **1** | **`max_num_seqs` 32 → 512** | 吞吐 **~7×**（2,237 → ~15,500 tok/s） | TPOT 14 → 33 ms |
| 2 | 开 prefix caching | 取决于 workload 的前缀重复率 | 显存 |
| 3 | 量化（FP8/INT8） | 权重减半 → step 时间减半 → **吞吐翻倍** | 精度 |
| 4 | 投机解码 | 1.5–2×（取决于接受率） | 复杂度、draft 模型显存 |
| 5 | 调度优化（本项目） | **吞吐 0**，p50 TTFT −86% | 预测器 −2% 吞吐 |

**把调度排在最后是有意的**：我自己的实验证明了它不改变吞吐。**在 batch 只开到容量 2.8% 的情况下谈调度优化，是在优化一个不是瓶颈的东西。**

<details><summary><b>追问：那什么情况下调度才值得做？</b></summary>

三个条件同时成立：

1. **已经饱和**——ρ 接近或超过 1，队列真有积压
2. **SLO 关心的分位数低于 $p^\*=100/\rho$**——否则调度反而让它更差
3. **吞吐已经优化到位**——否则应该先去拿那 7 倍

另外，如果并发是**显存约束**而不是配置约束，调度还能通过"短请求占更少 KV → 装下更多条"提高吞吐。我的配置下这条路被堵死了。
</details>

<details><summary><b>追问：量化为什么能让吞吐翻倍？</b></summary>

因为 decode 是**带宽瓶颈**（Q4.1）。step 时间 ≈ 读取字节数 / 带宽，而权重占 91%。

FP8 权重从 16.4 GB 降到 8.2 GB → 每 step 读取从 18 GB 降到 9.8 GB → **step 时间几乎减半** → 吞吐翻倍。

**这就是 roofline 分析的实用价值**：它直接告诉你哪个优化有效、能有多少。不做这个分析，"量化能加速"只是个口号。
</details>

### Q15.2 怎么给这个服务定 SLO 和容量？

见 Q5.6。核心是：**不是看吞吐天花板，是看 SLO 对应的 ρ。**

| SLO | 可用容量 |
|---|---|
| p90 TTFT < 100 ms | ~6 req/s |
| p90 TTFT < 1 s | ~8 req/s |
| 离线批处理 | 8.8 req/s |

而且 ρ 接近 1 时延迟急剧恶化，生产上留 20–30% 余量。

### Q15.3 如果要做多租户 / 优先级怎么办？

vLLM 自带 `--scheduling-policy priority`，按外部传入的 priority 字段排。

但**朴素的优先级有饥饿问题**——低优先级可能永远排不上。我的队列里那套**乘性时间衰减**（Q7.2）正好能用：

$$\text{effective} = \text{priority} \times \gamma^{t_w/\tau}$$

把 τ 调到和 SLO 同量级（比如低优先级保证 10 秒内被调度 → τ 设成几秒），就能在"优先级有效"和"不饿死"之间取平衡。

**但要注意我实测的教训**（Q7.2 追问）：当前的 γ=0.9/τ=30s 要**等待相差 18 分钟才能翻转 48 倍的分数差**——实际上几乎没起作用。做多租户的话这个参数必须重新标定。

### Q15.4 这套方法能推广到别的场景吗？

能，但**边界条件很明确**（Q10.4）：

> SJF 类调度的收益需要 **E[X] 有足够的跨度**（我这里是 48.5×）；
> CVaR 类风险感知的额外收益需要 **corr(μ,σ) 显著为负**。

所以：

| 场景 | SJF 有用吗 | CVaR 额外有用吗 |
|---|---|---|
| 混合长度的聊天服务 | ✅ 长度跨度大 | ❌ corr(μ,σ)≈0 |
| 定长任务（翻译、分类） | ❌ 长度都差不多 | ❌ |
| **翻译 API + 聊天端点混部** | ✅ | ✅ **corr(μ,σ) 为负** |
| 开了 reasoning 的服务 | ✅ | 可能 ✅（思考链长度不可预测） |

---

## 16. 数字速查

| 类别 | | |
|---|---|---|
| **硬件** | A100-**40GB**，HBM 1,555 GB/s | |
| | Qwen3-8B bf16 权重 | 16.4 GB |
| | 模型几何 | 36 层，32 Q head，**8 KV head（GQA）**，head_dim 128 |
| **配置** | `max_num_seqs` | **32**（vLLM 官方默认 128） |
| | `max_model_len` / `max_num_batched_tokens` | 8192 / 2048 |
| | `gpu-memory-utilization` | 0.88 |
| | prefix caching | 关闭 |
| **KV cache** | 容量 | **394,544 token** |
| | 每 token（36×2×8×128×2B） | **144 KB** |
| | 平均序列（输入 102 + 输出 249） | 351 token |
| | 理论可容纳并发 | **1,124 条** |
| | **实测利用率** | **2.0%**（峰值 2.6%） |
| | vLLM 自报满上下文并发 | 48.16× |
| **吞吐** | 天花板 | **8.8–9.0 req/s，2,237 tok/s** |
| | TPOT | **14 ms** |
| | roofline 下界（16.4 GB ÷ 1555 GB/s） | **10.5 ms** |
| | **带宽利用率** | **75%** |
| | 验算误差（32÷14ms vs 实测） | **2%** |
| | 权重占每 step 读取量 | **91%** |
| | 顺序对吞吐的理论上界 | **4.5%** |
| **批次** | Running 占用 | 31.60–31.89 / 32（跨臂差 <1%） |
| | 饱和期 Waiting 均值 | 237–248 |
| | **抢占次数** | **0**（全部五臂） |
| **排队** | ρ（rate 8 / 16 / 32） | 1.02 / 1.83 / 3.64 |
| | $W(i)=i(1/\mu-1/\lambda)$ 误差 | **4–11%** |
| | $p^\*=100/\rho$ 命中 | **8 / 9** |
| **主结果** | rate 16 p50 TTFT | **24,382 → 3,303 ms（−86.5%，7.4×）** |
| | 同条件 mean TTFT | +4.3%（**守恒**） |
| | 同条件 p90 / p99 TTFT | +65.7% / +103.6% |
| | rate 8 p90 TTFT（TIE） | **−30.1%** |
| | TIE vs SJF | p50 −4.5% ~ +13.8%，**噪声** |
| | oracle vs TIE | p50 −7.8% ~ +1.3%，**噪声** |
| **预测器** | DeBERTa-v3-base | 184M，fp16 |
| | 预测批延迟 | **44.8–48.3 ms** |
| | `max_batch_size`（参考实现 128） | **32** |
| | 服务开销 | TPOT **+2~5%**，吞吐 **−2.2%** |
| | μ / σ 的 test R² | **0.7668 / 0.0501** |
| | σ 标签信度（Spearman-Brown） | **0.8740** |
| | 采样次数影响（20→100） | **+2.9%**（n=20 已达渐近 96.5%） |
| **打分** | 分布 / 自由度 | log-t，ν=3.5 |
| | 截断 | E[X] 2028，CVaR 2048 |
| | 蒙特卡洛 | 10,000 次，~0.3–0.5 ms |
| | β 范围 | clip(0.1·L_q/B, **0.1, 0.5**) |
| | E[X] 跨度 vs 打分乘子跨度 | **48.5×** vs 1.07–1.26× |
| | 换序上界 | **3.3×** |
| | 离线预测 Spearman / top-32 | **0.9963 / 94%** |
| **标签** | 规模 | 9,998 条（唯一 prompt 9,081，**重复 9.2%**） |
| | 采样 | 20 次/prompt，temp 0.7 / top_p 0.8 / top_k 20 |
| | 退化 σ bug 影响 | **1,228 条（12.3%）** + 214 条 MLE 病态 |
| | 重新拟合改变 | **1,561 条** |
| | σ 分位 | p50 0.097，p90 0.248，p99 0.691，max 1.978 |
| | corr(μ, σ) | **+0.04** |
| **规模** | 压测 | **30 轮，30,000 请求，0 失败** |

---

## 17. 已知局限 / 不许硬编

被问到就直说，然后讲会怎么补。**编一个数字比承认没测更糟。**

### 17.1 实验本身的局限

| # | 局限 | 影响 |
|---|---|---|
| 1 | **没有直接采样 GPU 利用率** | roofline 是推算，但用实测 TPOT 验证到 2% 误差。要补就加 `nvidia-smi -l 5` 后台采样 |
| 2 | **rate 8（ρ=1.02）到 rate 16（ρ=1.83）之间没采样** | 而那是生产最常见的区间 |
| 3 | **`max_num_seqs=512` 的实验没做** | 吞吐 7× 是 roofline 外推，不是实测。而且这是唯一能验证"调度能否提升吞吐"的配置 |
| 4 | **没有逐请求数据**（`--save-detailed` 当时没加） | 算不了论文口径的 PTL 及其分位数 |
| 5 | **只测了一种 workload** | corr(μ,σ) 的边界条件只有离线排序分析，无端到端验证 |

### 17.2 ⚠️ 采样配置不一致

**打标签和服务用的解码参数不同：**

| | 阶段一打标签 | 阶段三压测 |
|---|---|---|
| temperature | **0.7** | **0.6** |
| top_p | **0.8** | **0.95** |
| thinking mode | **显式关闭** | **未指定（Qwen3 默认）** |

压测客户端传的是 `temperature=None`，所以 vLLM 用了模型 `generation_config` 的默认值。而 `generate_predictor_labels.py` 的注释**明确写了**必须一致。

**影响评估**：

- 标签描述的是 temp 0.7/top_p 0.8 下的长度分布，而服务时的实际分布不同 → **预测信号系统性偏移**
- 但**所有臂的服务配置相同**，所以**臂间对比仍然有效**
- 偏移的方向：top_p 0.95 比 0.8 更宽松 → 实际 σ 可能**高于**标签所述 → 这会**低估**调度的可用信号
- 所以 **7.4× 的 p50 改善是个下界**；但 **CVaR 项的 null 结果可能部分由这个不一致造成**，我没有单独验证

**这条是我写这份文档时才发现的**，不是实验时就知道的。要修就是重跑一轮把采样参数显式对齐。

### 17.3 对参考实现的未解问题

| # | 问题 | 现状 |
|---|---|---|
| 1 | **Table 3 的 2.3× 吞吐提升机制** | 我读了他们的代码，**公布的配置排除了我能识别的唯一机制**（Q10.5 追问）。压测 harness 没发布，无法进一步排查。**不编解释** |
| 2 | **他们的 σ 为什么比我的宽** | 采样次数这个解释我**测过并排除了**。剩下候选是模型不同、采样参数不同、或我关掉了 thinking mode。**倾向最后一个，没验证** |
| 3 | **他们报告的 σ 预测精度** | 没下载论文正文核对，所以不知道我的 R²=0.05 是"复现失败"还是"复现了一个已知难点" |
| 4 | **σ 为什么学不出来的最终解释** | "信号不在 prompt 文本里"是排除法剩下的，**不是直接测出来的**。要验证需要新的采样实验（比如固定 seed 贪心解码看 σ 是否塌缩） |

---

## 附：复现与相关文档

```bash
# 1. 关口检查：外挂能不能挂上去（纯 CPU，18 项）
python scripts/verify_vllm_integration.py

# 2. 生成 workload 与 oracle 标签（检查标签覆盖率）
python scripts/export_benchmark_dataset.py

# 3. 实验前的排序一致性分析——这一步决定了要加第 3 臂
TIE_SCORE_SEED=0 python scripts/rank_agreement_check.py

# 4. 采样次数对 sigma 的影响（不要 GPU，约 1 分钟）
python scripts/sigma_sample_size_bias.py

# 5. 单臂冒烟（GPU）
bash scripts/smoke_test.sh tie 60 32

# 6. 全量基准：5 臂 × 6 速率
sbatch hpc/benchmark_serving.slurm

# 7. 分析 + 出图
python scripts/analyze_benchmark.py results/bench_<jobid> --plot
```

| 文件 | 内容 |
|---|---|
| [`Interview_Deep_Dive.md`](Interview_Deep_Dive.md) | 按项目阶段组织，含更多 ML 细节和研究结论 |
| [`Phase3_Results.md`](Phase3_Results.md) | 30 轮完整数据和机制分析 |
| [`Phase3_Scheduling_Evaluation_Plan.md`](Phase3_Scheduling_Evaluation_Plan.md) | 五臂设计的推理过程 |
| [`Phase2_Execution_Log.md`](Phase2_Execution_Log.md) | 阶段二全部消融实验的配置和结果 |
