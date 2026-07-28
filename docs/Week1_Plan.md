# Week 1 详细计划:Simulator 框架搭建 + FCFS MVP

## 总体目标

搭好一个可扩展的仿真框架(Request 类、Scheduler 基类、Event-driven Simulator、Metrics 统计),
用 FCFS 跑通整条链路,后续算法(Oracle SJF / Predicted SJF / Priority)只需继承
Scheduler 基类、重写 `schedule()`,不用动主循环。

---

## 每日任务拆解

### Day 1:设计确认 + 项目骨架

**任务**
- 确认字段设计(Request 类字段、状态机、metric 定义)——已在讨论中定稿,直接落地
- 搭建项目目录结构:

```
llm-scheduler-sim/
├── src/
│   ├── request.py          # Request 类 + RequestStatus 枚举
│   ├── scheduler.py         # Scheduler 抽象基类
│   ├── schedulers/
│   │   ├── fcfs.py          # FCFS 具体实现
│   ├── simulator.py          # Event-driven 主循环
│   ├── event.py              # Event 类 + 事件类型枚举
│   ├── workload.py           # Request 生成器(到达时间、output_len 分布)
│   ├── metrics.py            # Waiting/Response time, Throughput, Fairness 计算
├── experiments/
│   ├── run_fcfs_baseline.py  # Week1 的跑批脚本
├── tests/
│   ├── test_simulator.py     # 回归测试
├── README.md
```

- 写 `request.py` 和 `event.py`(数据结构,量小但基础,先写清楚方便后续调用)

**产出**:目录结构 + 两个数据类文件
**预估代码量**:~80 行
**预估工时**:2–3 小时(含设计讨论)

---

### Day 2:Event-driven Simulator 主循环

**任务**
- 实现基于 min-heap 的事件队列(`heapq`)
- 定义核心事件类型:`ARRIVAL`(request 到达)、`DEPARTURE`(request 完成一个 decode step 或整体完成)
- 实现主循环:pop 最早事件 → 处理 → 可能产生新事件 → push 回队列 → 循环直到队列空
- **这一步先不接调度算法**,用一个"傻瓜调度"(比如"来一个跑一个,不排队")跑通,验证事件驱动机制本身没问题

**产出**:`simulator.py` 核心循环 + `event.py` 补丁
**预估代码量**:~120 行
**预估工时**:4–5 小时(event-driven 比 time-step 复杂,这是本周最费脑的一天)

---

### Day 3:Scheduler 基类 + FCFS 实现

**任务**
- 定义 `Scheduler` 抽象基类(`add_request()` / `schedule()` 接口)
- 实现 `FCFSScheduler`(标准 FIFO 队列)
- 实现 `max_batch_size` 限制逻辑(放在基类里,子类可复用,不用每个算法都重复实现)
- 把 Simulator 主循环接入真正的 Scheduler(替换掉 Day 2 的"傻瓜调度")

**产出**:`scheduler.py` + `schedulers/fcfs.py`
**预估代码量**:~100 行
**预估工时**:3–4 小时

---

### Day 4:Workload 生成器

**任务**
- 实现 `arrival_time` 生成(建议用泊松过程 / 指数分布间隔,这是文献标准做法,比"随机撒点"更严谨,评委也好评)
- 实现 `output_len` 生成(采用长尾分布,体现长尾特征——之前讨论过这样更贴近真实 LLM workload,也更能凸显 SJF 的优势场景)
- 支持可配置参数(到达率 λ、平均 output_len、request 总数),方便 Week 3 准备做对照实验

**产出**:`workload.py`
**预估代码量**:~60 行
**预估工时**:2–3 小时

---

### Day 5:Metrics 统计模块

**任务**
- 实现 waiting time / response time / throughput 计算
- 实现 fairness 指标(建议同时算标准差和 Jain's Fairness Index,方便评委可以对比两种定义)
- 实现按 priority、按 output_len 分桶的分组统计(为 Week 2/3 的 starvation 分析预留接口,虽然 FCFS 阶段用不上分组,但接口先做好)
- 输出格式:同时打到 console,顺便存一份 CSV(方便 Week 3 用 pandas/matplotlib 画图)

**产出**:`metrics.py`
**预估代码量**:~90 行
**预估工时**:2–3 小时

---

### Day 6:回归测试 + Bug 修复

**任务**
- 写 `experiments/run_fcfs_baseline.py`:生成一批 workload → 跑 FCFS → 输出 metrics
- 写基础回归测试(至少要覆盖:事件顺序是否正确、FCFS 是否严格按到达顺序执行、metrics 计算是否和手算结果一致)
- 用一个"手算验证"的小规模案例(比如 3–5 个 request,手动算出预期的 waiting time)校验整个流程没有逻辑错误——这一步很重要,能避免 Week 2 加新算法时,底层框架的 bug 混进新代码 debug 的干扰

**产出**:`test_simulator.py` + `run_fcfs_baseline.py`
**预估代码量**:~100 行
**预估工时**:3–4 小时

---

### Day 7:文档 + 缓冲期

**任务**
- 写 README(架构说明、如何运行、如何加新算法——为 Week 2 分工做准备)
- 处理 Day 1–6 遗留的 bug 或设计上的欠工(经验上一个事件驱动仿真第一次写基本都会出些细节问题,预留缓冲很有必要)
- 同步和对齐 Week 2 分工细节(Teammate A 负责 Oracle SJF / Predicted SJF,Teammate B 负责 Priority Scheduling,基于同一套框架各自开工)

**产出**:README.md + 修复清单
**预估工时**:2–4 小时(弹性,取决于前面进度)

---

## Week 1 代码量需求总结

| 模块 | 预估行数 |
|---|---|
| request.py + event.py | 80 |
| simulator.py(事件驱动主循环) | 120 |
| scheduler.py + fcfs.py | 100 |
| workload.py | 60 |
| metrics.py | 90 |
| 测试 + 实验脚本 | 100 |
| **合计** | **约 550 行** |

(不含 README、注释、Week 2/3 待加的三个算法)

## Week 1 工时总结

| 类型 | 预估工时 |
|---|---|
| 核心编码(Day1–5) | 13–18 小时 |
| 回归测试/debug(Day6) | 3–4 小时 |
| 文档/缓冲(Day7) | 2–4 小时 |
| **合计** | **约 18–26 小时**(两人协作,人均 9–13 小时/周) |

---

## 整个项目(三周)代码量与工时预估

| 阶段 | 新增代码量 | 说明 |
|---|---|---|
| Week 1(框架 + FCFS) | ~550 行 | 已详细拆解如上 |
| Week 2(+Oracle SJF, Predicted SJF, Priority) | ~250–300 行 | 一个算法各自 60–80 行(继承基类,逻辑简单),加上 Predicted SJF 的噪声预测模块 ~40 行 |
| Week 2(可选加分项:Aging/防饥饿机制) | ~50–80 行 | 之前讨论过的改进方向,建议作为加分项而非硬性任务 |
| Week 3(实验脚本 + 绘图) | ~150–200 行 | 多组对照实验(长尾程度/到达率/预测误差)+ matplotlib 绘图脚本,不算"核心系统"代码,但工作量不小 |
| **项目总计** | **约 1000–1150 行 Python** | 不含最终报告和 PPT |

**总工时估计:约 55–70 小时**(两人协作,人均 27–35 小时,分摊到 3 周大约每人每周 9–12 小时,和一门 3 学分课的 final project 工作量级相符,不算轻,但也不夸张)

---

## 风险提示(建议提前预判)

1. **Event-driven simulator 是本周唯一真正"烧脑"的部分**——如果之前没写过这类仿真,Day 2 可能超预算,建议把 Day 7 的缓冲时间优先留给它
2. **不要在 Week 1 就想着"顺便"把 starvation/aging 加进去**——先用 FCFS 验证框架本身没问题,加分项放到 Week 2 框架稳定之后再加,顺序颠倒容易两头debug
3. **提前约定 metrics 输出格式(CSV 列名等)**,因为 Week 2 两个队友要各自跑实验,输出格式不统一会导致 Week 3 汇总数据时返工
