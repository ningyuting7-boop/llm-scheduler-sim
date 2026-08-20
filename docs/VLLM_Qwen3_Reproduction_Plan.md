# 把 llm-scheduler-sim 扩展成真实的 vLLM + Qwen3-8B TIE 复现项目

## 背景

现在的项目(`llm-scheduler-sim/`)是一个纯 Python 的 event-driven **仿真**,改编自 TIE 论文(Zheng et al. 2026, arXiv:2604.00499)——见 `docs/Report.md`。目前用的是一个合成的、统计校准出来的预测器(`src/predictor.py`),不是真实训练出来的模型;`decode_time_per_step` 也是一个假设的常数,不是真实测得的 GPU 耗时。

用户想把这个项目扩展成真正的系统级复现:服务一个真实模型(**Qwen3-8B**),配一个**真正训练出来的 DeBERTa 预测器**(不是合成噪声模型),在用户学校的 HPC(SLURM 集群,可申请多张 GPU,用户对 SLURM/module/conda 流程比较熟悉)上跑基准测试。

到目前为止,用户在每个分岔点的选择:
- 真正训练一个 DeBERTa 预测器(不复用现有的合成校准预测器逻辑)
- **调度器集成方式(2026-08-20,反复讨论后最终定案)**:中途一度考虑改用 vLLM 官方可插拔的 `--scheduler-cls` 接口(继承一个类,不改源码,维护成本低),但查证后发现 vLLM 官方自己就在日志里明确警告"This scheduler interface is not public and compatibility may not be maintained"——也就是说这条路径**和 fork 源码面临的是同一类版本不稳定风险**,只是代码量少一点,并不是真的更"稳"。同时确认了论文官方实现(github.com/Hyzheng-code/TIE)用的就是**直接 fork vLLM v1 源码**,不是插件接口。综合考虑,**最终决定完全对齐论文,直接 fork vLLM 源码**,可以直接复用/改编他们的 `ua_predictor.py`/`ua_score_calculator.py`,不用自己按公式重新写一遍打分逻辑。**这个决定只影响阶段三,不影响阶段一/二已经在做的事**(阶段一是纯离线生成,完全不涉及 scheduler)。

用户要求按三个阶段、一步步来做:

1. **阶段一**(当前重点):在 HPC 上接通 vLLM + Qwen3-8B,生成预测器的训练数据集。
2. **阶段二**:训练预测器。
3. **阶段三**:端到端跑 benchmark(TIE 改造后的调度器 vs. vLLM 默认调度),写进报告。

这份计划目前只把阶段一定到可执行的细节;阶段二/三先给出大致框架,等阶段一跑通、产出实际数据后再细化(因为阶段二/三的具体做法依赖阶段一产出的标签格式、实际能跑出多大规模的数据)。

**当前状态**:两个后台调研任务都已经回来,下面是最终确定下来的调研结论。用户已确认:阶段一先用**小规模试跑**(~2,000–5,000 个 prompt × 20 次采样),而不是论文原始的 45,000×20 规模,目的是先低成本验证"生成 → 拟合"整条 pipeline 没问题,如果后面时间/资源allow 再考虑扩大规模。

### 调研任务(b)的发现:Qwen3-8B / vLLM / HPC 相关

- **Qwen3-8B 是 Apache 2.0,完全开放**——HF 仓库 `Qwen/Qwen3-8B`(instruct)或 `Qwen/Qwen3-8B-Base`。不需要申请权限。bf16 大约 16.5GB。这是好消息,不是阻塞项。
- **版本锁定风险**:Qwen3 架构要求 `vllm >= 0.8.5` 和 `transformers >= 4.51.0`;HPC 上如果 CUDA 模块比较旧(11.6/11.8)可能装不上。要先确认 HPC 上能 module load 到的 CUDA 版本。
- **TIE fork 的版本风险**:TIE 仓库的 README 没有锁定具体的 vLLM commit/版本。`vllm/v1/core/sched/scheduler.py` 目前在 vLLM `main` 分支上还在这个路径,但 V1 内部接口(Request/SchedulerOutput 这些)即使文件路径没变,也可能在小版本之间发生变化——需要找到 TIE 实际是基于哪个 vLLM 版本/commit 做的(看他们的 requirements.txt / CI 配置 / git 历史),而不是直接装"最新版"。
- **GPU/显存**:Qwen3-8B 权重本身约 16.5GB;要支撑 8-16 个并发请求的 continuous batching,现实需要 **≥40GB 显存**(A100 40/80GB、A6000 48GB、H100 都可以;V100 32GB 比较紧张;消费级 24GB 卡就得上量化,会改变benchmark 本身测的是什么)。理想情况下预测器最好用**独立 GPU**(模型本身很小,~350MB,显存不是问题——单独给一块卡的目的是避免和 Qwen3-8B 主模型抢计算/显存分配器资源,这也是 TIE 官方 `start-server.sh` 的设计)。**更新(2026-08-20)**:实际确认账号能用的是单卡 V100 32GB,一度考虑降级模型;后来发现学校 Explorer 集群其实有 **H200(141GB)**,已改用 H200,不需要再纠结显存/降级模型这件事——详见阶段一步骤 1 的更新说明。
- **阶段一生成标签用的 prompt 数据集选项**:
  - `sharegpt`——真实对话数据,不需要申请权限,vLLM 自带 benchmark 工具直接支持(`vllm bench serve --dataset-name sharegpt`)。
  - `alpaca`(`tatsu-lab/alpaca`,5.2万条)——不需要申请权限,通过 vLLM 的通用 `--hf-name` 加载器。
  - `lmsys-chat-1m`——**需要申请权限**,要签使用协议、填个人信息,审批时间不确定——这里先标记为风险项。**下面会被推翻**:检查我们自己已有的代码后发现,这个权限其实已经有了,所以最终还是用 LMSYS-Chat-1M——见下面"Prompt 数据来源"这一节。
  - `burstgpt`——真实到达时间轨迹,可以留到阶段三用来做请求到达模式(和 prompt 内容本身是两件事)。

### 调研任务(a)的发现:TIE 仓库实际的标签生成方法

- **TIE 仓库里没有标签生成代码。**`train/model_train.py` 只是读一个现成的 CSV,列是 `prompt, logt_mu, logt_sigma`。这一步的 pipeline 得我们自己写——这是确认过的空白,不用再去他们仓库里找了。
- **论文正文(不是代码仓库)记录了具体方法**:从 LMSYS-Chat-1M 采样 prompt,对每个 prompt 用被服务的模型**生成 20 次**(论文主要用 Llama-3-8B-Instruct 做这个"打标签"的模型;我们统一用 **Qwen3-8B**——这不算偏离论文精神,因为论文自己更大范围的评测里本来就包含了"Qwen3 变体"这个模型,我们只是把同一个模型型号一致地用在打标签和最终 serving 两个环节)。论文用了 45,000 个 prompt(我们先试跑 2,000-5,000 个)。
- **分布拟合**:对每个 prompt,用它 20 次采样得到的 output token 长度,拟合一个**固定 ν=3.5 的 log-t 分布**,通过联合 MLE、`scipy.optimize.minimize(method="L-BFGS-B")` 求解。论文的精确目标函数(Eq. 5-6):在固定 ν 的情况下,对 `μ∈ℝ, σ>0` 最大化
  `ℓ(μ,σ,ν) = Σᵢ [ln t_ν((ln xᵢ − μ)/σ) − ln σ − ln xᵢ]`,
  也就是 `ln(xᵢ)` 在 `μ + σ·T_ν` 分布下的对数似然。
- **输出粒度**:CSV 每行对应一个**prompt**(不是每个 completion 一行)——20 次原始采样被拟合过程"消耗掉",预测器训练时看不到单次采样。
- **采样温度**:仓库和论文里都没有明确写。我们的假设:用 temperature=1.0(标准随机采样)——刻意不用贪心/低温采样,因为TIE 和我们的核心前提都是"output length 的不确定性来自 EOS token 何时被采样到"这件事本身是随机的;温度太低会让方差塌缩,log-t 拟合就会退化。这个假设会写进脚本注释里,和这个项目一贯的做法一致(比如 `docs/Report.md` §2 里就明确写了 log-normal vs log-t 的简化取舍)。
- **一个记到阶段二的仓库 bug**(不影响阶段一):`ua_predictor.py` 推理时用 `max_length=2048` 给 DeBERTa tokenizer,但 `train/model_train.py` 训练时用的是 `MAX_LENGTH=512`——这是仓库代码里一个真实的不一致。这个不影响阶段一(阶段一关心的是被服务模型自己的输出,不是 DeBERTa 预测器的输入截断),但阶段二训练前必须先把这个对齐好。

### Prompt 数据来源:最终确定用 LMSYS-Chat-1M,不用 ShareGPT/Alpaca

我们已经有 `lmsys/lmsys-chat-1m` 的 HF 访问权限——`scripts/extract_lmsys_lengths.py` 现有的 docstring 里写着:"Requires a HuggingFace account that has accepted the lmsys-chat-1m usage terms and is logged in locally."。这意味着调研任务(b)提到的权限申请风险,对这个项目来说**已经解决了**(同一个账号),只要在跑标签生成的 HPC 节点上用同一个 HF 账号登录(`hf auth login` 或设置 `HF_TOKEN` 环境变量)即可。而且 LMSYS-Chat-1M 正是论文里这一步实际用的数据集,所以这不只是"图方便",也是更贴近论文的选择。

## 阶段一(最终版):vLLM + Qwen3-8B → 预测器训练标签

**产出物**:`data/qwen3_8b_logt_labels.csv`,列为 `prompt, logt_mu, logt_sigma`,每个 prompt 一行,覆盖约 2,000-5,000 个 LMSYS-Chat-1M 的 prompt,数据来自真实 Qwen3-8B 的输出——为阶段二的预测器训练做准备。

**范围说明**:这些代码可以在这个仓库里本地写好并测试,但**真正跑起来需要 HPC 的 GPU**,这个环境只存在于集群上——需要把代码同步到 HPC 账号(`git pull`/`scp`/`rsync`),自己提交 SLURM 作业。

### 具体步骤

1. **HPC 环境与硬件确认**(2026-08-20 更新):最初以为只有单卡 V100 32GB,查证后发现:
   - vLLM 只有 **0.20.0 及以后**才彻底放弃 sm_70(V100)支持,论文锁定的 **0.11.1** 还在这个 cutoff 之前,理论上能跑,但会自动退化到 XFormers backend(非 FlashAttention),且 V100 硬件不支持 bf16,需要 `--dtype float16` 转换,有一定数值风险。
   - 但学校 Northeastern 的 Explorer 集群其实**有 H200**(Hopper 架构,141GB 显存,原生支持 bf16 和 FlashAttention-3,完全对齐 vLLM 0.11.1 想用的技术栈)——参考 [Explorer H200 quickstart](https://rc-docs.northeastern.edu/en/explorer-main/gpus/quickstart-h200.html)。**最终决定:用 H200,不降级模型,继续用 Qwen3-8B**,V100 相关的风险(sm_70/bf16/显存紧张)全部不再适用。
   - 申请方式:关键 flag `--gres=gpu:h200:1`;分区选 `gpu-interactive`(交互式调试,一次限 1 张卡,适合先跑通阶段一步骤 2 的小范围验证)或 `gpu`(正式提交批处理作业)。以后阶段三如果想让预测器用独立物理 GPU(而不是和主模型共享同一张 H200),可以走 `multigpu` 分区。
   - 新建一个 conda/venv 环境,Python 3.10-3.13。`pip install vllm==0.11.1`(锁定论文实际用的版本)加上 `torch==2.9.0`、`scipy`(做 MLE 拟合)、`datasets`(`scripts/extract_lmsys_lengths.py` 已经在用)。需要 CUDA 12.3+,H200 节点上应该没问题,但装之前仍建议 `module avail cuda` 确认一下。
   这个阶段 SLURM 申请:**1 张 H200**——预测器专用的第二张卡到阶段二/三才需要(而且 141GB 显存下,即使阶段三想把预测器和主模型放同一张卡上共用也完全没问题)。

2. **先小范围验证 Qwen3-8B 能跑起来**,再上规模:用 vLLM 的离线生成接口跑一个写死的 prompt,确认环境/模型下载都没问题。这样能用很小的代价先排掉安装/CUDA/下载方面的坑。

3. **新脚本 `scripts/generate_predictor_labels.py`**,沿用 `scripts/extract_lmsys_lengths.py` 现有的写法风格(`load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)` 流式读取,类似 MAX_WORD_COUNT 的过滤方式):
   - 流式读取 `--num-prompts`(默认对应试跑规模,比如 3000)个对话,取每个对话里第一个 human 回合当作 prompt(这是一个明确记录下来的简化:忽略多轮上下文,和 `ua_predictor.py` 自己的 `extract_original_prompt()` 一样,只需要一个扁平的 prompt 字符串)。
   - 对每个 prompt,用 vLLM 的离线 `LLM.generate()`,配 `SamplingParams(n=20, temperature=1.0, max_tokens=2048)`(2048 这个上限对应 `ua_predictor.py` 自己的 `MAX_GENERATED_TOKENS`/`CVAR_MAX_GENERATED_TOKENS` 常量)——把多个 prompt 一起批量跑,而不是一个个跑,提高吞吐。
   - 记录每个 prompt 20 次生成各自的 output token 数。
   - 用一个新的小模块 `src/logt_fit.py`,按论文 Eq. 5-6 精确实现,通过 `scipy.optimize.minimize(method="L-BFGS-B")` 对每个 prompt 拟合 `(μ, σ)`,`ν` 固定为 3.5(用 `scipy.stats.t` 算对数似然项)。
   - 写出 `data/qwen3_8b_logt_labels.csv`(prompt, logt_mu, logt_sigma);同时为了方便调试和复现,把每个 prompt 的原始 20 次采样长度也存一份到 `data/qwen3_8b_length_samples.csv`(prompt_id, sample_lengths),这样如果拟合代码有 bug,不需要重新跑一遍很贵的生成步骤。

4. **测试**(`tests/test_logt_fit.py`,沿用这个项目一贯的"手算可验证"风格——参考 `tests/test_simulator.py`):从一个**已知**的 `(μ, σ, ν=3.5)` log-t 分布用 `scipy.stats.t` 生成合成样本,跑拟合,断言拟合出来的 `(μ̂, σ̂)` 接近真实值。这一步不需要 GPU、不需要真实数据集就能验证拟合代码本身对不对。

5. **SLURM 作业脚本 `hpc/generate_labels.slurm`**:把第 3 步包起来,申请 1 张 ≥40GB 的 GPU,按试跑规模估算 wall-time(粗略估算:3,000 个 prompt × 20 次采样 × 最多 2048 token,最坏情况约 1.23 亿 token,但 LMSYS 的中位数长度只有 89 词,实际大概率远低于这个上限;先按几个小时的 wall-time 规划,等看到具体 GPU 的真实吞吐再调整)。

6. **验收标准**:`data/qwen3_8b_logt_labels.csv` 存在且行数符合预期,`logt_mu`/`logt_sigma` 都是有限值且在合理范围内(抽查几个 prompt,看拟合出的 `μ` 是否和它们采样长度的中位数的对数大致吻合),`tests/test_logt_fit.py` 全部通过——这样阶段一就算完成,再去细化阶段二(在这份 CSV 上训练 DeBERTa 预测器)。

## 阶段二(框架,待细化):训练 DeBERTa 预测器

用 TIE 仓库的 `train/model_train.py`(或者一个接近的改编版本)在阶段一产出的 CSV 上训练。具体细节等阶段一跑完再定。

## 阶段三(框架,待细化):端到端 benchmark

**调度器集成方式(2026-08-20 最终定案:完全对齐论文,fork vLLM 源码)**:直接 fork/patch vLLM v1 源码(`vllm/v1/core/sched/scheduler.py`、`request_queue.py`),复用/改编 TIE 仓库的 `ua_predictor.py`(接阶段二训练出的预测器 checkpoint)、`ua_score_calculator.py`(论文公式 `score = E[X] + β·CVaR_0.9[X]`,可选 `− α·waiting_time` aging 项)。需要锁死一个具体的 vLLM 版本/commit(候选:论文写的 0.11.1,或去 TIE 仓库的 requirements.txt/git 历史找他们实际锁定的版本),升级 vLLM 时要手动 rebase 这几个文件——这是有意接受的维护成本,换来和论文一致的复现度、可以直接复用他们已经写好的打分/预测器调用代码。

在 HPC 上,让这个 fork 后的 scheduler(接上阶段二训练出的预测器 checkpoint)和原生 vLLM(默认调度)同时服务 Qwen3-8B;用一致的请求流量去打两边(理想情况下沿用现有仿真里已经用过的 Poisson 到达设置,方便对比——`src/workload.py`);收集真实的 per-token latency/throughput;在 `docs/Report.md` 里新增一节,把真实测量结果和仿真的预测结果做对比。具体细节(具体 fork 哪个 vLLM commit、怎么改编 `ua_predictor.py` 接自己训练的 checkpoint 等)等前两阶段完成后再细化研究。

## 下一步

阶段一的方案现在已经具体到可以直接实施:写 `src/logt_fit.py`、`tests/test_logt_fit.py`、`scripts/generate_predictor_labels.py`、`hpc/generate_labels.slurm`;跑本地测试套件做验证(不需要 GPU);然后把 SLURM 脚本交给账号本人,在 HPC 上跑起来(真正的执行发生在集群上,不在本地开发环境)。阶段二、三继续保持"框架待细化"状态,等阶段一有了真实产出(标签 CSV)之后再细化,这是"一步步来"的方式。
