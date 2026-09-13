# 阶段二执行记录：训练 DeBERTa 长度分布预测器

本文档记录阶段二的**实际执行过程**——做了什么、遇到什么问题、怎么诊断、怎么解决、最终结论是什么。设计方案本身见 [`Phase2_Predictor_Training_Plan.md`](Phase2_Predictor_Training_Plan.md)；这份文档记录的是计划落地时真实发生的事，包括几个计划里没预料到的问题。

**一句话总结**：μ 预测成功（测试集 R²=0.767），σ 预测失败（R²≈0.05，五种配置都一样）。通过一系列消融实验排除了标签噪声、loss 权重、过拟合、多任务干扰四种解释；剩下的候选解释（数据规模 / 信号不在文本里 / 架构）需要学习曲线实验才能进一步分辨，尚未执行。

---

## 0. 起点与目标

- **输入**：`data/qwen3_8b_logt_labels.csv`——9,998 条 `(prompt, logt_mu, logt_sigma)`，来自阶段一用 Qwen3-8B 在 LMSYS-Chat-1M 前 9,998 条 prompt 上各采样 20 次、拟合 log-t(ν=3.5) 得到
- **目标**：微调 DeBERTa-v3-base，从 prompt 文本预测 `(μ̂, σ̂)`，供阶段三 TIE scheduler 打分用（`score = E[X] + β·CVaR_0.9[X]`）
- **环境**：Northeastern Discovery 集群，conda 环境 `/scratch/ning.yuti/envs/tie-repro`，Python 3.11.15

---

## 1. 训练前发现并修复的数据问题

### 1.1 `logt_fit.py` 的退化样本兜底逻辑从未生效（真 bug）

**现象**：标签文件里出现大量接近机器精度的 σ，例如 prompt `"Hi"` 的 `logt_sigma = 4.4408920985006143e-16`，而代码本意是给这类样本一个 0.05 的下限。

**根因**：

```python
sample_std = float(np.std(log_samples))
if sample_std == 0.0:                      # 这个判断永远不成立
    return LogTFit(mu=mu0, sigma=0.05, ...)
```

直接验证：

```python
>>> np.std(np.log([12]*20))
4.440892098500626e-16        # 不是 0.0
```

20 个完全相同的浮点数，理论方差为 0，但 `np.std` 因为均值相减的舍入误差会返回 ~1e-16 量级的噪声。`== 0.0` 精确比较从未触发，代码落到后面的 L-BFGS-B 优化器——而在这种退化场景下目标函数在 σ→0 方向单调下降、没有极小值点，优化器停在哪纯属偶然。

**影响面**：`logt_sigma < 1e-6` 的行有 **1,228 / 9,998（12.3%）**。

**进一步发现（更严重）**：不止"完全相同"的样本有问题。交叉检验"拟合出的 σ"和"原始样本在 log 空间的真实标准差"，发现另有 **214 行（2.1%）样本本身有明显真实波动，优化器却拟合出接近 0 的 σ**。最极端的例子：

```
raw_std = 0.2091    fitted_sigma = 7.59e-06
samples = [2048, 2048, 2048, 1384, 2048, 2048, 2048, 2048, 2048, 1281,
           2048, 2048, 2048, 2048, 1216, 2048, 1057, 2048, 2048, 2048]
```

20 次采样里 5 次明显更短（1057~1384 vs 主体 2048），拟合结果却说"几乎没有不确定性"。**原因不是阈值写错，是重尾似然本身的病态**：ν=3.5 的 log-t 允许极端值的"代价"很低，优化器发现"把 σ 收到极小、把主体 2048 拟合得死死的，剩下几个不一样的甩给重尾去解释"在纯似然值上更划算。

**修复**（三层，`src/logt_fit.py`）：

1. 阈值判断：`sample_std < 1e-8` 替代 `== 0.0`
2. 给 L-BFGS-B 加下界：`log_sigma >= log(0.01)`，不让优化器进入病态区域
3. 拟合后合理性检查：若 `sigma_hat < sample_std × 0.1`，判定拟合失败，回退到矩估计 `sample_std × sqrt((ν-2)/ν)`

**修复验证**（三个已知病例）：

| 案例 | 修复前 σ | 修复后 σ |
|---|---|---|
| `"Hi"`（20 次全是 12） | 4.44e-16 | 0.0500（正确触发下限） |
| 上述 2048/1057 混合案例 | 7.59e-06 | 0.1369（走了回退分支） |
| `[15,16,15,...]` 轻微波动 | 1.16e-06 | 0.0100（优化器触界） |

**重新拟合全量**：用修复后的代码对已有的 20 次采样原始数据重跑（不需要重新调 Qwen3），**1,561 / 9,998 行的 σ 发生了实质变化**，`data/qwen3_8b_logt_labels.csv` 已更新。

**同一个 bug 的第二处实例**：`scripts/ks_pass_rate.py` 里也有 `if float(np.std(np.log(arr))) == 0.0`，同样修正为 `< 1e-8`。

### 1.2 修复后重跑 KS 拟合优度检验

旧的 KS 结论是基于有问题的 (μ,σ) 算的，必须刷新：

```
测试 8,574 个 prompt（1,424 个退化样本被正确排除）
log-t      KS 通过率 (p>0.05): 90.3%    （论文报告 93.1%）
log-normal KS 通过率 (p>0.05): 84.7%    （论文报告 60.3%）
```

**解读**：log-t 通过率与论文接近，选择 log-t 站得住。但 log-normal 的通过率比论文高出一大截——原因是**统计功效**：论文每个 prompt 采样 100 次，我们只有 20 次，更难探测出"真实是重尾、被误判成正态"这种细微偏差。**报告里不能直接引用论文的 93.1%/60.3%，必须用我们自己这组数字并说明采样预算差异。**

> ⚠️ 注意：上面"论文用 100 次采样"这个前提**未经验证**——我们手上只有 TIE 的代码仓库（不含论文 PDF），而代码里没有数据生成脚本。这个数字来自项目早期的记录，来源不明，引用前需要核对论文正文。

### 1.3 右截断（censoring）——已知限制，选择不修

全量 199,960 个采样中，**8,338 个（4.17%）精确等于 2048**，即 `max_tokens` 上限（该值与 TIE 自己 `ua_score_calculator.py` 的 `CVAR_MAX_GENERATED_TOKENS = 2048.0` 对齐）。

- **901 / 9,998 个 prompt（9.0%）** 至少有一次采样撞到上限
- **681 个（6.8%）** 是"部分撞上限、部分没撞"的混合情况——正是 1.1 节那批病态拟合的来源

这些观测的真实长度是"≥2048，具体未知"，当成精确值拟合会系统性扭曲 (μ,σ)。

**决定：不丢弃这批 prompt。** 它们恰恰是输出最长的一批，而长尾请求正是 TIE 的 CVaR 机制最需要预测器认识的对象——丢掉它们等于把训练数据里最关键的一段砍掉。当前 `fit_logt` 把截断值当精确值处理，是一个明确记录在案的近似。真正的修法是实现截断似然（把撞上限的样本按"X ≥ 2048"计入似然），留待未来。

> 补充：TIE 自己的打分代码**在推理侧**处理了截断（`mean()` 和 `cvar()` 都用蒙特卡洛采样后 `np.minimum(x_samples, MAX)` 截断），但那是打分阶段的处理，和训练标签拟合阶段的截断是两件不同的事。

### 1.4 重复 prompt 导致的 train/val/test 泄漏

**发现**：9,998 行里有 **205 个 prompt 文本重复出现，共 1,122 行（11.2%）**。例如 `"Hi"` 出现 50 次，某越狱模板 prompt 出现几十次。

- 42 组重复的标签完全相同（同一结果被复用）
- 163 组标签不同（同一 prompt 被独立重新采样，属正常采样方差）

**风险**：按行随机切分会让同一个 prompt 同时出现在 train 和 val/test——模型在训练时"背过"这个文本，验证时看到一模一样的输入，分数会被人为抬高。

**修复**：`grouped_split()`——先按去重后的 prompt 文本分组，在**prompt 层面**切 60/20/20，再展开回行。同一 prompt 的所有行必然落在同一个 split。

**为什么不直接去重**：分组切分和去重对防泄漏的效果**完全相同**，但去重会白白扔掉数据——163 组重复是**独立重新采样**的结果，两次采样都是这个 prompt 真实行为的合法观测，没有理由留一个扔一个。分组切分是零数据代价的方案。

**实际切分结果**（seed=42）：

| Split | 行数 | 去重 prompt 数 |
|---|---|---|
| Train | 5,987 | 5,448 |
| Val | 2,029 | 1,816 |
| Test | 1,982 | 1,817 |
| 合计 | 9,998 | 9,081 |

prompt 层面是精确 60/20/20；行层面因重复大户整块落入某一侧而略有偏移（train 占 59.9%），这是分组切分的预期代价。

> 对照：TIE 的参考实现用的是 `sklearn.train_test_split` 按行随机切，**没有**做重复分组。这一点我们比参考实现更严格；这也意味着他们报告的指标可能被轻微泄漏抬高过。

---

## 2. 实现

新增三个文件，架构严格按计划文档第 1 节：

- **`src/predictor_model.py`**：DeBERTa-v3-base encoder → multi-pooling（CLS + masked mean + masked max，768×3=2304）→ μ/σ 两个独立分支（`Linear→LayerNorm→GELU→Dropout(0.2)` ×2，hidden=256）→ 各自回归头（256→128→1）。另含 `NormalizeStats`（μ 直接 z-score；σ 先 log1p 再 z-score，统计量只在 train split 上拟合）和 `predictor_loss`。
- **`scripts/train_predictor.py`**：分组切分 → 加权采样 → 训练循环 → early stopping → 存 checkpoint / `normalize_stats.json` / `test_prompts.json` / `train_log.json`。
- **`tests/test_predictor_model.py`**：8 个结构性测试（带 padding 的池化正确性、前向输出维度、loss 有限、梯度能传到 encoder、归一化 round-trip、存取 round-trip）。用随机初始化的迷你 DebertaV2 config，不下载真实权重、不需要 GPU。**HPC 上实测 8/8 通过。**

另有 `scripts/evaluate_predictor.py`：在 held-out test split 上算 μ/σ 各自的 MAE/RMSE/R²（与 ELIS 报告格式一致，便于横向对比）。

---

## 3. 训练过程中遇到的工程问题

### 3.1 CUDA OOM（两次才解决）

**现象**：32GB GPU 上 batch_size=32 直接 OOM，报错栈停在 `deberta_v2/modeling_deberta_v2.py` 的 `disentangled_attention_bias` → `p2c_att = torch.gather(...)`。

**根因分析**：
1. `PromptLengthDataset.__getitem__` 里用了 `padding="max_length"`，**每条 prompt 都被 pad 到满 512 token**——而实测 prompt token 长度中位数只有 25、p95 才 371，绝大部分算力花在无意义的 padding 上
2. DeBERTa-v2 的 disentangled attention 要额外计算 content-to-position 和 position-to-content 两套偏置张量，比标准 attention 吃显存得多，把上面这份浪费放大了很多倍

**第一次修复**：改成动态 padding——`__getitem__` 不再 pad，自定义 `collate_fn` 里按每个 batch 内最长的那条 pad。

**仍然 OOM，而且报错数字一模一样**（"Tried to allocate 384.00 MiB"、"31.37 GiB in use"）。原因：动态 padding 只在**平均意义**上省显存，挡不住"某个 batch 里恰好混进一条接近 512 token 的长 prompt"——那一整个 batch 还是会被 pad 到 ~512。而 smoke test 只有 122 条训练数据、batch_size=32 只分成 3-4 个 batch、seed 固定，**每次抽到的都是同一个倒霉 batch**，所以报错数字完全一致。

**第二次修复（最终方案）**：`--batch-size 8 --grad-accum-steps 4`——物理上一次只在显存里放 8 条，梯度累积 4 次再更新一次权重，**等效 batch size 仍是 32**（与 TIE 对齐），优化器步进节奏和学习率调度不受影响。loss 在反向传播前除以 `grad_accum_steps`，保证累积后的梯度等于真实 32 条的平均梯度而非 4 倍。

### 3.2 fast tokenizer 效率警告

日志里出现：`You're using a DebertaV2TokenizerFast tokenizer... using the __call__ method is faster than using a method to encode the text followed by a call to the pad method`。

原因：我们逐条 tokenize（`__getitem__` 里一次一条）再单独 `tokenizer.pad()`，绕开了 Rust 后端对整批文本并行编码的快速路径。

**修复**：把 tokenize 也挪进 `collate_fn`，`__getitem__` 只返回原始 prompt 文本，collate 时对整批原始字符串一次性调用 `tokenizer(batch_texts, truncation=True, padding=True)`。

**验证**：修改前后 loss 数值**逐位相同**（`2.2179/1.4169` → `2.5728/1.3976`），确认这是纯效率改动、没有改变任何计算结果。

### 3.3 SSH 断线杀掉了一次完整训练

交互式 `srun --pty` 会话在训练跑到 epoch 8 时断开（`Connection reset by peer` / `Broken pipe`）。

**关键教训**：`nohup` / `disown` 在这个场景**不管用**——`srun --pty /bin/bash` 这个 shell 本身就是 SLURM 任务的主体，shell 一死，SLURM 会把整个任务下的所有进程一起清理掉，这不是 SIGHUP 层面的问题，是资源分配被回收。真正有效的只有 tmux/screen（让 pty 活在服务器端）或者 `sbatch`（完全不依赖终端）。

**损失评估**：
- ✅ `best_model.pt` 完整保住：740,938,159 字节，与理论值吻合（184M 参数 × 4 字节 ≈ 736MB + 两个头 ~5.5MB）；加载验证通过（222 个参数张量，key 命名正常）
- ❌ 逐 epoch 的 loss 历史丢失：`train_log.json` 只在训练循环**正常结束后**才写盘，中途被杀从未执行到那一步

**改进**：后续运行走 `sbatch`；若必须用交互式，输出用 `| tee logs/xxx.log` 重定向留存。

### 3.4 HPC 排队与分区

- `gpu` 分区（批处理）提交后长时间 `PD (Priority)`，`squeue --start` 给出的预估开始时间是次日凌晨 03:48
- `gpu-interactive` 分区分配很快，但**时间上限约 2 小时**（申请 4 小时直接被拒：`Requested time limit is invalid`），撑不住 2-4 小时的完整训练
- `sinfo -p gpu` 显示 17 个节点处于 `mix`（部分占用）、无 `idle`——说明不是资源枯竭，是优先级排队
- **有效技巧**：参考同集群上另一个项目（`~/projects/if/ifncm/slurm/run_eloo.sbatch`，同为 `gpu` 分区、资源需求相近）的做法，把 `--gres=gpu:1` 改成 `--gres=gpu:v100-sxm2:1` **指定具体卡型**后排队明显变快。反直觉（限定型号本该缩小可选池），但实测有效，推测是该集群上 V100 竞争相对不激烈

### 3.5 登录节点会杀掉 torch 进程

在登录节点跑 `python3 -c "import torch; ..."` 直接返回 `Killed`——登录节点有严格的内存限制，光是 import torch 就可能触发。**检查文件内容用 `grep` 这类轻量命令，需要真正 import 的验证放到计算节点。**

---

## 4. 核心调查：σ 为什么学不会

### 4.1 五次训练的完整结果

所有测试集指标来自 `scripts/evaluate_predictor.py`，在同一份 held-out test split（1,982 行 / 1,817 个 prompt）上计算：

| 编号 | checkpoint 目录 | 配置 | 最佳 epoch | mu R² | **sigma R²** |
|---|---|---|---|---|---|
| **v1** | `predictor_full` | σ权重=1.0（等权重），不冻结 | 8（被断线打断） | **0.7668** | **0.0501** |
| v2 | `predictor_sigma_weighted` | σ权重=2.0 | 2 | 0.6337 | 0.0139 |
| v3 | `predictor_sigma_weighted_v2` | σ权重 3.0→3.2 动态 + 梯度裁剪 | 1 | 0.5808 | 0.0473 |
| v3′ | `predictor_sigma_weighted_v3` | 同上 + 修复监控 loss | 1 | 未单独评估¹ | — |
| v4 | `predictor_frozen_v4` | σ权重 3.0→3.35 + epoch 2 冻结 encoder | 4 | 0.6803 | 0.0495 |
| v5 | `predictor_sigma_only_v5` | **只训 σ**（`--mu-weight 0`），不冻结 | 1 | 0.0042² | 0.0316 |

¹ v3′ 与 v3 用相同 seed/数据/训练权重，且监控修复后仍选中同一个 epoch（epoch 1），checkpoint 实质相同，未重复评估。
² μ 头未收到任何梯度，该数值仅用于确认 `--mu-weight 0` 生效。

**σ 的 R² 在 0.014~0.050 之间，五种差异极大的配置下没有任何实质变化。** 同期 μ 的 R² 在 0.58~0.77 之间明显随配置变动——证明训练流程有能力学习、指标有能力移动，就是 σ 不动。

### 4.2 各次运行的逐 epoch 曲线

**v1（等权重，被断线打断）**：

```
epoch 0  train 2.4986  val 1.5139
epoch 1  train 1.6120  val 1.4465
epoch 2  train 1.2858  val 1.3737
epoch 3  train 1.0471  val 1.3712
epoch 4  train 0.8579  val 1.3732
epoch 5  train 0.6728  val 1.3788
epoch 6  train 0.5607  val 1.2630
epoch 7  train 0.5110  val 1.3158
epoch 8  train 0.4199  val 1.2391     ← 断线，best_model.pt 停在这里
```

**v3′（动态 σ 权重 + 修复后的监控 loss）**：

```
epoch 0  train 5.4123  val 1.5945  σw 3.00
epoch 1  train 4.0605  val 1.3602  σw 3.05     ← best
epoch 2  train 3.3780  val 1.3897  σw 3.10
epoch 3  train 2.6852  val 1.4034  σw 3.15
epoch 4  train 2.3044  val 1.3772  σw 3.20
early stopping at epoch 4
```

**v4（epoch 2 冻结 encoder）**：

```
epoch 0  train 5.4121  val 1.5945  σw 3.00
epoch 1  train 4.0650  val 1.3471  σw 3.05
epoch 2  froze encoder; 1,379,842 trainable params left, lr -> 5e-05
epoch 2  train 3.5347  val 1.4145  σw 3.10
epoch 3  train 3.5441  val 1.3589  σw 3.15
epoch 4  train 3.6822  val 1.3021  σw 3.20     ← best
epoch 5  train 3.4271  val 1.3405  σw 3.25
epoch 6  train 3.6367  val 1.4040  σw 3.30
epoch 7  train 3.6802  val 1.4027  σw 3.35
early stopping at epoch 7
```

**v5（只训 σ）**：

```
epoch 0  train 1.4220  val 1.0201
epoch 1  train 1.1450  val 0.9751     ← best
epoch 2  train 0.9359  val 1.1147
epoch 3  train 0.7310  val 1.0721
epoch 4  train 0.6209  val 1.0761
early stopping at epoch 4
```

> v5 的 val_loss 有一个直接可读的含义：目标是 z-score 归一化的，方差恰好为 1，所以"永远预测平均值"对应 MSE = 1.0、R² = 0。最佳 val_loss = 0.9751 → val R² ≈ 0.025，与测试集的 0.0316 一致。

### 4.3 途中修掉的一个监控 bug

**问题**：`val_loss = loss_mu + sigma_weight(epoch)·loss_sigma`，而 `sigma_weight` 每个 epoch 都在变大（3.0→4.0）。即使 `loss_mu` 和 `loss_sigma` 都在下降，乘积项也可能因为权重涨得更快而上升——**early stopping 拿一把自己在变长的尺子去量"有没有进步"，可能被假象误导而提前停止**。

v3 那次运行正是可疑案例：train_loss 一路 5.41→2.30 猛降（模型显然还在学），val_loss 却从 epoch 1 之后就"不再改善"。

**修复**：引入 `MONITORING_SIGMA_WEIGHT = 1.0`，验证阶段固定用它，训练阶段继续用动态权重。这样 val_loss 跨 epoch 可比。

**结果：修复是对的，但结论没变。** 用稳定的尺子重新量（v3′），early stopping 依然在同一位置触发（best 仍是 epoch 1，仍在 epoch 4 停）。说明**那次的平台期是真实的过拟合，不是测量假象**——我原先的假设被自己的实验证伪了。

### 4.4 与 TIE 参考实现的逐项对比

读了 `~/Documents/AI_inference/TIE/train/model_train.py`（799 行）后发现的差异：

| 项 | TIE 参考实现 | 我们 | 影响 |
|---|---|---|---|
| **σ loss 权重** | `3 + (epoch/NUM_EPOCHS)×1.0`，即 3.0→4.0 动态 | 原本只试过 1.0、2.0 | **已采纳**（v3/v4） |
| **梯度裁剪** | `clip_grad_norm_(max_norm=1.0)` | 原本没有 | **已采纳**——高权重会放大 σ 梯度，没有裁剪容易不稳定 |
| NaN/Inf loss 保护 | 检测到就跳过该 batch | 没有 | 未采纳（实测未出现 NaN） |
| encoder 冻结 | epoch 12/20 冻结，lr 2e-5→5e-5 | 原本不冻结 | **已采纳但改了时机**（见下） |
| 样本加权阈值 | 绝对值（μ>5.5、σ>1.0 → 1.5；μ>6.0 或 σ>1.3 → 2.0） | 百分位（p90/p95） | 未对齐，影响存疑 |
| train/test 切分 | `train_test_split` 按行随机，**不处理重复** | 按 prompt 分组切分 | 我们更严格 |
| 硬件 | 2 张 GPU + DataParallel，`padding='max_length'` | 单卡 + 梯度累积 + 动态 padding | 环境差异 |

**关于冻结时机的推理**：论文冻在 epoch 12/20（约 60% 处），但那是针对**他们 4 倍大的数据集**——过拟合来得晚得多。可迁移的是**原则**（"encoder 不再帮忙时就冻住"）而非数字。按我们自己的 val_loss 曲线（epoch 1 见底、epoch 2 开始退化，同时 train_loss 继续猛降），等价时机是 **epoch 2**。

**冻结实验的结果（v4）**：机制完全按预期生效——
- 可训练参数从 1.84 亿降到 1,379,842
- **过拟合确实被压住**：冻结后 train_loss 停止下降，在 3.43~3.68 之间横盘（对比冻结前 5.41→2.30 的猛降）
- **多训练了 3 个 epoch**：早停从 epoch 4 推到 epoch 7，最佳 checkpoint 从 epoch 1 推到 epoch 4
- **val_loss 改善**：1.3021 vs v3′ 的 1.3602（两者监控权重都固定为 1.0，可直接比较）
- **但 σ 测试集 R² 纹丝不动**：0.0495 vs 0.0473

**关于 3-4 这个权重值的出处**：TIE 代码里这行**没有任何注释说明依据**，`train/` 目录下也只有 `model_train.py` 这一个文件，没有消融脚本、没有超参搜索记录，仓库里也没有论文 PDF。因此**不能在报告里写成"论文实验验证 3-4 最优"**——诚实的表述是"我们在参考实现中发现该值被硬编码使用，代码未给出依据"。

### 4.5 决定性诊断：标签本身的噪声天花板

**动机**：如果 σ 标签本身主要是采样噪声，那任何模型都学不到——低 R² 就是数据问题而非模型问题。这个假设必须先证伪或证实。

**方法**（`scripts/sigma_label_reliability.py`，纯 CPU，本地几分钟跑完）：把每个 prompt 的 20 次采样随机劈成两半（各 10 次），**各自独立拟合** log-t，然后在 prompt 之间计算两半估计值的一致性。这给出任何预测器的**理论上限**——你不可能预测得比"标签预测它自己"更准。

**结果**（2,000 个 prompt × 5 次随机劈分）：

| 目标 | r²（相关系数平方） | r²（恒等预测） | Spearman-Brown 校正 |
|---|---|---|---|
| **mu** | 0.9950 ± 0.0003 | 0.9950 ± 0.0004 | 0.9988 ± 0.0001 |
| **sigma** | 0.6031 ± 0.0285 | 0.5585 ± 0.0537 | 0.8740 ± 0.0116 |
| sigma（排除全退化样本） | 0.5788 ± 0.0298 | 0.5273 ± 0.0583 | 0.8638 ± 0.0126 |

（2,000 个 prompt 中有 310 个（15.5%）20 次采样完全相同，这类样本两半都会被兜底到 σ=0.05、天然完美一致，因此单列一行排除它们的版本。Spearman-Brown 校正把"10 次采样估计"的可靠性外推到真实标签的"20 次采样"规模。）

**结论：σ 标签不是噪声。** 用一半数据估出的 σ 能解释另一半估计约 58-60% 的方差，校正到真实标签规模约 0.87。**σ 的可学上限在 0.6~0.87 量级，而不是我们测到的 0.05——模型只抓到了不到十分之一的可用信号。**

作为对照，μ 的 split-half 可靠性接近 1.0，与它 0.77 的测试集 R² 相符（标签极稳定，模型也确实学到了大部分）。

> **重要限定**：这个可靠性测的是"σ 是不是 prompt 的稳定属性"（同一 prompt 两次独立估计是否一致），而模型做的是更难的事——**只看 prompt 文本、不看任何采样结果**去预测 σ。所以 0.6-0.87 是"标签有真实信号"的**必要条件**，排除了"标签全是噪声"这个解释，但并不证明"这个信号能从文本推断出来"。

---

## 5. 结论

### 5.1 已排除的解释

| 假设 | 证伪方式 | 结论 |
|---|---|---|
| σ 标签主要是采样噪声 | split-half 可靠性 r²≈0.60，校正后 0.87 | ❌ 标签有真实信号 |
| σ 的 loss 权重不够 | 1.0 / 2.0 / 3.0→4.0 三档 | ❌ σ R² 在 0.014~0.050 间随机波动 |
| 过拟合导致 σ 没时间被学到 | 冻结 encoder：过拟合确实被压住、多训 3 轮、val_loss 改善 | ❌ σ R² 仍为 0.0495 |
| μ 与 σ 争夺共享 encoder | `--mu-weight 0` 单训 σ，encoder 完全专用 | ❌ σ R² 仅 0.0316 |

### 5.2 尚未区分的候选解释

1. **数据规模不足**——6k 条训练数据学不会"文本→不确定性"这种微妙映射（μ 这种强信号够用，σ 不够）
2. **信号不在 prompt 文本里**——σ 反映的是模型采样过程的波动，可能部分取决于解码动态而非 prompt 语义
3. **架构/配方**——DeBERTa-base + 当前头设计抓不住这类信号

**能分辨前两者的实验（未执行）**：学习曲线——用 1k / 2k / 4k / 6k 条数据分别训练，观察 σ R² 随数据量的变化趋势。若单调上升则是数据规模问题（可外推"加数据有用"）；若完全平坦则信号不在文本里，加再多数据也无济于事。命令已准备好，成本约 1.5 小时。

**另一个未验证的前提**：我们从未确认论文自己的 σ R² 是多少。若他们的 σ 也在 0.05-0.1 量级，则我们是**成功复现了一个"σ 难预测"的已知现象**，而非复现失败——这会显著改变报告的措辞。需要查论文正文的预测器指标表。

### 5.3 交付物

**最终采用 v1（`checkpoints/predictor_full/`）**——μ R²=0.7668 明显领先，σ 与其余版本实质打平（0.0501，五者中最高）。该 checkpoint 虽因断线停在 epoch 8、未走完早停流程，但其选择过程有效（σ 权重恒为 1.0，val_loss 尺度稳定，保存的是截至当时的最佳版本）。

阶段三需要该目录下三个文件：

| 文件 | 用途 |
|---|---|
| `best_model.pt` | 模型权重（740,938,159 字节，222 个参数张量） |
| `normalize_stats.json` | 推理时反归一化必须使用训练时的同一份统计量 |
| `test_prompts.json` | 配合阶段一记录的约束：阶段三 benchmark 须取 LMSYS index ≥ 10,000 的 prompt，避开预测器训练见过的 9,998 条 |

`normalize_stats.json` 的实际取值（在 train split 的 5,987 行上拟合）：

```json
{
  "mu_mean": 5.235280789925398,
  "mu_std": 1.667900155893575,
  "log1p_sigma_mean": 0.11762295942774698,
  "log1p_sigma_std": 0.09737729256722726
}
```

这组数字本身对 4.x 节的讨论有补充意义：**σ 的分布非常集中**。`log1p` 空间的均值 0.1176 反算回原空间约 `expm1(0.1176) ≈ 0.125`，标准差仅 0.097——绝大多数 prompt 的 σ 都挤在 0.05~0.2 这个很窄的区间里（其中还有 15.5% 因为 20 次采样完全相同而被兜底到 0.05）。这解释了为什么测试集上 σ 的 MAE≈0.07 "看起来很小"却对应 R²≈0.05：**MAE 小只是因为 σ 本身的动态范围就小，模型实质上是在输出一个接近全局均值的常数**，并没有区分出高不确定性的少数类——而那批少数类恰恰是 TIE 的 CVaR 机制唯一关心的对象。

对比 μ：`mu_std = 1.668`（log 空间），动态范围比 σ 大一个量级以上，是个容易学的目标。

**对阶段三的直接影响**：TIE 的 `score = E[X] + β·CVaR_0.9[X]` 中，CVaR 项完全由 σ̂ 驱动。当前 σ̂ 的 R²≈0.05 意味着**风险对冲项拿到的基本是无信息量的输入**，因此阶段三若观察不到 TIE 相对 Predicted-SJF 的优势，必须区分两种可能：(a) 方法本身在真实预测器下不成立，(b) 仅仅是本阶段 σ 预测器不合格。这一点必须在阶段三的结论中显式说明。

---

## 6. 复现命令

```bash
# 环境
module load anaconda3/2024.06
source activate /scratch/ning.yuti/envs/tie-repro

# 结构测试（HPC 上 8/8 通过；本地无 torch 会自动跳过）
python3 -m unittest tests/test_predictor_model.py -v

# 标签可靠性诊断（纯 CPU，本地可跑）
python3 scripts/sigma_label_reliability.py

# KS 拟合优度检验（纯 CPU）
python3 scripts/ks_pass_rate.py

# 训练（v1 配置 = 最终采用的版本）
python3 scripts/train_predictor.py \
    --batch-size 8 --grad-accum-steps 4 \
    --sigma-weight-start 1.0 --sigma-weight-end 1.0 \
    --freeze-encoder-epoch 99 \
    --output-dir checkpoints/predictor_full

# 批处理提交（推荐，不怕断线）
sbatch hpc/train_predictor.slurm

# 测试集评估
python3 scripts/evaluate_predictor.py --checkpoint-dir checkpoints/predictor_full
```
