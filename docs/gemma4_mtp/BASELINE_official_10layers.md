# Google 官方 MTP Draft — 10 层 maiprofile 完整基线

> 2026-07-30 · config `26b_e011_mtp.json` · **官方 assistant（未 maiprofile 训练）**
> target `gemma-4-26B-A4B` (text_only, fp8) · assistant = 官方 `models/assistant`
> spec_tokens=5 · 1000 prompts/layer · max_concurrency=**none（无限并发，饱和）** · rate=inf · tp=1 · max_tokens=8192

这是**官方 Google draft 的零训练基线**，用于对比我们训练的 draft。10 个 maiprofile 短层全跑。

---

## 0. 汇总表（按 accept_len 降序）

| # | layer | accept% | accept_len | out tok/s | QPS (req/s) | TPOT ms | TTFT ms (mean) | duration s | gen tokens | 成功/失败 |
|---|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **layer3_seasonality** | **99.03** | **5.95** | **5724.4** | 5.45 | **20.87** | 78028 | 183.4 | 1,049,801 | 1000/0 |
| 2 | layer3_commercial_interests | 81.49 | 5.07 | 2322.6 | 1.84 | 54.17 | 244440 | 543.0 | 1,261,101 | 1000/0 |
| 3 | layer1_actual | 77.06 | 4.85 | 2909.5 | 9.56 | 38.55 | 46152 | 104.7 | 304,466 | 1000/0 |
| 4 | layer4_commercial_preference | 68.91 | 4.45 | 2572.4 | 5.10 | 47.09 | 90974 | 196.0 | 504,310 | 1000/0 |
| 5 | layer1_delta | 65.52 | 4.28 | 1248.0 | 3.79 | 107.62 | 116778 | 260.1 | 324,606 | 987/13 |
| 6 | layer1_intent | 58.60 | 3.93 | 2357.3 | 12.69 | 47.77 | 34877 | 78.8 | 185,712 | 1000/0 |
| 7 | layer2_coarse_interest | 57.54 | 3.88 | 2108.5 | 2.89 | 61.32 | 149417 | 346.4 | 730,297 | 1000/0 |
| 8 | layer3_persona | 53.88 | 3.69 | 1913.8 | 1.90 | 61.31 | 225043 | 526.0 | 1,006,618 | 1000/0 |
| 9 | layer2_temporal | 52.68 | 3.63 | 2107.1 | 8.87 | 57.07 | 51284 | 112.7 | 237,444 | 1000/0 |
| 10 | **layer4_biography** | **36.01** | **2.80** | **230.7** | 2.73 | **454.35** | 185730 | 366.0 | 84,459 | 1000/0 |

**等权均值**：accept% ≈ **65.1**，accept_len ≈ **4.25**，out tok/s ≈ **2149**。

---

## 1. Per-position acceptance（%）

| layer | pos0 | pos1 | pos2 | pos3 | pos4 |
|-------|:---:|:---:|:---:|:---:|:---:|
| layer3_seasonality | 99.84 | 99.65 | 99.25 | 98.77 | 97.66 |
| layer3_commercial_interests | 93.49 | 87.63 | 81.58 | 75.71 | 69.05 |
| layer1_actual | 90.04 | 82.44 | 76.23 | 70.70 | 65.91 |
| layer4_commercial_preference | 85.75 | 75.42 | 67.52 | 61.07 | 54.80 |
| layer1_delta | 86.04 | 75.36 | 64.62 | 55.14 | 46.43 |
| layer1_intent | 80.01 | 65.37 | 56.10 | 49.31 | 42.18 |
| layer2_coarse_interest | 79.94 | 64.42 | 54.49 | 47.22 | 41.65 |
| layer3_persona | 78.40 | 62.19 | 50.41 | 42.14 | 36.27 |
| layer2_temporal | 77.88 | 60.83 | 49.75 | 40.99 | 33.96 |
| layer4_biography | 67.78 | 47.79 | 33.90 | 20.58 | 10.00 |

---

## 2. 关键观察

- **难度谱系极宽**：从 layer3_seasonality（accept 99%、accept_len 5.95、pos4 仍 97.66%，模板化极易）到 layer4_biography（accept 36%、accept_len 2.80、pos4 仅 10%，长输入自由生成极难）。**官方 draft 在难层几乎失效。**
- **layer4_biography 是异常点**：out tok/s 仅 **230.7**、TPOT **454ms**——输入 4.4M tokens（最长 prompt）+ 生成极少（84k），prefill 极重、accept 崩到 36%，投机基本没用。
- **out tok/s 受输出长度 + accept 双重影响**，不能单看：seasonality 生成 1.05M tokens 又 accept 99% → 5724 tok/s（最高）；biography 生成少 + accept 低 → 230（最低）。
- **layer1_delta（我们主训练层）官方基线**：accept 65.52、accept_len 4.28、pos0 86.04、pos4 46.43——**这正是我们训练后提到 accept 77 / pos4 64 的对照基线**。
- **TTFT 普遍 30–240 秒**：无限并发饱和，绝大部分时间在排队/prefill（见 WORKFLOW §0 吞吐分析）。

## 3. 与训练后 draft 的衔接

本表是**官方未训练**基线。我们训练的 draft 对比见 `WORKFLOW_train_infer_parity.md §0`：
- layer1_delta：官方 accept 65.5 → 训练后 77.3（conc32 spec5，吞吐 +6.5%）。
- 训练主要拉高中低分层（layer1/2/3 非模板层）；seasonality 已饱和无提升空间。

---

## 4. 官方 vs 训练后（full_4096_s8001）10 层完整对比

同 harness（conc=none 饱和, spec=5, 1000 prompts/层）。训练模型 = `full_4096_s8001`（全量 26b 数据）。

| layer | 官方 accept% | 训练 accept% | 官方 acc_len | 训练 acc_len | Δacc_len | 官方 pos4 | 训练 pos4 | Δpos4 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **layer4_commercial_preference** | 68.91 | **76.69** | 4.45 | **4.83** | **+0.38** | 54.80 | **66.20** | **+11.4** |
| **layer1_delta** | 65.52 | **72.68** | 4.28 | **4.63** | **+0.35** | 46.43 | **57.73** | **+11.3** |
| layer3_commercial_interests | 81.49 | **84.99** | 5.07 | **5.25** | +0.18 | 69.05 | **76.40** | +7.4 |
| layer2_temporal | 52.68 | **55.17** | 3.63 | **3.76** | +0.13 | 33.96 | **38.30** | +4.3 |
| layer2_coarse_interest | 57.54 | **59.42** | 3.88 | **3.97** | +0.09 | 41.65 | **44.69** | +3.0 |
| layer4_biography | 36.01 | **37.55** | 2.80 | **2.88** | +0.08 | 10.00 | **15.27** | +5.3 |
| layer1_intent | 58.60 | **59.35** | 3.93 | **3.97** | +0.04 | 42.18 | **44.16** | +2.0 |
| layer3_seasonality | 99.03 | 99.03 | 5.95 | 5.95 | 0.00 | 97.66 | 98.10 | +0.4 |
| layer1_actual | 77.06 | 75.87 | 4.85 | 4.79 | −0.06 | 65.91 | 65.30 | −0.6 |
| **layer3_persona** | 53.88 | 49.48 | 3.69 | 3.47 | **−0.22** | 36.27 | 32.21 | −4.1 |
| **等权均值** | **65.07** | **67.02** | **4.253** | **4.350** | **+0.10** | — | — | — |

### 观察
- **训练净赢 7/10 层**，等权 accept_len +0.10（4.25→4.35），accept% +1.95。
- **最大赢家 = 我们重点训练的分布**：layer1_delta（+0.35, pos4 +11.3）、layer4_commercial_preference（+0.38, pos4 +11.4）——**尾部 accept 大幅改善**，正是训练针对性拉高难 token。
- **seasonality 零变化**（已饱和 5.95）；layer1_actual 微降（−0.06，噪声内）。
- **layer3_persona 唯一明显退化**（−0.22, pos4 −4.1）——full 数据里 persona 分布可能被其它层稀释/冲突，值得单独查（是否 persona 样本占比低或分布偏移）。
- **out tok/s 几乎不变**（饱和 bench，见 WORKFLOW §0：收益进 TPOT 不进吞吐）；seasonality 5724→5746、commercial_pref 2572→2698。

### 结论
full 数据训练在**大多数层普涨**，尤其重点层的**尾部 accept**（pos3/pos4）大幅提升——这正是投机解码 accept_len 的关键。persona 退化是唯一需要跟进的点。
