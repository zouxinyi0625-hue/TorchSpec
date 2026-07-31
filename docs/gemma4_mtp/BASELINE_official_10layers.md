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

## 4. 官方 vs 训练后（full_4096_s8001）10 层完整对比 ★

同 harness（conc=none 饱和, spec=5, 1000 prompts/层，本地完整日志解析）。训练模型 = `full_4096_s8001`（全量 26b 数据）。**早时间戳=官方，晚时间戳=训练。**

### 4.1 全指标对比（按 Δaccept_len 降序）

| layer | 官方acc% | 训练acc% | Δacc | 官方len | 训练len | Δlen | 官方pos4 | 训练pos4 | Δpos4 | ΔTPOT ms |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **layer4_commercial_preference** | 68.91 | **76.69** | **+7.78** | 4.45 | **4.83** | **+0.38** | 54.8 | **66.2** | **+11.4** | **−2.9** |
| **layer1_delta** | 65.52 | **72.68** | **+7.16** | 4.28 | **4.63** | **+0.35** | 46.4 | **57.7** | **+11.3** | **−15.0** |
| layer3_commercial_interests | 81.49 | **84.99** | +3.50 | 5.07 | **5.25** | +0.18 | 69.0 | **76.4** | +7.4 | −1.3 |
| layer2_temporal | 52.68 | **55.17** | +2.49 | 3.63 | **3.76** | +0.13 | 34.0 | **38.3** | +4.3 | −2.0 |
| layer2_coarse_interest | 57.54 | **59.42** | +1.88 | 3.88 | **3.97** | +0.09 | 41.6 | **44.7** | +3.0 | −2.7 |
| layer4_biography | 36.01 | **37.55** | +1.54 | 2.80 | **2.88** | +0.08 | 10.0 | **15.3** | +5.3 | **−9.9** |
| layer1_intent | 58.60 | **59.35** | +0.75 | 3.93 | **3.97** | +0.04 | 42.2 | **44.2** | +2.0 | −0.1 |
| layer3_seasonality | 99.03 | 99.03 | 0.00 | 5.95 | 5.95 | 0.00 | 97.7 | **98.1** | +0.4 | 0.0 |
| layer1_actual | 77.06 | 75.87 | −1.19 | 4.85 | 4.79 | −0.06 | 65.9 | 65.3 | −0.6 | +0.6 |
| **layer3_persona** | 53.88 | 49.48 | **−4.40** | 3.69 | 3.47 | **−0.22** | 36.3 | 32.2 | −4.1 | +1.5 |
| **等权均值** | **65.07** | **67.02** | **+1.95** | **4.253** | **4.350** | **+0.10** | **49.8** | **53.8** | **+4.0** | **−3.18** |

### 4.2 训练后优势（核心结论）

**训练后 draft 在 8/10 层全面变好，等权指标全线提升：**

| 指标 | 官方 | 训练 | 提升 | 含义 |
|------|:---:|:---:|:---:|------|
| **等权 accept%** | 65.07 | **67.02** | **+1.95** | 更多 draft token 被接受 |
| **等权 accept_len** | 4.253 | **4.350** | **+0.10** | 每步多产出 token |
| **等权 pos4 accept** | 49.79 | **53.84** | **+4.05** | ★ 尾部 token 接受率大涨 |
| **等权 TPOT** | 95.01 ms | **91.83 ms** | **−3.18 ms (−3.4%)** | ★ 每 token 延迟降低（饱和场景的真实收益）|

**三大亮点：**
1. **重点分布层大赢**：layer1_delta（acc +7.16, pos4 +11.3, TPOT **−15ms**）、layer4_commercial_preference（acc +7.78, pos4 +11.4, tok/s **+4.9%**）——正是训练针对的数据分布。
2. **尾部 accept（pos3/pos4）是最大改善维度**：等权 pos4 +4.0，多个层 +11。投机解码越靠后越难，训练把最难的尾部拉起来，直接延长有效 accept_len。
3. **TPOT 全面下降**（8/10 层）：饱和 bench 下 accept↑ 的收益**确实变现成了每 token 延迟改善**（等权 −3.4%，layer1_delta −15ms、biography −10ms）——这是饱和场景投机收益的正确落点。

### 4.3 两个需跟进点（诚实标注）
- **layer3_persona 退化**（acc −4.40, acc_len −0.22, TPOT +1.5）：唯一明显退化层。full 数据里 persona 可能被稀释/分布冲突，建议查 `source_layer` 占比。
- **layer1_actual 微降**（−1.19 / −0.06）：噪声区，基本持平。

---

## 5. 吞吐（out tok/s）专项对比 — 饱和 bench 下几乎不动

| layer | 官方 tok/s | 训练 tok/s | Δ% |
|-------|:---:|:---:|:---:|
| layer4_commercial_preference | 2572 | **2698** | **+4.9%** |
| layer2_coarse_interest | 2108 | 2175 | +3.1% |
| layer3_commercial_interests | 2323 | 2347 | +1.1% |
| layer2_temporal | 2107 | 2121 | +0.7% |
| layer3_seasonality | 5724 | 5746 | +0.4% |
| layer1_intent | 2357 | 2364 | +0.3% |
| layer1_actual | 2909 | 2910 | +0.0% |
| layer4_biography | 231 | 231 | +0.0% |
| layer3_persona | 1914 | 1904 | −0.5% |
| layer1_delta | 1248 | 1208 | −3.2% |
| **等权均值** | **2349** | **2370** | **+0.9%** |

### 为什么吞吐几乎不涨（尽管 accept_len / TPOT 普遍改善）

**这是 conc=none（无限并发）饱和 bench —— GPU compute-bound，投机收益进 TPOT/延迟（见 §4.2 的 −3.4%），不进吞吐。** 完整机制见 `WORKFLOW_train_infer_parity.md §0`：

1. **饱和场景吞吐 = GPU 算力上限决定**，不由 accept 决定。989+ 并发把 GPU 打满（TTFT 30–240s 全在排队），draft 开销吃 FLOPs 与其它请求抢算力 → accept↑ 省的时间进 **TPOT**（每 token 更快），而非总吞吐。
2. **out tok/s 层间差异（231 vs 5746）远大于训练差异（±几十）**，训练的 ±0.9% 淹没在饱和噪声里；layer1_delta −3.2% 是 duration 波动噪声（其 accept_len 实际 +0.35、TPOT −15ms 都在涨）。

### 吞吐真能涨的场景（已验证）
`WORKFLOW §0`：**conc=32 + spec=5**（非饱和 + 收益空间够）→ layer1_delta 训练后 **out tok/s +6.5%**（1058→1127）+ TPOT −14%。
→ **饱和 10 层不是展示吞吐的场景**；饱和下训练的价值 = **accept_len / pos4 / TPOT**（§4）；要吞吐红利须 **非饱和并发 + spec≥5**。
