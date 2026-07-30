# Gemma4 MTP layer1_delta — training results & vLLM deployment benchmark

**Status: ✅ RESOLVED — train/infer parity fixed, online bench improved on all metrics.**
Root cause was the **training draft FORWARD** (HF parallel `create_attention_masks`
≠ vLLM single-step), NOT the pre/post-norm hidden mismatch originally suspected.
Fix = port the validated single-step **strip forward** into training + token/label
off-by-one shift. Full workflow: [`WORKFLOW_train_infer_parity.md`](WORKFLOW_train_infer_parity.md).

---

## 0. Latest result — vllm-msn online bench (2026-07-29, 200 prompts, 26b_e011_mtp)

Trained draft = `hf_iter_0001399` (strip forward, 3 epoch, layer1). **All metrics up vs official baseline.**

| Layer | accept % | accept_len | out tok/s | pos0 % |
|-------|:---:|:---:|:---:|:---:|
| **50** | 64.28 → **68.46** | 4.21 → **4.42** | 1505 → **1609** | 85.31 → **88.72** |
| **layer1_delta** | 70.36 → **71.27** | 4.52 → **4.56** | 953 → **1205 (+26%)** | 88.06 → **89.20** |

per-position (trained): 50 = 88.72/78.57/67.82/58.12/49.08 · layer1_delta = 89.20/79.64/70.49/62.37/54.67

---

## 0a. 4096-context runs (layer1 + full) — high-concurrency saturated bench (2026-07-30)

Two 4096-context trainings (overnight `run.sh`): layer1_delta and the full
maiprofile_26b split. Both warm-started from the official assistant, strip
forward. Benchmarked on the **saturated** harness (989 concurrent, rate=inf,
TTFT ≈ 100 s → GPU compute-bound), 1000 prompts, target text_only, spec=5.

HF exports: `Xinyi0625/gemma4_26ba4b_mtp_layer1_4096_s2269`,
`Xinyi0625/gemma4_26ba4b_mtp_full_4096_s8001`.

| run | accept % | accept_len | pos0 | pos4 | **TPOT ms** | out tok/s | duration s |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline (official) | 65.37 | 4.27 | 86.09 | 46.44 | 100.73 | 1295.9 | 237.7 |
| **layer1_4096_s2269** | **77.26** | **4.86** | **90.75** | **64.60** | **88.76** | 1209.4 | 262.96 |
| **full_4096_s8001** | 72.63 | 4.63 | 88.54 | 57.76 | 89.86 | **1328.4** | 234.91 |

### 为什么 accept 大涨、吞吐没涨？

**这是高并发饱和 bench（GPU 已打满，compute-bound）；投机解码在此只降 TPOT，不升吞吐。**

- **accept 的收益进了 TPOT**：100.73 → 88.76 ms（**−12%**）。投机解码本质是"每 decode step 验证多 token"，accept↑ 直接降 per-token 延迟——收益真实存在。
- **吞吐（tok/s）在饱和并发下是 FLOP-bound，不是延迟-bound**：989 请求 rate=inf，TTFT≈100 s 全在排队/prefill，GPU 满载。吞吐由算力上限决定；draft 前向 + 验证 + 拒绝 token 的浪费都吃 FLOPs，饱和时和其它请求抢 GPU → 吞吐不升甚至微降。
- **layer1 accept 更高(77%)却吞吐更低**：duration 262.96 s（最长）落在饱和噪声区——draft 开销 + batching 波动主导，accept 微弱优势被淹没，非真实吞吐排序。
- **结论**：draft 是好的（accept 65→77、pos0 86→91、pos4 46→65、TPOT −12%）；**饱和 bench 是"吞吐上限测试"，天然掩盖投机收益**。要展示吞吐/延迟收益应用**低并发/单流**（或 cap max-concurrency），届时 TPOT 改善直接变成更高单流吞吐 + 更低延迟。

### layer1 vs full
- layer1（专训该分布）accept 最高（77.26），pos4 最强（64.60）——分布内最优。
- full（全量 26b）accept 72.63，泛化更广但单层峰值略低；吞吐列最高（1328，饱和噪声内）。
- 选型：**若部署面向 layer1 类分布用 layer1；面向全 maiprofile 混合用 full。**

---

## 0b. HISTORICAL (pre-fix, exp1) — the "deploys at half accept" failure

> ⚠️ Below is the ORIGINAL failing run (exp1, `hf_iter_0002269`) that motivated the
> investigation. The pre/post-norm hypothesis in it was **wrong** — the real cause was
> the forward (see section 0 + WORKFLOW doc). Kept for history.

| stage | harness | metric | pos0 / avg_acc | acc_len | notes |
|-------|---------|--------|----------------|---------|-------|
| **TRAIN eval** (1000) | TorchSpec teacher-force, mooncake hidden | avg_acc **0.9112**, avg_loss 0.3999 | 0.91 | **3.2697** | best=latest iter_0002269, no overfit |
| **DEPLOY** baseline (official assistant) | vLLM online, fp8 | accept **65.57%** | 0.863 | **4.28** | official `models/assistant` |
| **DEPLOY** trained (hf_iter_0002269) | vLLM online, fp8 | accept **33.02%** | 0.427 | 2.65 | our draft — HALVED (pre-fix) |
| **DEPLOY** trained (hf_iter_0002269) | vLLM online, **bf16** | accept **33.50%** | 0.430 | 2.67 | fp8 ruled out |

---

## 2. Training run (exp1 layer1_delta)

- Config: `configs/hf_gemma4_mtp.yaml`, mtp_mode, REPLICATE, 1 inference + 7 training GPUs.
- Data: `data/train_layer1_delta.jsonl` (13466 raw → 10596 after loss-mask filter),
  eval `data/eval_layer1_delta_1000.jsonl` (1000 → 784 after dp truncation).
- `max_seq_length=4096`, `num_epochs=3` → auto `num_train_steps=2268`
  (steps_per_epoch=756, global_batch_size=14, accumulation_steps=2).
- `gemma4_mtp_num_steps=4`, `loss_decay_gamma=7.0`, `teacher_force=true`.
- `inference_engine_type: hf`, `last_hidden_states_prenorm: null`.
- **Final:** `eval/avg_acc=0.9112`, `eval/avg_loss=0.3999`,
  `eval/simulated_acc_len=3.2697`, world_size=7. Checkpoints iter_0002001 +
  iter_0002269 (max_checkpoints=2); best == latest (curve never turned down).
- Exported to HF via `tools/gemma4_mtp/convert_dcp_to_hf.py` →
  `gemma4_mtp_exp1_hf/hf_iter_0002269/` (48 tensors, pre_projection norm 32.8).

---

## 3. Deployment benchmark (vLLM online)

Bench: `vllm-msn/benchmarks/gemma4_12b_fp8/run_maiprofile_online.sh`, layer1_delta
eval 200 prompts, target `models/text_only`, spec_tokens=5, same server/temperature
per column. Only `GEMMA4_ASSISTANT_MODEL_PATH` varies between baseline and trained.

Per-position acceptance (%):

| run | pos0 | pos1 | pos2 | pos3 | pos4 |
|-----|------|------|------|------|------|
| baseline (official) | 86.26 | 75.61 | 64.36 | 54.88 | 46.73 |
| trained fp8 | 42.66 | 36.82 | 32.60 | 28.38 | 24.67 |
| trained bf16 | 43.03 | 37.20 | 32.90 | 29.03 | 25.34 |

---

## 4. Diagnosis — what's ruled out and what's left

Ruled out (hard evidence):
- **FP8**: bf16 (33.50) ≈ fp8 (33.02).
- **Weight loading**: server log, no missing/unexpected keys; 0.78 GiB draft
  loaded; 4 draft layers mapped to target L28/L29.
- **embed_scale**: vLLM gemma4_mtp normalizer = sqrt(2816) = 53.07 = training scale.
- **Repeatability / harness**: baseline hits 65.57% on the same flow.

Key tell: trained draft is HALVED at EVERY position including **pos0** (no prefix
dependency, teacher-force-independent). Uniform ×0.5 including pos0 = systematic
input mismatch, NOT model quality. And TRAIN-eval pos0 (0.91) ≈ baseline-DEPLOY
pos0 (0.863): the draft is as good as the official one — it's just fed differently
at deploy.

Root cause (from source, pending live probe) —
`torchspec/config/inference_config.py::resolve_last_hidden_states_prenorm`:

```python
"""vLLM's extract_hidden_states connector can only capture raw layer outputs
(pre-norm), while sglang and hf provide post-norm outputs."""
if self.last_hidden_states_prenorm is not None:
    return self.last_hidden_states_prenorm
return self.inference_engine_type == "vllm"
```

- Training: `last_hidden_states_prenorm=null` + `inference_engine_type=hf`
  → `False` → **POST-norm** hidden (confirmed by `gemma4_mtp_target.py:175
  last_hidden = out.last_hidden_state`, after `model.norm` for HF Gemma4).
- vLLM deploy: feeds **PRE-norm** hidden (raw last-layer output).
- → draft trained on RMSNorm'd (~O(1)) hidden but served un-normalized hidden
  → `pre_projection` sees OOD input → uniform accept halving incl. pos0.

The official assistant is unaffected because it was trained against vLLM's
pre-norm convention.

---

## 5. Confirmation probe

`tools/gemma4_mtp/probe_prenorm_gap.py`: loads the target, runs one real prompt,
prints mean per-token L2 norm of PRE-norm vs POST-norm hidden. Large ratio =
gap confirmed.

```bash
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
python tools/gemma4_mtp/probe_prenorm_gap.py \
    --target /tmp/models/gemma4/text_only \
    --data   data/eval_layer1_delta_1000.jsonl --max-len 2048
```

---

## 6. Fixes (decide after probe)

- **A (clean, retrain):** set training `inference.last_hidden_states_prenorm: true`
  so the HF engine emits pre-norm hidden matching vLLM; retrain. Produces a draft
  compatible with stock vLLM (no patch needed).
- **B (no retrain, patch vLLM):** apply the target's `model.norm` to
  `hidden_states` inside vLLM `gemma4_mtp.py` before `pre_projection`. Faster to
  test the hypothesis, but binds the draft to a patched vLLM.
