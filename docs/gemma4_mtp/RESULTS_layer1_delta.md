# Gemma4 MTP layer1_delta — training results & vLLM deployment benchmark

**Status: root cause IDENTIFIED (pending probe confirmation).** Training converged
cleanly (eval acc 0.91, sim_acc_len 3.27), but the draft deploys in vLLM at ~half
the official baseline's acceptance. Evidence points to a **train/deploy
hidden-state mismatch**: training feeds the draft POST-norm target hidden, vLLM
deployment feeds PRE-norm.

---

## 1. Master table — training vs deployment

| stage | harness | metric | pos0 / avg_acc | acc_len | notes |
|-------|---------|--------|----------------|---------|-------|
| **TRAIN eval** (1000) | TorchSpec teacher-force, mooncake hidden | avg_acc **0.9112**, avg_loss 0.3999 | 0.91 | **3.2697** | best=latest iter_0002269, no overfit |
| **DEPLOY** baseline (official assistant) | vLLM online, fp8 | accept **65.57%** | 0.863 | **4.28** | official `models/assistant` |
| **DEPLOY** trained (hf_iter_0002269) | vLLM online, fp8 | accept **33.02%** | 0.427 | 2.65 | our draft — HALVED |
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
