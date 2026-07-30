# Gemma4 MTP Draft — 端到端训练/推理一致工作流

> 最后更新: 2026-07-29 · 分支 `dev/gemma4_mtp_offline` · 状态: ✅ **上线验证通过（vllm-msn online bench 全指标提升）**

---

## 0. 一句话结论（Master Table）

| 项 | 状态 | 结论 |
|----|:----:|------|
| **训推一致根因** | ✅ 已修 | 训练用的 HF 并行 forward ≠ vLLM 单步 → 换成验证过的 **strip forward** |
| **token off-by-one** | ✅ 已修 | 喂 `token_{t+1}` 预测 `token_{t+2}`（token+label 一起 shift） |
| **端到端训练** | ✅ 跑通 | `configs/hf_gemma4_mtp.yaml`, layer1, 3 epoch, acc 0.88–0.89 |
| **转换 DCP→HF** | ✅ 跑通 | `convert_dcp_to_hf.py` |
| **上线 bench** | ✅ **变好** | vllm-msn online bench, **两个 layer 全指标提升** |

### 上线 bench 结果（vllm-msn online, 200 prompts, 26b_e011_mtp）

| Layer | 指标 | 官方 baseline | **我们训练** | Δ |
|-------|------|:---:|:---:|:---:|
| **50** | accept_rate % | 64.28 | **68.46** | **+4.18** |
| | accept_len | 4.21 | **4.42** | +0.21 |
| | out tok/s | 1504.9 | **1609.3** | +104 |
| | pos0 % | 85.31 | **88.72** | +3.41 |
| **layer1_delta** | accept_rate % | 70.36 | **71.27** | **+0.91** |
| | accept_len | 4.52 | **4.56** | +0.04 |
| | out tok/s | 953.4 | **1204.9** | **+251 (+26%)** |
| | pos0 % | 88.06 | **89.20** | +1.14 |

per-position accept（trained）：
- **50**: pos0=88.72 pos1=78.57 pos2=67.82 pos3=58.12 pos4=49.08
- **layer1_delta**: pos0=89.20 pos1=79.64 pos2=70.49 pos3=62.37 pos4=54.67

### 4096-context runs — 三方对比（高并发饱和 bench, 989 concurrent, 1000 prompts, 2026-07-30）

| 指标 | baseline (官方) | layer1_4096_s2269 | full_4096_s8001 |
|------|:---:|:---:|:---:|
| **accept rate %** | 65.37 | **77.26** ✅ | 72.63 |
| **accept_len** | 4.27 | **4.86** ✅ | 4.63 |
| **TPOT ms**（↓好） | 100.73 | **88.76** ✅ (−12%) | 89.86 |
| out tok/s | 1295.9 | 1209.4 | **1328.4** |
| duration s | 237.7 | 262.96 | 234.91 |
| pos0 % | 86.09 | **90.75** | 88.54 |
| pos1 % | 75.13 | **83.47** | 79.89 |
| pos2 % | 64.34 | **76.77** | 72.01 |
| pos3 % | 54.87 | **70.69** | 64.95 |
| pos4 % | 46.44 | **64.60** | 57.76 |

HF 导出：`Xinyi0625/gemma4_26ba4b_mtp_layer1_4096_s2269` · `Xinyi0625/gemma4_26ba4b_mtp_full_4096_s8001`

**accept↑ / 吞吐平的原因**：此为**饱和 bench**（989 并发，TTFT≈100s，GPU compute-bound）。投机解码收益进 **TPOT（−12%）**，非吞吐——饱和时吞吐由 GPU 算力上限决定，draft 开销反而抢 FLOPs。要展示吞吐/延迟收益须用**低并发/单流**。详见 `RESULTS_layer1_delta.md §0a`。

**选型**：分布内（layer1 类）用 **layer1**（accept 峰值最高）；全 maiprofile 混合用 **full**（泛化更广）。

---

## 1. 怎么跑训练

### 1.1 前置

| 项 | 值 |
|----|----|
| 训练脚本 | `python -m torchspec.train_entry` |
| config | `configs/hf_gemma4_mtp.yaml` |
| draft config | `configs/draft_models/gemma4_mtp.json`（`assistant_model_path` 已指向官方 assistant，warm start） |
| target | `$AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only` |
| 官方 draft（warm start） | `$AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant` |
| 训练数据 | `data/train_layer1_delta.jsonl` |
| eval 数据 | `data/eval_layer1_delta_1000.jsonl` |
| GPU | 单机 8×A100（7 训练 + 1 HF inference engine） |
| max_seq_length | 4096（默认，config 已改） |

### 1.2 启动命令

```bash
cd $WD/TorchSpec
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH

python -m torchspec.train_entry --config configs/hf_gemma4_mtp.yaml \
    model.target_model_path=$AZURE_ML_INPUT_UKWDATA/maiprofile/models/text_only \
    dataset.train_data_path=$WD/TorchSpec/data/train_layer1_delta.jsonl \
    dataset.eval_data_path=$WD/TorchSpec/data/eval_layer1_delta_1000.jsonl \
    output_dir=./outputs/gemma4-mtp-strip-layer1 2>&1 | tee /tmp/train_strip_layer1.log
```

（`$WD` = `/scratch/azureml/cr/j/27d05d9e68d844bd812d0ba6ed5c0577/exe/wd`）

### 1.3 数据流（不用先 prepare cache）

这是 **online HF 管线**（边训边生成，非离线）：

```
HF Gemma4MTPTargetModel (1 GPU, return_shared_kv_states=True)
    → 实时产 last_hidden + shared_kv{sliding L28, full L29}
    → mooncake 流式传给 7 个 training rank
    → draft forward（strip）算 loss
```

- shared_kv 形状：sliding `(B,8,T,256)`、full `(B,2,T,512)` —— 直接对接 strip forward
- **HF-target 的 shared_kv 够用**：实测喂正确 strip forward，argmax 复现 vLLM（cos 0.97 的数值差不致命）。**无需离线 dump vLLM 数据**。

### 1.4 关键参数（config）

```yaml
gemma4_mtp_num_steps: 4
gemma4_mtp_loss_decay_gamma: 7.0
gemma4_mtp_teacher_force: true
learning_rate: 1.0e-4      # cosine, warmup_ratio 0.04
num_epochs: 3
save_interval: 1000 / save_per_epoch: true
eval_interval: 500
```

### 1.5 训练健康信号

- `[Rank 0] Loaded HF assistant weights from .../assistant` ← warm start（**不是 from scratch**）
- trainable = **614M**（含 tied embed/lm_head 268M，**全参训，不冻结** —— assistant embed 1024d 是 draft 输出层，独立于 target 2816d）
- acc 稳定 **0.88–0.89**，loss ~0.5，`acc_per_step` 单调递减（vLLM accept 形状）
- 1398 步（6537 train / 3 epoch / global_batch 14）

---

## 2. 怎么转换模型（DCP → HF）

训练存的是 DCP 分片 checkpoint（`iter_XXXXXXX/model/`），要转成 HF `Gemma4AssistantForCausalLM` 才能 vLLM 部署。

```bash
python tools/gemma4_mtp/convert_dcp_to_hf.py \
    --ckpt outputs/gemma4-mtp-strip-layer1/checkpoints/iter_0001399 \
    --assistant-config $AZURE_ML_INPUT_UKWDATA/maiprofile/models/assistant \
    --out outputs/gemma4-mtp-strip-layer1/hf_iter_0001399
```

- `--assistant-config`：只取原始 assistant 的 **config**（layer_types/rope/维度），权重来自 checkpoint
- CPU 单进程，无需分布式 init
- **正常输出**：`kept 49 assistant weights, dropped 97 non-assistant keys`
  - dropped 的 `draft_model._strip_fwd.m.*` 是 strip forward 对 assistant 的**引用别名**（同一份参数），drop 重复项、保留 `assistant.*`，**不丢训练成果**
- 产物：`hf_iter_0001399/{config.json, *.safetensors, tokenizer}`

---

## 3. 怎么测评（vllm-msn online bench）

标准 vllm-msn online alignment benchmark：

```bash
# 服务端起 vLLM（26b_e011_mtp config），draft 指向转换后的 hf_iter_0001399
# 客户端跑 per-layer online bench：
#   vllm bench serve --backend openai-chat --base-url http://localhost:8100 \
#       --model gemma4 --tokenizer google/gemma-4-26B-A4B-it \
#       --dataset-path .../sc1_maiprofile_<layer>.jsonl --num-prompts 200 --output-len 8192
```

关注输出的 **Speculative Decoding** 段：Acceptance rate %、Acceptance length、Per-position acceptance。

---

## 4. 修复原理（为什么之前掉一半）

| 维度 | ❌ 之前（掉一半） | ✅ 现在 |
|------|------|------|
| **draft forward** | HF `Gemma4AssistantForCausalLM` 并行 + `create_attention_masks` | `Gemma4MTPStripForward`（每 position 单 query，正确 sliding/full mask） |
| **实测 argmax** | 1201 ≠ vLLM | 188357 == vLLM |
| **token/label** | 喂 `token_t` 预测 `token_{t+1}` | 喂 `token_{t+1}` 预测 `token_{t+2}`（全 shift） |
| **rope** | — | HF apply_rotary（实测 == vLLM，去 vLLM 依赖） |
| **shared_kv 来源** | HF target（cos 0.97） | 同左 —— **不是差异关键**，forward 才是 |

**根因**：HF 的并行 attention mask 语义 ≠ vLLM 单步 decode。这是"训练好但接 vLLM accept 掉一半"的真凶，**不是数据源**（HF vs vLLM 的 3% 数值差对 argmax 不致命）。

### 关键文件

| 文件 | 作用 |
|------|------|
| `torchspec/models/draft/gemma4_mtp_strip_forward.py` | batch strip forward（= vLLM，验证过 188357） |
| `torchspec/models/gemma4_mtp.py` | 训练 forward + token/label off-by-one shift |
| `torchspec/models/draft/gemma4_mtp.py` | draft model（`use_strip_forward` 开关接入） |
| `tools/gemma4_mtp/strip_vllm_draft_step0.py` | 纯 PyTorch 单步剥离（验证基准） |
| `tools/gemma4_mtp/probe_training_forward.py` | 训练前预检：官方权重过训练 forward，answer 区间 pos0=0.89 |
| `tools/gemma4_mtp/convert_dcp_to_hf.py` | DCP → HF |
| `tools/gemma4_mtp/verify_hf_shared_kv.py` | 证明 HF shared_kv 够用 |
| `tools/gemma4_mtp/compare_assistant_target_embed.py` | 证明 embed 该全参训（1024d≠2816d） |

---

## 5. 复现顺序（checklist）

1. `probe_training_forward.py` 预检 → answer 区间 pos0 ≈ 0.89（确认 forward 对齐再训）
2. `train_entry --config configs/hf_gemma4_mtp.yaml ...` → 训到 `iter_0001399`
3. `convert_dcp_to_hf.py` → `hf_iter_0001399`
4. vllm-msn online bench → 对比官方 baseline（accept/len/tok-s/pos0 全升）
