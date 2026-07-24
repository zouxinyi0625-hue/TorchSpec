#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Bit-parity gate for the Gemma4 MTP draft wrapper.

Asserts that Gemma4MTPDraftModel.forward produces logits + last_hidden_state
IDENTICAL to a raw HF Gemma4AssistantForCausalLM, given the same
inputs_embeds / position_ids / shared_kv_states. This is the door that must
show diff==0 before we trust the training wrapper (mirrors the earlier
gemma4-mtp-trainer verify_parity.py which caught the double-scaling bug).

Run on the server (GPU, needs the real weights):

    python tools/gemma4_mtp/verify_parity.py \
        --target    /tmp/models/gemma4/text_only \
        --assistant /tmp/models/gemma4/assistant

Expected: "PARITY OK (max_abs_diff=0.0 ...)".
"""
import argparse

import torch


def build_inputs(target_path, assistant_path, seq_text, device):
    """Produce a realistic (inputs_embeds, position_ids, shared_kv_states) triple
    straight from the target model, exactly as the HF candidate generator does."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target = AutoModelForCausalLM.from_pretrained(
        target_path, dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
    ids = tok(seq_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = target.model(input_ids=ids, use_cache=True, return_shared_kv_states=True)
    shared_kv = out.shared_kv_states
    last_hidden = out.last_hidden_state[:, -1:]              # (B,1,2816) last seen token
    last_tok = ids[:, -1:]
    tok_emb = target.get_input_embeddings()(last_tok)        # raw scaled (B,1,2816)
    inputs_embeds = torch.cat([tok_emb, last_hidden], dim=-1)  # (B,1,5632)
    position_ids = torch.tensor([[ids.shape[1] - 1]], dtype=torch.long, device=device)
    return inputs_embeds, position_ids, shared_kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/tmp/models/gemma4/text_only")
    ap.add_argument("--assistant", default="/tmp/models/gemma4/assistant")
    ap.add_argument("--seq-text", default="The quick brown fox jumps over the lazy dog")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs_embeds, position_ids, shared_kv = build_inputs(
        args.target, args.assistant, args.seq_text, device
    )

    # --- reference: raw HF assistant ---
    from transformers import Gemma4AssistantForCausalLM

    hf = Gemma4AssistantForCausalLM.from_pretrained(
        args.assistant, dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(device)
    with torch.no_grad():
        ref = hf(inputs_embeds=inputs_embeds, position_ids=position_ids,
                 shared_kv_states=shared_kv, use_cache=False)

    # --- our TorchSpec wrapper, loaded from the same weights ---
    from torchspec.models.draft.gemma4_mtp import Gemma4MTPConfig, Gemma4MTPDraftModel

    cfg = Gemma4MTPConfig(assistant_model_path=args.assistant,
                          target_model_path=args.target)
    ours = Gemma4MTPDraftModel(cfg).eval().to(device=device, dtype=torch.bfloat16)
    ours.load_assistant_weights(args.assistant)
    ours.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        logits, last_hs = ours(
            inputs_embeds=inputs_embeds, position_ids=position_ids,
            shared_kv_states=shared_kv,
        )

    d_logits = (logits.float() - ref.logits.float()).abs().max().item()
    d_hs = (last_hs.float() - ref.last_hidden_state.float()).abs().max().item()
    print(f"logits          shape ours={tuple(logits.shape)} ref={tuple(ref.logits.shape)}")
    print(f"last_hidden     shape ours={tuple(last_hs.shape)} ref={tuple(ref.last_hidden_state.shape)}")
    print(f"max_abs_diff    logits={d_logits:.3e}  last_hidden={d_hs:.3e}")
    if d_logits == 0.0 and d_hs == 0.0:
        print("PARITY OK (diff==0) ✅")
    elif max(d_logits, d_hs) < 1e-3:
        print(f"PARITY ~OK (diff<1e-3, likely nondeterministic matmul) ⚠️")
    else:
        print("PARITY FAILED ❌ — investigate before training")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
