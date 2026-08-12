# MTP 训练结论：来源与依据（从原始 online_results 自动重算）

> 生成时间：2026-08-12 11:47:45
> 数据目录：`C:\Users\xinyizou\Downloads\logs\server_logs\online_results`
> 规则：同一 layer 取两个时间戳，**早=官方 baseline**，**晚=训练后模型**

## 1) 原始证据文件（10层 × 2 次）

| layer | 官方(早) | 训练(晚) |
|---|---|---|
| layer1_actual | `26b_e011_mtp_layer1_actual_online_20260730_085313.txt` | `26b_e011_mtp_layer1_actual_online_20260730_100414.txt` |
| layer1_delta | `26b_e011_mtp_layer1_delta_online_20260730_085528.txt` | `26b_e011_mtp_layer1_delta_online_20260730_100628.txt` |
| layer1_intent | `26b_e011_mtp_layer1_intent_online_20260730_090020.txt` | `26b_e011_mtp_layer1_intent_online_20260730_101138.txt` |
| layer2_coarse_interest | `26b_e011_mtp_layer2_coarse_interest_online_20260730_090210.txt` | `26b_e011_mtp_layer2_coarse_interest_online_20260730_101327.txt` |
| layer2_temporal | `26b_e011_mtp_layer2_temporal_online_20260730_090829.txt` | `26b_e011_mtp_layer2_temporal_online_20260730_101928.txt` |
| layer3_commercial_interests | `26b_e011_mtp_layer3_commercial_interests_online_20260730_091052.txt` | `26b_e011_mtp_layer3_commercial_interests_online_20260730_102151.txt` |
| layer3_persona | `26b_e011_mtp_layer3_persona_online_20260730_092036.txt` | `26b_e011_mtp_layer3_persona_online_20260730_103135.txt` |
| layer3_seasonality | `26b_e011_mtp_layer3_seasonality_online_20260730_092957.txt` | `26b_e011_mtp_layer3_seasonality_online_20260730_104101.txt` |
| layer4_biography | `26b_e011_mtp_layer4_biography_online_20260730_093329.txt` | `26b_e011_mtp_layer4_biography_online_20260730_104433.txt` |
| layer4_commercial_preference | `26b_e011_mtp_layer4_commercial_preference_online_20260730_094010.txt` | `26b_e011_mtp_layer4_commercial_preference_online_20260730_105112.txt` |

## 2) 抽取字段（每个原始 txt 直接解析）

- Acceptance rate (%)
- Acceptance length
- Output token throughput (tok/s)
- Request throughput (req/s)
- Mean TPOT (ms)
- Mean TTFT (ms)
- Benchmark duration (s)
- Per-position acceptance (pos0~pos4)

## 3) 逐层对比（官方 vs 训练）

| layer | off acc | tr acc | Δacc | off len | tr len | Δlen | off pos4 | tr pos4 | Δpos4 | off tok/s | tr tok/s | Δtok/s | off TPOT | tr TPOT | ΔTPOT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| layer1_actual | 77.06 | 75.87 | -1.19 | 4.85 | 4.79 | -0.06 | 65.91 | 65.30 | -0.61 | 2909.47 | 2910.01 | +0.54 | 38.55 | 39.19 | +0.64 |
| layer1_delta | 65.52 | 72.68 | +7.16 | 4.28 | 4.63 | +0.35 | 46.43 | 57.73 | +11.30 | 1247.96 | 1207.93 | -40.03 | 107.62 | 92.65 | -14.97 |
| layer1_intent | 58.60 | 59.35 | +0.75 | 3.93 | 3.97 | +0.04 | 42.18 | 44.16 | +1.98 | 2357.33 | 2363.82 | +6.49 | 47.77 | 47.64 | -0.13 |
| layer2_coarse_interest | 57.54 | 59.42 | +1.88 | 3.88 | 3.97 | +0.09 | 41.65 | 44.69 | +3.04 | 2108.50 | 2174.55 | +66.05 | 61.32 | 58.60 | -2.72 |
| layer2_temporal | 52.68 | 55.17 | +2.49 | 3.63 | 3.76 | +0.13 | 33.96 | 38.30 | +4.34 | 2107.14 | 2121.34 | +14.20 | 57.07 | 55.09 | -1.98 |
| layer3_commercial_interests | 81.49 | 84.99 | +3.50 | 5.07 | 5.25 | +0.18 | 69.05 | 76.40 | +7.35 | 2322.58 | 2347.10 | +24.52 | 54.17 | 52.86 | -1.31 |
| layer3_persona | 53.88 | 49.48 | -4.40 | 3.69 | 3.47 | -0.22 | 36.27 | 32.21 | -4.06 | 1913.77 | 1904.42 | -9.35 | 61.31 | 62.78 | +1.47 |
| layer3_seasonality | 99.03 | 99.03 | +0.00 | 5.95 | 5.95 | +0.00 | 97.66 | 98.10 | +0.44 | 5724.42 | 5746.09 | +21.67 | 20.87 | 20.88 | +0.01 |
| layer4_biography | 36.01 | 37.55 | +1.54 | 2.80 | 2.88 | +0.08 | 10.00 | 15.27 | +5.27 | 230.74 | 230.80 | +0.06 | 454.35 | 444.41 | -9.94 |
| layer4_commercial_preference | 68.91 | 76.69 | +7.78 | 4.45 | 4.83 | +0.38 | 54.80 | 66.20 | +11.40 | 2572.43 | 2698.32 | +125.89 | 47.09 | 44.22 | -2.87 |

## 4) 等权汇总（10层平均）

| 指标 | 官方均值 | 训练均值 | Δ |
|---|---:|---:|---:|
| accept rate (%) | 65.072 | 67.023 | +1.951 |
| accept length | 4.253 | 4.350 | +0.097 |
| pos4 (%) | 49.791 | 53.836 | +4.045 |
| out tok/s | 2349.434 | 2370.438 | +21.004 |
| TPOT (ms) | 95.012 | 91.832 | -3.180 |
| QPS (req/s) | 5.482 | 5.511 | +0.029 |
| TTFT (ms) | 122272.366 | 121814.161 | -458.205 |
| duration (s) | 271.703 | 271.064 | -0.639 |

## 5) 结论（仅依据上述原始日志）

1. 训练后在 10 层中多数层 accept/accept_len 提升，等权 `accept +1.951`、`accept_len +0.097`。
2. 尾部 token 接受率明显提升：等权 `pos4 +4.045`。
3. 饱和并发下 out tok/s 只小幅提升（等权 `+21 tok/s`，约 `+0.9%`），但 TPOT 改善明显（等权 `-3.18 ms`）。
4. 个别层退化（如 layer3_persona）在表中保留，结论是“总体提升 + 局部退化需跟进”，非只报喜。
