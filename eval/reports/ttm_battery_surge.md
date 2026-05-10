# Granite TTM r2 Battery Surge

## Sliding-window evaluation (post-training-cutoff range)

- windows: 40
- fine-tune MAE: 0.1318 m
- zero-shot TTM r2 MAE: 0.1291 m
- persistence MAE: 0.1866 m
- fine-tune vs persistence: +29.4% MAE reduction
- fine-tune vs zero-shot:   -2.1% MAE reduction

## Named-window breakdown

| window | fine-tune MAE (m) | zero-shot TTM MAE (m) | persistence MAE (m) | n |
|---|---:|---:|---:|---:|
| noreaster_2024_12 | 0.1735 | 0.1535 | 0.1851 | 96 |
| feb_2026_post_training | 0.1380 | 0.1487 | 0.1828 | 96 |
| may_2026_calm_post_training | 0.0976 | 0.0767 | 0.0666 | 96 |

## Provenance

```json
{
  "model_name": "msradam/Granite-TTM-r2-Battery-Surge",
  "model_revision": null,
  "inputs": [
    {
      "sliding_n": 40
    },
    {
      "window": "noreaster_2024_12"
    },
    {
      "window": "feb_2026_post_training"
    },
    {
      "window": "may_2026_calm_post_training"
    }
  ],
  "code_sha": "53e291e9454ccec46f85608b45cfeb5798aeee92",
  "platform": "Darwin arm64 py3.12.12",
  "captured_at_utc": "2026-05-10T12:26:00.637217+00:00"
}
```

```yaml measurements
model: Granite TTM r2 Battery Surge
card_metric: "0.1091 m MAE"
reproduced: "0.1318 m MAE all-windows; 0.3239 m MAE on storm windows (peak >=0.7m), -10% vs zero-shot"
method: "NOAA 8518750 hourly, sliding n=40 + named n=3"
m3: "yes (cpu fp32, ~3M params)"
j_per_call: "0.2125 J (estimated, 17.7 ms)"
```


## Benchmark

- n_calls: 30
- avg_duration_s: 0.0177
- avg_joules: 0.2125 (estimated)

## Performance by surge magnitude

The fine-tune was trained to predict storms; the post-2024 sliding eval is dominated by calm weather. Stratifying by target-window peak surge reveals the model's actual operating regime.

| target peak |window-n| fine-tune MAE | zero-shot MAE | persistence MAE | ft vs zs |
|---|---:|---:|---:|---:|---:|
| all windows | 40 | 0.1318 | 0.1291 | 0.1866 | -2.1% |
| all (≥0.30m) | 30 | 0.1521 | 0.1473 | 0.2205 | -3.3% |
| all (≥0.50m) | 9 | 0.2238 | 0.2377 | 0.3526 | +5.9% |
| all (≥0.70m) | 3 | 0.3239 | 0.3615 | 0.6715 | +10.4% |

**Headline finding**: the fine-tune's edge over zero-shot **scales with surge magnitude**. On calm windows the two are tied (the fine-tune actually trails zero-shot by ~3% on aggregate); at peak ≥ 0.5 m the fine-tune leads by ~6%; at peak ≥ 0.7 m it leads by ~10%. This is exactly the regime the model was trained for. Persistence is uncompetitive at any storm threshold (~50–100% worse).

Practical read: for nor'easter / hurricane nowcasts (the use case), use the fine-tune. For routine fair-weather projections, zero-shot TTM r2 is fine.

