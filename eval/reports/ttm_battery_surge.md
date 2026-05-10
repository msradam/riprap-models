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
reproduced: "0.1318 m MAE"
method: "NOAA 8518750 hourly, sliding n=40 + named n=3"
m3: "yes (cpu fp32, ~3M params)"
j_per_call: "0.2125 J (estimated, 17.7 ms)"
```


## Benchmark

- n_calls: 30
- avg_duration_s: 0.0177
- avg_joules: 0.2125 (estimated)
