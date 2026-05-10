# Sniff-test probe

Each row runs a real input through the model and asserts the output falls in the expected qualitative range. Pass means the output makes intuitive sense for the input (e.g. dense Manhattan → many building pixels; Jamaica Bay → dominantly water; calm summer week → small surge forecast). The thresholds are intentionally generous; this is a smoke test, not an accuracy benchmark.

Known interpretable failures:

- **TTM at Kings Point / Sandy Hook on calm windows predicts 'moderate' instead of 'calm'.** The fine-tune was trained on the Battery (8518750) only. Slightly larger residuals at unfamiliar stations = correct distribution-shift behaviour, not a bug. For those stations use zero-shot TTM r2 or station-specific fine-tunes.

_Generated: 2026-05-10T14:30:50.121390+00:00_

## Overall: 38 / 40 pass (95%)

## ttm-battery-surge: 8/10 pass

| case | expected | got | peak (m) | elapsed | result | notes |
|---|---|---|---:|---:|---|---|
| `hurricane_ida_2021` | storm | storm | 0.3591 | 1.64s | ✅ | Ida arrived 2021-09-01 night |
| `december_2024_noreaster` | storm | storm | 0.3389 | 1.57s | ✅ | known nor'easter 2024-12-18 |
| `calm_summer_2025_july` | calm | calm | 0.1494 | 1.48s | ✅ | fair-weather Atlantic |
| `calm_winter_2025_feb` | calm | calm | 0.1291 | 1.45s | ✅ | fair-weather midwinter |
| `kings_point_calm_2025` | calm | moderate | 0.1779 | 1.31s | ❌ | Long Island Sound, fair |
| `sandy_hook_calm_2025` | calm | moderate | 0.1617 | 1.57s | ❌ | NJ Bight, fair |
| `battery_recent_30d` | any | calm | 0.096 | 1.19s | ✅ | live last 30 days |
| `kings_point_recent_30d` | any | moderate | 0.1771 | 1.12s | ✅ | Kings Point live |
| `battery_pre_storm_jan2026` | storm | storm | 0.3485 | 2.78s | ✅ | Feb 2026 nor'easter window |
| `battery_calm_april_2026` | calm | calm | 0.119 | 1.24s | ✅ | spring fair-weather |

## prithvi-pluvial: 10/10 pass

| case | expected | pred px | pred % | elapsed | result | notes |
|---|---|---:|---:|---:|---|---|
| `ida_largest_si` | flood | 4941 | 9.85% | 5.06s | ✅ | Staten Island — largest Ida polygon |
| `ida_2nd_si` | flood | 6592 | 13.14% | 1.76s | ✅ | Staten Island — 2nd largest |
| `ida_si_north` | flood | 7981 | 15.91% | 2.0s | ✅ | Staten Island North — large polygon |
| `ida_si_west` | flood | 4234 | 8.44% | 0.85s | ✅ | Staten Island West — large polygon |
| `ida_first_polygon` | flood | 1843 | 3.67% | 2.08s | ✅ | Polygon 0 — Staten Island |
| `manhattan_midtown_clear` | no-flood | 0 | 0.0% | 4.61s | ✅ | dense urban, no Ida polygon |
| `central_park_clear` | no-flood | 0 | 0.0% | 2.26s | ✅ | Central Park, no flood |
| `yankee_stadium_clear` | no-flood | 0 | 0.0% | 2.29s | ✅ | stadium + parking |
| `pelham_bay_clear` | no-flood | 0 | 0.0% | 2.25s | ✅ | Bronx upland, no Ida |
| `forest_hills_clear` | no-flood | 0 | 0.0% | 3.04s | ✅ | Queens upland |

## terramind-buildings: 10/10 pass

| case | expected | pred px | pred % | elapsed | result | notes |
|---|---|---:|---:|---:|---|---|
| `manhattan_midtown` | many | 49901 | 99.45% | 47.58s | ✅ | dense urban core |
| `central_park` | few | 29960 | 59.71% | 35.02s | ✅ | Manhattan park, mixed |
| `jamaica_bay` | none | 92 | 0.18% | 27.94s | ✅ | open water |
| `jfk_airport_runways` | few | 18537 | 36.94% | 54.34s | ✅ | runways = impervious |
| `pelham_bay_park` | few | 736 | 1.47% | 28.9s | ✅ | Bronx park, marshland |
| `brooklyn_industrial` | many | 49292 | 98.24% | 50.71s | ✅ | dense industrial |
| `coney_island_beach` | many | 33477 | 66.72% | 27.13s | ✅ | beach + boardwalk + dense residential |
| `hudson_yards` | many | 35560 | 70.87% | 31.27s | ✅ | dense Manhattan |
| `staten_island_greenbelt` | few | 21652 | 43.15% | 34.54s | ✅ | SI park |
| `queens_residential` | many | 42255 | 84.21% | 42.13s | ✅ | residential mix |

## terramind-lulc: 10/10 pass

| case | expected | dominant | counts (water/imp/veg/bare/bld) | elapsed | result | notes |
|---|---|---|---|---:|---|---|
| `manhattan_midtown` | ['impervious', 'building'] | impervious | 722/49015/307/132/0 | 0.51s | ✅ | dense urban core |
| `central_park` | ['vegetation', 'impervious'] | impervious | 4462/29448/13703/2563/0 | 0.51s | ✅ | Manhattan park, mixed |
| `jamaica_bay` | ['water'] | water | 48328/554/1192/102/0 | 0.51s | ✅ | open water |
| `jfk_airport_runways` | ['impervious'] | impervious | 3082/45800/312/982/0 | 0.5s | ✅ | runways = impervious |
| `pelham_bay_park` | ['vegetation', 'impervious'] | vegetation | 18499/5769/18970/6938/0 | 0.51s | ✅ | Bronx park, marshland |
| `brooklyn_industrial` | ['impervious', 'building'] | impervious | 0/49564/515/97/0 | 0.5s | ✅ | dense industrial |
| `coney_island_beach` | ['water', 'impervious'] | impervious | 15783/29284/165/777/4167 | 0.5s | ✅ | beach + boardwalk + dense residential |
| `hudson_yards` | ['impervious', 'building'] | impervious | 12851/36227/899/199/0 | 0.52s | ✅ | dense Manhattan |
| `staten_island_greenbelt` | ['vegetation', 'impervious'] | impervious | 6/22683/22539/4948/0 | 0.5s | ✅ | SI park |
| `queens_residential` | ['impervious', 'building', 'vegetation'] | impervious | 1902/37139/10645/490/0 | 0.51s | ✅ | residential mix |

