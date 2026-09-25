| config | size | perplexity | legal move rate | Stockfish top-1 | value corr |
|---|---|---|---|---|---|
| `baseline_fp16` | — | 4.65 | 65.0% [56.1–72.9] | 15.8% [10.4–23.4] | 0.864 |
| `quant_8bit` | 1.64 GB | 4.65 (1.00x) | 65.0% [56.1–72.9] | 17.5% [11.7–25.3] | 0.867 |
| `quant_4bit` | 0.87 GB | 4.67 (1.01x) | 57.5% [48.6–66.0] | 14.2% [9.0–21.5] | 0.841 |
| `quant_3bit` | 0.68 GB | 5.63 (1.21x) | 50.8% [42.0–59.6] | 12.5% [7.7–19.6] | 0.777 |
| `svd_keep0.9` | 1428M (7% smaller) | 15.05 (3.24x) | 0.0% [0.0–3.1] | 6.7% [3.4–12.6] | 0.269 |
| `svd_keep0.8` | 1312M (15% smaller) | 34.02 (7.32x) | 0.0% [0.0–3.1] | 5.0% [2.3–10.5] | -0.019 |
| `svd_keep0.7` | 1196M (23% smaller) | 280.64 (60.41x) | 0.0% [0.0–3.1] | 4.2% [1.8–9.4] | -0.014 |
| `svd_keep0.6` | 1081M (30% smaller) | 459.95 (99.00x) | 0.0% [0.0–3.1] | 6.7% [3.4–12.6] | -0.032 |
| `svd_keep0.5` | 965M (37% smaller) | 400.82 (86.27x) | 0.0% [0.0–3.1] | 10.8% [6.4–17.7] | -0.079 |
| `svd_keep0.4` | 850M (45% smaller) | 2036019.59 (438240.08x) | 0.0% [0.0–3.1] | 5.8% [2.9–11.6] | -0.001 |
| `svd_keep0.3` | 734M (52% smaller) | 1004144.58 (216135.64x) | 0.0% [0.0–3.1] | 4.2% [1.8–9.4] | -0.023 |
| `asvd_keep0.9` | 1428M (7% smaller) | 6.54 (1.41x) | 51.7% [42.8–60.4] | 5.8% [2.9–11.6] | 0.809 |
| `asvd_keep0.8` | 1312M (15% smaller) | 8.52 (1.83x) | 19.2% [13.1–27.1] | 8.3% [4.6–14.7] | 0.751 |
| `asvd_keep0.7` | 1196M (23% smaller) | 9.84 (2.12x) | 17.5% [11.7–25.3] | 4.2% [1.8–9.4] | 0.377 |
| `asvd_keep0.6` | 1081M (30% smaller) | 22.42 (4.83x) | 3.3% [1.3–8.3] | 4.2% [1.8–9.4] | 0.082 |
| `asvd_keep0.5` | 965M (37% smaller) | 33.54 (7.22x) | 1.7% [0.5–5.9] | 7.5% [4.0–13.6] | -0.043 |
| `asvd_keep0.4` | 850M (45% smaller) | 71.50 (15.39x) | 0.8% [0.1–4.6] | 5.8% [2.9–11.6] | -0.008 |
| `asvd_keep0.3` | 734M (52% smaller) | 477.24 (102.72x) | 0.0% [0.0–3.1] | 5.0% [2.3–10.5] | -0.116 |
| `healed_keep0.7` | — | 8.70 (1.87x) | 9.2% [5.2–15.7] | 6.7% [3.4–12.6] | -0.198 |
| `healed_keep0.5` | — | 8.88 (1.91x) | 9.2% [5.2–15.7] | 5.8% [2.9–11.6] | -0.079 |

### Divergence

- **`quant_8bit`**: perplexity 1.00x; legality retains 100% (within noise); top-1 retains 111% (within noise); value corr retains 100%
- **`quant_4bit`**: perplexity 1.01x; legality retains 88% (within noise); top-1 retains 90% (within noise); value corr retains 97%
- **`quant_3bit`**: perplexity 1.21x; legality retains 78% (within noise); top-1 retains 79% (within noise); value corr retains 90%
- **`svd_keep0.9`**: perplexity 3.24x; legality retains 0% (significant); top-1 retains 42% (within noise); value corr retains 31%
- **`svd_keep0.8`**: perplexity 7.32x; legality retains 0% (significant); top-1 retains 32% (within noise); value corr retains -2%
- **`svd_keep0.7`**: perplexity 60.41x; legality retains 0% (significant); top-1 retains 26% (significant); value corr retains -2%
- **`svd_keep0.6`**: perplexity 99.00x; legality retains 0% (significant); top-1 retains 42% (within noise); value corr retains -4%
- **`svd_keep0.5`**: perplexity 86.27x; legality retains 0% (significant); top-1 retains 68% (within noise); value corr retains -9%
- **`svd_keep0.4`**: perplexity 438240.08x; legality retains 0% (significant); top-1 retains 37% (within noise); value corr retains -0%
- **`svd_keep0.3`**: perplexity 216135.64x; legality retains 0% (significant); top-1 retains 26% (significant); value corr retains -3%
- **`asvd_keep0.9`**: perplexity 1.41x; legality retains 79% (within noise); top-1 retains 37% (within noise); value corr retains 94%
- **`asvd_keep0.8`**: perplexity 1.83x; legality retains 29% (significant); top-1 retains 53% (within noise); value corr retains 87%
- **`asvd_keep0.7`**: perplexity 2.12x; legality retains 27% (significant); top-1 retains 26% (significant); value corr retains 44%
- **`asvd_keep0.6`**: perplexity 4.83x; legality retains 5% (significant); top-1 retains 26% (significant); value corr retains 9%
- **`asvd_keep0.5`**: perplexity 7.22x; legality retains 3% (significant); top-1 retains 47% (within noise); value corr retains -5%
- **`asvd_keep0.4`**: perplexity 15.39x; legality retains 1% (significant); top-1 retains 37% (within noise); value corr retains -1%
- **`asvd_keep0.3`**: perplexity 102.72x; legality retains 0% (significant); top-1 retains 32% (within noise); value corr retains -13%
- **`healed_keep0.7`**: perplexity 1.87x; legality retains 14% (significant); top-1 retains 42% (within noise); value corr retains -23%
- **`healed_keep0.5`**: perplexity 1.91x; legality retains 14% (significant); top-1 retains 37% (within noise); value corr retains -9%
