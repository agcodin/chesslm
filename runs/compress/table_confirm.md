| config | size | perplexity | legal move rate | Stockfish top-1 | value corr |
|---|---|---|---|---|---|
| `baseline_fp16` | — | 4.64 | 57.8% [54.3–61.1] | 15.5% [13.2–18.2] | 0.865 |
| `quant_4bit` | 0.87 GB | 4.66 (1.00x) | 57.8% [54.3–61.1] | 13.5% [11.3–16.0] | 0.843 |
| `quant_3bit` | 0.68 GB | 5.42 (1.17x) | 51.2% [47.8–54.7] | 12.0% [9.9–14.4] | 0.761 |
| `healedlong_keep0.7` | 1196M (23% smaller) | 4.53 (0.98x) | 63.2% [59.9–66.5] | 11.9% [9.8–14.3] | 0.848 |
| `healedlong_keep0.7` | 1196M (23% smaller) | 4.53 (0.98x) | 63.2% [59.9–66.5] | 11.9% [9.8–14.3] | 0.848 |

### Divergence

- **`quant_4bit`**: perplexity 1.00x; legality retains 100% (within noise); top-1 retains 87% (within noise); value corr retains 97%
- **`quant_3bit`**: perplexity 1.17x; legality retains 89% (within noise); top-1 retains 77% (within noise); value corr retains 88%
- **`healedlong_keep0.7`**: perplexity 0.98x; legality retains 110% (within noise); top-1 retains 77% (within noise); value corr retains 98%
- **`healedlong_keep0.7`**: perplexity 0.98x; legality retains 110% (within noise); top-1 retains 77% (within noise); value corr retains 98%
