Index: 41646 chunks, embedder `st:BAAI/bge-small-en-v1.5`, chunker `structure` 350/50 tokens. 113 answerable questions.

| mode | R@1 | R@3 | R@5 | R@10 | R@20 | MRR | doc@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| vector | 0.593 | 0.823 | 0.885 | 0.903 | 0.938 | 0.719 | 0.956 | 121.0 | 169.9 |
| hybrid | 0.628 | 0.805 | 0.841 | 0.947 | 0.991 | 0.745 | 0.974 | 116.2 | 176.8 |
| hybrid_rerank | 0.655 | 0.841 | 0.885 | 0.938 | 0.991 | 0.771 | 0.982 | 2568.2 | 3376.7 |
