# Chapter 2 — Models

Book reference: [Chapter 2: Models](https://www.baseten.co/inference-engineering/book/02-models/).

These notes primarily expand the neural-network and LLM-mechanics topics using our own worked examples.

1. [From word vectors to attention to next-token prediction](attention-from-scratch.md)
2. [How query, key, value, and output matrices are learned](how-matrices-learn.md)
3. [Runnable numerical companion](../../examples/chapter_02_attention.py)

Read the first explanation sequentially. It covers why transformations are needed, why query and key differ, what values contribute, and why the final dot product is a decoding operation.

## Learning log

| Date | Added |
|---|---|
| 2026-10-18 | Consolidated the October 14–16 discussion: Word2Vec versus contextual states; learned matching; Q/K/V; softmax; residual updates; layers; next-token decoding; matrix training. |

## Possible next topics

Multi-head attention; feed-forward networks and nonlinear activations; positional information; causal masking and the KV cache; prefill versus decode. These are not yet covered in depth by these notes.
