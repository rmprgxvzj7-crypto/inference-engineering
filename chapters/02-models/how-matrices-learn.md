# How the matrices are learned

The worked attention example chose matrix entries to make their effects visible. This note explains where useful entries come from in training.

## 1. Who chooses the values?

For a model trained from scratch, parameters generally start with small random values chosen using an initialization scheme. During training, the model sees text and predicts the next token at eligible positions. The actual next tokens in the training text provide targets.

If the target token receives probability p, the usual token loss is -log(p). Assigning the actual token more probability reduces that loss. A training batch aggregates many such errors.

Backpropagation calculates how changing each parameter would locally change the loss. An optimizer uses those gradients to update the parameters. Embeddings, query/key/value matrices, feed-forward parameters, and output projections are trained together; there is no separate answer sheet of correct attention weights.

During ordinary inference, those learned parameters stay fixed. New prompts change the temporary hidden states and the query/key/value vectors calculated from them.

## 2. Why don't query and key become identical?

They are initialized separately and affect the loss through different uses in the computation. One shared objective does not imply identical parameter updates.

Use one scalar parameter for each matrix to see this without matrix notation:

| Quantity | Value |
|---|---:|
| Receiver input, bank | 2 |
| Source input, money | 3 |
| Query parameter a | 0.20 |
| Key parameter b | 0.50 |

The query is 2a, the key is 3b, and their score is:

$$s=(2a)(3b)=6ab.$$

The initial score is 6 × 0.20 × 0.50 = 0.60.

### Change only the query parameter

Increase a from 0.20 to 0.21, leaving b = 0.50:

$$s_{new}=6\times0.21\times0.50=0.63.$$

The score increases by 0.03.

The shorthand 6 × 0.01 × 0.50 = 0.03 means:

- 6: the product of the two token inputs, 2 × 3.
- 0.01: the increase in a.
- 0.50: the unchanged b.
- 0.03: the change in score, not the score itself.

### Reset, then change only the key parameter

Start again at a = 0.20 and b = 0.50. Increase b to 0.51:

$$s_{new}=6\times0.20\times0.51=0.612.$$

The score increases by 0.012.

| Experiment | Parameter increase | Score increase |
|---|---:|---:|
| Change a only | 0.01 | 0.030 |
| Change b only | 0.01 | 0.012 |

The score has different sensitivities to a and b. Backpropagation follows those sensitivities through softmax, the value mixture, and later layers all the way to prediction loss. It calculates derivatives rather than physically trying each perturbation. The optimizer need not update both entries by equal amounts.

This scalar example shows how different updates can arise; it is not a full training simulation. Real matrices also act on different token positions and participate in many coupled comparisons. Score increases are not universally good: the direction of a useful update comes from the loss.

## 3. What drives each matrix toward its role?

The architecture defines where each matrix is used:

| Matrix | Computational use | What training can improve |
|---|---|---|
| Query | Receiver side of matching | How receiver features affect source selection |
| Key | Source side of matching | How source features affect matching |
| Value | Information being mixed | What source information gets transmitted |
| Output head | Scores vocabulary tokens from final state | How contextual features predict continuations |

There is no explicit objective “make queries into questions.” All these parameters cooperate to reduce prediction loss, and their positions in the calculation determine the error signals they receive.

Different trained matrices can implement the same behavior; there is no unique required endpoint. For example, in our scalar score, doubling a and halving b preserves the product ab. The matrix case has analogous freedoms. Distinct labels do not require that every parameter be different.

If parameters were initialized identically and treated perfectly symmetrically by a computation, they could remain identical. Standard query/key roles and independently initialized parameters do not impose that constraint.

## 4. What is learned versus calculated?

| Learned parameters, ordinarily fixed at inference | Calculated for the current input |
|---|---|
| Embedding table | Starting token representations |
| Query/key/value matrices | Query/key/value vectors |
| Layer transformations | Contextual hidden states |
| Output head weights | Vocabulary logits and probabilities |

“The representation changed because of context” means the right-hand column changed. It does not mean the model retrained itself while answering.
