# From word vectors to attention to next-token prediction

Companion to [Chapter 2 of Inference Engineering](https://www.baseten.co/inference-engineering/book/02-models/). This is an original explanation built from our conversation.

## 1. What problem are we solving?

Consider:

> She deposited money at the bank

A fixed Word2Vec embedding gives “bank” the same vector in this sentence and in “She walked beside the river bank.” It cannot, by itself, distinguish which meaning this occurrence uses.

A transformer starts with token representations and updates them using the available context. We want the representation at “bank” to incorporate information from “money,” “deposited,” and other relevant tokens.

For these notes, pretend each displayed word is one token. Actual tokenizers can split words. Token IDs are lookup keys, not measures of meaning. An embedding lookup maps an ID to a vector. A vector with 4,096 numbers has 4,096 dimensions; a sequence of six such vectors has shape [6, 4096]. Hidden states are the intermediate representations calculated inside the network.

Moving from IDs to embeddings increases representation dimensionality. Successive transformer layers commonly keep the same main representation width while changing its values.

## 2. Why not just add every word equally?

Vector addition can mix information. But equal contributions give us no way to favor useful clues. We need a learned calculation of how much to take from each source for a particular receiver.

In our example, **bank is the receiver** and **money is a source**. Neither role permanently belongs to a word. Each position has its own attention calculation.

“Money should matter more than the” is an illustrative intuition, not a programmed rule. Real heads can use function words, position, syntax, or other features. Large attention weights are not by themselves a complete explanation of a model's reasoning.

## 3. Why transform vectors before comparing them?

Start with invented vectors:

| Token | Current vector |
|---|---|
| bank | [2, 3] |
| money | [1, 4] |
| the | [1, 3] |

A direct dot product compares bank and money:

$$[2,3]\cdot[1,4]=(2\times1)+(3\times4)=14.$$

The second coordinate contributes 12 of that score. Suppose we first transform bank by multiplying its second coordinate by 0.1:

$$[2,3]\longrightarrow[2,0.3].$$

Now the score is:

$$[2,0.3]\cdot[1,4]=2+1.2=3.2.$$

We changed which features influence the comparison. A matrix implements that adjustable transformation; training learns its entries. Real matrices can mix coordinates as well as rescale them. The dot-product operation itself stays unchanged.

The 0.1 transformation above only illustrates the motivation. The next section defines the transformations used throughout our main worked example.

## 4. Why two transformations: query and key?

What matters about the receiving token can differ from what matters about the source token. Let our toy transformations be:

| Role | Transformation | Name |
|---|---|---|
| Receiver side | Select the first coordinate | Query matrix |
| Source side | Select the second coordinate | Key matrix |

Using row vectors, the actual matrices are:

$$W_Q=\begin{bmatrix}1\\0\end{bmatrix},\qquad W_K=\begin{bmatrix}0\\1\end{bmatrix}.$$

Bank's query is calculated by applying the query matrix to bank:

$$q_{bank}=[2,3]W_Q=[2].$$

Money's key is calculated by applying the key matrix to money:

$$k_{money}=[1,4]W_K=[4].$$

Then compare those outputs:

$$\underbrace{2}_{\text{bank's query}}\times\underbrace{4}_{\text{money's key}}=\boxed{8}.$$

**The query matrix was applied to the receiver. The resulting query vector is then compared with the source's key vector. These are separate steps.**

If we reverse roles, money's query is [1] and bank's key is [3], giving 1 × 3 = 3. This shows that the score function can be directional. Reordering the same two vectors in a dot product would not change the answer; changing which token supplies the query uses different vectors.

In this particular causal sentence, money cannot actually attend forward to bank. The reverse calculation illustrates the scoring function only; a causal mask would disallow that connection.

## 5. How do matrices acquire these roles?

The architecture assigns the jobs; training learns useful parameters for those jobs.

- The matrix applied on the receiver side of the score calculation is called the query matrix.
- The matrix applied on the source side is called the key matrix.
- The matrix producing the information that gets mixed is called the value matrix.

Random matrices already occupy these computational roles before training. Their initial outputs are simply unhelpful. They do not need to discover that they are “questions” or “clues.” Those words are optional intuitions, not literal definitions of their entries.

Within a head and layer, the same query transformation is applied to every token. There is no bank-specific query matrix. During ordinary inference, the matrices remain fixed; their output vectors vary with the token representations. Later layers use already-contextual representations and have their own learned parameters.

## 6. Scores are not weights yet

Bank's query is [2]. Using the same key transformation:

| Source | Key | Raw score |
|---|---|---|
| money | [4] | 2 × 4 = 8 |
| the | [3] | 2 × 3 = 6 |

For hand arithmetic, **restrict this toy attention calculation to these two sources**. An ordinary causal attention head would also include the other allowed positions and bank itself. Adding those positions changes the softmax denominator and the resulting numbers.

Scores are divided by the square root of the query/key dimension. Here that dimension is one, so the divisor is sqrt(1) = 1. Larger heads use larger divisors to help control score scale.

Softmax converts scores into positive relative weights summing to one:

$$w_{money}=\frac{e^8}{e^8+e^6}\approx0.880797,$$

$$w_{the}=\frac{e^6}{e^8+e^6}\approx0.119203.$$

We will round these to **0.881 and 0.119** for hand calculations. A weight depends on the receiver and all the competing allowed scores; it is not a permanent importance attached to a word.

Dot products are not Euclidean distances. Their values depend on vector lengths and alignment. Cosine similarity normalizes away lengths; it is useful for comparing embeddings but is not this attention-weight calculation.

## 7. What do we actually retrieve? Values

Keys helped determine the amounts. We now need the vectors to mix. Let the value transformation double the first coordinate and retain the second:

$$W_V=\begin{bmatrix}2&0\\0&1\end{bmatrix}.$$

| Source | Current vector | Value vector |
|---|---|---|
| money | [1, 4] | [2, 4] |
| the | [1, 3] | [2, 3] |

Multiply each value by its attention weight:

$$0.881[2,4]=[1.762,3.524],$$

$$0.119[2,3]=[0.238,0.357].$$

Add the contributions:

$$z_{bank}=[1.762,3.524]+[0.238,0.357]=\boxed{[2,3.881]}.$$

This is the attention output for bank, not a new token.

### Why add money and the?

We are combining their information into one vector for the receiver. This is a weighted average of value vectors, coordinate by coordinate. Both first coordinates are 2, so the result is 2. The second result, 3.881, lies between 3 and 4 and is closer to money's 4 because money has more weight.

Multiple sources can contribute. A separate value transformation lets the model learn what information to transmit independently of the features used to calculate matching scores.

## 8. Update bank's representation

A transformer combines attention-head outputs and applies an output projection. Use one toy head and an identity output projection here, so [2, 3.881] stays unchanged. Then the residual addition gives:

$$\underbrace{[2,3]}_{\text{bank's incoming representation}}+\underbrace{[2,3.881]}_{\text{attention contribution}}=\boxed{[4,6.881]}.$$

This is a temporary hidden-state update for this occurrence of bank. It does not rewrite the embedding table or retrain the model.

Real blocks also contain normalization and a feed-forward network with nonlinear operations and a residual connection. Their order depends on the architecture. We omit their numerical effects here to isolate how information moves between tokens.

## 9. How can the next layer use that context?

It receives the updated hidden states, not fresh embedding lookups. Temporarily ignoring the omitted operations, suppose the next layer's query matrix selects the second coordinate:

| Input to the next query transformation | Query output |
|---|---|
| Original bank [2, 3] | [3] |
| Contextual bank [4, 6.881] | [6.881] |

The new query indirectly depends on money and the because their contributions changed the input. Information gathered in one layer can therefore change attention in subsequent layers.

## 10. How does this predict a next token?

After the final layer and any final normalization, the last position's hidden state feeds a learned vocabulary projection, also called an output head or unembedding. It produces one score (logit) for every vocabulary token.

Treat [4, 6.881] as the final state for our toy example. Invent two output scoring vectors:

| Candidate next token | Learned output vector | Dot product with final state |
|---|---|---|
| . | [1, 0] | 4 × 1 + 6.881 × 0 = 4 |
| to | [0, 0.6] | 4 × 0 + 6.881 × 0.6 = 4.1286 |

“To” scores slightly higher. With the original [2, 3], the scores would have been 2 and 1.8, reversing that ranking. These arbitrary numbers show how context can change a prediction, not which continuation a trained model would actually prefer.

Vocabulary softmax converts all output logits to probabilities. This is a different softmax from attention softmax: attention normalizes across source positions; output softmax normalizes across vocabulary tokens. A decoding policy selects a token, which is appended to the sequence. The process repeats. Implementations usually cache prior keys and values instead of recomputing everything.

### Why would a dot product decode anything useful?

It is a learned weighted sum of features. The arithmetic is simple; the learned weights supply the useful interpretation.

In a separate interpretive example, imagine h = [2, 5], where the first coordinate represents sentence-completion evidence and the second represents evidence that a purpose will follow. These semantic labels are invented; actual features are generally distributed.

| Candidate | Learned decoding weights | Weighted evidence score |
|---|---|---|
| . | [3, -1] | 3 × 2 - 1 × 5 = 1 |
| to | [-1, 2] | -1 × 2 + 2 × 5 = 8 |

For “to,” completion evidence reduces the score and purpose evidence increases it. Training jointly learns the hidden features and output weights. The dot product is therefore a **decoding operation**, not a magical similarity test that inherently knows the next word.

Decoding here predicts what can follow; it does not recover a next word already stored inside bank's vector. Earlier tokens remain available as context. The final hidden state is neither lossless nor guaranteed to capture every detail of the input.

## 11. One diagram of the core calculation

```mermaid
flowchart TD
    R["Receiver: bank vector"] --> Q["Query transformation"]
    S["Source token vectors"] --> K["Key transformation"]
    S --> V["Value transformation"]
    Q --> M["Dot products, scaling, softmax"]
    K --> M
    M --> A["Weighted sum of values"]
    V --> A
    A --> O["Output projection and residual addition"]
    R --> O
```

## 12. What next-token prediction does and does not imply

Predicting tokens is the objective and output mechanism, not a description of every internal computation learned to serve it. Learning representations of systems, relationships, and procedures can improve prediction. That does not prove that next-token training alone produces general intelligence or reliable reasoning. Capabilities must be tested rather than inferred from fluent text.

Continue with [how the matrices are learned](how-matrices-learn.md).
