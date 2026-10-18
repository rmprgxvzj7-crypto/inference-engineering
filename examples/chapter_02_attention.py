"""Dependency-free numerical companion to Chapter 2.

One receiver, two explicitly selected sources, one attention head.
This is pedagogical arithmetic, not a complete transformer implementation.
"""

from math import exp, isclose, sqrt


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("Dot product requires equal vector lengths")
    return sum(x * y for x, y in zip(a, b))


def matmul_row(vector, matrix):
    if len(vector) != len(matrix):
        raise ValueError("Vector width must equal matrix row count")
    return [dot(vector, column) for column in zip(*matrix)]


def softmax(scores):
    # Subtracting a shared maximum preserves softmax and avoids overflow.
    largest = max(scores)
    numerators = [exp(score - largest) for score in scores]
    total = sum(numerators)
    return [value / total for value in numerators]


def main():
    bank = [2.0, 3.0]
    sources = {"money": [1.0, 4.0], "the": [1.0, 3.0]}
    wq = [[1.0], [0.0]]       # Select receiver's first coordinate.
    wk = [[0.0], [1.0]]       # Select source's second coordinate.
    wv = [[2.0, 0.0], [0.0, 1.0]]  # Double first, retain second.

    query = matmul_row(bank, wq)
    keys = [matmul_row(vector, wk) for vector in sources.values()]
    values = [matmul_row(vector, wv) for vector in sources.values()]
    scores = [dot(query, key) / sqrt(len(query)) for key in keys]
    weights = softmax(scores)
    attention = [sum(w * v[j] for w, v in zip(weights, values)) for j in range(2)]
    # Identity output projection; normalization and FFN omitted.
    updated_bank = [x + y for x, y in zip(bank, attention)]

    print("Bank query:", query)
    for name, key, value, score, weight in zip(sources, keys, values, scores, weights):
        print(f"{name}: key={key}, value={value}, score={score:g}, weight={weight:.6f}")
    print("Attention output:", [round(x, 6) for x in attention])
    print("Bank after residual addition:", [round(x, 6) for x in updated_bank])

    # Treat the updated state as final only for this toy output-head example.
    output_vectors = {".": [1.0, 0.0], "to": [0.0, 0.6]}
    before = [dot(bank, vector) for vector in output_vectors.values()]
    after = [dot(updated_bank, vector) for vector in output_vectors.values()]
    probabilities = softmax(after)
    for token, old, new, probability in zip(output_vectors, before, after, probabilities):
        print(f"Next-token candidate {token!r}: old logit={old:.6f}, "
              f"new logit={new:.6f}, two-candidate probability={probability:.6f}")

    # Numerical checks of the lesson's key claims, using unrounded arithmetic.
    assert scores == [8.0, 6.0]
    assert isclose(sum(weights), 1.0)
    assert isclose(attention[0], 2.0)
    assert isclose(attention[1], 3.880797077977882)
    assert before[0] > before[1] and after[1] > after[0]
    assert isclose(6 * 0.21 * 0.5 - 0.6, 0.03)
    assert isclose(6 * 0.2 * 0.51 - 0.6, 0.012)
    print("All worked-example checks passed.")


if __name__ == "__main__":
    main()
