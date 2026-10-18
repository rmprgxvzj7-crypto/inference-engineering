# Study repository instructions

## Purpose and authorization

This repository accompanies Philip Kiely's Baseten book, Inference Engineering.
The user explicitly requested: create inference-engineering, add the attention explanation under Chapter 2, and update Git as more questions are asked.

For future substantive questions about this book or inference engineering in sessions where this instruction is available:

1. Explain the concept in conversation at the user's pace.
2. Add the resolved explanation and important follow-up clarifications to the relevant chapter. If understanding is still evolving, label the unresolved point rather than claiming mastery.
3. Update the chapter index and learning log when needed.
4. Verify changed arithmetic or runnable examples; avoid irrelevant test suites.
5. Commit and push to the established remote when access is available. Report the actual commit/link; if blocked, state that it is local only. No new permission is needed for these requested note updates. Respect remote protections and never force-push.

Do not create scheduled tasks for ordinary conversational questions. Do not automatically add unrelated personal conversations. Do not promise cross-session execution unless this repository or its instructions are available in that session.

## Chapter organization

Use exactly one Markdown file per chapter: `chapters/chapter-NN-topic.md`.
Keep that chapter's explanations, diagrams, questions, learning log, and executable examples in the same file. Embed runnable examples in code blocks. Do not create per-topic chapter files, separate chapter index files, or standalone example files for that chapter. Add future questions as sections in the existing chapter file.

## Explanation style that works

- Begin with why an operation is needed, then show how it works.
- Use tiny, concrete numerical examples and label every number by its role.
- Explain one new operation at a time. Do not jump from embeddings to all of Q/K/V and backpropagation at once.
- Distinguish matrices (learned parameters) from vectors calculated using those matrices.
- Maintain one consistent example; explicitly announce when switching to a separate toy example.
- Use tables, equations, and compact diagrams. Do not replace a numerical explanation with an elaborate database or person/expertise analogy.
- The example that clicked: receiver bank [2,3], source money [1,4]; query selects bank's first coordinate (2), key selects money's second coordinate (4), score = 2 × 4 = 8.
- State precisely which token receives information and which supplies it.
- Treat query/key/value names as computational roles, not literal semantic labels stored in matrix entries.
- Distinguish raw scores, scaled scores, attention weights, model parameters, and next-token probabilities.
- Explain simplifications, especially causal masking, omitted tokens, normalization, output projections, and final-layer assumptions.

## Editing discipline

Preserve the user's existing work. Read current files and remote state before editing. Keep source links, original explanations, and book quotations distinct. Link to the book; do not copy its chapters.
