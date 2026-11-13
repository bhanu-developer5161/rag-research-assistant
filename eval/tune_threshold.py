"""Choose the refusal threshold from data.

Prints the best dense similarity among the retrieved passages for on-topic
questions and for off-topic ones, then how many of each a given cutoff gets wrong.
Run from the project root (index must exist):  python -m eval.tune_threshold
"""
from src.retrieve import retrieve
from eval.questions import TESTS

ON_TOPIC = [q for q, _ in TESTS]

# Nothing to do with the three papers.
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the 2018 football world cup?",
    "What is the boiling point of water?",
    "How do I renew a passport?",
    "What is the best way to learn guitar?",
]

# Machine-learning topics I believe the three papers do not cover (assumption: verify).
RELATED_ABSENT = [
    "How many parameters does GPT-4 have?",
    "How do convolutional neural networks classify images?",
    "What is reinforcement learning from human feedback?",
    "How does the Llama model handle long context?",
    "How does diffusion work for image generation?",
    "What GPU was used to train ResNet?",
]


def best_score(question):
    return max(h["score"] for h in retrieve(question))


groups = {
    "on-topic (should answer)": [(q, best_score(q)) for q in ON_TOPIC],
    "off-topic (should refuse)": [(q, best_score(q)) for q in OFF_TOPIC],
    "related but absent (should refuse)": [(q, best_score(q)) for q in RELATED_ABSENT],
}

for name, rows in groups.items():
    print(f"\n{name}")
    for q, s in sorted(rows, key=lambda r: r[1]):
        print(f"  {s:.3f}  {q}")

on = [s for _, s in groups["on-topic (should answer)"]]
off = [s for _, s in groups["off-topic (should refuse)"]]
rel = [s for _, s in groups["related but absent (should refuse)"]]

print("\ncutoff | on-topic wrongly refused | off-topic wrongly answered | related-absent wrongly answered")
for cut in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    print(f" {cut:.2f}  |  {sum(s < cut for s in on)}/{len(on)}"
          f"  |  {sum(s >= cut for s in off)}/{len(off)}"
          f"  |  {sum(s >= cut for s in rel)}/{len(rel)}")
