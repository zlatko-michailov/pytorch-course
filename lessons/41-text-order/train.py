"""An exact representational collision, not a failed optimizer."""
import torch
from torch import nn


def main():
    torch.manual_seed(41)
    torch.set_num_threads(1)
    vocabulary = {"<pad>": 0, "<unk>": 1, "cat": 2, "chases": 3, "dog": 4}
    examples = [("cat chases dog", 1), ("dog chases cat", 0)]
    ids = torch.tensor([[vocabulary[word] for word in text.split()]
                        for text, _ in examples])
    labels = torch.tensor([label for _, label in examples], dtype=torch.float32)
    embedding = nn.Embedding(len(vocabulary), 6)
    classifier = nn.Linear(6, 1)
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + list(classifier.parameters()), lr=0.03
    )
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        means = embedding(ids).mean(1)
        logits = classifier(means).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        means = embedding(ids).mean(1)
        logits = classifier(means).squeeze(-1)
        torch.testing.assert_close(means[0], means[1])
        torch.testing.assert_close(logits[0], logits[1])
        probabilities = logits.sigmoid()
        accuracy = (logits.ge(0) == labels.bool()).float().mean().item()
    assert accuracy == 0.5
    # Concatenation keeps a slot for each position; this is a counterexample,
    # not the variable-length architecture we will build next.
    ordered = embedding(ids).detach().flatten(1)
    assert not torch.allclose(ordered[0], ordered[1])
    for example, probability in zip(examples, probabilities.tolist()):
        print(example, f"predicted P(cat is actor)={probability:.6f}")
    print(f"Best shared-probability BCE is log(2)={torch.log(torch.tensor(2.)).item():.6f}")
    print(f"Observed BCE={loss.item():.6f}, accuracy={accuracy:.1%}")
    print("Mean vectors collide; ordered flattened vectors differ:", ordered.shape)
    print("This is a proof about this representation, not about all text models.")


if __name__ == "__main__":
    main()
