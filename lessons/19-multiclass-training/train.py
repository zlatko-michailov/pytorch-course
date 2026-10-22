import torch

from course.fashion_data import arguments, partitions
from course.image_training import artifact, fit, mlp, save_model


def main():
    args = arguments("Session 19: ten raw logits, one class index").parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1900)
    train, val = partitions(args)
    model = mlp()
    scores = model(train.tensors[0][:4])
    labels = train.tensors[1][:4]
    loss = torch.nn.functional.cross_entropy(scores, labels)
    shifted = torch.nn.functional.cross_entropy(scores + 7.0, labels)
    assert torch.allclose(loss, shifted, atol=1e-6)
    assert scores.shape == (len(labels), 10)
    print("CE is invariant to a common logit offset; no softmax before CE")
    fit(model, train, val, args.epochs)
    save_model(artifact(args, "mlp"), model, "mlp-784-64-10", args)


if __name__ == "__main__":
    main()
