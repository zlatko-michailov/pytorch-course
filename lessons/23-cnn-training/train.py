import torch

from course.fashion_data import arguments, partitions
from course.image_models import cnn
from course.image_training import artifact, fit, mlp, save_model


def main():
    args = arguments("Session 23: compare architectures on one fixed split").parse_args()
    torch.set_num_threads(1)
    train, val = partitions(args)
    for name, builder in [("mlp", mlp), ("cnn", cnn)]:
        torch.manual_seed(1900)
        model = builder()
        print(name, "parameters:", sum(p.numel() for p in model.parameters()))
        result = fit(model, train, val, args.epochs)
        print(name, "selected validation CE:", result["loss"])
        if name == "cnn":
            save_model(artifact(args, "cnn"), model, "cnn-8-16", args, dropout=0.0)
    print("Same partitions, epochs, batch size, optimizer and seeds; no official test")


if __name__ == "__main__":
    main()
