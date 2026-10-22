import torch

from course.fashion_data import CLASSES, arguments, loader, partitions
from course.image_training import artifact, evaluate, fit, load_model, mlp


def confusion(actual, predicted):
    return torch.bincount(10 * actual + predicted, minlength=100).reshape(10, 10)


def main():
    args = arguments("Session 20: inspect validation errors, not the test set").parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1900)
    train, val = partitions(args)
    if args.smoke:
        model = mlp()
        fit(model, train, val, 1)
    else:
        model, payload = load_model(artifact(args, "mlp"), mlp, False)
        if payload["architecture"] != "mlp-784-64-10":
            raise ValueError("Expected session 19's MLP")
    result = evaluate(model, loader(val))
    matrix = confusion(result["actual"], result["predicted"])
    assert matrix.sum().item() == len(val)
    assert matrix.trace().item() == (result["actual"] == result["predicted"]).sum().item()
    print("Rows=actual, columns=predicted; class order:", CLASSES)
    print(matrix)
    for index, name in enumerate(CLASSES):
        support = matrix[index].sum().item()
        recall = matrix[index, index].item() / support if support else None
        print(f"{index}: {name:12s} support={support} recall={recall}")
    mistakes = (result["actual"] != result["predicted"]).nonzero().flatten()[:5]
    for index in mistakes.tolist():
        actual = result["actual"][index].item()
        predicted = result["predicted"][index].item()
        print(f"validation position={index}: actual={CLASSES[actual]} predicted={CLASSES[predicted]}")
        thumbnail = val[index][0][0][::4, ::4]
        print("\n".join("".join("#" if value > 0 else "." for value in row)
                        for row in thumbnail.tolist()))


if __name__ == "__main__":
    main()
