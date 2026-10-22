import torch
from torch import nn

from course.fashion_data import arguments, loader, partitions, test_partition
from course.image_models import cnn
from course.image_training import artifact, evaluate, fit, save_model


def main():
    parser = arguments("Session 25: dropout ablation and execution modes")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2500)
    demonstration = nn.Dropout(0.5)
    ones = torch.ones(256, requires_grad=True)
    demonstration.train()
    with torch.no_grad():
        stochastic = demonstration(ones)
    assert not torch.equal(stochastic, ones)
    demonstration.eval()
    deterministic = demonstration(ones)
    assert deterministic.requires_grad and torch.equal(deterministic, ones)
    print("train()+no_grad(): dropout still active; eval(): gradients still possible")
    train, val = partitions(args)
    best_loss, selected, selected_p = float("inf"), None, None
    for probability in [0.0, 0.4]:
        torch.manual_seed(1900)
        model = cnn(probability)
        print("dropout probability:", probability)
        result = fit(model, train, val, args.epochs)
        if result["loss"] < best_loss:
            best_loss, selected, selected_p = result["loss"], model, probability
    save_model(artifact(args, "selected-cnn"), selected, "cnn-8-16", args,
               dropout=selected_p, augmentation="none")
    print("Selected by validation CE only:", selected_p)
    if args.final_test:
        result = evaluate(selected, loader(test_partition(args)))
        print(f"FINAL {'SYNTHETIC' if args.smoke else 'OFFICIAL'} test "
              f"CE={result['loss']:.4f} accuracy={result['accuracy']:.3f}")
    else:
        print("Official test unopened; use --final-test only after fixing this experiment")


if __name__ == "__main__":
    main()
