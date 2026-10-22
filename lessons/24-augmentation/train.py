import torch

from course.fashion_data import arguments, partitions
from course.image_augmentation import make_augment
from course.image_models import cnn
from course.image_training import artifact, fit, save_model


def main():
    args = arguments("Session 24: training-only augmentation").parse_args()
    torch.set_num_threads(1)
    train, val = partitions(args)
    original = val.tensors[0].clone()
    sample = train.tensors[0][:8]
    changed = make_augment()(sample)
    assert changed.shape == sample.shape and changed.min() >= -1 and changed.max() <= 1
    assert not torch.equal(changed, sample)
    for enabled in [False, True]:
        torch.manual_seed(1900)
        model = cnn()
        print("augmentation:", enabled)
        fit(model, train, val, args.epochs,
            augment=make_augment() if enabled else None)
        if enabled:
            save_model(artifact(args, "augmented-cnn"), model, "cnn-8-16", args,
                       dropout=0.0, augmentation="pad2-crop28-horizontal-flip")
    assert torch.equal(original, val.tensors[0])
    print("Validation pixels are unchanged; no augmentation in evaluation")


if __name__ == "__main__":
    main()
