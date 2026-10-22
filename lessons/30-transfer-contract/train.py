import torch

from course.fashion_data import CLASSES, partitions
from course.transfer_contract import build_initial, preprocess, transfer_arguments


def main():
    args = transfer_arguments("Session 30: local pretrained weight/input contract").parse_args()
    torch.set_num_threads(1)
    train, _ = partitions(args)
    model, provenance = build_initial(args)
    images = preprocess(train.tensors[0][:2], args.smoke)
    with torch.no_grad():
        logits = model(images)
    assert images.shape[1:] == (3, 224, 224) and logits.shape == (len(images), 10)
    print("source [N,1,28,28] -> pretrained input", tuple(images.shape))
    print("replacement head:", model.fc)
    print("class mapping:", CLASSES)
    print("provenance:", provenance)
    print("The replacement head is random; no transfer-quality evaluation yet")


if __name__ == "__main__":
    main()
