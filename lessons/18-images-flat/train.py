import torch

from course.fashion_data import CLASSES, arguments, loader, partitions


def main():
    args = arguments("Session 18: images are structured tensors").parse_args()
    torch.set_num_threads(1)
    train, val = partitions(args)
    images, labels = next(iter(loader(train)))
    flat = images.flatten(start_dim=1)
    assert images.shape[1:] == (1, 28, 28)
    assert flat.shape == (len(labels), 784)
    assert torch.equal(images, flat.reshape_as(images))
    assert labels.dtype == torch.long
    print(f"images={tuple(images.shape)} flat={tuple(flat.shape)} labels={tuple(labels.shape)}")
    print(f"pixel range=[{images.min().item():.3f}, {images.max().item():.3f}]")
    print("first class:", labels[0].item(), CLASSES[labels[0].item()])
    print(f"MLP first layer: 784*64 + 64 = {784 * 64 + 64} parameters")
    print("validation untouched by fitting:", len(val), "examples")


if __name__ == "__main__":
    main()
