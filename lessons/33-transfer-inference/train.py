from pathlib import Path

import torch

from course.fashion_data import normalize, partitions, read_idx, test_partition
from course.transfer_contract import (
    build_initial, load_transfer, preprocess, save_transfer, transfer_arguments, transfer_path,
)
from course.transfer_training import train_stage, transfer_evaluate


def main():
    parser = transfer_arguments("Session 33: rebuild preprocessing and class meaning")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--input-images", type=Path,
                        help="Optional local 28x28 grayscale IDX image file, no labels needed")
    args = parser.parse_args()
    torch.set_num_threads(1)
    train, val = partitions(args)
    if args.smoke and args.checkpoint is None:
        original, provenance = build_initial(args)
        train_stage(original, train, val, args)
        train_stage(original, train, val, args, fine_tune=True)
        save_transfer(transfer_path(args, "finetuned"), original, provenance, args, "finetuned")
    path = args.checkpoint or transfer_path(args, "finetuned")
    model, payload = load_transfer(path, args.smoke)
    if payload["stage"] not in ("head", "finetuned"):
        raise ValueError("Expected an adapted head or fine-tuned checkpoint")
    source = val.tensors[0][:4]
    if args.input_images is not None:
        source = normalize(read_idx(args.input_images.parent, args.input_images.name, True))
    images = preprocess(source, args.smoke)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (len(images), len(payload["classes"]))
    if args.smoke and args.checkpoint is None:
        with torch.no_grad():
            reference = original(images)
        torch.testing.assert_close(reference, logits, rtol=0, atol=0)
        print("Exact fixture save/reconstruction roundtrip verified")
    predicted = logits.argmax(dim=1)
    names = [payload["classes"][index] for index in predicted.tolist()]
    assert all(name in payload["classes"] for name in names)
    print("Predicted class names:", names)
    print("Input contract:", payload["preprocess"])
    print("Original source:", payload["provenance"])
    if args.final_test:
        result = transfer_evaluate(model, test_partition(args), args.smoke)
        print(f"FINAL {'SYNTHETIC' if args.smoke else 'OFFICIAL'} test:", result)
    else:
        print("Official test unopened; --final-test only after validation decisions are fixed")


if __name__ == "__main__":
    main()
