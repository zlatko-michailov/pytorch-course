import torch

from course.fashion_data import arguments, loader, partitions, test_partition
from course.image_training import artifact, evaluate, fit, load_model, mlp, save_model


def main():
    parser = arguments("Session 21: reconstruct the image inference contract")
    parser.add_argument("--final-test", action="store_true",
                        help="Explicit final evaluation after fixing the configuration")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1900)
    train, val = partitions(args)
    path = artifact(args, "mlp")
    if args.smoke:
        model = mlp()
        fit(model, train, val, args.epochs)
        save_model(path, model, "mlp-784-64-10", args)
    model, payload = load_model(path, mlp, args.smoke)
    if payload["architecture"] != "mlp-784-64-10":
        raise ValueError("Expected session 19's architecture")
    images = val.tensors[0][:8]
    with torch.no_grad():
        before = model(images)
    copy_path = artifact(args, "roundtrip")
    torch.save(payload, copy_path)
    restored, _ = load_model(copy_path, mlp, args.smoke)
    with torch.no_grad():
        after = restored(images)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    print("Exact CPU state roundtrip verified; predictions:", after.argmax(dim=1).tolist())
    print("Saved class mapping:", payload["classes"])
    if args.final_test:
        result = evaluate(restored, loader(test_partition(args)))
        print(f"FINAL {'SYNTHETIC' if args.smoke else 'OFFICIAL'} test "
              f"CE={result['loss']:.4f} accuracy={result['accuracy']:.3f}")
    else:
        print("Official test remains unopened; --final-test is an explicit final decision")


if __name__ == "__main__":
    main()
