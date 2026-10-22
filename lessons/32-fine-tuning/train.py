import torch

from course.fashion_data import partitions
from course.transfer_contract import (
    build_initial, load_transfer, save_transfer, transfer_arguments, transfer_path,
)
from course.transfer_training import train_stage, transfer_evaluate


def main():
    args = transfer_arguments("Session 32: unfreeze the last block with a smaller LR").parse_args()
    torch.set_num_threads(1)
    train, val = partitions(args)
    if args.smoke:
        model, provenance = build_initial(args)
        train_stage(model, train, val, args)
    else:
        model, payload = load_transfer(args.checkpoint or transfer_path(args, "head"), False)
        if payload["stage"] != "head":
            raise ValueError("Start this stage from session 31's selected head checkpoint")
        provenance = payload["provenance"]
    head_result = transfer_evaluate(model, val, args.smoke)
    train_stage(model, train, val, args, fine_tune=True)
    fine_result = transfer_evaluate(model, val, args.smoke)
    print("Head validation CE:", head_result["loss"], "fine-tuned CE:", fine_result["loss"])
    if fine_result["loss"] >= head_result["loss"]:
        print("No validation improvement: prefer the head checkpoint for final inference")
    save_transfer(transfer_path(args, "finetuned"), model, provenance, args, "finetuned")


if __name__ == "__main__":
    main()
