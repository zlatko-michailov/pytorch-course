import torch

from course.fashion_data import partitions
from course.transfer_contract import build_initial, save_transfer, transfer_arguments, transfer_path
from course.transfer_training import train_stage


def main():
    args = transfer_arguments("Session 31: train a head, truly freeze the backbone").parse_args()
    torch.set_num_threads(1)
    train, val = partitions(args)
    model, provenance = build_initial(args)
    train_stage(model, train, val, args)
    save_transfer(transfer_path(args, "head"), model, provenance, args, "head")


if __name__ == "__main__":
    main()
