"""Offline model/data boundary introduced in session 30; never downloads weights."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from course.fashion_data import CLASSES, PREPROCESS, SPLIT, arguments


TRANSFER_PREPROCESS = {
    "source": dict(PREPROCESS), "rgb": "repeat grayscale channel three times",
    "resize_shorter": 256, "crop": 224, "interpolation": "bilinear",
    "antialias": True, "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def transfer_arguments(description):
    parser = arguments(description)
    parser.add_argument("--weights", type=Path, help="Local official ResNet18 V1 .pth")
    parser.add_argument("--weights-sha256", help="Full trusted local weight file SHA-256")
    parser.add_argument("--checkpoint", type=Path)
    return parser


def torchvision_parts():
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "Real transfer requires a torchvision build compatible with this torch/Python. "
            "No fallback or installation is attempted; --smoke uses an explicit fixture."
        ) from error
    return resnet18, ResNet18_Weights.IMAGENET1K_V1


class TinyFixture(nn.Module):
    """Random initialization, not pretrained and not a ResNet."""

    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4, padding=1),
                                  nn.BatchNorm2d(4), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(4, 8, 3, stride=2, padding=1),
                                    nn.BatchNorm2d(8), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(8, classes)

    def forward(self, images):
        return self.fc(self.layer4(self.stem(images)).flatten(start_dim=1))


def build_initial(args):
    torch.manual_seed(3000)
    if args.smoke:
        model = TinyFixture()
        provenance = {"kind": "tiny-random-fixture", "pretrained": False,
                      "source": "local seeded random initialization"}
    else:
        if args.weights is None or args.weights_sha256 is None:
            raise ValueError("Real mode requires --weights and --weights-sha256; no downloads")
        expected = args.weights_sha256.lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("--weights-sha256 must be a complete hexadecimal SHA-256")
        digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
        if digest != expected or not digest.startswith("f37072fd"):
            raise ValueError("Weights must match the supplied digest and official ResNet18 V1 prefix")
        resnet18, _ = torchvision_parts()
        model = resnet18(weights=None)
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        provenance = {
            "kind": "resnet18-imagenet1k-v1", "pretrained": True,
            "source": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "sha256": digest, "original_head": "1000 ImageNet classes; discarded",
        }
    model.eval()
    return model, provenance


def preprocess(images, smoke):
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("Expected source images [N,1,28,28] in Fashion [-1,1] convention")
    if not torch.isfinite(images).all() or images.min() < -1 or images.max() > 1:
        raise ValueError("Source pixels must be finite and in [-1,1]")
    rgb = ((images + 1.0) / 2.0).repeat(1, 3, 1, 1)
    if smoke:
        resized = F.interpolate(rgb, size=(256, 256), mode="bilinear",
                                align_corners=False, antialias=True)
        crop = resized[:, :, 16:240, 16:240]
        mean = torch.tensor(TRANSFER_PREPROCESS["mean"]).reshape(1, 3, 1, 1)
        std = torch.tensor(TRANSFER_PREPROCESS["std"]).reshape(1, 3, 1, 1)
        result = (crop - mean) / std
    else:
        _, weights = torchvision_parts()
        result = weights.transforms()(rgb)
    assert result.shape == (len(images), 3, 224, 224)
    return result


def transfer_path(args, stage):
    prefix = "smoke" if args.smoke else "fashion"
    return Path(".artifacts/transfer") / f"{prefix}-{stage}.pt"


def save_transfer(path, model, provenance, args, stage):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1, "stage": stage, "state": model.state_dict(),
        "provenance": provenance, "classes": list(CLASSES),
        "preprocess": dict(TRANSFER_PREPROCESS), "split": SPLIT,
        "smoke": args.smoke, "seed": 3000,
        "training": {"epochs": args.epochs, "max_train": args.max_train,
                     "max_val": args.max_val, "batchnorm": "running statistics frozen"},
    }
    torch.save(payload, path)
    print("saved", path, "provenance:", provenance["kind"])


def load_transfer(path, expected_smoke):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["schema"] != 1 or payload["classes"] != CLASSES
            or payload["preprocess"] != TRANSFER_PREPROCESS or payload["split"] != SPLIT
            or payload["smoke"] != expected_smoke):
        raise ValueError("Transfer checkpoint violates this input/class/split contract")
    provenance = payload["provenance"]
    if expected_smoke:
        if provenance["kind"] != "tiny-random-fixture" or provenance["pretrained"]:
            raise ValueError("Smoke must not masquerade as pretrained ImageNet")
        model = TinyFixture(len(payload["classes"]))
    else:
        if provenance["kind"] != "resnet18-imagenet1k-v1" or not provenance["pretrained"]:
            raise ValueError("Real inference requires the declared pretrained provenance")
        resnet18, _ = torchvision_parts()
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(payload["classes"]))
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, payload
