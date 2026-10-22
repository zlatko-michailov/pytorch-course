"""Completed synthetic maintenance-triage reference, introduced in lesson 50.

No file I/O, data generation, training, or test evaluation occurs at import time.
"""
import math
from pathlib import Path
import random

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RELEASE_PATH = ROOT / "lessons/53-capstone-selection/artifacts/release.pt"
SELECTION_PATH = RELEASE_PATH.with_name("selection.pt")
SCHEMA = {
    "version": 1,
    "numeric": ["temperature_c", "vibration_mm_s", "load_pct"],
    "bounds": [[0., 150.], [0., 30.], [0., 100.]],
    "category": "machine_type",
    "allow_numeric_null": True,
    "max_batch": 128,
    "target": "synthetic_next_shift_review",
}
SPLIT_UNITS = {"train": range(0, 120), "validation": range(120, 160),
               "test": range(160, 200)}
CANDIDATES = (
    {"name": "logistic", "hidden": 0, "weight_decay": 0.001},
    {"name": "mlp16", "hidden": 16, "weight_decay": 0.002},
)
SEED = 5054
UPDATES = 160
LEARNING_RATE = 0.025
THRESHOLD = 0.5


def validate_record(record):
    if not isinstance(record, dict):
        raise TypeError("Each input must be a JSON object/dict.")
    expected = set(SCHEMA["numeric"]) | {SCHEMA["category"]}
    if set(record) != expected:
        raise ValueError(f"Expected exactly fields {sorted(expected)}; got {sorted(record)}.")
    for key, (low, high) in zip(SCHEMA["numeric"], SCHEMA["bounds"]):
        value = record[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be a number or null, not {type(value).__name__}.")
        if not low <= value <= high or not math.isfinite(value):
            raise ValueError(f"{key} must be finite and in [{low}, {high}].")
    category = record[SCHEMA["category"]]
    if not isinstance(category, str) or not category.strip() or len(category) > 64:
        raise ValueError("machine_type must be a nonempty string of at most 64 characters.")
    if category != category.strip() or category == "<unknown>":
        raise ValueError("machine_type must be trimmed and cannot use the reserved sentinel.")


def make_split(split):
    """Generate six correlated readings per unit; split by unit, not by reading."""
    if split not in SPLIT_UNITS:
        raise ValueError(f"Unknown split: {split}")
    rows = []
    categories = ("pump", "fan", "compressor")
    effects = {"pump": 0.25, "fan": -0.25, "compressor": 0.1}
    for unit in SPLIT_UNITS[split]:
        rng = random.Random(9000 + unit)
        category = categories[unit % len(categories)]
        temperature_offset = rng.gauss(0, 6)
        vibration_offset = rng.uniform(-0.4, 0.4)
        hidden_unit_risk = rng.gauss(0, 0.25)
        for _ in range(6):
            temperature = min(130., max(15., rng.gauss(70 + temperature_offset, 15)))
            vibration = min(8., max(0.05, rng.gauss(2 + vibration_offset, 0.9)))
            load = rng.uniform(10, 98)
            interaction = 1.5 if vibration > 2.8 and load > 70 else 0.
            risk_logit = (-1.15 + (temperature - 70) / 15
                          + 0.9 * (vibration - 2) + 0.5 * (load - 55) / 25
                          + effects[category] + interaction + hidden_unit_risk)
            label = int(rng.random() < 1 / (1 + math.exp(-risk_logit)))
            values = [temperature, vibration, load]
            record = {name: (None if rng.random() < 0.06 else value)
                      for name, value in zip(SCHEMA["numeric"], values)}
            record["machine_type"] = category
            rows.append({"unit_id": f"unit-{unit:03d}", "inputs": record, "target": label})
    return rows


def fit_preprocessor(training_rows):
    if not training_rows:
        raise ValueError("Cannot fit preprocessing without training rows.")
    for row in training_rows:
        validate_record(row["inputs"])
    means, scales = [], []
    for name in SCHEMA["numeric"]:
        observed = [row["inputs"][name] for row in training_rows
                    if row["inputs"][name] is not None]
        if not observed:
            raise ValueError(f"Training field {name} has no observed values.")
        mean = math.fsum(observed) / len(observed)
        variance = math.fsum((value - mean) ** 2 for value in observed) / len(observed)
        means.append(mean)
        scales.append(max(math.sqrt(variance), 1e-6))
    categories = ["<unknown>"] + sorted({
        row["inputs"]["machine_type"] for row in training_rows
    })
    return {"version": 1, "means": means, "scales": scales, "categories": categories,
            "feature_names": ([f"{name}:z" for name in SCHEMA["numeric"]]
                              + [f"{name}:missing" for name in SCHEMA["numeric"]]
                              + [f"machine_type:{name}" for name in categories])}


def transform(records, preprocessing):
    if not isinstance(records, list) or not records:
        raise ValueError("Provide a nonempty list of records.")
    features = []
    categories = preprocessing["categories"]
    for record in records:
        validate_record(record)
        standardized, missing = [], []
        for i, name in enumerate(SCHEMA["numeric"]):
            value = record[name]
            is_missing = value is None
            if is_missing:
                value = preprocessing["means"][i]
            standardized.append((value - preprocessing["means"][i]) / preprocessing["scales"][i])
            missing.append(float(is_missing))
        category = record["machine_type"]
        index = categories.index(category) if category in categories else 0
        one_hot = [float(i == index) for i in range(len(categories))]
        features.append(standardized + missing + one_hot)
    result = torch.tensor(features, dtype=torch.float32)
    assert result.shape[1] == len(preprocessing["feature_names"])
    assert torch.isfinite(result).all().item()
    return result


def tensor_batch(rows, preprocessing):
    return (transform([row["inputs"] for row in rows], preprocessing),
            torch.tensor([row["target"] for row in rows], dtype=torch.float32))


def make_model(input_width, candidate):
    hidden = candidate["hidden"]
    if hidden not in (0, 16):
        raise ValueError("Reference supports only the two predeclared architectures.")
    first = nn.Linear(input_width, hidden or 1)
    # The unknown-category indicator never appears in training. A zero initial
    # coefficient makes its unseen-category contribution neutral, not random.
    with torch.no_grad():
        first.weight[:, 6].zero_()
    return nn.Sequential(first, nn.ReLU(), nn.Linear(hidden, 1)) if hidden else first


@torch.no_grad()
def evaluate(model, batch):
    model.eval()
    features, labels = batch
    logits = model(features).squeeze(-1)
    assert logits.shape == labels.shape and torch.isfinite(logits).all().item()
    predictions, positives = logits.ge(0), labels.bool()
    tp = (predictions & positives).sum().item()
    fn = (~predictions & positives).sum().item()
    tn = (~predictions & ~positives).sum().item()
    fp = (predictions & ~positives).sum().item()
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {"bce": nn.functional.binary_cross_entropy_with_logits(logits, labels).item(),
            "accuracy": (tp + tn) / len(labels), "positive_recall": recall,
            "balanced_accuracy": balanced, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "count": len(labels), "positive_count": int(labels.sum().item())}


def train_candidate(candidate, train, validation):
    torch.manual_seed(SEED)
    model = make_model(train[0].shape[1], candidate)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=candidate["weight_decay"])
    initial_loss = None
    for step in range(UPDATES):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train[0]).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train[1])
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        optimizer.step()
    return model, {"candidate": dict(candidate), "initial_train_bce": initial_loss,
                   "train": evaluate(model, train), "validation": evaluate(model, validation),
                   "parameter_count": sum(p.numel() for p in model.parameters())}


def checkpoint(model, preprocessing, candidate, experiment_record):
    return {
        "format_version": 1, "schema": SCHEMA, "preprocessing": preprocessing,
        "candidate": dict(candidate), "threshold": THRESHOLD, "seed": SEED,
        "updates": UPDATES, "learning_rate": LEARNING_RATE,
        "selection_rule": "lowest validation BCE; first declared candidate breaks ties",
        "state_dict": {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()},
        "experiment_record": experiment_record,
    }


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    if payload.get("schema") != SCHEMA or payload.get("threshold") != THRESHOLD:
        raise ValueError("Checkpoint schema/threshold differs from this reference.")
    preprocessing = payload["preprocessing"]
    if preprocessing.get("version") != 1:
        raise ValueError("Unsupported preprocessing version.")
    if len(preprocessing["means"]) != 3 or len(preprocessing["scales"]) != 3:
        raise ValueError("Expected three numeric preprocessing columns.")
    if any(not math.isfinite(x) for x in preprocessing["means"]):
        raise ValueError("Preprocessing means must be finite.")
    if any(not math.isfinite(x) or x <= 0 for x in preprocessing["scales"]):
        raise ValueError("Preprocessing scales must be finite and positive.")
    categories = preprocessing["categories"]
    if (not isinstance(categories, list) or not categories or categories[0] != "<unknown>"
            or not all(isinstance(item, str) for item in categories)
            or len(set(categories)) != len(categories)):
        raise ValueError("Invalid saved category vocabulary.")
    expected_names = ([f"{name}:z" for name in SCHEMA["numeric"]]
                      + [f"{name}:missing" for name in SCHEMA["numeric"]]
                      + [f"machine_type:{name}" for name in categories])
    if preprocessing["feature_names"] != expected_names:
        raise ValueError("Saved feature order does not match the schema.")
    candidate = payload["candidate"]
    if candidate not in CANDIDATES:
        raise ValueError("Unknown candidate configuration.")
    model = make_model(len(expected_names), candidate)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("Checkpoint parameters must be finite.")
    model.eval()
    return model, preprocessing, payload


def predict_records(path, records):
    if not isinstance(records, list) or not 1 <= len(records) <= SCHEMA["max_batch"]:
        raise ValueError(f"Prediction batch must contain 1..{SCHEMA['max_batch']} records.")
    model, preprocessing, payload = load_checkpoint(path)
    features = transform(records, preprocessing)
    with torch.no_grad():
        probabilities = model(features).squeeze(-1).sigmoid().tolist()
    return [{"probability": probability,
             "decision": "review" if probability >= payload["threshold"] else "routine",
             "unknown_machine_type": record["machine_type"] not in preprocessing["categories"],
             "missing_numeric_fields": sum(record[name] is None for name in SCHEMA["numeric"])}
            for record, probability in zip(records, probabilities)]


def prepared_development_data():
    """The development path deliberately has no request for the test partition."""
    train_rows, validation_rows = make_split("train"), make_split("validation")
    preprocessing = fit_preprocessor(train_rows)
    return (preprocessing, tensor_batch(train_rows, preprocessing),
            tensor_batch(validation_rows, preprocessing))
