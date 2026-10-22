"""Train-only preprocessing and a completed logistic reference baseline."""
import copy
import json

import torch

from course.capstone import (
    CANDIDATES, evaluate, make_split, prepared_development_data, train_candidate, transform,
)


def main():
    torch.set_num_threads(1)
    preprocessing, train, validation = prepared_development_data()
    original = copy.deepcopy(preprocessing)
    modified = dict(make_split("validation")[0]["inputs"])
    modified["temperature_c"] = 149.
    transform([modified], preprocessing)
    assert preprocessing == original
    model, record = train_candidate(CANDIDATES[0], train, validation)
    assert record["train"]["bce"] < record["initial_train_bce"]
    assert record["parameter_count"] <= 250
    rate = train[1].mean().item()
    constant_logits = torch.full_like(validation[1], torch.logit(torch.tensor(rate)).item())
    constant_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        constant_logits, validation[1]
    ).item()
    print("SYNTHETIC baseline; no test data was generated.")
    print("Train-only preprocessing:", json.dumps(preprocessing, indent=2))
    print("Feature/label shapes:", tuple(train[0].shape), tuple(train[1].shape))
    print(f"Train-prevalence constant baseline validation BCE={constant_bce:.4f}")
    print("Completed logistic experiment:", json.dumps(record, indent=2))
    print("Re-evaluation contract:", evaluate(model, validation) == record["validation"])
    print("Changed validation inputs did not change fitted preprocessing.")


if __name__ == "__main__":
    main()
