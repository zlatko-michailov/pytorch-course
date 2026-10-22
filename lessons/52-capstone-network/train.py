"""One justified nonlinear candidate; validation is not a final score."""
import json

import torch

from course.capstone import CANDIDATES, prepared_development_data, train_candidate


def main():
    torch.set_num_threads(1)
    preprocessing, train, validation = prepared_development_data()
    model, record = train_candidate(CANDIDATES[1], train, validation)
    assert record["parameter_count"] <= 250
    assert record["train"]["bce"] < record["initial_train_bce"]
    with torch.no_grad():
        original_logits = model(validation[0]).squeeze(-1)
        reordered = validation[0].flip(0)
        torch.testing.assert_close(model(reordered).squeeze(-1).flip(0), original_logits)
    print("SYNTHETIC nonlinear candidate; no test data generated.")
    print(model)
    print("Feature order:", preprocessing["feature_names"])
    print("Completed 10 -> 16 -> 1 experiment:", json.dumps(record, indent=2))
    print("Motivation: vibration/load interaction; costs: more parameters and variance.")
    print("No guaranteed win. The predeclared two-candidate validation comparison happens in 53.")
    print("Per-record prediction is invariant to batch reordering.")


if __name__ == "__main__":
    main()
