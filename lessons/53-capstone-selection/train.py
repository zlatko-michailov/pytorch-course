"""Freeze validation choice, evaluate final test once, persist a safe checkpoint."""
import json

import torch

from course.capstone import (
    CANDIDATES, RELEASE_PATH, SELECTION_PATH, checkpoint, evaluate, load_checkpoint,
    make_split, prepared_development_data, tensor_batch, train_candidate,
)


def main():
    torch.set_num_threads(1)
    if RELEASE_PATH.exists():
        _, _, payload = load_checkpoint(RELEASE_PATH)
        print("CACHED RELEASE: no training, candidate selection, or new test evaluation.")
        print(json.dumps(payload["experiment_record"], indent=2))
        return
    if SELECTION_PATH.exists():
        raise RuntimeError(
            "A selection receipt exists without a release. Investigate the interrupted run; "
            "refusing to silently repeat a potentially already observed final test."
        )
    preprocessing, train, validation = prepared_development_data()
    experiments = []
    trained = []
    for candidate in CANDIDATES:
        model, record = train_candidate(candidate, train, validation)
        trained.append(model)
        experiments.append(record)
        print("Development experiment:", json.dumps(record, indent=2))
    index = min(range(len(experiments)),
                key=lambda i: experiments[i]["validation"]["bce"])
    selected = trained[index]
    record = {"development": experiments, "selected_candidate": CANDIDATES[index]["name"],
              "selection_partition": "validation", "threshold": 0.5,
              "test_evaluations": 0}
    payload = checkpoint(selected, preprocessing, CANDIDATES[index], record)
    RELEASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Persist the decision BEFORE the test partition is even constructed.
    torch.save(payload, SELECTION_PATH)
    reloaded, saved_preprocessing, _ = load_checkpoint(SELECTION_PATH)
    assert saved_preprocessing == preprocessing
    with torch.no_grad():
        before = selected(validation[0])
        after = reloaded(validation[0])
    assert torch.equal(before, after), "Save/reload must reproduce logits exactly on this CPU."
    final_test = tensor_batch(make_split("test"), preprocessing)
    final_metrics = evaluate(reloaded, final_test)
    record["test_evaluations"] = 1
    record["final_test"] = final_metrics
    record["educational_goal_met"] = (
        final_metrics["bce"] <= 0.58 and final_metrics["balanced_accuracy"] >= 0.68
    )
    payload["experiment_record"] = record
    torch.save(payload, RELEASE_PATH)
    _, _, persisted = load_checkpoint(RELEASE_PATH)
    assert persisted["experiment_record"] == record
    report_path = RELEASE_PATH.with_name("experiment.json")
    report_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Frozen choice:", record["selected_candidate"])
    print("ONE final test:", json.dumps(final_metrics, indent=2))
    print("Educational goal met:", record["educational_goal_met"])
    print("Exact CPU checkpoint roundtrip passed. Saved:", RELEASE_PATH.relative_to(RELEASE_PATH.parents[3]))
    print("Rerunning this session reports the cached release, never another final-test pass.")


if __name__ == "__main__":
    main()
