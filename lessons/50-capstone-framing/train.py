"""Problem and data contracts; attempt a design before reading the reference."""
import json

from course.capstone import (
    CANDIDATES, LEARNING_RATE, SCHEMA, SEED, SPLIT_UNITS, THRESHOLD, UPDATES, make_split,
)


def main():
    train, validation = make_split("train"), make_split("validation")
    train_units = {row["unit_id"] for row in train}
    validation_units = {row["unit_id"] for row in validation}
    assert train_units.isdisjoint(validation_units)
    assert all(set(row) == {"unit_id", "inputs", "target"} for row in train)
    print("SYNTHETIC maintenance triage; never use this as an equipment-safety system.")
    print("Input schema:", json.dumps(SCHEMA, indent=2))
    print("One inspectable training row:", json.dumps(train[0], indent=2))
    for name, rows in (("train", train), ("validation", validation)):
        positives = sum(row["target"] for row in rows)
        print(f"{name}: {len(rows)} readings, {len({r['unit_id'] for r in rows})} units, "
              f"positive fraction={positives / len(rows):.3f}")
    print(f"Reserved test specification: {len(SPLIT_UNITS['test'])} units x 6 readings; NOT generated.")
    print("Frozen experiment: candidates=", CANDIDATES)
    print(f"seed={SEED}, Adam lr={LEARNING_RATE}, updates={UPDATES}, threshold={THRESHOLD}")
    print("Select lowest validation BCE; ties favor first candidate; evaluate final test only in 53.")
    print("Engineering budget: CPU/offline, <=250 parameters, inference batches 1..128.")
    print("Educational success goal: final BCE <=.58 and balanced accuracy >=.68.")
    print("A missed goal is a reported result, never a reason to tune on the final test.")


if __name__ == "__main__":
    main()
