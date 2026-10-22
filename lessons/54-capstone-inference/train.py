"""Independent prediction interface: never silently generates data or trains."""
import argparse
import json

from course.capstone import RELEASE_PATH, predict_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", help="One input object or a list of up to 128 objects.")
    args = parser.parse_args()
    if not RELEASE_PATH.exists():
        raise SystemExit(
            "Missing release checkpoint. From the course root run:\n"
            ".venv/bin/python -m lessons.53-capstone-selection.train\n"
            "Inference refuses to train or invent fallback predictions."
        )
    if args.input_json is not None:
        inputs = json.loads(args.input_json)
        records = [inputs] if isinstance(inputs, dict) else inputs
        print(json.dumps(predict_records(RELEASE_PATH, records), indent=2))
        return
    records = [
        {"temperature_c": 60., "vibration_mm_s": 1., "load_pct": 40., "machine_type": "fan"},
        {"temperature_c": 96., "vibration_mm_s": 4., "load_pct": 90., "machine_type": "pump"},
        {"temperature_c": None, "vibration_mm_s": 2., "load_pct": 65., "machine_type": "prototype"},
    ]
    predictions = predict_records(RELEASE_PATH, records)
    assert predictions == predict_records(RELEASE_PATH, records)
    assert predictions[-1]["unknown_machine_type"]
    assert predictions[-1]["missing_numeric_fields"] == 1
    print("SYNTHETIC triage only; new input predictions, no training or test evaluation:")
    for record, prediction in zip(records, predictions):
        print(json.dumps({"input": record, "prediction": prediction}))
    invalid = [
        {**records[0], "temperature_c": float("nan")},
        {**records[0], "load_pct": True},
        {**records[0], "vibration_mm_s": -1},
        {**records[0], "unexpected_field": 123},
        {key: value for key, value in records[0].items() if key != "machine_type"},
        {**records[0], "machine_type": ""},
    ]
    for record in invalid:
        try:
            predict_records(RELEASE_PATH, [record])
        except (TypeError, ValueError) as error:
            print("Expected input rejection:", error)
        else:
            raise AssertionError("Invalid input was accepted.")
    try:
        predict_records(RELEASE_PATH, [])
    except ValueError as error:
        print("Expected empty-batch rejection:", error)
    else:
        raise AssertionError("Empty batch was accepted.")
    print("Deterministic safe-load, schema, missing-value, and unknown-category checks passed.")


if __name__ == "__main__":
    main()
