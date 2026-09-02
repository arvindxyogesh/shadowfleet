def build_training_manifest(base_records: list[dict], hard_example_records: list[dict]) -> list[dict]:
    """Union of the base dataset and newly labeled hard examples, keyed by
    input_id. A hard-example record overrides a base record with the same
    input_id (e.g. a corrected label for an image already in the base set).
    """
    combined: dict[str, dict] = {r["input_id"]: r for r in base_records}
    for record in hard_example_records:
        combined[record["input_id"]] = record
    return list(combined.values())


def select_labeled_hard_examples(hard_examples: list[dict]) -> list[dict]:
    """Filter a /hard-examples API response down to inputs that are ready
    to train on (status == "labeled") and reshape each into a manifest
    record: {input_id, label}.
    """
    return [
        {"input_id": example["input_id"], "label": example["label"]}
        for example in hard_examples
        if example.get("status") == "labeled" and example.get("label") is not None
    ]
