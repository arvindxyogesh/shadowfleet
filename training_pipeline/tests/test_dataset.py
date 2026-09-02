from training_pipeline.app.dataset import build_training_manifest, select_labeled_hard_examples


def test_build_training_manifest_unions_base_and_hard_examples():
    base = [{"input_id": "a", "label": "base-a"}, {"input_id": "b", "label": "base-b"}]
    hard = [{"input_id": "c", "label": "hard-c"}]

    manifest = build_training_manifest(base, hard)

    ids = {r["input_id"] for r in manifest}
    assert ids == {"a", "b", "c"}


def test_build_training_manifest_hard_examples_override_base_records():
    base = [{"input_id": "a", "label": "base-a"}]
    hard = [{"input_id": "a", "label": "corrected-a"}]

    manifest = build_training_manifest(base, hard)

    assert len(manifest) == 1
    assert manifest[0]["label"] == "corrected-a"


def test_build_training_manifest_with_no_hard_examples_returns_base():
    base = [{"input_id": "a", "label": "base-a"}]
    assert build_training_manifest(base, []) == base


def test_select_labeled_hard_examples_keeps_only_labeled_with_a_label():
    hard_examples = [
        {"input_id": "a", "status": "labeled", "label": {"boxes": []}},
        {"input_id": "b", "status": "pending", "label": None},
        {"input_id": "c", "status": "labeled", "label": None},  # inconsistent state, still excluded
    ]

    selected = select_labeled_hard_examples(hard_examples)

    assert selected == [{"input_id": "a", "label": {"boxes": []}}]
