import pytest

from control_plane.app.canary import assign_canary_nodes


def test_empty_fleet_returns_empty_groups():
    assert assign_canary_nodes([], 20) == ([], [])


def test_splits_fleet_by_percentage():
    nodes = [f"node-{i}" for i in range(10)]
    canary, control = assign_canary_nodes(nodes, 20)
    assert len(canary) == 2
    assert len(control) == 8
    assert set(canary) | set(control) == set(nodes)
    assert set(canary).isdisjoint(control)


def test_rounds_up_to_at_least_one_canary_node():
    nodes = [f"node-{i}" for i in range(10)]
    canary, control = assign_canary_nodes(nodes, 5)
    assert len(canary) == 1
    assert len(control) == 9


def test_hundred_percent_puts_every_node_in_canary():
    nodes = ["a", "b", "c"]
    canary, control = assign_canary_nodes(nodes, 100)
    assert canary == ["a", "b", "c"]
    assert control == []


def test_assignment_is_deterministic_for_the_same_input():
    nodes = ["node-3", "node-1", "node-2"]
    first = assign_canary_nodes(nodes, 33)
    second = assign_canary_nodes(list(reversed(nodes)), 33)
    assert first == second


def test_rejects_out_of_range_percentage():
    with pytest.raises(ValueError):
        assign_canary_nodes(["a"], 0)
    with pytest.raises(ValueError):
        assign_canary_nodes(["a"], 101)
