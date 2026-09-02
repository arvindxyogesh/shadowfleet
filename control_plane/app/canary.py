def assign_canary_nodes(node_ids: list[str], target_percentage: int) -> tuple[list[str], list[str]]:
    """Split the fleet into a canary group and a control group for a rollout.

    Deterministic given the same (sorted) node list and percentage, so a
    rollout's assignment is reproducible and auditable rather than
    re-randomized on every call. At least one node is assigned to canary
    whenever the fleet is non-empty and target_percentage > 0.
    """
    if not node_ids:
        return [], []
    if not 0 < target_percentage <= 100:
        raise ValueError("target_percentage must be between 1 and 100")

    ordered = sorted(node_ids)
    canary_count = max(1, round(len(ordered) * target_percentage / 100))
    canary_count = min(canary_count, len(ordered))

    canary = ordered[:canary_count]
    control = ordered[canary_count:]
    return canary, control
