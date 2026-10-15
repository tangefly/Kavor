from collections.abc import Sequence
from typing import NamedTuple


class Shard(NamedTuple):

    item_start: int
    item_end: int
    row_start: int
    row_end: int


def plan_shards(row_counts: Sequence[int], world_size: int) -> list[Shard]:
    num_items = len(row_counts)
    cum_rows = [0]
    for count in row_counts:
        cum_rows.append(cum_rows[-1] + count)
    total_rows = cum_rows[-1]

    cuts = [0]
    for r in range(1, world_size):
        target = total_rows * r / world_size
        best = cuts[-1]
        for i in range(cuts[-1], num_items + 1):
            if abs(cum_rows[i] - target) < abs(cum_rows[best] - target):
                best = i
        best = min(best, max(cuts[-1], num_items - (world_size - r)))
        cuts.append(best)
    cuts.append(num_items)

    return [
        Shard(cuts[r], cuts[r + 1], cum_rows[cuts[r]], cum_rows[cuts[r + 1]])
        for r in range(world_size)
    ]
