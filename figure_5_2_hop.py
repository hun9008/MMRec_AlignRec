
from __future__ import annotations

import argparse
import csv
from pathlib import Path


AnchorRec_top123 = [6405, 6407, 6406]
AlignRec_top123 = [6976, 6995, 6994]


def resolve_interactions_path(path: Path) -> Path:
    candidates = [
        path,
        Path("data/baby/baby.inter"),
        Path("/home/hun/data/data/baby/baby.inter"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find interactions file. Tried: {candidates}")


def read_interactions(path: Path) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(f, dialect=dialect)

        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        if "userID" not in reader.fieldnames or "itemID" not in reader.fieldnames:
            raise ValueError(f"{path} must contain userID and itemID columns.")

        item_to_users: dict[int, set[int]] = {}
        user_to_items: dict[int, set[int]] = {}
        for row in reader:
            user_id = int(row["userID"])
            item_id = int(row["itemID"])
            item_to_users.setdefault(item_id, set()).add(user_id)
            user_to_items.setdefault(user_id, set()).add(item_id)

    return item_to_users, user_to_items


def count_two_hop_items(
    item_id: int,
    item_to_users: dict[int, set[int]],
    user_to_items: dict[int, set[int]],
) -> int:
    users = item_to_users.get(item_id)
    if not users:
        return 0

    two_hop_items: set[int] = set()
    for user_id in users:
        two_hop_items.update(user_to_items.get(user_id, ()))
    two_hop_items.discard(item_id)
    return len(two_hop_items)


def count_one_hop_users(item_id: int, item_to_users: dict[int, set[int]]) -> int:
    return len(item_to_users.get(item_id, ()))


def print_top123_two_hop_counts(
    model_name: str,
    item_ids: list[int],
    item_to_users: dict[int, set[int]],
    user_to_items: dict[int, set[int]],
) -> None:
    print(f"\n[{model_name}]")
    print("rank\titemID\t1-hop user count\t2-hop item count")
    for rank, item_id in enumerate(item_ids, start=1):
        one_hop_count = count_one_hop_users(item_id, item_to_users)
        two_hop_count = count_two_hop_items(item_id, item_to_users, user_to_items)
        print(f"top{rank}\t{item_id}\t{one_hop_count}\t{two_hop_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print 2-hop item counts for Figure 5 AnchorRec/AlignRec top-1/2/3 itemIDs."
    )
    parser.add_argument("--interactions", type=Path, default=Path("data/baby/baby.inter"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactions_path = resolve_interactions_path(args.interactions)
    item_to_users, user_to_items = read_interactions(interactions_path)

    print(f"interactions: {interactions_path}")
    print_top123_two_hop_counts("AnchorRec", AnchorRec_top123, item_to_users, user_to_items)
    print_top123_two_hop_counts("AlignRec", AlignRec_top123, item_to_users, user_to_items)


if __name__ == "__main__":
    main()
