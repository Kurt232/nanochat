from scripts.train_1b_after_06b import prune_node_checkpoint_history


def test_prune_node_checkpoint_history_keeps_final_and_unrelated_files(tmp_path):
    checkpoint_dir = tmp_path / "base_checkpoints" / "model-tag"
    checkpoint_dir.mkdir(parents=True)
    sizes = {}
    for step in (5_000, 10_000):
        for filename in (
            f"model_{step:06d}.pt",
            f"meta_{step:06d}.json",
            f"optim_{step:06d}_rank0.pt",
        ):
            payload = f"payload-{filename}".encode()
            (checkpoint_dir / filename).write_bytes(payload)
            sizes[filename] = len(payload)
    (checkpoint_dir / "notes.txt").write_text("keep me", encoding="utf-8")

    removed_files, removed_bytes = prune_node_checkpoint_history(tmp_path, "model-tag", 10_000)

    assert removed_files == 3
    assert removed_bytes == sum(size for name, size in sizes.items() if "005000" in name)
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == [
        "meta_010000.json",
        "model_010000.pt",
        "notes.txt",
        "optim_010000_rank0.pt",
    ]
