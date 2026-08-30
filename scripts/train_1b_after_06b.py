"""Wait for the active 0.6B run, compact it on HDFS, then launch 1B/50B."""

import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


NASTK = "/opt/tiger/nastk/bin/nastk"
STORAGE_ROOT = os.environ.get(
    "NANOCHAT_STORAGE_URI",
    "hdfs://harunavaali/home/byte_search_aisearch_strategy/wenjiedu/nanochat",
).rstrip("/")
CHECKPOINT_ROOT = f"{STORAGE_ROOT}/base_checkpoints"


def list_remote(directory):
    completed = subprocess.run(
        [NASTK, "ls", directory],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return {line.strip().rsplit("/", 1)[-1] for line in completed.stdout.splitlines() if line.strip()}


def checkpoint_is_complete(model_tag, step, world_size=32):
    filenames = list_remote(f"{CHECKPOINT_ROOT}/{model_tag}")
    required = {f"model_{step:06d}.pt", f"meta_{step:06d}.json"}
    required.update(f"optim_{step:06d}_rank{rank}.pt" for rank in range(world_size))
    return required.issubset(filenames)


def wait_for_checkpoint(model_tag, step):
    print(f"Waiting for complete {model_tag} checkpoint step {step}", flush=True)
    while True:
        try:
            if checkpoint_is_complete(model_tag, step):
                print(f"Found complete {model_tag} checkpoint step {step}", flush=True)
                return
        except subprocess.CalledProcessError:
            pass
        time.sleep(60)


def retain_hdfs_step(model_tag, keep_step):
    """Stage and verify the final checkpoint, then atomically replace the history."""
    directory = f"{CHECKPOINT_ROOT}/{model_tag}"
    filenames = list_remote(directory)
    checkpoint_pattern = re.compile(r"(?:model|meta|optim)_(\d+)(?:_rank\d+)?\.(?:pt|json)$")
    final_files = sorted(
        filename
        for filename in filenames
        if (match := checkpoint_pattern.fullmatch(filename)) and int(match.group(1)) == keep_step
    )
    expected_count = 34  # model + metadata + one optimizer shard per rank
    if len(final_files) != expected_count:
        raise RuntimeError(f"Expected {expected_count} final checkpoint files, found {len(final_files)}")
    staging = f"{directory}.final-{keep_step:06d}"
    subprocess.run([NASTK, "mkdir", "-p", staging], check=True)

    def stage(filename):
        subprocess.run([NASTK, "cp", f"{directory}/{filename}", f"{staging}/{filename}"], check=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(stage, final_files))
    if set(final_files) != list_remote(staging):
        raise RuntimeError(f"Staged final checkpoint verification failed: {staging}")
    subprocess.run([NASTK, "rm", "-r", directory], check=True)
    subprocess.run([NASTK, "mv", staging, directory], check=True)
    print(f"Retained only complete step {keep_step} for {model_tag} on HDFS", flush=True)


def prune_node_checkpoint_history(local_base_dir, model_tag, keep_step):
    """Delete only recognized intermediate checkpoint files on one node."""
    directory = Path(local_base_dir) / "base_checkpoints" / model_tag
    if not directory.is_dir():
        return 0, 0
    pattern = re.compile(r"(?:model|meta|optim)_(\d+)(?:_rank\d+)?\.(?:pt|json)$")
    removed_files = 0
    removed_bytes = 0
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_file() and match and int(match.group(1)) != keep_step:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
    return removed_files, removed_bytes


def prune_cluster_checkpoint_history(model_tag, keep_step):
    import ray

    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False)
    prune_remote = ray.remote(num_cpus=0.1)(prune_node_checkpoint_history)
    refs = []
    for node in (node for node in ray.nodes() if node.get("Alive")):
        node_key = next(
            key for key in node["Resources"] if key.startswith("node:") and key != "node:__internal_head__"
        )
        refs.append(
            prune_remote.options(resources={node_key: 0.001}).remote(
                os.environ.get("NANOCHAT_BASE_DIR", "/opt/tiger/nanochat_runtime"),
                model_tag,
                keep_step,
            )
        )
    results = ray.get(refs)
    ray.shutdown()
    removed_files = sum(result[0] for result in results)
    removed_gib = sum(result[1] for result in results) / 2**30
    print(f"Pruned {removed_files} local intermediate files ({removed_gib:.2f} GiB) for {model_tag}", flush=True)


def main():
    wait_for_checkpoint("qwen3-0.6b-30b", 14_305)
    retain_hdfs_step("qwen3-0.6b-30b", 14_305)
    prune_cluster_checkpoint_history("qwen3-0.6b-30b", 14_305)
    # Give the prior Ray trainer time to release its placement group.
    time.sleep(30)
    print("Launching Qwen3-1B 50B-token training", flush=True)
    subprocess.run(["bash", "runs/qwen3_1b_50b_ray.sh"], check=True)
    wait_for_checkpoint("qwen3-1b-50b", 23_842)
    retain_hdfs_step("qwen3-1b-50b", 23_842)
    prune_cluster_checkpoint_history("qwen3-1b-50b", 23_842)


if __name__ == "__main__":
    main()
