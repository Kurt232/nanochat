"""Copy the active 1B run's compacted final checkpoint to its durable owner path."""

import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


HDFS = "/opt/tiger/yarn_deploy/hadoop/bin/hdfs"
SOURCE_ROOT = os.environ.get(
    "NANOCHAT_MIGRATION_SOURCE_URI", "hdfs://harunavaali/user/tiger/nanochat"
).rstrip("/")
DEST_ROOT = os.environ.get(
    "NANOCHAT_STORAGE_URI",
    "hdfs://harunavaali/home/byte_search_aisearch_strategy/wenjiedu/nanochat",
).rstrip("/")
MODEL_TAG = "qwen3-1b-50b"
FINAL_STEP = 23_842
WORLD_SIZE = 32


def list_remote(directory):
    completed = subprocess.run(
        [HDFS, "dfs", "-ls", directory],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    filenames = set()
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0][0] in {"-", "d"}:
            filenames.add(fields[-1].rsplit("/", 1)[-1])
    return filenames


def expected_files():
    files = {f"model_{FINAL_STEP:06d}.pt", f"meta_{FINAL_STEP:06d}.json"}
    files.update(
        f"optim_{FINAL_STEP:06d}_rank{rank}.pt" for rank in range(WORLD_SIZE)
    )
    return files


def source_is_compacted():
    directory = f"{SOURCE_ROOT}/base_checkpoints/{MODEL_TAG}"
    files = list_remote(directory)
    checkpoint_pattern = re.compile(
        r"(?:model|meta|optim)_(\d+)(?:_rank\d+)?\.(?:pt|json)$"
    )
    checkpoint_files = {name for name in files if checkpoint_pattern.fullmatch(name)}
    return checkpoint_files == expected_files()


def wait_for_compacted_source():
    print(
        f"Waiting for compacted {MODEL_TAG} step {FINAL_STEP} at {SOURCE_ROOT}",
        flush=True,
    )
    while True:
        try:
            if source_is_compacted():
                return
        except subprocess.CalledProcessError:
            pass
        time.sleep(60)


def copy_final_checkpoint():
    source = f"{SOURCE_ROOT}/base_checkpoints/{MODEL_TAG}"
    checkpoint_root = f"{DEST_ROOT}/base_checkpoints"
    destination = f"{checkpoint_root}/{MODEL_TAG}"
    staging = f"{checkpoint_root}/.{MODEL_TAG}.migrating-{os.getpid()}"
    files = sorted(expected_files())

    subprocess.run([HDFS, "dfs", "-mkdir", "-p", checkpoint_root], check=True)
    try:
        if list_remote(destination) == set(files):
            print(f"Destination is already complete: {destination}", flush=True)
            return
        raise RuntimeError(f"Destination already exists but is incomplete: {destination}")
    except subprocess.CalledProcessError:
        pass

    subprocess.run([HDFS, "dfs", "-mkdir", "-p", staging], check=True)

    def copy_one(filename):
        subprocess.run(
            [
                HDFS,
                "dfs",
                "-cp",
                "-p",
                f"{source}/{filename}",
                f"{staging}/{filename}",
            ],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy_one, files))
    if list_remote(staging) != set(files):
        raise RuntimeError(f"Migration verification failed: {staging}")
    subprocess.run([HDFS, "dfs", "-mv", staging, destination], check=True)
    print(f"Migrated complete final checkpoint to {destination}", flush=True)


def main():
    wait_for_compacted_source()
    copy_final_checkpoint()


if __name__ == "__main__":
    main()
