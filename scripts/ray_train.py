"""Launch an existing nanochat training module on a Ray GPU worker group.

Example:
    python -m scripts.ray_train --num-workers 32 -- --model-architecture qwen3-0.6b

Arguments after ``--`` are passed verbatim to scripts.base_train.
"""

import argparse
import os
import runpy
import subprocess
import sys

from scripts.ray_prepare import fetch_tokenizer, load_shell_network_env, prepare_node, train_and_upload_tokenizer


def train_loop_per_worker(config):
    # TorchTrainer has already configured RANK/LOCAL_RANK/WORLD_SIZE and NCCL.
    # Ray owns the process group lifecycle and will destroy it during backend
    # shutdown. Avoid destroying it a second time in scripts.base_train.
    os.environ["NANOCHAT_RAY_MANAGED_PROCESS_GROUP"] = "1"
    # Resolve network settings on the worker itself. In particular, do not put
    # proxy credentials in Ray's serialized runtime environment or its logs.
    os.environ.update(load_shell_network_env())
    sys.argv = ["scripts.base_train", *config["train_args"]]
    runpy.run_module("scripts.base_train", run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description="Run nanochat base training with Ray Train")
    parser.add_argument("--address", default="auto", help="Ray address (default: auto)")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--cpus-per-worker", type=float, default=2)
    parser.add_argument("--run-name", default="nanochat-ray")
    parser.add_argument("--prepare-data-shards", type=int, default=0, help="download this many train shards across node-local disks")
    parser.add_argument("--reuse-tokenizer", action="store_true", help="reuse the tokenizer already in storage instead of retraining it after data preparation")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--local-base-dir", default="/opt/tiger/nanochat_runtime")
    parser.add_argument("--storage-uri", default="", help="durable HDFS/ByteNAS URI for tokenizer and checkpoints")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    if not train_args:
        parser.error("pass scripts.base_train arguments after --")

    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    runtime_env = {
        "working_dir": os.getcwd(),
        "env_vars": {
            "NANOCHAT_BASE_DIR": args.local_base_dir,
            "NANOCHAT_DATA_DIR": os.path.join(args.local_base_dir, "base_data_climbmix"),
            "NANOCHAT_NODE_LOCAL_DATA": "1",
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        },
    }
    if args.storage_uri:
        runtime_env["env_vars"]["NANOCHAT_CHECKPOINT_SYNC_URI"] = args.storage_uri
    for name in (
        "NANOCHAT_DTYPE",
        "WANDB_API_KEY",
        "WANDB_MODE",
    ):
        if value := os.environ.get(name):
            runtime_env["env_vars"][name] = value
    ray.init(address=args.address, runtime_env=runtime_env)

    if args.prepare_data_shards > 0:
        if not args.storage_uri:
            parser.error("--storage-uri is required when preparing node-local data")
        ca_bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("CURL_CA_BUNDLE")
        if not ca_bundle or not os.path.isfile(ca_bundle):
            raise RuntimeError("The driver needs a valid SSL_CERT_FILE or CURL_CA_BUNDLE")
        bootstrap_uri = f"{args.storage_uri.rstrip('/')}/bootstrap"
        nastk = "/opt/tiger/nastk/bin/nastk"
        subprocess.run([nastk, "mkdir", "-p", bootstrap_uri], check=True)
        subprocess.run(
            [nastk, "cp", "-s", ca_bundle, f"{bootstrap_uri}/bytedance-ca-bundle.pem"],
            check=True,
        )
        # Task arguments are stored as an opaque Ray object and are not rendered
        # as runtime-env metadata. Each node persists its copy with mode 0600.
        driver_network_env = load_shell_network_env()
        network_env_ref = ray.put(driver_network_env)
        nodes = sorted(
            (node for node in ray.nodes() if node.get("Alive")),
            key=lambda node: node["NodeManagerAddress"],
        )
        prepare_remote = ray.remote(num_cpus=1)(prepare_node)
        prepare_refs = []
        node_keys = []
        for node_index, node in enumerate(nodes):
            node_key = next(
                key for key in node["Resources"] if key.startswith("node:") and key != "node:__internal_head__"
            )
            node_keys.append(node_key)
            prepare_refs.append(
                prepare_remote.options(resources={node_key: 0.001}).remote(
                    node_index,
                    len(nodes),
                    args.prepare_data_shards,
                    args.local_base_dir,
                    args.storage_uri,
                    network_env_ref,
                    args.download_workers,
                )
            )
        for result in ray.get(prepare_refs):
            print(f"Prepared data: {result}", flush=True)

        if not args.reuse_tokenizer:
            tokenizer_train_remote = ray.remote(num_cpus=4)(train_and_upload_tokenizer)
            ray.get(
                tokenizer_train_remote.options(resources={node_keys[0]: 0.001}).remote(
                    args.local_base_dir, args.storage_uri
                )
            )
        tokenizer_fetch_remote = ray.remote(num_cpus=0.5)(fetch_tokenizer)
        ray.get(
            [
                tokenizer_fetch_remote.options(resources={node_key: 0.001}).remote(
                    args.local_base_dir, args.storage_uri
                )
                for node_key in node_keys
            ]
        )
        print("Tokenizer distributed to every node", flush=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"train_args": train_args},
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=True,
            resources_per_worker={"CPU": args.cpus_per_worker},
            placement_strategy="SPREAD",
        ),
        run_config=RunConfig(name=args.run_name),
        metadata={"base_train_args": train_args},
    )
    result = trainer.fit()
    print(result)


if __name__ == "__main__":
    main()
