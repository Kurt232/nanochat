"""Prepare node-local nanochat data and tokenizer for a Ray cluster."""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542

NETWORK_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "PIP_CERT",
    "NODE_EXTRA_CA_CERTS",
)


def load_shell_network_env(local_base_dir=None):
    """Load proxy/CA settings on this node without putting values in Ray metadata."""
    # The stock .bashrc returns immediately for non-interactive shells, so source
    # only its proxy exports (in their original order) in a short-lived shell.
    export_patterns = (
        "proxy_pwd",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "no_proxy",
        "NO_PROXY",
    )
    sed_args = " ".join(f"-e '/^[[:space:]]*export {name}=/p'" for name in export_patterns)
    shell_command = f"source <(sed -n {sed_args} ~/.bashrc); env -0"
    completed = subprocess.run(
        ["bash", "-lc", shell_command],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    shell_env = {}
    for entry in completed.stdout.split(b"\0"):
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        shell_env[key.decode(errors="replace")] = value.decode(errors="replace")
    result = {name: shell_env[name] for name in NETWORK_ENV_NAMES if shell_env.get(name)}
    base_dir = local_base_dir or os.environ.get("NANOCHAT_BASE_DIR")
    if base_dir:
        network_env_file = Path(base_dir) / "bootstrap" / "network-env.json"
        if network_env_file.is_file():
            with open(network_env_file, encoding="utf-8") as handle:
                saved_env = json.load(handle)
            for name in NETWORK_ENV_NAMES:
                if saved_env.get(name):
                    result[name] = saved_env[name]
        ca_bundle = Path(base_dir) / "bootstrap" / "bytedance-ca-bundle.pem"
        if ca_bundle.is_file():
            for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT"):
                result[name] = str(ca_bundle)
    return result


def _download_one(index, data_dir):
    filename = f"shard_{index:05d}.parquet"
    destination = data_dir / filename
    if destination.exists() and destination.stat().st_size > 0:
        return "cached"
    tmp = destination.with_suffix(f".parquet.tmp.{os.getpid()}")
    url = f"{BASE_URL}/{filename}"
    for attempt in range(1, 6):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(tmp, destination)
            return "downloaded"
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def prepare_node(
    node_index,
    num_nodes,
    num_train_shards,
    local_base_dir,
    storage_uri,
    fallback_network_env,
    download_workers=8,
):
    """Install the one missing wheel and cache this node's disjoint data subset."""
    bootstrap_dir = Path(local_base_dir) / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    ca_bundle = bootstrap_dir / "bytedance-ca-bundle.pem"
    if not ca_bundle.is_file():
        subprocess.run(
            [
                "/opt/tiger/nastk/bin/nastk",
                "cp",
                "-s",
                f"{storage_uri.rstrip('/')}/bootstrap/bytedance-ca-bundle.pem",
                str(ca_bundle),
            ],
            check=True,
        )
    network_env_file = bootstrap_dir / "network-env.json"
    local_network_env = load_shell_network_env(local_base_dir)
    if not (local_network_env.get("HTTPS_PROXY") or local_network_env.get("https_proxy")):
        safe_fallback = {
            name: fallback_network_env[name]
            for name in NETWORK_ENV_NAMES
            if fallback_network_env.get(name) and "CERT" not in name and "BUNDLE" not in name
        }
        with open(network_env_file, "w", encoding="utf-8") as handle:
            json.dump(safe_fallback, handle)
        os.chmod(network_env_file, 0o600)
    network_env = load_shell_network_env(local_base_dir)
    if not (network_env.get("HTTPS_PROXY") or network_env.get("https_proxy")):
        raise RuntimeError("No HTTPS proxy is configured on this node")
    os.environ.update(network_env)
    command_env = os.environ.copy()
    subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "--user",
            "--no-deps",
            "--index-url",
            "https://pypi.org/simple",
            "rustbpe==0.1.0",
            "kernels==0.11.7",
        ],
        check=True,
        env=command_env,
    )
    base_dir = Path(local_base_dir)
    data_dir = base_dir / "base_data_climbmix"
    data_dir.mkdir(parents=True, exist_ok=True)
    indices = list(range(node_index, num_train_shards, num_nodes)) + [MAX_SHARD]
    counts = {"cached": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {pool.submit(_download_one, index, data_dir): index for index in indices}
        for future in as_completed(futures):
            counts[future.result()] += 1
    return {
        "node_index": node_index,
        "data_dir": str(data_dir),
        "shards": len(indices),
        **counts,
    }


def train_and_upload_tokenizer(local_base_dir, storage_uri):
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = local_base_dir
    env["NANOCHAT_DATA_DIR"] = os.path.join(local_base_dir, "base_data_climbmix")
    subprocess.run(
        ["python", "-m", "scripts.tok_train", "--max-chars", "2000000000", "--vocab-size", "32768"],
        check=True,
        env=env,
    )
    tokenizer_dir = Path(local_base_dir) / "tokenizer"
    remote_dir = f"{storage_uri.rstrip('/')}/tokenizer"
    nastk = "/opt/tiger/nastk/bin/nastk"
    subprocess.run([nastk, "mkdir", "-p", remote_dir], check=True)
    for filename in ("tokenizer.pkl", "token_bytes.pt"):
        subprocess.run([nastk, "cp", "-s", str(tokenizer_dir / filename), f"{remote_dir}/{filename}"], check=True)
    return str(tokenizer_dir)


def fetch_tokenizer(local_base_dir, storage_uri):
    tokenizer_dir = Path(local_base_dir) / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"{storage_uri.rstrip('/')}/tokenizer"
    nastk = "/opt/tiger/nastk/bin/nastk"
    for filename in ("tokenizer.pkl", "token_bytes.pt"):
        subprocess.run([nastk, "cp", "-s", f"{remote_dir}/{filename}", str(tokenizer_dir / filename)], check=True)
    return str(tokenizer_dir)
