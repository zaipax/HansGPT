"""Bounded, model-free communication diagnostics; run on idle server GPUs only.

No driver, topology, training configuration, or persistent NCCL setting is changed.
Each case has its own process group and timeout. Raw logs stay under artifacts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("smoke", "matrix", "extended", "p2p", "dma", "transport")
    )
    parser.add_argument("--worker", choices=("collective", "dma", "peer"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes-mib", default="64,256")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gradient", action="store_true")
    parser.add_argument("--affinity", choices=("inherit", "local", "remote"), default="inherit")
    return parser.parse_args()


def affinity(mode):
    allowed = sorted(os.sched_getaffinity(0))
    if mode == "inherit":
        return allowed
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    physical = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank])
    node = (0 if physical < 4 else 1) ^ (mode == "remote")
    text = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    cpus = set()
    for item in text.split(","):
        limits = [int(value) for value in item.split("-")]
        cpus.update(range(limits[0], limits[-1] + 1))
    chosen = sorted(cpus.intersection(allowed))
    if not chosen:
        raise RuntimeError("No allowed CPUs on requested NUMA node")
    os.sched_setaffinity(0, chosen)
    return chosen


def write_result(path, result):
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)


def worker(args):
    cpus = affinity(args.affinity)
    import torch

    torch.set_num_threads(1)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", rank if args.worker == "collective" else 0)
    torch.cuda.set_device(device)
    metadata = dict(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=torch.cuda.nccl.version(),
        visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        affinity=args.affinity,
        cpu_affinity=cpus,
        warmup=args.warmup,
        steps=args.steps,
        environment={key: value for key, value in os.environ.items() if key.startswith("NCCL_")},
    )
    rows = []
    if args.worker == "collective":
        import torch.distributed as dist

        from hansgpt_research.distributed_sync import sync_gradients

        dist.init_process_group("nccl", timeout=timedelta(seconds=35), device_id=device)
        world = dist.get_world_size()
        metadata["world_size"] = world

        def measure(name, tensor, operation, expected, payload_bytes):
            wall_times, event_times = [], []
            for iteration in range(args.warmup + args.steps):
                tensor.fill_(rank + 1)
                dist.barrier()
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                before = time.perf_counter()
                operation()
                end.record()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                # Full check outside timing, including every synthetic gradient element.
                if not bool((tensor == expected).all()):
                    raise RuntimeError(f"Incorrect result: {name}")
                if iteration >= args.warmup:
                    wall_times.append(elapsed)
                    event_times.append(start.elapsed_time(end) / 1000)
            samples = [None] * world
            dist.all_gather_object(samples, dict(wall=wall_times, event=event_times, cpus=cpus))
            maximum = [max(sample["wall"][i] for sample in samples) for i in range(args.steps)]
            median = statistics.median(maximum)
            rows.append(
                dict(
                    name=name,
                    bytes=payload_bytes,
                    median_seconds=median,
                    min_seconds=min(maximum),
                    max_seconds=max(maximum),
                    algorithm_GBps=payload_bytes / median / 1e9,
                    ring_equivalent_bus_GBps=payload_bytes / median / 1e9 * 2 * (world - 1) / world,
                    per_rank_samples=samples,
                )
            )
            if rank == 0:
                print(
                    json.dumps(
                        {key: value for key, value in rows[-1].items() if key != "per_rank_samples"}
                    ),
                    flush=True,
                )

        expected = world * (world + 1) / 2
        for mib in map(int, args.sizes_mib.split(",")):
            tensor = torch.empty(mib * 1024**2 // 4, dtype=torch.float32, device=device)
            measure(
                f"allreduce_{mib}mib",
                tensor,
                lambda tensor=tensor: dist.all_reduce(tensor),
                expected,
                tensor.numel() * 4,
            )
            del tensor
        if args.gradient:
            tensor = torch.empty(1_515_243_008, dtype=torch.float32, device=device)
            scratch = torch.empty(256 * 1024**2 // 4, device=device)
            parameter = SimpleNamespace(grad=tensor)

            def direct_buckets():
                for piece in tensor.split(scratch.numel()):
                    dist.all_reduce(piece)

            def copies():
                for piece in tensor.split(scratch.numel()):
                    scratch[: piece.numel()].copy_(piece)
                    piece.copy_(scratch[: piece.numel()])

            measure(
                "gradient_contiguous",
                tensor,
                lambda: dist.all_reduce(tensor),
                expected,
                tensor.numel() * 4,
            )
            measure(
                "gradient_direct_buckets256", tensor, direct_buckets, expected, tensor.numel() * 4
            )
            measure(
                "gradient_project_sync256",
                tensor,
                lambda: sync_gradients([parameter], scratch),
                expected,
                tensor.numel() * 4,
            )
            measure("gradient_copy_only", tensor, copies, rank + 1, tensor.numel() * 4)
        if rank == 0:
            write_result(args.output, dict(metadata=metadata, rows=rows))
        dist.destroy_process_group()
        return

    count = 64 * 1024**2 // 4
    gpu = torch.full((count,), 7.0, device=device)
    if args.worker == "dma":
        host = torch.full((count,), 7.0, pin_memory=True)
        operations = [
            ("H2D", lambda: gpu.copy_(host, non_blocking=True)),
            ("D2H", lambda: host.copy_(gpu, non_blocking=True)),
        ]
    else:
        metadata["can_access_peer"] = torch.cuda.can_device_access_peer(0, 1)
        other = torch.zeros_like(gpu, device="cuda:1")
        operations = [
            ("copy_0_to_1", lambda: other.copy_(gpu, non_blocking=True)),
            ("copy_1_to_0", lambda: gpu.copy_(other, non_blocking=True)),
        ]
    for name, operation in operations:
        timings = []
        for iteration in range(args.warmup + args.steps):
            torch.cuda.synchronize(0)
            if args.worker == "peer":
                torch.cuda.synchronize(1)
            before = time.perf_counter()
            operation()
            torch.cuda.synchronize(0)
            if args.worker == "peer":
                torch.cuda.synchronize(1)
            elapsed = time.perf_counter() - before
            if iteration >= args.warmup:
                timings.append(elapsed)
        if not bool((gpu == 7).all()):
            raise RuntimeError("Copy failed")
        if args.worker == "dma" and not bool((host == 7).all()):
            raise RuntimeError("Host copy failed")
        if args.worker == "peer" and not bool((other == 7).all()):
            raise RuntimeError("Peer copy failed")
        median = statistics.median(timings)
        rows.append(
            dict(
                name=name,
                bytes=count * 4,
                median_seconds=median,
                GBps=count * 4 / median / 1e9,
                samples=timings,
            )
        )
    write_result(args.output, dict(metadata=metadata, rows=rows))
    print(json.dumps(rows), flush=True)


def suite(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError("GPU processes exist; inspect before using this idle-GPU suite")
    args.output.mkdir(parents=True)
    cases = []

    def add(gpus, label, *, mode="collective", env=None, extra=(), timeout=150):
        cases.append((gpus, label, mode, env or {}, list(extra), timeout))

    if args.suite == "smoke":
        add("0,1", "smoke_pair", extra=("--sizes-mib", "1,64", "--steps", "3"))
    elif args.suite == "matrix":
        for left, right in itertools.combinations(range(8), 2):
            add(f"{left},{right}", f"pair_{left}_{right}", extra=("--sizes-mib", "64"))
        for gpus in ("0,1,2,3", "4,5,6,7", "0,1,4,5", "0,1,2,3,4,5,6,7"):
            add(gpus, "group_" + gpus.replace(",", "_"))
    elif args.suite == "extended":
        for gpus in ("0,1", "0,2", "0,4", "0,1,2,3", "4,5,6,7", "0,2,4,6", "0,1,2,3,4,5,6,7"):
            add(gpus, "gradient_" + gpus.replace(",", "_"), extra=("--gradient",))
        for setting in ("local", "remote"):
            for gpus in ("0,1", "0,4", "4,5,6,7", "0,1,2,3,4,5,6,7"):
                add(gpus, setting + "_" + gpus.replace(",", "_"), extra=("--affinity", setting))
        for gpus in ("0,1", "0,4", "4,5,6,7"):
            add(gpus, "socket_" + gpus.replace(",", "_"), env={"NCCL_SHM_DISABLE": "1"})
    elif args.suite == "p2p":
        for gpus in ("0,1", "0,2", "0,4"):
            add(gpus, "copy_" + gpus.replace(",", "_"), mode="peer", timeout=45)
            add(gpus, "p2p_" + gpus.replace(",", "_"), env={"NCCL_P2P_DISABLE": "0"}, timeout=60)
        add("0,1", "cumem_host_only", env={"NCCL_CUMEM_HOST_ENABLE": "1"}, timeout=60)
    elif args.suite == "transport":
        for gpus in ("0,1", "0,4", "4,5,6,7", "0,1,2,3,4,5,6,7"):
            for mode in ("1", "2", "3"):
                add(
                    gpus,
                    "memcpy" + mode + "_" + gpus.replace(",", "_"),
                    env={"NCCL_SHM_USE_CUDA_MEMCPY": "1", "NCCL_SHM_MEMCPY_MODE": mode},
                )
            for channels in ("4", "8"):
                add(
                    gpus,
                    "channels" + channels + "_" + gpus.replace(",", "_"),
                    env={"NCCL_MIN_NCHANNELS": channels, "NCCL_MAX_NCHANNELS": channels},
                )
    elif args.suite == "dma":
        for gpu in range(8):
            for setting in ("local", "remote"):
                add(
                    str(gpu),
                    f"dma_{gpu}_{setting}",
                    mode="dma",
                    extra=("--affinity", setting),
                    timeout=60,
                )
    else:
        raise ValueError("Select --suite or --worker")
    results = []
    for gpus, label, mode, overrides, extra, deadline in cases:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("NCCL_"):
                del env[key]
        env.update(
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES=gpus,
            NCCL_P2P_DISABLE="1",
            NCCL_CUMEM_HOST_ENABLE="0",
            NCCL_DEBUG="INFO",
            NCCL_DEBUG_SUBSYS="INIT,GRAPH,P2P,SHM,NET,TUNING",
            OMP_NUM_THREADS="1",
        )
        env.update(overrides)
        command = ["timeout", "--signal=TERM", "--kill-after=10s", f"{deadline}s", sys.executable]
        if mode == "collective":
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(len(gpus.split(","))),
            ]
        command += [
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            "--output",
            str((args.output / f"{label}.json").resolve()),
            *extra,
        ]
        print(f"START {label}", flush=True)
        before = time.perf_counter()
        with (args.output / f"{label}.log").open("x") as stream:
            completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        result = dict(
            case=label,
            devices=gpus,
            returncode=completed.returncode,
            elapsed=time.perf_counter() - before,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
        # Only terminate suite-owned children via timeout; never terminate training jobs.
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if active:
            raise RuntimeError("GPU process remains after case; stop and inspect")
    write_result(args.output / "index.json", results)


if __name__ == "__main__":
    args = arguments()
    if args.worker:
        worker(args)
    else:
        suite(args)
