"""NPU-only bitwise JIT dispatch and specialization-cache validation.

Run: python -m scripts.jit.validate_dispatch --output /tmp/run1 --cache-root /tmp/cache
Pass --cases cases.json to exercise several configurations in the same ACL context.
The first case is revisited after the matrix to catch stale-handle reuse.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

from tests.elastic import test_dispatch as oracle


def summarize(samples):
    return {
        field: statistics.median(sample[field] for sample in samples)
        for field in ("wall_us", "kernel_us", "device_total_us")
    }


def check_all_ranks(dist, world, validate):
    error = None
    try:
        validate()
    except Exception as exc:
        error = repr(exc)
    errors = [None] * world
    dist.all_gather_object(errors, error)
    if any(errors):
        raise AssertionError(f"Validation failed: {errors}")


def specialization_key(world, case):
    # Tokens, routes, expert count, rank and addresses deliberately are not here.
    return (world, case["hidden"], case["topk"], case["dtype"] == "fp8", case["weights"], case["profile"])


def select_jit_backend(group):
    """Keep this Mesh+Clos kernel qualification behind upstream topology checks."""
    from deep_ep._machine import select

    selection = select(group=group)
    if selection["machine_type"] != "mesh_clos":
        raise NotImplementedError(
            f"Dispatch JIT validation requires mesh_clos; selected {selection['machine_type']} uses Direct kernels"
        )
    return selection


def rank_dedup_enabled(world, mode):
    if mode not in ("auto", "rank", "server"):
        raise ValueError("dedup_mode must be auto, rank or server")
    enabled = world in (8, 16) if mode == "auto" else mode == "rank"
    if enabled and world not in (8, 16):
        raise ValueError("Rank-deduplicated Dispatch supports EP8/16")
    return enabled


def device_timing(diagnostics):
    names, values = diagnostics["device_phase_names"], diagnostics["device_ms"]
    if len(names) != 4 or len(values) != 4 or len(set(names)) != 4:
        raise ValueError("Expected four named dispatch timing phases")
    phases = dict(zip(names, (float(value) * 1000 for value in values)))
    if "kernel" not in phases or "device_total" not in phases:
        raise ValueError("Dispatch timing must identify kernel and device_total phases")
    return dict(kernel_us=phases["kernel"], device_total_us=phases["device_total"], device_phase_us=phases)


def validate_cache_record(history, spec, record, previous_count=None):
    count = record["context_kernel_count"]
    if type(count) is not int or count < 1:
        raise AssertionError("Invalid context kernel count")
    if spec in history:
        if history[spec] != record["key"]:
            raise AssertionError("Same spec selected a different artifact")
        if record["handle_reused"] is not True:
            raise AssertionError("Repeated spec did not reuse its loaded handle")
        if previous_count is not None and count != previous_count:
            raise AssertionError("Repeated spec changed the context kernel count")
    else:
        if any(key == record["key"] for key in history.values()):
            raise AssertionError("Distinct specialization specs shared one artifact")
        # A context may have unrelated preloaded handles, so this is a lower bound.
        if count < len(history) + 1:
            raise AssertionError("Distinct specs did not create distinct context handles")
    history[spec] = record["key"]
    return count


def load_cases(args):
    defaults = {
        name: getattr(args, name)
        for name in (
            "tokens",
            "hidden",
            "topk",
            "experts",
            "dtype",
            "weights",
            "profile",
            "route",
            "seed",
            "routing",
            "dedup_mode",
            "benchmark_barriers",
        )
    }
    rows = [{}] if args.cases is None else json.loads(args.cases.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("--cases must contain a nonempty JSON list")
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - set(defaults):
            raise ValueError(f"Unknown case fields: {row!r}")
        case = defaults | row
        if (
            type(case["tokens"]) is not int
            or not 1 <= case["tokens"] <= 16384
            or type(case["hidden"]) is not int
            or not 0 < case["hidden"] <= 7168
            or type(case["topk"]) is not int
            or not 1 <= case["topk"] <= 16
            or type(case["experts"]) is not int
            or not args.ep_size <= case["experts"] <= 1024
            or case["experts"] % args.ep_size
            or case["dtype"] not in ("bf16", "fp8")
            or type(case["weights"]) is not bool
            or type(case["profile"]) is not bool
            or type(case["benchmark_barriers"]) is not bool
            or case["route"] not in oracle._ROUTES
            or case["routing"] not in ("fresh", "cached", "alternating")
        ):
            raise ValueError(f"Invalid HT case: {case}")
        alignment = 128 if case["dtype"] == "fp8" else 16
        if case["hidden"] % alignment:
            raise ValueError(f"hidden must be aligned to {alignment}: {case}")
        rank_dedup_enabled(args.ep_size, case["dedup_mode"])
        result.append(case)
    # A -> B -> ... -> A checks that the active handle follows its complete spec.
    return result + [dict(result[0])] if len(result) > 1 else result


def worker(local_rank, args):
    import torch.distributed as dist

    try:
        _worker(local_rank, args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker(local_rank, args):
    rank, world = oracle._worker_identity(local_rank, args)
    cases = load_cases(args)
    deep_ep = oracle._import_installed_deep_ep()
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep._native import require_native
    from deep_ep._jit import initialize
    from deep_ep.buffers._elastic_layout import dispatch_layout

    torch.npu.set_device(local_rank)
    torch.npu.init()
    os.environ["EP_JIT_CACHE_DIR"] = str(args.cache_root.resolve() / f"rank-{rank}")
    master = os.environ.get("MASTER_ADDR", args.master_addr)
    port = int(os.environ.get("MASTER_PORT", args.master_port))
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{master}:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=args.timeout),
    )
    selection = select_jit_backend(dist.group.WORLD)
    native = require_native("dispatch JIT validation")
    if not hasattr(native, "prepare_dispatch_jit"):
        raise RuntimeError("Reinstall the current wheel with template-specialized dispatch support")
    # Guard against accidentally launching differing matrices across machines.
    peer_cases = [None] * world
    dist.all_gather_object(peer_cases, cases)
    if any(value != cases for value in peer_cases):
        raise ValueError("All ranks must use the same validation cases")
    check_all_ranks(dist, world, lambda: initialize(native, Path(deep_ep.__file__).resolve().parent))
    output_dir = args.output / f"rank-{rank}"
    output_dir.mkdir(parents=True, exist_ok=False)
    key_history, all_cases, cache_count = {}, [], None
    for index, case in enumerate(cases):
        spec = specialization_key(world, case)
        cache = {}
        check_all_ranks(dist, world, lambda: cache.update(native.dispatch_jit_metrics(*spec)))

        def validate_key():
            nonlocal cache_count
            cache_count = validate_cache_record(key_history, spec, cache, cache_count)

        check_all_ranks(dist, world, validate_key)
        result = _run_case(torch, dist, deep_ep, native, dispatch_layout, rank, world, args, case, index, spec, cache)
        all_cases.append(result)
        (output_dir / f"case-{index}.json").write_text(json.dumps(result, indent=2) + "\n")
    result = dict(
        rank=rank,
        passed=True,
        machine_selection=selection,
        dispatch_backend="mesh_clos_ht_jit",
        torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__),
        cases=all_cases,
    )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    all_results = [None] * world
    dist.all_gather_object(all_results, result)
    if rank == 0:
        aggregate = []
        for index, case in enumerate(cases):
            times = {
                field: {
                    "max_rank_median": max(item["cases"][index]["median"][field] for item in all_results),
                    "median_rank_median": statistics.median(
                        item["cases"][index]["median"][field] for item in all_results
                    ),
                }
                for field in ("wall_us", "kernel_us", "device_total_us")
            }
            aggregate.append(dict(case=case, aggregate=times))
        (args.output / "summary.json").write_text(
            json.dumps(dict(passed=True, ranks=world, unit="us", cases=aggregate), indent=2) + "\n"
        )


def _run_case(torch, dist, deep_ep, native, dispatch_layout, rank, world, args, case, index, spec, cache):
    tokens, hidden, topk, experts = (case[name] for name in ("tokens", "hidden", "topk", "experts"))
    fp8 = case["dtype"] == "fp8"
    routes = oracle.make_routes(world, tokens, topk, experts, 0, case["route"], seed=case["seed"])
    rank_dedup = rank_dedup_enabled(world, case["dedup_mode"])
    notify_golden = oracle.build_notify_golden(routes, experts, 256, rank_dedup=rank_dedup)
    entries = oracle.expert_rows(routes, experts, rank)
    rows = oracle.receive_capacity(routes, experts)
    capacity = (
        2
        * dispatch_layout(world, tokens, topk, experts, rows, fp8, weights=case["weights"], hidden=hidden)["num_bytes"]
    )
    buffer = deep_ep.ElasticBuffer(
        group=dist.group.WORLD,
        num_bytes=capacity,
        num_max_tokens_per_rank=tokens,
        hidden=hidden,
        num_topk=topk,
        use_fp8_dispatch=fp8,
        explicitly_destroy=True,
        machine_type="mesh_clos",
    )
    buffer._notify_rank_dedup = rank_dedup
    samples, handle, retained, checks = [], None, [], 0
    try:
        for iteration in range(args.warmup + args.iterations):
            generation = iteration + index * (args.warmup + args.iterations)
            local, scales = oracle.payload(rank, tokens, generation, fp8, hidden)
            x = (local.npu(), scales.npu()) if fp8 else local.npu()
            w = oracle.weights_for(rank, tokens, topk, generation).npu() if case["weights"] else None
            expected, expected_scales, expected_weights = oracle.expected_payload(
                routes, experts, rank, generation, fp8, entries=entries, width=hidden
            )
            if case["routing"] == "fresh" or (case["routing"] == "alternating" and iteration % 2 == 0):
                handle = None
            diagnostics = {"benchmark_barriers": case["benchmark_barriers"]}
            if case["profile"]:
                diagnostics["profile"] = torch.zeros((64, 32), dtype=torch.int64, device="npu")
            buffer._dispatch_diagnostics = diagnostics
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            received = buffer.dispatch(
                x,
                topk_weights=w,
                handle=handle,
                topk_idx=torch.tensor(routes[rank], dtype=torch.int64, device="npu") if handle is None else None,
                num_experts=experts if handle is None else None,
                do_expand=True,
                async_with_compute_stream=True,
            )  # Explicit HT, even with small token counts.
            received[-1].current_stream_wait()
            torch.npu.synchronize()
            wall_us = (time.perf_counter_ns() - start) / 1000
            handle = received[3]
            phases = device_timing(buffer._dispatch_diagnostics)
            data = received[0][0] if fp8 else received[0]

            def validate():
                if bool(handle._rank_return_plan) != rank_dedup:
                    raise AssertionError("Dispatch selected the wrong rank/server dedup plan")
                oracle.check_handle(handle, notify_golden, rank, routes, experts)
                oracle.assert_bytes(data, expected, "JIT hidden")
                if fp8:
                    oracle.assert_bytes(received[0][1], expected_scales, "JIT scales")
                if case["weights"]:
                    oracle.assert_bytes(received[2], expected_weights, "JIT weights")
                elif received[2] is not None:
                    raise AssertionError("Unweighted dispatch returned weights")

            check_all_ranks(dist, world, validate)
            if iteration >= args.warmup:
                samples.append(
                    dict(
                        iteration=iteration,
                        wall_us=wall_us,
                        **phases,
                    )
                )
            checks += 1
            retained = [(data, expected)]
            if fp8:
                retained.append((received[0][1], expected_scales))
            if case["weights"]:
                retained.append((received[2], expected_weights))
    finally:
        buffer.destroy()
    check_all_ranks(
        dist,
        world,
        lambda: [oracle.assert_bytes(value, expected, "retained after destroy") for value, expected in retained],
    )
    after = {}
    check_all_ranks(dist, world, lambda: after.update(native.dispatch_jit_metrics(*spec)))
    check_all_ranks(dist, world, lambda: _same_key(cache, after))
    return dict(
        case=case,
        spec=list(spec),
        passed=True,
        bitwise_checks=checks,
        cache_before=cache,
        cache_after=after,
        samples=samples,
        median=summarize(samples),
    )


def _same_key(before, after):
    if before["key"] != after["key"]:
        raise AssertionError("Runtime data changed the specialization key")
    if after["handle_reused"] is not True:
        raise AssertionError("Runtime data caused the loaded handle to be replaced")
    if before["context_kernel_count"] != after["context_kernel_count"]:
        raise AssertionError("Runtime data increased the context kernel count")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    oracle._add_launch_args(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, help="JSON list; override any per-case option")
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--route", choices=oracle._ROUTES, default="random")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--routing", choices=("fresh", "cached", "alternating"), default="alternating")
    parser.add_argument("--weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dedup-mode", choices=("auto", "rank", "server"), default="auto")
    parser.add_argument(
        "--benchmark-barriers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicit benchmark alignment; production completion is tested by default",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=900, help="Gloo timeout, including first JIT compile")
    args = parser.parse_args()
    if not __debug__ or args.iterations < 1 or args.warmup < 1 or args.timeout < 1:
        parser.error("Enable assertions; use warmup>=1, iterations>=1 and timeout>=1")
    oracle._launch(args, worker)


if __name__ == "__main__":
    main()
