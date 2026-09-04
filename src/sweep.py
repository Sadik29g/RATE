import json
import multiprocessing as mp
import pickle
import traceback

from dataclasses import asdict, is_dataclass
from pathlib import Path

from .runner import RunConfig, run_single
from .dqn import DQNConfig, run_dqn

from pathlib import Path


def cfg_tag(cfg):
    if not is_dataclass(cfg):
        raise TypeError(
            "cfg_tag() expects a dataclass configuration."
        )

    data = asdict(cfg)

    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True
    )

    return (
        f"{type(cfg).__name__}|"
        f"{payload}"
    )


def result_tag(result):
    if not isinstance(result, dict):
        raise TypeError(
            "result_tag() expects a result dictionary."
        )

    key = result.get("key")

    if not isinstance(key, str) or not key:
        raise KeyError(
            "Result record does not contain a valid 'key'."
        )

    return key


def _run_config(cfg):
    if isinstance(cfg, RunConfig):
        return run_single(cfg)

    if isinstance(cfg, DQNConfig):
        return run_dqn(cfg)

    raise TypeError(
        f"Unsupported configuration type: "
        f"{type(cfg).__name__}"
    )


def _worker(cfg):
    key = cfg_tag(cfg)
    config_type = type(cfg).__name__
    config_snapshot = asdict(cfg)

    try:
        result = _run_config(cfg)

        if not isinstance(result, dict):
            raise TypeError(
                "Runner must return a result dictionary."
            )

        result = dict(result)

        result["key"] = key
        result["config_type"] = config_type
        result["config"] = config_snapshot

        return result

    except Exception as e:
        return {
            "key": key,
            "config_type": config_type,
            "config": config_snapshot,
            "error_type": type(e).__name__,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def _save_atomic(obj, path):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")

    with open(tmp, "wb") as f:
        pickle.dump(obj, f)

    tmp.replace(path)


def run_stage(
    configs,
    out_path,
    label,
    workers,
    save_every=10,
    retry_errors=False
):
    configs = list(configs)
    out_path = Path(out_path)
    label = str(label).strip() or "sweep"
    workers = int(workers)
    save_every = int(save_every)

    if workers <= 0:
        raise ValueError("workers must be positive.")

    if save_every <= 0:
        raise ValueError("save_every must be positive.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    if out_path.exists():
        with open(out_path, "rb") as f:
            results = pickle.load(f)

        if not isinstance(results, list):
            raise TypeError(
                "Existing results file must contain a list."
            )

    result_index = {}

    for i, result in enumerate(results):
        key = result_tag(result)

        if key in result_index:
            raise ValueError(
                "Existing results file contains duplicate "
                f"configuration key: {key}"
            )

        result_index[key] = i

    if retry_errors:
        done = {
            result_tag(result)
            for result in results
            if "error" not in result
        }
    else:
        done = set(result_index)

    pending = []
    queued = set()

    for cfg in configs:
        key = cfg_tag(cfg)

        if key in done or key in queued:
            continue

        pending.append(cfg)
        queued.add(key)

    total = len(pending)

    print(
        f"[{label}] "
        f"loaded={len(results)} "
        f"pending={total} "
        f"workers={workers}"
    )

    if total == 0:
        return results

    ctx = mp.get_context("spawn")

    try:
        with ctx.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(
                _worker,
                pending,
                chunksize=1
            )

            for i, result in enumerate(iterator, 1):
                key = result_tag(result)

                if key in result_index:
                    results[result_index[key]] = result
                else:
                    result_index[key] = len(results)
                    results.append(result)

                status = (
                    "ERROR"
                    if "error" in result
                    else "OK"
                )

                print(
                    f"[{label}] "
                    f"{i}/{total} "
                    f"{status}"
                )

                if i % save_every == 0 or i == total:
                    _save_atomic(results, out_path)

    except KeyboardInterrupt:
        _save_atomic(results, out_path)
        print(
            f"[{label}] interrupted; "
            f"checkpoint saved to {out_path}"
        )
        raise

    except Exception:
        _save_atomic(results, out_path)
        raise

    return results
