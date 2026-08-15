#!/usr/bin/env python3
"""Capture default and highest-precision Frontier LSTM dots without training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping


PROBE_ROOT = Path(__file__).resolve().parent
FORWARD_ROOT = PROBE_ROOT.parent
REPO_ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = PROBE_ROOT / "PRECISION_PROTOCOL.json"
PAYLOAD = FORWARD_ROOT / "FORWARD_PAYLOAD.json"
BASE_CAPTURE = FORWARD_ROOT / "capture_forward_only.py"
PROTOCOL_SHA256 = "0abdb46a7b56986756a31f3d4cc1793af20fc6ca53d2b397720386aab7f5b820"
PAYLOAD_SHA256 = "845a34ae40fb762e72b4c6ec569ef16ab6531b241eeaf6cecbc0523059f3bc78"
BASE_CAPTURE_SHA256 = "437e65d445b42d78430c7f84f2e2c4dfe8e2d31ad0973acf031f8831ae40d5a4"
RUN_ROOT = Path("/data/robotixx/ued_bench/runs/blackwell_precision_0abdb46a")


class PrecisionCaptureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionCaptureError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_environment(backend: str) -> None:
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    expected = {
        "JAX_PLATFORMS": "cpu" if backend == "cpu" else "cuda",
        "JAX_PLATFORM_NAME": backend,
        "JAX_THREEFRY_PARTITIONABLE": "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "TMPDIR": "/data/robotixx/ued_bench/tmp",
        "PIP_CACHE_DIR": "/data/robotixx/ued_bench/cache/pip",
    }
    for name, value in expected.items():
        require(os.environ.get(name, "").lower() == value, f"{name} drift")


def load_frozen_base() -> Any:
    require(sha256(BASE_CAPTURE) == BASE_CAPTURE_SHA256, "base forward capture drift")
    specification = importlib.util.spec_from_file_location(
        "_frozen_forward_capture_base", BASE_CAPTURE
    )
    require(specification is not None and specification.loader is not None, "base import failed")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def numeric_match(np: Any, left: Any, right: Any, rtol: float, atol: float) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.isposinf(left), np.isposinf(right))
        and np.array_equal(np.isneginf(left), np.isneginf(right))
        and bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
    )


def capture(args: argparse.Namespace) -> dict[str, Any]:
    validate_environment(args.backend)
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, "precision protocol drift")
    require(sha256(PAYLOAD) == PAYLOAD_SHA256, "payload drift")
    protocol = json.loads(PROTOCOL.read_text())
    payload = json.loads(PAYLOAD.read_text())
    require(protocol["payload"]["sha256"] == PAYLOAD_SHA256, "protocol/payload drift")
    base = load_frozen_base()
    source = args.source.resolve()
    source_record = base.validate_source(source)
    source_capture_path, source_capture = base.validate_payload_files(payload)
    output = args.output.resolve()
    require(output.is_relative_to(RUN_ROOT.resolve()), "output outside precision run root")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "unsafe output parent")
    gpu_pid_before = base.gpu_pid_present() if args.backend == "gpu" else None
    if args.backend == "gpu":
        require(gpu_pid_before is True, "preserved GPU process missing before capture")

    sys.path[:0] = [str(source / "src"), str(REPO_ROOT)]
    import jax
    import jax.numpy as jnp
    import numpy as np
    from minimax.models.maze.gridworld_models import GridWorldACStudentModel

    require(jax.default_backend() == args.backend, "requested backend is not active")
    require(jax.config.jax_threefry_partitionable is False, "Threefry mode drift")
    devices = jax.devices(args.backend)
    require(len(devices) == 1 and devices[0].platform == args.backend, "device drift")
    minimax_path = Path(sys.modules["minimax"].__file__).resolve()
    require(minimax_path.is_relative_to(source), "minimax imported outside isolated source")

    selectors = payload["record_selectors"]
    params_np = base.nested_parameters(
        np,
        source_capture_path,
        source_capture,
        selectors["population_parameters"],
    )

    def array_for(name: str) -> Any:
        return base.load_array(
            np,
            source_capture_path,
            base.selected_record(source_capture, selectors[name]),
        )

    runner_rng_np = array_for("runner_rng")
    loss_rng_np = array_for("loss_rng")
    image_np = array_for("image")
    agent_dir_np = array_for("agent_dir")
    dones_np = array_for("dones")
    carry_c_np = array_for("carry_c")[int(selectors["carry_c"]["time_index"])]
    carry_h_np = array_for("carry_h")[int(selectors["carry_h"]["time_index"])]
    reset_np = np.zeros_like(dones_np, dtype=np.bool_)
    reset_np[1:] = dones_np[:-1].astype(np.bool_)

    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    image = jnp.asarray(image_np)
    agent_dir = jnp.asarray(agent_dir_np)
    reset = jnp.asarray(reset_np)
    initial_carry = (jnp.asarray(carry_c_np), jnp.asarray(carry_h_np))
    parameter_digest_before = base.tree_digest(jax, np, params)

    model_spec = payload["model"]
    model = GridWorldACStudentModel(
        output_dim=int(model_spec["output_dim"]),
        n_hidden_layers=int(model_spec["n_hidden_layers"]),
        hidden_dim=int(model_spec["hidden_dim"]),
        n_conv_filters=int(model_spec["n_conv_filters"]),
        conv_kernel_size=int(model_spec["conv_kernel_size"]),
        n_scalar_embeddings=int(model_spec["n_scalar_embeddings"]),
        scalar_embed_dim=int(model_spec["scalar_embed_dim"]),
        recurrent_arch=model_spec["recurrent_arch"],
        recurrent_hidden_dim=int(model_spec["recurrent_hidden_dim"]),
        base_activation=model_spec["base_activation"],
        head_activation=model_spec["head_activation"],
    )
    canonical_value, canonical_logits, canonical_carry = jax.jit(
        lambda model_params, obs_image, obs_dir, carry, done: model.apply(
            model_params,
            {"image": obs_image, "agent_dir": obs_dir},
            carry,
            done,
        )
    )(params, image, agent_dir, initial_carry, reset)

    def decomposed_forward(
        model_params: Mapping[str, Any],
        obs_image: Any,
        obs_dir: Any,
        carry: Any,
        done: Any,
        precision: Any,
    ) -> tuple[Any, Any, Any]:
        leaves = model_params["params"]
        time_steps, batch_size = obs_image.shape[:2]
        flat_image = obs_image.reshape((-1,) + obs_image.shape[2:])
        conv_preactivation = jax.lax.conv_general_dilated(
            flat_image,
            leaves["cnn"]["kernel"],
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        conv_preactivation = conv_preactivation + leaves["cnn"]["bias"]
        conv_preactivation = conv_preactivation.reshape(
            time_steps, batch_size, *conv_preactivation.shape[1:]
        )
        visual_flat = jax.nn.relu(conv_preactivation).reshape(time_steps, batch_size, -1)
        scalar_embedding = leaves["fc_scalar"]["embedding"][obs_dir]
        features = jnp.concatenate((visual_flat, scalar_embedding), axis=-1)

        cell = leaves["rnn"]["OptimizedLSTMCell_0"]
        gate_names = ("i", "f", "g", "o")
        input_kernel = jnp.concatenate(
            tuple(cell[f"i{name}"]["kernel"] for name in gate_names), axis=-1
        )
        hidden_kernel = jnp.concatenate(
            tuple(cell[f"h{name}"]["kernel"] for name in gate_names), axis=-1
        )
        hidden_bias = jnp.concatenate(
            tuple(cell[f"h{name}"]["bias"] for name in gate_names), axis=-1
        )

        def recurrent_step(previous: Any, step: Any) -> tuple[Any, Any]:
            previous_c, previous_h = previous
            step_features, step_reset = step
            selected_c = jnp.where(step_reset[:, None], jnp.zeros_like(previous_c), previous_c)
            selected_h = jnp.where(step_reset[:, None], jnp.zeros_like(previous_h), previous_h)
            input_affine = jnp.dot(step_features, input_kernel, precision=precision)
            hidden_affine = (
                jnp.dot(selected_h, hidden_kernel, precision=precision) + hidden_bias
            )
            input_parts = jnp.split(input_affine, 4, axis=-1)
            hidden_parts = jnp.split(hidden_affine, 4, axis=-1)
            preactivation = tuple(
                input_part + hidden_part
                for input_part, hidden_part in zip(input_parts, hidden_parts)
            )
            gate_i = jax.nn.sigmoid(preactivation[0])
            gate_f = jax.nn.sigmoid(preactivation[1])
            gate_g = jnp.tanh(preactivation[2])
            gate_o = jax.nn.sigmoid(preactivation[3])
            new_c = gate_f * selected_c + gate_i * gate_g
            new_h = gate_o * jnp.tanh(new_c)
            intermediates = {
                "input_affine": dict(zip(gate_names, input_parts)),
                "hidden_affine": dict(zip(gate_names, hidden_parts)),
                "preactivation": dict(zip(gate_names, preactivation)),
                "activation": {
                    "i": gate_i,
                    "f": gate_f,
                    "g": gate_g,
                    "o": gate_o,
                },
                "cell_state": new_c,
                "hidden_state": new_h,
            }
            return (new_c, new_h), intermediates

        final_carry, recurrent = jax.lax.scan(
            recurrent_step,
            carry,
            (features, done),
        )
        shared = {
            "conv_preactivation": conv_preactivation,
            "features": features,
        }
        return final_carry, shared, recurrent

    default_forward = jax.jit(
        lambda model_params, obs_image, obs_dir, carry, done: decomposed_forward(
            model_params, obs_image, obs_dir, carry, done, None
        )
    )
    highest_forward = jax.jit(
        lambda model_params, obs_image, obs_dir, carry, done: decomposed_forward(
            model_params,
            obs_image,
            obs_dir,
            carry,
            done,
            jax.lax.Precision.HIGHEST,
        )
    )
    default_carry, default_shared, default_recurrent = default_forward(
        params, image, agent_dir, initial_carry, reset
    )
    highest_carry, highest_shared, highest_recurrent = highest_forward(
        params, image, agent_dir, initial_carry, reset
    )
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        (
            canonical_value,
            canonical_logits,
            canonical_carry,
            default_carry,
            default_shared,
            default_recurrent,
            highest_carry,
            highest_shared,
            highest_recurrent,
        ),
    )
    cpu_tolerance = protocol["comparison"]["cpu"]
    cpu_rtol = float(cpu_tolerance["rtol"])
    cpu_atol = float(cpu_tolerance["atol"])
    require(
        numeric_match(np, canonical_carry[0], default_carry[0], cpu_rtol, cpu_atol)
        and numeric_match(np, canonical_carry[1], default_carry[1], cpu_rtol, cpu_atol),
        "default decomposition does not match canonical carry",
    )
    require(
        numeric_match(
            np,
            default_shared["conv_preactivation"],
            highest_shared["conv_preactivation"],
            0.0,
            0.0,
        )
        and numeric_match(
            np,
            default_shared["features"],
            highest_shared["features"],
            0.0,
            0.0,
        ),
        "precision intervention changed shared features",
    )
    if args.backend == "cpu":
        for left, right in zip(
            jax.tree_util.tree_leaves(default_recurrent),
            jax.tree_util.tree_leaves(highest_recurrent),
        ):
            require(
                numeric_match(np, left, right, cpu_rtol, cpu_atol),
                "CPU default/highest recurrent output drift",
            )
        require(
            numeric_match(np, default_carry[0], highest_carry[0], cpu_rtol, cpu_atol)
            and numeric_match(np, default_carry[1], highest_carry[1], cpu_rtol, cpu_atol),
            "CPU default/highest final carry drift",
        )

    output.mkdir(mode=0o700)
    bundle = base.ArrayBundle(np, jax, output)
    bundle.add(
        "input_payload",
        "exact_inputs",
        {
            "params": params,
            "runner_rng": jnp.asarray(runner_rng_np),
            "loss_rng": jnp.asarray(loss_rng_np),
            "image": image,
            "agent_dir": agent_dir,
            "dones": jnp.asarray(dones_np),
            "reset": reset,
            "initial_carry": initial_carry,
        },
    )
    bundle.add(
        "shared_convolution_preactivation",
        "conv_valid_nhwc_hwio",
        default_shared["conv_preactivation"],
    )
    bundle.add("shared_features", "visual_plus_scalar", default_shared["features"])
    for mode, recurrent, final_carry in (
        ("default", default_recurrent, default_carry),
        ("highest", highest_recurrent, highest_carry),
    ):
        bundle.add(
            f"{mode}_lstm_input_affine",
            "concatenated_input_dot",
            recurrent["input_affine"],
        )
        bundle.add(
            f"{mode}_lstm_hidden_affine",
            "concatenated_hidden_dot_bias",
            recurrent["hidden_affine"],
        )
        bundle.add(
            f"{mode}_lstm_gate_preactivation",
            "hidden_plus_input",
            recurrent["preactivation"],
        )
        bundle.add(
            f"{mode}_lstm_gate_activation",
            "sigmoid_i_f_o_tanh_g",
            recurrent["activation"],
        )
        bundle.add(f"{mode}_lstm_cell_state", "new_c", recurrent["cell_state"])
        bundle.add(f"{mode}_lstm_hidden_state", "new_h", recurrent["hidden_state"])
        bundle.add(
            f"{mode}_final_carry",
            "decomposed",
            {"c": final_carry[0], "h": final_carry[1]},
        )
    bundle.add(
        "canonical_default_output",
        "gridworld_student_apply",
        {
            "carry": canonical_carry,
            "logits": canonical_logits,
            "value": canonical_value,
        },
    )

    parameter_digest_after = base.tree_digest(jax, np, params)
    require(parameter_digest_after == parameter_digest_before, "parameters changed")
    gpu_pid_after = base.gpu_pid_present() if args.backend == "gpu" else None
    if args.backend == "gpu":
        require(gpu_pid_after is True, "preserved GPU process missing after capture")
    receipt = {
        "schema_version": 1,
        "status": "captured_default_and_highest_forward_only",
        "lane": "modern",
        "backend": args.backend,
        "paper_evidence": False,
        "performance_endpoint": False,
        "training_steps": 0,
        "experiment_step_calls": 0,
        "agent_update_calls": 0,
        "gradient_calculations": 0,
        "gradient_transformation_proposals": 0,
        "optimizer_applications": 0,
        "parameter_mutations": 0,
        "forward_calls": {
            "canonical_default": 1,
            "decomposed_default": 1,
            "decomposed_highest": 1,
        },
        "rng_consumed": False,
        "runtime": {
            "python": platform.python_version(),
            "jax": importlib.metadata.version("jax"),
            "jaxlib": importlib.metadata.version("jaxlib"),
            "flax": importlib.metadata.version("flax"),
            "device_platform": devices[0].platform,
            "device_kind": devices[0].device_kind,
            "jax_platforms_env": os.environ["JAX_PLATFORMS"],
            "jax_platform_name_env": os.environ["JAX_PLATFORM_NAME"],
            "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
            "xla_python_client_preallocate_env": os.environ[
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ],
            "minimax_module": str(minimax_path),
        },
        "precision": {
            "default": "None",
            "highest": "jax.lax.Precision.HIGHEST",
            "shared_features_exact_within_backend": True,
            "default_decomposition_matches_canonical": True,
            "cpu_default_highest_match": args.backend == "cpu",
        },
        "source": source_record,
        "payload": {
            "source_capture": str(source_capture_path),
            "source_capture_sha256": base.sha256(source_capture_path),
            "parameter_sha256_before": parameter_digest_before,
            "parameter_sha256_after": parameter_digest_after,
            "unchanged": True,
        },
        "hashes": {
            "protocol_sha256": PROTOCOL_SHA256,
            "payload_sha256": PAYLOAD_SHA256,
            "base_forward_capture_sha256": BASE_CAPTURE_SHA256,
            "capture_script_sha256": sha256(Path(__file__).resolve()),
        },
        "preserved_gpu_process": {
            "pid": base.PRESERVED_GPU_PID,
            "present_before": gpu_pid_before,
            "present_after": gpu_pid_after,
        },
        "groups": bundle.groups,
        "records": bundle.records,
        "record_count": len(bundle.records),
    }
    base.atomic_json(output / "capture.json", receipt)
    print(
        "PRECISION_CAPTURE_OK "
        f"backend={args.backend} records={len(bundle.records)} training_steps=0"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()
    try:
        capture(args)
    except (
        PrecisionCaptureError,
        RuntimeError,
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"PRECISION_CAPTURE_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
