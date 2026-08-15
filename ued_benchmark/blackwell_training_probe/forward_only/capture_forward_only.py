#!/usr/bin/env python3
"""Capture a frozen Frontier CNN/LSTM forward pass without training APIs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping


PROBE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = PROBE_ROOT / "FORWARD_ONLY_PROTOCOL.json"
PAYLOAD = PROBE_ROOT / "FORWARD_PAYLOAD.json"
PROTOCOL_SHA256 = "024239a6b659097198a6d902b1bb63698849d38e340ac033fa21537b0e5888ce"
PAYLOAD_SHA256 = "845a34ae40fb762e72b4c6ec569ef16ab6531b241eeaf6cecbc0523059f3bc78"
RUN_ROOT = Path("/data/robotixx/ued_bench/runs/blackwell_forward_only_024239a6")
UPSTREAM_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
UPSTREAM_TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"
MODERN_MANIFEST_SHA256 = "ea5fb73c0072cd95829630344e559f02a83f65b0f8b479845ef4dff8921ff65c"
PRESERVED_GPU_PID = 2786996


class CaptureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def validate_source(source: Path) -> dict[str, Any]:
    require(source.is_dir() and not source.is_symlink(), "unsafe source directory")
    require(git(source, "rev-parse", "HEAD") == UPSTREAM_COMMIT, "source commit drift")
    require(git(source, "rev-parse", "HEAD^{tree}") == UPSTREAM_TREE, "source tree drift")
    modern_manifest = source / ".blackwell_training_overlay.json"
    require(
        modern_manifest.is_file() and not modern_manifest.is_symlink(),
        "missing modernization manifest",
    )
    require(sha256(modern_manifest) == MODERN_MANIFEST_SHA256, "modern source drift")
    removed_count = sum(
        path.read_text().count("jax.tree_map")
        for path in (source / "src/minimax").rglob("*.py")
    )
    require(removed_count == 0, "modern source retains removed JAX API")
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "modernization_manifest_sha256": sha256(modern_manifest),
        "removed_jax_tree_map_occurrences": removed_count,
    }


def gpu_pid_present() -> bool:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return str(PRESERVED_GPU_PID) in {
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    }


def selected_record(
    capture: Mapping[str, Any], selector: Mapping[str, Any]
) -> Mapping[str, Any]:
    index = int(selector["index"])
    require(0 <= index < len(capture["records"]), "payload record index out of range")
    record = capture["records"][index]
    require(record["index"] == index, "capture record index drift")
    for field in ("stage", "label", "path"):
        require(record[field] == selector[field], f"payload selector {field} drift")
    return record


def validate_payload_files(payload: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    capture_path = Path(payload["source_capture"]["path"]).resolve()
    require(capture_path.is_file() and not capture_path.is_symlink(), "unsafe source capture")
    require(
        sha256(capture_path) == payload["source_capture"]["sha256"],
        "source capture digest drift",
    )
    capture = json.loads(capture_path.read_text())
    require(capture["status"] == "captured_without_optimizer_application", "bad source capture")
    require(capture["lane"] == "modern" and capture["backend"] == "cpu", "bad payload lane")
    require(capture["optimizer_applications"] == 0, "payload source applied optimizer")
    require(capture["parameter_mutations"] == 0, "payload source mutated parameters")
    require(
        capture["hashes"]["protocol_sha256"]
        == payload["source_capture"]["protocol_sha256"],
        "parent component protocol drift",
    )
    require(
        capture["hashes"]["capture_script_sha256"]
        == payload["source_capture"]["capture_script_sha256"],
        "parent capture script drift",
    )
    selectors = payload["record_selectors"]
    selected_indices = list(selectors["population_parameters"]["indices"])
    selected_indices.extend(
        int(selector["index"])
        for name, selector in selectors.items()
        if name != "population_parameters"
    )
    for index in selected_indices:
        record = capture["records"][index]
        require(record["index"] == index, "selected record ordering drift")
        relative = Path(record["file"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe array path")
        array_path = capture_path.parent / relative
        require(array_path.is_file() and not array_path.is_symlink(), "missing payload array")
        require(sha256(array_path) == record["file_sha256"], "payload array digest drift")
    param_selector = selectors["population_parameters"]
    param_records = [capture["records"][index] for index in param_selector["indices"]]
    require(len(param_records) == 23, "parameter leaf count drift")
    for record in param_records:
        require(record["stage"] == param_selector["stage"], "parameter stage drift")
        require(record["label"] == param_selector["label"], "parameter label drift")
        require(record["path"].startswith(param_selector["path_prefix"]), "parameter path drift")
        require(record["shape"][0] == 1, "parameter population axis drift")
    return capture_path, capture


def load_array(np: Any, capture_path: Path, record: Mapping[str, Any]) -> Any:
    array = np.load(capture_path.parent / record["file"], allow_pickle=False)
    require(list(array.shape) == record["shape"], "loaded array shape drift")
    require(str(array.dtype) == record["dtype"], "loaded array dtype drift")
    require(hashlib.sha256(array.tobytes()).hexdigest() == record["raw_sha256"], "raw array drift")
    return np.ascontiguousarray(array)


def nested_parameters(
    np: Any,
    capture_path: Path,
    capture: Mapping[str, Any],
    selector: Mapping[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for index in selector["indices"]:
        record = capture["records"][index]
        tokens = re.findall(r"\['([^']+)'\]", record["path"])
        require(tokens[:2] == ["params", "params"], "parameter root drift")
        tokens = tokens[2:]
        require(tokens, "empty parameter path")
        target = params
        for token in tokens[:-1]:
            target = target.setdefault(token, {})
        array = load_array(np, capture_path, record)
        require(array.shape[0] == 1, "non-singleton population parameter")
        target[tokens[-1]] = array[int(selector["population_index"])]
    return {"params": params}


def tree_digest(jax: Any, np: Any, tree: Any) -> str:
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.ascontiguousarray(np.asarray(leaf))
        descriptor = json.dumps(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(array.tobytes())
    return digest.hexdigest()


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


class ArrayBundle:
    def __init__(self, np: Any, jax: Any, output: Path):
        self.np = np
        self.jax = jax
        self.directory = output / "arrays"
        self.directory.mkdir(mode=0o700)
        self.records: list[dict[str, Any]] = []
        self.groups: dict[str, list[dict[str, Any]]] = {}

    def add(self, stage: str, label: str, tree: Any) -> None:
        flattened, _ = self.jax.tree_util.tree_flatten_with_path(tree)
        structure = []
        first = len(self.records)
        for path, leaf in flattened:
            array = self.np.ascontiguousarray(self.np.asarray(leaf))
            require(array.dtype.kind != "O", f"object array forbidden: {stage}/{label}")
            index = len(self.records)
            filename = f"{index:05d}.npy"
            target = self.directory / filename
            with target.open("wb") as handle:
                self.np.save(handle, array, allow_pickle=False)
            finite = self.np.isfinite(array)
            finite_abs = self.np.abs(array[finite].astype(self.np.float64, copy=False))
            record = {
                "index": index,
                "stage": stage,
                "label": label,
                "path": self.jax.tree_util.keystr(path),
                "file": f"arrays/{filename}",
                "file_sha256": sha256(target),
                "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size": int(array.size),
                "nan_count": int(self.np.isnan(array).sum()),
                "infinite_count": int(self.np.isinf(array).sum()),
                "positive_inf_count": int(self.np.isposinf(array).sum()),
                "negative_inf_count": int(self.np.isneginf(array).sum()),
                "abs_sum": float(finite_abs.sum(dtype=self.np.float64)),
                "squared_l2": float(self.np.square(finite_abs).sum(dtype=self.np.float64)),
                "max_abs": float(finite_abs.max(initial=0.0)),
            }
            self.records.append(record)
            structure.append(
                {
                    "path": record["path"],
                    "shape": record["shape"],
                    "dtype": record["dtype"],
                    "size": record["size"],
                }
            )
        structure_sha = hashlib.sha256(
            json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.groups.setdefault(stage, []).append(
            {
                "label": label,
                "first_record": first,
                "record_count": len(flattened),
                "structure_sha256": structure_sha,
            }
        )


def capture(args: argparse.Namespace) -> dict[str, Any]:
    validate_environment(args.backend)
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, "forward protocol drift")
    require(sha256(PAYLOAD) == PAYLOAD_SHA256, "forward payload drift")
    protocol = json.loads(PROTOCOL.read_text())
    payload = json.loads(PAYLOAD.read_text())
    require(protocol["payload"]["sha256"] == PAYLOAD_SHA256, "protocol/payload drift")
    source = args.source.resolve()
    source_record = validate_source(source)
    capture_path, source_capture = validate_payload_files(payload)
    output = args.output.resolve()
    require(output.is_relative_to(RUN_ROOT.resolve()), "output outside forward run root")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "unsafe output parent")
    gpu_pid_before = gpu_pid_present() if args.backend == "gpu" else None
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
    require(len(devices) == 1, "exactly one backend device required")
    require(devices[0].platform == args.backend, "device platform drift")
    minimax_path = Path(sys.modules["minimax"].__file__).resolve()
    require(minimax_path.is_relative_to(source), "minimax imported outside isolated source")

    selectors = payload["record_selectors"]
    params_np = nested_parameters(
        np,
        capture_path,
        source_capture,
        selectors["population_parameters"],
    )

    def array_for(name: str) -> Any:
        return load_array(
            np,
            capture_path,
            selected_record(source_capture, selectors[name]),
        )

    runner_rng_np = array_for("runner_rng")
    loss_rng_np = array_for("loss_rng")
    image_np = array_for("image")
    agent_dir_np = array_for("agent_dir")
    dones_np = array_for("dones")
    carry_c_np = array_for("carry_c")[int(selectors["carry_c"]["time_index"])]
    carry_h_np = array_for("carry_h")[int(selectors["carry_h"]["time_index"])]
    expected_cpu = {
        "carry_c": array_for("expected_carry_c"),
        "carry_h": array_for("expected_carry_h"),
        "logits": array_for("expected_logits"),
        "value": array_for("expected_value"),
    }
    reset_np = np.zeros_like(dones_np, dtype=np.bool_)
    reset_np[1:] = dones_np[:-1].astype(np.bool_)

    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    image = jnp.asarray(image_np)
    agent_dir = jnp.asarray(agent_dir_np)
    reset = jnp.asarray(reset_np)
    initial_carry = (jnp.asarray(carry_c_np), jnp.asarray(carry_h_np))
    parameter_digest_before = tree_digest(jax, np, params)

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
    ) -> tuple[Any, Any]:
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
        conv_activation = jax.nn.relu(conv_preactivation)
        visual_flat = conv_activation.reshape(time_steps, batch_size, -1)
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
            input_affine = jnp.dot(step_features, input_kernel)
            hidden_affine = jnp.dot(selected_h, hidden_kernel) + hidden_bias
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
            forget_term = gate_f * selected_c
            input_term = gate_i * gate_g
            new_c = forget_term + input_term
            tanh_c = jnp.tanh(new_c)
            new_h = gate_o * tanh_c
            intermediates = {
                "selected_c": selected_c,
                "selected_h": selected_h,
                "input_affine": dict(zip(gate_names, input_parts)),
                "hidden_affine": dict(zip(gate_names, hidden_parts)),
                "preactivation": dict(zip(gate_names, preactivation)),
                "activation": {
                    "i": gate_i,
                    "f": gate_f,
                    "g": gate_g,
                    "o": gate_o,
                },
                "forget_term": forget_term,
                "input_term": input_term,
                "cell_state": new_c,
                "tanh_cell_state": tanh_c,
                "hidden_state": new_h,
            }
            return (new_c, new_h), intermediates

        final_carry, recurrent = jax.lax.scan(
            recurrent_step,
            carry,
            (features, done),
        )
        feature_intermediates = {
            "conv_preactivation": conv_preactivation,
            "conv_activation": conv_activation,
            "visual_flat": visual_flat,
            "scalar_embedding": scalar_embedding,
            "features": features,
        }
        return final_carry, (feature_intermediates, recurrent)

    manual_carry, (feature_intermediates, recurrent) = jax.jit(decomposed_forward)(
        params,
        image,
        agent_dir,
        initial_carry,
        reset,
    )
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        (
            canonical_value,
            canonical_logits,
            canonical_carry,
            manual_carry,
            feature_intermediates,
            recurrent,
        ),
    )

    if args.backend == "cpu":
        cpu_tolerance = protocol["comparison"]["cpu"]
        rtol = float(cpu_tolerance["rtol"])
        atol = float(cpu_tolerance["atol"])
        for name, actual in (
            ("value", canonical_value),
            ("logits", canonical_logits),
            ("carry_c", canonical_carry[0]),
            ("carry_h", canonical_carry[1]),
        ):
            require(
                numeric_match(np, expected_cpu[name], actual, rtol, atol),
                f"canonical CPU output does not reproduce frozen payload: {name}",
            )
        require(
            numeric_match(np, canonical_carry[0], manual_carry[0], rtol, atol),
            "manual CPU cell-state carry does not match canonical carry",
        )
        require(
            numeric_match(np, canonical_carry[1], manual_carry[1], rtol, atol),
            "manual CPU hidden-state carry does not match canonical carry",
        )

    output.mkdir(mode=0o700)
    bundle = ArrayBundle(np, jax, output)
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
        "convolution_preactivation",
        "conv_valid_nhwc_hwio",
        feature_intermediates["conv_preactivation"],
    )
    bundle.add("convolution_activation", "relu", feature_intermediates["conv_activation"])
    bundle.add("visual_flatten", "reshape", feature_intermediates["visual_flat"])
    bundle.add("scalar_embedding", "embedding_lookup", feature_intermediates["scalar_embedding"])
    bundle.add("concatenated_features", "visual_plus_scalar", feature_intermediates["features"])
    bundle.add(
        "reset_selected_carry",
        "lax_select_equivalent",
        {"c": recurrent["selected_c"], "h": recurrent["selected_h"]},
    )
    bundle.add("lstm_input_affine", "concatenated_input_dot", recurrent["input_affine"])
    bundle.add("lstm_hidden_affine", "concatenated_hidden_dot_bias", recurrent["hidden_affine"])
    bundle.add("lstm_gate_preactivation", "hidden_plus_input", recurrent["preactivation"])
    bundle.add("lstm_gate_activation", "sigmoid_i_f_o_tanh_g", recurrent["activation"])
    bundle.add(
        "lstm_cell_terms",
        "forget_and_input_terms",
        {"forget": recurrent["forget_term"], "input": recurrent["input_term"]},
    )
    bundle.add("lstm_cell_state", "new_c", recurrent["cell_state"])
    bundle.add(
        "lstm_hidden_state",
        "output_gate_times_tanh_c",
        {"new_h": recurrent["hidden_state"], "tanh_c": recurrent["tanh_cell_state"]},
    )
    bundle.add("manual_final_carry", "decomposed", {"c": manual_carry[0], "h": manual_carry[1]})
    bundle.add(
        "canonical_model_output",
        "gridworld_student_apply",
        {
            "carry": canonical_carry,
            "logits": canonical_logits,
            "value": canonical_value,
        },
    )
    parameter_digest_after = tree_digest(jax, np, params)
    require(parameter_digest_after == parameter_digest_before, "parameters changed")
    gpu_pid_after = gpu_pid_present() if args.backend == "gpu" else None
    if args.backend == "gpu":
        require(gpu_pid_after is True, "preserved GPU process missing after capture")

    receipt = {
        "schema_version": 1,
        "status": "captured_forward_only_without_training",
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
        "forward_calls": {"canonical_model": 1, "decomposed_probe": 1},
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
        "source": source_record,
        "payload": {
            "source_capture": str(capture_path),
            "source_capture_sha256": sha256(capture_path),
            "parameter_sha256_before": parameter_digest_before,
            "parameter_sha256_after": parameter_digest_after,
            "unchanged": True,
        },
        "hashes": {
            "protocol_sha256": PROTOCOL_SHA256,
            "payload_sha256": PAYLOAD_SHA256,
            "capture_script_sha256": sha256(Path(__file__).resolve()),
        },
        "cpu_reproduction_gates": (
            {
                "canonical_matches_frozen_capture": True,
                "manual_final_carry_matches_canonical": True,
            }
            if args.backend == "cpu"
            else {"status": "not_applicable_to_gpu_capture"}
        ),
        "preserved_gpu_process": {
            "pid": PRESERVED_GPU_PID,
            "present_before": gpu_pid_before,
            "present_after": gpu_pid_after,
        },
        "groups": bundle.groups,
        "records": bundle.records,
        "record_count": len(bundle.records),
    }
    atomic_json(output / "capture.json", receipt)
    print(
        "FORWARD_ONLY_CAPTURE_OK "
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
        CaptureError,
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"FORWARD_ONLY_CAPTURE_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
