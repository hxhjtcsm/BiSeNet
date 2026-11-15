#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export frozen RUGD3 A / D-BGA models to ONNX and verify
PyTorch <-> ONNXRuntime numerical consistency.

Deployment contract:
    input:
        name  = input
        dtype = float32
        shape = [1, 3, 576, 704]
        layout = NCHW

    output:
        name  = logits
        dtype = float32
        expected shape = [1, 3, 576, 704]

Important:
    - preprocessing is NOT embedded in ONNX
    - argmax is NOT embedded in ONNX
    - D-BGA boundary head is training-only and should not
      be part of the executed export graph
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


REPO_ROOT = Path(
    "/root/autodl-tmp/projects/BiSeNet"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from lib.models import model_factory


MODEL_TYPES = {
    "A":
        "bisenetv2",

    "DBGA":
        "bisenetv2_boundary_full_bga",
}


INPUT_SHAPE = (
    1,
    3,
    576,
    704,
)

N_CLASSES = 3

DEFAULT_OPSET = 13


class SemanticLogitsWrapper(nn.Module):
    """
    Export only the semantic inference output.

    Existing evaluator convention:
        model(x)
        -> tuple/list
        -> outputs[0]
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        outputs = self.model(
            x
        )

        if isinstance(
            outputs,
            (tuple, list),
        ):
            logits = outputs[0]
        else:
            logits = outputs

        return logits


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def git_head() -> str:

    try:
        return (
            subprocess
            .check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=REPO_ROOT,
                text=True,
            )
            .strip()
        )

    except Exception:
        return "UNKNOWN"


def load_state_dict(
    path: Path,
):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model" in checkpoint
        and
        isinstance(
            checkpoint["model"],
            dict,
        )
    ):
        checkpoint = (
            checkpoint[
                "model"
            ]
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint did not contain "
            "a state_dict."
        )

    return checkpoint


def build_model(
    variant: str,
    checkpoint_path: Path,
) -> SemanticLogitsWrapper:

    model_type = (
        MODEL_TYPES[
            variant
        ]
    )

    print(
        f"model_type = {model_type}"
    )

    model = (
        model_factory[
            model_type
        ](
            N_CLASSES
        )
    )

    state_dict = (
        load_state_dict(
            checkpoint_path
        )
    )

    result = (
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    )

    if (
        len(
            result.missing_keys
        ) != 0
        or
        len(
            result.unexpected_keys
        ) != 0
    ):
        raise RuntimeError(
            "Strict checkpoint load "
            "did not produce clean result."
        )

    if hasattr(
        model,
        "aux_mode",
    ):
        model.aux_mode = "eval"

    model.cpu()
    model.eval()

    wrapper = (
        SemanticLogitsWrapper(
            model
        )
    )

    wrapper.cpu()
    wrapper.eval()

    return wrapper


def make_input(
    seed: int,
) -> torch.Tensor:

    generator = (
        torch.Generator(
            device="cpu"
        )
    )

    generator.manual_seed(
        seed
    )

    # Roughly represents normalized image tensor distribution.
    x = torch.randn(
        INPUT_SHAPE,
        generator=generator,
        dtype=torch.float32,
    )

    return x


def preflight_model(
    model: nn.Module,
) -> None:

    x = make_input(
        20261113
    )

    with torch.inference_mode():

        y = model(
            x
        )

    print(
        "PyTorch preflight input :",
        tuple(
            x.shape
        ),
        x.dtype,
    )

    print(
        "PyTorch preflight output:",
        tuple(
            y.shape
        ),
        y.dtype,
    )

    expected = (
        1,
        N_CLASSES,
        576,
        704,
    )

    if tuple(
        y.shape
    ) != expected:

        raise RuntimeError(
            "Unexpected semantic "
            f"logit shape: {y.shape}; "
            f"expected={expected}"
        )

    if not torch.isfinite(
        y
    ).all():

        raise RuntimeError(
            "PyTorch output "
            "contains NaN/Inf."
        )


def export_onnx(
    model: nn.Module,
    output_path: Path,
    opset: int,
    artifacts_dir: Path,
) -> None:

    x = make_input(
        20261113
    )

    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Exporting ONNX..."
    )

    print(
        "opset =",
        opset,
    )

    print(
        "static input =",
        INPUT_SHAPE,
    )

    # New torch.export-based exporter.
    torch.onnx.export(
        model,
        (
            x,
        ),
        f=str(
            output_path
        ),
        input_names=[
            "input",
        ],
        output_names=[
            "logits",
        ],
        opset_version=opset,
        dynamo=True,
        external_data=False,
        optimize=True,
        report=True,
        artifacts_dir=str(
            artifacts_dir
        ),
    )

    if not output_path.is_file():

        raise RuntimeError(
            f"ONNX was not created: "
            f"{output_path}"
        )


def shape_of_value_info(
    value_info,
):

    dims = []

    tensor_type = (
        value_info
        .type
        .tensor_type
    )

    for dim in (
        tensor_type
        .shape
        .dim
    ):

        if (
            dim.HasField(
                "dim_value"
            )
        ):
            dims.append(
                int(
                    dim.dim_value
                )
            )

        elif (
            dim.dim_param
        ):
            dims.append(
                dim.dim_param
            )

        else:
            dims.append(
                None
            )

    return dims


def audit_onnx(
    path: Path,
) -> dict:

    print()
    print(
        "Running onnx.checker..."
    )

    model = onnx.load(
        str(
            path
        )
    )

    onnx.checker.check_model(
        model
    )

    print(
        "onnx.checker: PASS"
    )

    inputs = [
        {
            "name":
                x.name,

            "shape":
                shape_of_value_info(
                    x
                ),
        }
        for x in
        model.graph.input
    ]

    outputs = [
        {
            "name":
                x.name,

            "shape":
                shape_of_value_info(
                    x
                ),
        }
        for x in
        model.graph.output
    ]

    opsets = {
        (
            item.domain
            or
            "ai.onnx"
        ):
            int(
                item.version
            )
        for item in
        model.opset_import
    }

    operator_counts = (
        collections.Counter(
            node.op_type
            for node in
            model.graph.node
        )
    )

    print(
        "Inputs :",
        inputs,
    )

    print(
        "Outputs:",
        outputs,
    )

    print(
        "Opsets :",
        opsets,
    )

    print(
        "Nodes  :",
        len(
            model.graph.node
        ),
    )

    print(
        "Initializers:",
        len(
            model.graph.initializer
        ),
    )

    if (
        len(inputs) != 1
        or
        inputs[0]["name"]
        != "input"
    ):
        raise RuntimeError(
            f"Unexpected ONNX inputs: "
            f"{inputs}"
        )

    if inputs[0][
        "shape"
    ] != list(
        INPUT_SHAPE
    ):
        raise RuntimeError(
            "ONNX input is not the "
            "required static shape: "
            f"{inputs[0]['shape']}"
        )

    if (
        len(outputs) != 1
        or
        outputs[0]["name"]
        != "logits"
    ):
        raise RuntimeError(
            f"Unexpected ONNX outputs: "
            f"{outputs}"
        )

    return {
        "inputs":
            inputs,

        "outputs":
            outputs,

        "opsets":
            opsets,

        "node_count":
            len(
                model.graph.node
            ),

        "initializer_count":
            len(
                model.graph.initializer
            ),

        "operator_counts":
            dict(
                sorted(
                    operator_counts
                    .items()
                )
            ),
    }


def verify_onnxruntime(
    model: nn.Module,
    onnx_path: Path,
    num_tests: int,
    max_abs_threshold: float,
    min_argmax_agreement: float,
) -> dict:

    print()
    print(
        "Creating ONNXRuntime "
        "CPU session..."
    )

    session = (
        ort.InferenceSession(
            str(
                onnx_path
            ),
            providers=[
                "CPUExecutionProvider",
            ],
        )
    )

    input_info = (
        session
        .get_inputs()[0]
    )

    output_info = (
        session
        .get_outputs()[0]
    )

    print(
        "ORT input :",
        input_info.name,
        input_info.shape,
        input_info.type,
    )

    print(
        "ORT output:",
        output_info.name,
        output_info.shape,
        output_info.type,
    )

    if (
        input_info.name
        != "input"
    ):
        raise RuntimeError(
            "ORT input name mismatch."
        )

    if (
        output_info.name
        != "logits"
    ):
        raise RuntimeError(
            "ORT output name mismatch."
        )

    tests = []

    for i in range(
        num_tests
    ):

        seed = (
            20261113
            + i
        )

        x = make_input(
            seed
        )

        with torch.inference_mode():

            torch_logits = (
                model(
                    x
                )
                .cpu()
                .numpy()
            )

        ort_logits = (
            session.run(
                [
                    "logits",
                ],
                {
                    "input":
                        x.numpy(),
                },
            )[0]
        )

        if (
            torch_logits.shape
            !=
            ort_logits.shape
        ):
            raise RuntimeError(
                "Output shape mismatch: "
                f"PyTorch={torch_logits.shape}, "
                f"ORT={ort_logits.shape}"
            )

        if not np.isfinite(
            ort_logits
        ).all():

            raise RuntimeError(
                "ONNXRuntime output "
                "contains NaN/Inf."
            )

        abs_error = np.abs(
            torch_logits
            -
            ort_logits
        )

        max_abs = float(
            abs_error.max()
        )

        mean_abs = float(
            abs_error.mean()
        )

        torch_pred = np.argmax(
            torch_logits,
            axis=1,
        )

        ort_pred = np.argmax(
            ort_logits,
            axis=1,
        )

        argmax_agreement = float(
            np.mean(
                torch_pred
                ==
                ort_pred
            )
        )

        print()
        print(
            f"[test {i + 1}] "
            f"seed={seed}"
        )

        print(
            f"max_abs_error      = "
            f"{max_abs:.10e}"
        )

        print(
            f"mean_abs_error     = "
            f"{mean_abs:.10e}"
        )

        print(
            f"argmax_agreement   = "
            f"{argmax_agreement:.10f}"
        )

        tests.append(
            {
                "seed":
                    seed,

                "max_abs_error":
                    max_abs,

                "mean_abs_error":
                    mean_abs,

                "argmax_agreement":
                    argmax_agreement,
            }
        )

    worst_max_abs = max(
        item[
            "max_abs_error"
        ]
        for item in tests
    )

    worst_mean_abs = max(
        item[
            "mean_abs_error"
        ]
        for item in tests
    )

    worst_agreement = min(
        item[
            "argmax_agreement"
        ]
        for item in tests
    )

    print()
    print(
        "Worst max abs error =",
        f"{worst_max_abs:.10e}",
    )

    print(
        "Worst mean abs error =",
        f"{worst_mean_abs:.10e}",
    )

    print(
        "Minimum argmax agreement =",
        f"{worst_agreement:.10f}",
    )

    if (
        worst_max_abs
        >
        max_abs_threshold
    ):
        raise RuntimeError(
            "ONNX numerical check failed: "
            f"max_abs={worst_max_abs} > "
            f"{max_abs_threshold}"
        )

    if (
        worst_agreement
        <
        min_argmax_agreement
    ):
        raise RuntimeError(
            "ONNX semantic agreement failed: "
            f"{worst_agreement} < "
            f"{min_argmax_agreement}"
        )

    return {
        "tests":
            tests,

        "worst_max_abs_error":
            worst_max_abs,

        "worst_mean_abs_error":
            worst_mean_abs,

        "minimum_argmax_agreement":
            worst_agreement,

        "max_abs_threshold":
            max_abs_threshold,

        "min_argmax_agreement_threshold":
            min_argmax_agreement,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "A",
            "DBGA",
        ],
    )

    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
    )

    parser.add_argument(
        "--num-tests",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-abs-threshold",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--min-argmax-agreement",
        type=float,
        default=0.99999,
    )

    args = parser.parse_args()

    weight = (
        args.weight
        .resolve()
    )

    output_dir = (
        args.output_dir
        .resolve()
    )

    if not weight.is_file():
        raise FileNotFoundError(
            weight
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    variant_lower = (
        args.variant
        .lower()
    )

    onnx_path = (
        output_dir
        /
        (
            f"rugd3_"
            f"{variant_lower}"
            f"_seed123_"
            f"576x704_fp32.onnx"
        )
    )

    report_dir = (
        output_dir
        /
        "export_report"
    )

    print(
        "=" * 100
    )

    print(
        "RUGD3 ONNX EXPORT"
    )

    print(
        "=" * 100
    )

    print(
        "Variant    :",
        args.variant,
    )

    print(
        "Weight     :",
        weight,
    )

    print(
        "Weight SHA :",
        sha256_file(
            weight
        ),
    )

    print(
        "Git HEAD   :",
        git_head(),
    )

    print(
        "Torch      :",
        torch.__version__,
    )

    print(
        "ONNX       :",
        onnx.__version__,
    )

    print(
        "ORT        :",
        ort.__version__,
    )

    print(
        "Input      :",
        INPUT_SHAPE,
    )

    print(
        "Opset      :",
        args.opset,
    )

    model = build_model(
        args.variant,
        weight,
    )

    preflight_model(
        model
    )

    export_onnx(
        model,
        onnx_path,
        args.opset,
        report_dir,
    )

    graph_info = audit_onnx(
        onnx_path
    )

    numerical_info = (
        verify_onnxruntime(
            model,
            onnx_path,
            args.num_tests,
            args.max_abs_threshold,
            args.min_argmax_agreement,
        )
    )

    manifest = {
        "variant":
            args.variant,

        "model_type":
            MODEL_TYPES[
                args.variant
            ],

        "seed":
            123,

        "checkpoint":
            str(
                weight
            ),

        "checkpoint_sha256":
            sha256_file(
                weight
            ),

        "git_head":
            git_head(),

        "torch_version":
            torch.__version__,

        "onnx_version":
            onnx.__version__,

        "onnxruntime_version":
            ort.__version__,

        "requested_opset":
            args.opset,

        "actual_onnx_opsets":
            graph_info[
                "opsets"
            ],

        "input_name":
            "input",

        "input_dtype":
            "float32",

        "input_layout":
            "NCHW",

        "input_shape":
            list(
                INPUT_SHAPE
            ),

        "output_name":
            "logits",

        "output_dtype":
            "float32",

        "onnx_path":
            str(
                onnx_path
            ),

        "onnx_sha256":
            sha256_file(
                onnx_path
            ),

        "onnx_size_bytes":
            onnx_path
            .stat()
            .st_size,

        "graph":
            graph_info,

        "numerical_verification":
            numerical_info,
    }

    manifest_path = (
        output_dir
        /
        "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    sha_path = (
        output_dir
        /
        "SHA256SUMS.txt"
    )

    sha_path.write_text(
        (
            f"{manifest['onnx_sha256']}  "
            f"{onnx_path.name}\n"
            f"{manifest['checkpoint_sha256']}  "
            f"{weight.name}\n"
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ONNX EXPORT + VERIFY: PASS"
    )

    print(
        "=" * 100
    )

    print(
        "ONNX:",
        onnx_path,
    )

    print(
        "SHA256:",
        manifest[
            "onnx_sha256"
        ],
    )

    print(
        "Size:",
        manifest[
            "onnx_size_bytes"
        ],
        "bytes",
    )

    print(
        "Manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()
