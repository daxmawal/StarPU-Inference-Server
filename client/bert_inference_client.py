#!/usr/bin/env python3
"""gRPC client used to send real inference requests to the StarPU server."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import grpc
import numpy as np

import grpc_service_pb2
import grpc_service_pb2_grpc

BASE_DTYPE_TO_NUMPY = {
    "FP32": np.float32,
    "FP64": np.float64,
    "FP16": np.float16,
    "BF16": np.float16,
    "INT64": np.int64,
    "INT32": np.int32,
    "INT16": np.int16,
    "INT8": np.int8,
    "UINT8": np.uint8,
    "BOOL": np.bool_,
    "UINT64": np.uint64,
}

CONTENT_FIELD_BY_BASE_DTYPE = {
    "FP32": "fp32_contents",
    "FP64": "fp64_contents",
    "INT32": "int_contents",
    "INT16": "int_contents",
    "INT8": "int_contents",
    "UINT8": "uint_contents",
    "UINT64": "uint64_contents",
    "INT64": "int64_contents",
    "BOOL": "bool_contents",
}

DEFAULT_SERVER = "127.0.0.1:50051"
DEFAULT_MODEL_NAME = "bert"
DEFAULT_MODEL_VERSION = "1"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_MESSAGE_MB = 32
DEFAULT_TOKENIZER = "bert-base-uncased"
DEFAULT_MAX_LENGTH = 128
DEFAULT_REQUESTED_OUTPUTS = ["output0"]
DEFAULT_PREVIEW_VALUES = 16


def _tokenize_texts(
    texts: Iterable[str], tokenizer_name: str, max_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch-tokenize sentences with a HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Run 'pip install transformers' to "
            "enable text tokenization."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int64, copy=False)
    attention_mask = encoded["attention_mask"].astype(np.int64, copy=False)
    return input_ids, attention_mask


def _normalise_dtype_label(dtype_label: str | None) -> str:
    if not dtype_label:
        return "FP32"
    label = dtype_label.upper()
    if label.startswith("TYPE_"):
        label = label[5:]
    return label


def _resolve_numpy_dtype(dtype_label: str | None) -> Tuple[str, np.dtype]:
    base = _normalise_dtype_label(dtype_label)
    np_dtype = BASE_DTYPE_TO_NUMPY.get(base)
    if np_dtype is None:
        raise ValueError(f"Datatype {dtype_label} is not supported for display.")
    return base, np_dtype


def _typed_contents_to_numpy(
    output: grpc_service_pb2.ModelInferResponse.InferOutputTensor,
    base_dtype: str,
    np_dtype: np.dtype,
) -> np.ndarray | None:
    field_name = CONTENT_FIELD_BY_BASE_DTYPE.get(base_dtype)
    if not field_name:
        return None
    contents = getattr(output.contents, field_name, None)
    if contents in (None, []):
        return None
    return np.array(contents, dtype=np_dtype)


def extract_response_tensors(
    response: grpc_service_pb2.ModelInferResponse,
) -> List[dict]:
    """Convert protobuf outputs into NumPy tensors."""
    tensors: List[dict] = []
    for idx, output in enumerate(response.outputs):
        name = output.name or f"output{idx}"
        base_dtype, np_dtype = _resolve_numpy_dtype(output.datatype)
        dtype_label = output.datatype or base_dtype
        shape = tuple(output.shape)

        array: np.ndarray | None = None
        if idx < len(response.raw_output_contents):
            raw = response.raw_output_contents[idx]
            if raw:
                array = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()

        if array is None:
            typed = _typed_contents_to_numpy(output, base_dtype, np_dtype)
            if typed is not None:
                array = typed.reshape(shape)

        if array is None:
            raise ValueError(
                f"No data retrieved for output '{name}'. "
                "Ensure the server returns raw contents."
            )

        tensors.append(
            {
                "name": name,
                "dtype_label": dtype_label,
                "base_dtype": base_dtype,
                "array": array,
            }
        )

    return tensors


def _add_input_tensor(
    request: grpc_service_pb2.ModelInferRequest, name: str, array: np.ndarray
) -> None:
    """Append a raw tensor to the gRPC message."""
    dtype_map = {
        np.dtype("float32"): "FP32",
        np.dtype("float64"): "FP64",
        np.dtype("float16"): "FP16",
        np.dtype("int64"): "INT64",
        np.dtype("int32"): "INT32",
        np.dtype("int16"): "INT16",
        np.dtype("int8"): "INT8",
        np.dtype("uint8"): "UINT8",
        np.dtype("bool"): "BOOL",
    }
    np_dtype = np.dtype(array.dtype)
    if np_dtype not in dtype_map:
        raise ValueError(f"Datatype {array.dtype} is not supported for input {name}.")

    array_c = np.ascontiguousarray(array)
    tensor = request.inputs.add()
    tensor.name = name
    tensor.datatype = dtype_map[np_dtype]
    tensor.shape.extend(array_c.shape)
    request.raw_input_contents.append(array_c.tobytes(order="C"))


def build_infer_request(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    model_name: str,
    model_version: str,
    request_id: str | None,
    requested_outputs: List[str],
) -> grpc_service_pb2.ModelInferRequest:
    """Build the `ModelInfer` request."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must share the same shape.")

    request = grpc_service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    if request_id:
        request.id = request_id
    request.client_send_ms = int(time.time() * 1000)

    _add_input_tensor(request, "input_ids", input_ids)
    _add_input_tensor(request, "attention_mask", attention_mask)

    for output_name in requested_outputs:
        output = request.outputs.add()
        output.name = output_name

    return request


def run_inference(
    server_addr: str,
    request: grpc_service_pb2.ModelInferRequest,
    timeout_s: float,
    max_message_bytes: int,
) -> grpc_service_pb2.ModelInferResponse:
    """Send the request and return the response."""
    options = [
        ("grpc.max_send_message_length", max_message_bytes),
        ("grpc.max_receive_message_length", max_message_bytes),
    ]
    with grpc.insecure_channel(server_addr, options=options) as channel:
        stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel)
        return stub.ModelInfer(request, timeout=timeout_s)


def _summarize_response(
    response: grpc_service_pb2.ModelInferResponse,
    outputs_info: List[dict],
    max_preview_values: int,
) -> None:
    """Print a simple summary of the outputs and optional value previews."""
    print(f"Model: {response.model_name} (version: {response.model_version})")
    for info in outputs_info:
        tensor = info["array"]
        dtype_label = info["dtype_label"]
        name = info["name"]

        print(f"- Output {name}: shape={tensor.shape} dtype={dtype_label}")
        print(
            f"  min={tensor.min():.6f} max={tensor.max():.6f} "
            f"mean={tensor.mean():.6f}"
        )

        if max_preview_values > 0 and tensor.size > 0:
            batch_index = 0
            sample = tensor
            if tensor.ndim >= 1 and tensor.shape[0] > 1:
                sample = tensor[batch_index]
            if sample.ndim >= 1:
                sample = sample.reshape(-1)
            flat = sample.flatten()
            count = min(max_preview_values, flat.size)
            preview = flat[:count]
            preview_str = np.array2string(
                preview,
                precision=6,
                separator=", ",
                suppress_small=False,
                max_line_width=80,
            )
            prefix = "  first elements"
            if tensor.ndim >= 1 and tensor.shape[0] > 1:
                prefix += " (batch 0)"
            print(f"{prefix} ({count}): {preview_str}")


def validate_with_reference(
    reference_path: Path,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    outputs_info: List[dict],
    atol: float,
    rtol: float,
) -> List[dict]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for --reference-model but is not installed in the "
            "current environment."
        ) from exc

    reference_path = reference_path.expanduser()
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference model not found: {reference_path}")

    model = torch.jit.load(str(reference_path), map_location="cpu")
    model.eval()

    ids_tensor = torch.from_numpy(np.ascontiguousarray(input_ids))
    mask_tensor = torch.from_numpy(np.ascontiguousarray(attention_mask))

    with torch.inference_mode():
        result = model(ids_tensor, mask_tensor)

    if isinstance(result, torch.Tensor):
        reference_tensors = [result]
    elif isinstance(result, (tuple, list)):
        reference_tensors = list(result)
    else:
        raise RuntimeError(
            "The reference model output is neither a tensor nor a tensor sequence."
        )

    stats: List[dict] = []
    for idx, info in enumerate(outputs_info):
        name = info["name"]
        server_tensor = info["array"]

        if idx >= len(reference_tensors):
            stats.append(
                {
                    "name": name,
                    "ok": False,
                    "error": "The reference model returned fewer outputs.",
                }
            )
            continue

        ref_tensor = reference_tensors[idx]
        if not isinstance(ref_tensor, torch.Tensor):
            stats.append(
                {
                    "name": name,
                    "ok": False,
                    "error": "Unexpected reference output type (non-Tensor).",
                }
            )
            continue

        ref_array = ref_tensor.detach().cpu().numpy()
        if ref_array.shape != server_tensor.shape:
            stats.append(
                {
                    "name": name,
                    "ok": False,
                    "error": (
                        f"Mismatched shape: server {server_tensor.shape} "
                        f"vs reference {ref_array.shape}"
                    ),
                }
            )
            continue

        ref_cast = ref_array.astype(server_tensor.dtype, copy=False)
        diff = np.abs(server_tensor - ref_cast)
        tol = atol + rtol * np.abs(ref_cast)
        ok = bool(np.all(diff <= tol))
        max_abs = float(diff.max()) if diff.size else 0.0
        denom = np.maximum(np.abs(ref_cast), atol)
        max_rel = float((diff / denom).max()) if diff.size else 0.0

        stats.append(
            {
                "name": name,
                "ok": ok,
                "max_abs": max_abs,
                "max_rel": max_rel,
            }
        )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gRPC client for the StarPU Inference Server (BERT)."
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help="gRPC address of the server (host:port).",
    )
    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="Sentence to infer (repeat the option for batching).",
    )
    parser.add_argument(
        "--reference-model",
        type=Path,
        help=("Path to a local TorchScript model used to validate the server output."),
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for validation (--reference-model).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for validation (--reference-model).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_ids, attention_mask = _tokenize_texts(
        args.text, DEFAULT_TOKENIZER, DEFAULT_MAX_LENGTH
    )

    request = build_infer_request(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model_name=DEFAULT_MODEL_NAME,
        model_version=DEFAULT_MODEL_VERSION,
        request_id=None,
        requested_outputs=DEFAULT_REQUESTED_OUTPUTS,
    )

    response = run_inference(
        server_addr=args.server,
        request=request,
        timeout_s=DEFAULT_TIMEOUT_S,
        max_message_bytes=DEFAULT_MAX_MESSAGE_MB * 1024 * 1024,
    )

    outputs_info = extract_response_tensors(response)
    _summarize_response(response, outputs_info, DEFAULT_PREVIEW_VALUES)

    if args.reference_model:
        stats = validate_with_reference(
            args.reference_model,
            input_ids,
            attention_mask,
            outputs_info,
            atol=args.atol,
            rtol=args.rtol,
        )
        for stat in stats:
            name = stat["name"]
            if "error" in stat:
                print(f"Validation {name}: ERROR – {stat['error']}")
                continue
            status = "OK" if stat["ok"] else "FAIL"
            print(
                f"Validation {name}: {status} "
                f"(max abs diff={stat['max_abs']:.3e}, "
                f"max rel diff={stat['max_rel']:.3e})"
            )


if __name__ == "__main__":
    main()
