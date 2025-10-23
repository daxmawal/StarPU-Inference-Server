#!/usr/bin/env python3
"""
Client gRPC pour envoyer une requête d'inférence réelle au serveur StarPU.

Deux modes d'utilisation :
  * Passage d'une ou plusieurs phrases via --text (tokenisation automatique
    avec HuggingFace).
  * Chargement d'entrées pré-encodées depuis un fichier .npz contenant
    `input_ids` et `attention_mask`.

Le script prépare les tenseurs au format attendu (`int64`, shape [batch, 128])
par la configuration `models/bert.yml`, envoie la requête `ModelInfer` et
affiche un résumé des sorties.
"""

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


def _load_encoded_inputs(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Charge `input_ids` et `attention_mask` depuis un fichier .npz."""
    data = np.load(npz_path)
    try:
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
    except KeyError as exc:
        raise ValueError(
            "Le fichier .npz doit contenir les clés 'input_ids' et " "'attention_mask'."
        ) from exc

    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("Les tenseurs doivent être de rang 2 (batch, sequence).")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids et attention_mask doivent avoir la même shape.")

    return (
        np.asarray(input_ids, dtype=np.int64, order="C"),
        np.asarray(attention_mask, dtype=np.int64, order="C"),
    )


def _tokenize_texts(
    texts: Iterable[str], tokenizer_name: str, max_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenise des phrases en batch avec un tokenizer HuggingFace."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers n'est pas installé. Exécutez 'pip install transformers'"
            " ou fournissez un fichier .npz pré-encodé via --encoded-npz."
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
        raise ValueError(f"Datatype {dtype_label} non pris en charge pour l'affichage.")
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
    """Convertit les sorties protobuf en tenseurs NumPy."""
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
                f"Aucune donnée récupérée pour la sortie '{name}'. "
                "Vérifiez que le serveur renvoie les contenus bruts."
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
    """Ajoute un tenseur brut au message gRPC."""
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
        raise ValueError(
            f"Datatype {array.dtype} non pris en charge pour l'entrée {name}."
        )

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
    """Construit la requête `ModelInfer`."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids et attention_mask doivent partager la shape.")

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
    """Envoie la requête et renvoie la réponse."""
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
    """Affiche un résumé simple des sorties et, optionnellement, quelques valeurs."""
    print(f"Modèle: {response.model_name} (version: {response.model_version})")
    for info in outputs_info:
        tensor = info["array"]
        dtype_label = info["dtype_label"]
        name = info["name"]

        print(f"- Sortie {name}: shape={tensor.shape} dtype={dtype_label}")
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
            prefix = "  premiers éléments"
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
            "PyTorch est requis pour --reference-model, mais n'est pas installé "
            "dans l'environnement courant."
        ) from exc

    reference_path = reference_path.expanduser()
    if not reference_path.exists():
        raise FileNotFoundError(f"Modèle de référence introuvable: {reference_path}")

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
            "La sortie du modèle de référence n'est pas un tenseur ou une séquence "
            "de tenseurs."
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
                    "error": "Le modèle de référence a renvoyé moins de sorties.",
                }
            )
            continue

        ref_tensor = reference_tensors[idx]
        if not isinstance(ref_tensor, torch.Tensor):
            stats.append(
                {
                    "name": name,
                    "ok": False,
                    "error": "Sortie de référence inattendue (non-Tensor).",
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
                        f"Shape différente: serveur {server_tensor.shape} "
                        f"vs référence {ref_array.shape}"
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
        description="Client gRPC pour StarPU Inference Server (BERT)."
    )
    parser.add_argument(
        "--server",
        default="127.0.0.1:50051",
        help="Adresse gRPC du serveur (host:port).",
    )
    parser.add_argument(
        "--model-name",
        default="bert",
        help="Nom du modèle exposé côté serveur.",
    )
    parser.add_argument(
        "--model-version",
        default="1",
        help="Version du modèle.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout RPC en secondes.",
    )
    parser.add_argument(
        "--max-message-mb",
        type=int,
        default=32,
        help="Taille max des messages gRPC (MiB).",
    )
    parser.add_argument(
        "--request-id",
        help="Identifiant optionnel transmis au serveur.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--encoded-npz",
        type=Path,
        help="Fichier .npz avec `input_ids` et `attention_mask` (int64).",
    )
    input_group.add_argument(
        "--text",
        action="append",
        help="Phrase à inférer (répéter l'option pour le batch).",
    )

    parser.add_argument(
        "--tokenizer",
        default="bert-base-uncased",
        help="Tokenizer HuggingFace utilisé pour --text.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Longueur maximale de séquence (padding/troncature).",
    )
    parser.add_argument(
        "--output",
        action="append",
        default=["output0"],
        help="Nom des sorties à récupérer (par défaut: output0).",
    )
    parser.add_argument(
        "--print-values",
        type=int,
        default=16,
        help="Nombre de valeurs à afficher par sortie (0 pour désactiver).",
    )
    parser.add_argument(
        "--reference-model",
        type=Path,
        help=(
            "Chemin vers un modèle TorchScript local pour valider la sortie du "
            "serveur."
        ),
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Tolérance relative pour la validation (--reference-model).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Tolérance absolue pour la validation (--reference-model).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.encoded_npz:
        input_ids, attention_mask = _load_encoded_inputs(args.encoded_npz)
    else:
        assert args.text is not None
        input_ids, attention_mask = _tokenize_texts(
            args.text, args.tokenizer, args.max_length
        )

    request = build_infer_request(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model_name=args.model_name,
        model_version=args.model_version,
        request_id=args.request_id,
        requested_outputs=list(dict.fromkeys(args.output)),
    )

    response = run_inference(
        server_addr=args.server,
        request=request,
        timeout_s=args.timeout,
        max_message_bytes=args.max_message_mb * 1024 * 1024,
    )

    outputs_info = extract_response_tensors(response)
    max_preview_values = max(0, args.print_values)
    _summarize_response(response, outputs_info, max_preview_values)

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
                print(f"Validation {name}: ERREUR – {stat['error']}")
                continue
            status = "OK" if stat["ok"] else "ECHEC"
            print(
                f"Validation {name}: {status} "
                f"(max abs diff={stat['max_abs']:.3e}, "
                f"max rel diff={stat['max_rel']:.3e})"
            )


if __name__ == "__main__":
    main()
