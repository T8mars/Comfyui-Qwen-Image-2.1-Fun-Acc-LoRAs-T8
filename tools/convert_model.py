"""Wrap the pinned Alibaba PAI PDD export for the companion ComfyUI node.

The 528 tensors and their payload bytes are unchanged. Only safetensors
metadata is added so the node can reject unrelated LoRAs and use the exact
four-step sigma schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path


SOURCE_SHA256 = "764c56ae94f330b6d06ccc322f95e1b8ce46424ddde5899a15432f95f720d558"
SOURCE_REVISION = "f7545234760e1847cd8e89e52bd951cb0b7e327f"
FORMAT = "t8_qwenimage21_funacc_pdd_4step_v1"
SIGMAS = [1.0, 0.9169867038726807, 0.7861579060554504, 0.5494909882545471, 0.0]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(stream):
    raw_size = stream.read(8)
    if len(raw_size) != 8:
        raise ValueError("Invalid safetensors file")
    size = struct.unpack("<Q", raw_size)[0]
    if not 0 < size < 100_000_000:
        raise ValueError("Invalid safetensors header size")
    raw_header = stream.read(size)
    if len(raw_header) != size:
        raise ValueError("Truncated safetensors header")
    return json.loads(raw_header), size + 8


def validate(header: dict, payload_size: int):
    names = {key for key in header if key != "__metadata__"}
    if len(names) != 528:
        raise ValueError(f"Expected 528 tensors, found {len(names)}")
    up = {name.removesuffix(".lora_up") for name in names if name.endswith(".lora_up")}
    down = {name.removesuffix(".lora_down") for name in names if name.endswith(".lora_down")}
    full = names - {name + ".lora_up" for name in up} - {name + ".lora_down" for name in down}
    if len(up) != 231 or up != down or len(full) != 66:
        raise ValueError("Unexpected LoRA pairs or full weights")
    if header["proj_out.weight"]["shape"] != [4, 64, 4096]:
        raise ValueError("Expected four 64 x 4096 output heads")
    spans = sorted(tuple(header[name]["data_offsets"]) for name in names)
    cursor = 0
    for start, end in spans:
        if start != cursor or end <= start:
            raise ValueError("Gap or overlap in tensor payload")
        cursor = end
    if cursor != payload_size:
        raise ValueError("Tensor payload size mismatch")
    if header.get("__metadata__", {}).get("format") != "qwenimage21_extracted_prefused_v1":
        raise ValueError("Unexpected PDD export format")


def convert(source: Path, output: Path):
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Source SHA256 does not match the pinned release")

    with source.open("rb") as src:
        header, source_offset = read_header(src)
        payload_size = source.stat().st_size - source_offset
        validate(header, payload_size)
        header["__metadata__"] = {
            **header.get("__metadata__", {}),
            "t8_format": FORMAT,
            "pdd_sigmas": json.dumps(SIGMAS, separators=(",", ":")),
            "source_repo": "alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs",
            "source_revision": SOURCE_REVISION,
            "hf_file_sha256": SOURCE_SHA256,
            "conversion": "Metadata only; all 528 tensors and data bytes unchanged",
        }
        encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".partial", delete=False) as dst:
                temp = Path(dst.name)
                dst.write(struct.pack("<Q", len(encoded)))
                dst.write(encoded)
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            with temp.open("rb") as check:
                checked_header, checked_offset = read_header(check)
                validate(checked_header, temp.stat().st_size - checked_offset)
                if temp.stat().st_size - checked_offset != payload_size:
                    raise ValueError("Converted payload size differs")
                output_data_hash = hashlib.sha256()
                for chunk in iter(lambda: check.read(8 * 1024 * 1024), b""):
                    output_data_hash.update(chunk)
            src.seek(source_offset)
            source_data_hash = hashlib.sha256()
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                source_data_hash.update(chunk)
            if output_data_hash.digest() != source_data_hash.digest():
                raise ValueError("Original tensor data changed")
            os.replace(temp, output)
        finally:
            if temp is not None and temp.exists():
                temp.unlink()
    print(f"Converted: {output}")
    print(f"SHA256: {digest(output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
