"""Refetches tokenizer.json, verified against the pin below.

Source: microsoft/xclip-base-patch16-kinetics-600, transformers revision
e4921c41fc296102aae210d43d4127c5e3e51928 - the same revision
scripts/export_xclip.py exports the two ONNX towers from, in the package
root. tokenizer.json is CLIP's byte-level BPE (vocab 49408, bos 49406,
eos 49407): the `tokenizers` crate reads it directly, compiled into
embed.wasm with include_bytes! since a wasm module has no filesystem to
read it from at runtime - only wasi:nn's load-by-name reaches the ONNX
graph, and that path is a name, never a file.

Run from anywhere:
    python fetch-tokenizer.py
Overwrites tokenizer.json beside this script once the download's sha256
matches SHA256 below; refuses to write otherwise.
"""

import hashlib
import pathlib
import urllib.request

REPO = "microsoft/xclip-base-patch16-kinetics-600"
REVISION = "e4921c41fc296102aae210d43d4127c5e3e51928"
FILE = "tokenizer.json"
SHA256 = "f3dba9c5f2100fc09edb20622a18083f1fbf5041e4c0af371959bd7502428d44"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}"
DEST = pathlib.Path(__file__).resolve().parent / FILE


def main() -> None:
    print(f"fetching {URL}")
    with urllib.request.urlopen(URL, timeout=30) as response:
        data = response.read()
    found = hashlib.sha256(data).hexdigest()
    if found != SHA256:
        raise SystemExit(f"{FILE} hashes to {found}, expected {SHA256}: refusing to write")
    DEST.write_bytes(data)
    print(f"wrote {DEST} ({len(data)} bytes, sha256 {found})")


if __name__ == "__main__":
    main()
