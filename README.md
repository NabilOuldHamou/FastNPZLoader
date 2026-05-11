# fast_npz (readme written by Claude)

Parallel NPZ loader written in Rust, exposed as a Python extension via PyO3.  
Loads an entire directory of `.npz` files concurrently across CPU cores.

## Features

- **Parallel across files** — rayon distributes NPZ files across all cores
- **Parallel within a file** — arrays inside each NPZ parsed in parallel after decompression
- **Float32, Float64, strings** — supports `float32`, `float64`, byte strings (`|S`), and Unicode strings (`<U` / `>U`)
- **Progress bar** — tqdm-style terminal output via `indicatif`
- **Zero Python dependencies** — pure Rust, no numpy required at load time

## Installation

Requires [maturin](https://github.com/PyO3/maturin) and a Rust toolchain.

```bash
pip install maturin
maturin develop --release   # install into current venv
```

For a wheel you can distribute:

```bash
maturin build --release
pip install target/wheels/fast_npz-*.whl
```

## Usage

```python
import fast_npz

result = fast_npz.load_from_directory("/path/to/npz/files")
# result: dict[file_path, dict[array_name, list]]

for path, arrays in result.items():
    for name, values in arrays.items():
        print(f"{path} / {name}: {len(values)} elements")
```

### Output format

```python
{
  "/data/train.npz": {
    "embeddings": [0.1, 0.2, ...],   # list[float]
    "labels":     ["cat", "dog", ...] # list[str]
  },
  "/data/val.npz": { ... }
}
```

Arrays are flattened to 1-D lists. Shape information is not preserved — store shapes separately if needed.

## Supported dtypes

| NumPy dtype | Python type |
|---|---|
| `float32` | `list[float]` |
| `float64` | `list[float]` |
| `\|S<n>` (byte string) | `list[str]` |
| `<U<n>` / `>U<n>` (Unicode) | `list[str]` |
| Other | silently skipped |

## Cross-compilation (ARM)

With [cargo-zigbuild](https://github.com/rust-cross/cargo-zigbuild) (no system toolchain needed):

```bash
pip install maturin
cargo install cargo-zigbuild
rustup target add aarch64-unknown-linux-gnu
maturin build --release --zig --target aarch64-unknown-linux-gnu
```

## Architecture

```
load_from_directory(directory)
  └── rayon::par_iter over .npz paths
        └── load_single_npz(path)
              ├── Phase 1 (sequential): open zip, decompress each .npy entry → Vec<u8>
              └── Phase 2 (parallel):  parse NPY header once, dispatch to:
                    ├── ndarray::ArrayD::<f64>  →  NpyData::Float64
                    ├── ndarray::ArrayD::<f32>  →  NpyData::Float32
                    ├── byte string decoder     →  NpyData::Strings
                    └── UTF-32 decoder          →  NpyData::Strings
```

## Dependencies

| Crate | Role |
|---|---|
| `pyo3` | Python ↔ Rust bindings |
| `ndarray-npy` | NPY format parsing for numeric arrays |
| `rayon` | Data parallelism |
| `zip` | ZIP/NPZ archive extraction |
| `indicatif` | Terminal progress bar |
| `ndarray` | N-dimensional array type |
