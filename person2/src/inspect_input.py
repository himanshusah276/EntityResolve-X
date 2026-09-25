"""
Person 2 — Input Inspection Tool
Inspects Person 1's preprocessed/blocked candidate pairs, normalized records, and ground truth files.
Extracts schema, column names, dtypes, approximate row counts, and sample records
without loading entire multi-million row datasets into memory.
"""

import os
import sys
import json
import argparse
from pathlib import Path


def inspect_file(file_path: Path):
    """Inspects a single file using lightweight metadata or first few lines."""
    print("=" * 60)
    print(f"FILE: {file_path.name}")
    print(f"Path: {file_path.resolve()}")
    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    if size_gb >= 1.0:
        print(f"File Size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
    else:
        print(f"File Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    ext = file_path.suffix.lower()

    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            schema = parquet_file.schema_arrow
            print(f"Format: Parquet (Arrow)")
            print(f"Total Rows: {metadata.num_rows:,}")
            print(f"Row Groups: {metadata.num_row_groups}")
            print(f"Columns ({len(schema.names)}):")
            for name, typ in zip(schema.names, schema.types):
                print(f"  - {name}: {typ}")
            
            # Read small sample (first 3 rows)
            sample_table = parquet_file.read_row_group(0).slice(0, 3)
            print("\nSample (first 3 rows):")
            for row in sample_table.to_pylist():
                print(f"  {row}")
        except ImportError:
            print("pyarrow not installed. Run 'pip install pyarrow' to inspect Parquet metadata.")
        except Exception as e:
            print(f"Error inspecting Parquet file: {e}")

    elif ext in [".tsv", ".csv", ".txt"]:
        sep = "\t" if ext == ".tsv" else ","
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline().rstrip("\r\n").split(sep)
                print(f"Format: Delimited ({'TSV' if sep == '\t' else 'CSV'})")
                print(f"Columns ({len(header)}): {header}")
                print("\nSample (first 3 rows):")
                for i in range(3):
                    line = f.readline()
                    if not line:
                        break
                    print(f"  Row {i+1}: {line.rstrip()[:200]}")
        except Exception as e:
            print(f"Error inspecting delimited file: {e}")
    else:
        print(f"Unknown extension: {ext}")
    print("=" * 60)


def scan_directory(data_dir: Path):
    """Scans the directory for candidate data files."""
    if not data_dir.exists():
        print(f"Directory not found: {data_dir.resolve()}")
        return

    files = [f for f in data_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not files:
        print(f"No files found in {data_dir.resolve()}")
        return

    print(f"\nFound {len(files)} files in {data_dir.resolve()}:\n")
    for f in sorted(files, key=lambda x: x.name):
        inspect_file(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Person 1 outputs and ground truth.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to file or directory to inspect (defaults to person2/data or workspace root)",
    )
    args = parser.parse_args()

    target_path = Path(args.path) if args.path else None
    if target_path is None:
        default_data = Path(__file__).resolve().parent.parent / "data"
        if default_data.exists() and any(default_data.iterdir()):
            target_path = default_data
        else:
            target_path = Path.cwd()

    if target_path.is_file():
        inspect_file(target_path)
    else:
        scan_directory(target_path)
