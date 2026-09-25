"""
Person 2 — Input Inspection Tool
Inspects the actual competition datasets (train/test TSV files) and Person 1 artifacts.
Extracts schema, column names, approximate row counts, file sizes, and sample records
without loading entire multi-million row datasets into memory.
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def count_lines_fast(file_path: Path) -> int:
    """Fast line count by reading binary chunks."""
    count = 0
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            count += chunk.count(b"\n")
    return count


def inspect_file(file_path: Path):
    """Inspects a single file using lightweight metadata or first few lines."""
    print("=" * 65)
    print(f"FILE: {file_path.name}")
    print(f"Path: {file_path.resolve()}")
    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    if size_gb >= 1.0:
        print(f"Size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
    else:
        print(f"Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

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
            
            sample_table = parquet_file.read_row_group(0).slice(0, 3)
            print("\nSample (first 3 rows):")
            for row in sample_table.to_pylist():
                print(f"  {row}")
        except Exception as e:
            print(f"Error inspecting Parquet file: {e}")

    elif ext in [".tsv", ".csv", ".txt"]:
        sep = "\t" if ext == ".tsv" else ","
        try:
            total_lines = count_lines_fast(file_path)
            print(f"Format: Delimited ({'TSV' if sep == '\t' else 'CSV'})")
            print(f"Total Rows: {max(0, total_lines - 1):,} (Lines: {total_lines:,})")

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline().rstrip("\r\n").split(sep)
                print(f"Columns ({len(header)}): {header}")
                print("\nSample (first 3 rows):")
                for i in range(3):
                    line = f.readline()
                    if not line:
                        break
                    clean_line = line.rstrip("\r\n").replace("\r", " ").replace("\t", "  |  ")
                    print(f"  Row {i+1}: {clean_line[:140]}")
        except Exception as e:
            print(f"Error inspecting delimited file: {e}")
    else:
        print(f"Unknown extension: {ext}")
    print("=" * 65)


def scan_directory(data_dir: Path):
    """Scans the directory for candidate data files."""
    if not data_dir.exists():
        print(f"Directory not found: {data_dir.resolve()}")
        return

    files = [f for f in data_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not files:
        print(f"No files found in {data_dir.resolve()}")
        return

    print(f"\nFound {len(files)} file(s) in {data_dir.resolve()}:\n")
    for f in sorted(files, key=lambda x: x.name):
        inspect_file(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect actual competition data and Person 1 outputs.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to file or directory to inspect (e.g. data/train or data/test)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    if args.path:
        target_path = Path(args.path)
    else:
        # Default to actual train data directory
        actual_train = project_root / "data" / "train"
        if actual_train.exists():
            target_path = actual_train
        else:
            target_path = Path.cwd()

    if target_path.is_file():
        inspect_file(target_path)
    elif target_path.is_dir():
        scan_directory(target_path)
    else:
        print(f"Path does not exist: {target_path}")
