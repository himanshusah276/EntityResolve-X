"""
Memory-efficient streaming data loader for large TSV files.
"""
import csv
from typing import Dict, Generator, Iterator, List, Optional, Tuple


def stream_tsv_records(
    file_path: str,
    limit: Optional[int] = None,
    filter_country: Optional[str] = None
) -> Generator[Dict[str, str], None, None]:
    """
    Stream records line-by-line from a TSV file to minimize memory overhead.
    """
    with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return
        
        count = 0
        for row in reader:
            if not row:
                continue
            # Map columns safely
            record = {header[i]: row[i] if i < len(row) else "" for i in range(len(header))}
            
            if filter_country and record.get("country") != filter_country:
                continue
                
            yield record
            count += 1
            if limit is not None and count >= limit:
                break


def load_ground_truth(file_path: str, limit: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Load ground truth mapping from source1_entity_id -> list of matched_entity_ids.
    """
    gt = {}
    with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        count = 0
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            matched_str = row[1] if len(row) > 1 else ""
            if matched_str.strip():
                gt[s1_id] = [m.strip() for m in matched_str.split(",") if m.strip()]
            else:
                gt[s1_id] = []
            count += 1
            if limit is not None and count >= limit:
                break
    return gt
