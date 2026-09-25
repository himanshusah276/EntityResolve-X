"""
Official evaluation metrics: macro-averaged Precision, Recall, and F0.5 per Source 1 entity.
"""
from typing import Dict, List, Set, Tuple


def calculate_entity_metrics(
    ground_truth_matches: Set[str],
    predicted_matches: Set[str]
) -> Tuple[float, float, float]:
    """
    Calculate Precision, Recall, and F0.5 for a single Source 1 entity.
    F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
    """
    # Case 1: Ground truth is empty (true singleton)
    if not ground_truth_matches:
        if not predicted_matches:
            # Correctly predicted no match
            return 1.0, 1.0, 1.0
        else:
            # Predicted false matches for a singleton
            return 0.0, 0.0, 0.0

    # Case 2: Ground truth has matches, but predicted is empty
    if not predicted_matches:
        return 0.0, 0.0, 0.0

    # Case 3: Both non-empty
    tp = len(ground_truth_matches & predicted_matches)
    fp = len(predicted_matches - ground_truth_matches)
    fn = len(ground_truth_matches - predicted_matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0 or (0.25 * precision + recall) == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

    return precision, recall, f05


def evaluate_macro_metrics(
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Macro-average metrics across all Source 1 entities in ground truth.
    """
    total_p = 0.0
    total_r = 0.0
    total_f05 = 0.0
    n = len(ground_truth)

    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f05": 0.0, "count": 0}

    singletons_total = 0
    singletons_correct = 0

    for s1_id, gt_list in ground_truth.items():
        gt_set = set(gt_list)
        pred_set = set(predictions.get(s1_id, []))
        
        p, r, f05 = calculate_entity_metrics(gt_set, pred_set)
        total_p += p
        total_r += r
        total_f05 += f05

        if not gt_set:
            singletons_total += 1
            if not pred_set:
                singletons_correct += 1

    singleton_acc = singletons_correct / singletons_total if singletons_total > 0 else 1.0

    return {
        "precision": total_p / n,
        "recall": total_r / n,
        "f05": total_f05 / n,
        "singleton_accuracy": singleton_acc,
        "total_entities": n,
        "singletons_total": singletons_total,
    }


def evaluate_candidate_recall(
    ground_truth: Dict[str, List[str]],
    candidates: Dict[str, List[str]]
) -> float:
    """
    Calculate upper-bound recall of true matches present in generated candidate pairs.
    """
    total_true_matches = 0
    recalled_matches = 0

    for s1_id, gt_list in ground_truth.items():
        cand_set = set(candidates.get(s1_id, []))
        for m in gt_list:
            total_true_matches += 1
            if m in cand_set:
                recalled_matches += 1

    return recalled_matches / total_true_matches if total_true_matches > 0 else 1.0
