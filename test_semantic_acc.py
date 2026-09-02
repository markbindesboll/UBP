import torch
from collections import Counter

# =============================================================================
# Configuration
# =============================================================================
EMBEDDINGS_PATH = '/data/thingseeg2/ubp_exp/eeg_intra-subject_ubp_EEGProjectLayer_RN50/sub-08_seed0/test_embeddings.pt'
CATS_PATH = '/data/thingseeg2/images/categories/test_cat_list.pth'
SUPER_CATS_PATH = '/data/thingseeg2/images/categories/test_super_cat_list.pth'
MARGIN = 0.001  # Similarity difference threshold for an inconclusive weighted vote

# =============================================================================
# Helper function for Error Formatting
# =============================================================================
def format_wrong_guesses(error_counter, top_n=3):
    """Formats the top N incorrectly predicted categories into a readable string."""
    total_errors = sum(error_counter.values())
    if total_errors == 0:
        return "None (100% Accuracy!)"
    
    sorted_errors = error_counter.most_common(top_n)
    parts = []
    for cat, count in sorted_errors:
        pct = (count / total_errors) * 100
        parts.append(f"'{cat}' ({pct:.1f}%)")
        
    return ", ".join(parts)

# =============================================================================
# Helper function for Evaluation
# =============================================================================
def evaluate_retrieval(sim_matrix, true_labels, k, margin):
    n_queries = sim_matrix.shape[0]

    top1_correct = 0
    hard_correct = 0
    hard_inc = 0
    weight_correct = 0
    weight_inc = 0

    # Counters to track wrong predictions
    top1_errors = Counter()
    hard_errors = Counter()
    weight_errors = Counter()

    for i in range(n_queries):
        true_label = true_labels[i]

        # Get top-k similarities and indices
        top_k_vals, top_k_idx = torch.topk(sim_matrix[i], k)
        top_k_vals = top_k_vals.tolist()
        top_k_idx = top_k_idx.tolist()

        # Look up the semantic labels for the retrieved images
        top_k_labels = [true_labels[idx] for idx in top_k_idx]

        # ---------------------------------------------------------------------
        # 0. TOP-1 CATEGORY MATCH
        # ---------------------------------------------------------------------
        pred_top1 = top_k_labels[0]
        if pred_top1 == true_label:
            top1_correct += 1
        else:
            top1_errors[pred_top1] += 1

        # ---------------------------------------------------------------------
        # 1. HARD MAJORITY VOTE (Simple Counting)
        # ---------------------------------------------------------------------
        counts = Counter(top_k_labels)
        mc_hard = counts.most_common()

        # Tie-breaker logic (count of #1 == count of #2)
        if len(mc_hard) > 1 and mc_hard[0][1] == mc_hard[1][1]:
            hard_inc += 1
        else:
            pred_hard = mc_hard[0][0]
            if pred_hard == true_label:
                hard_correct += 1
            else:
                hard_errors[pred_hard] += 1

        # ---------------------------------------------------------------------
        # 2. WEIGHTED SIMILARITY VOTE (Summing Cosine Similarities)
        # ---------------------------------------------------------------------
        weighted_scores = {}
        for val, lbl in zip(top_k_vals, top_k_labels):
            weighted_scores[lbl] = weighted_scores.get(lbl, 0.0) + val

        # Sort categories by their accumulated similarity scores (descending)
        sorted_weights = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)

        # Inconclusive if difference between #1 and #2 is less than or equal to margin
        if len(sorted_weights) > 1 and (sorted_weights[0][1] - sorted_weights[1][1]) <= margin:
            weight_inc += 1
        else:
            pred_weight = sorted_weights[0][0]
            if pred_weight == true_label:
                weight_correct += 1
            else:
                weight_errors[pred_weight] += 1

    # Calculate percentages
    top1_acc = top1_correct / n_queries
    hard_acc = hard_correct / n_queries
    weight_acc = weight_correct / n_queries

    return (top1_acc, hard_acc, hard_inc, weight_acc, weight_inc,
            top1_errors, hard_errors, weight_errors)

# =============================================================================
# Execution
# =============================================================================
def main():
    # Load labels and embeddings
    cats = torch.load(CATS_PATH, map_location='cpu', weights_only=False)
    super_cats = torch.load(SUPER_CATS_PATH, map_location='cpu', weights_only=False)
    embeds = torch.load(EMBEDDINGS_PATH, map_location='cpu', weights_only=False)

    # Extract backbones for title formatting
    folder_name = EMBEDDINGS_PATH.split('/')[-3]
    parts = folder_name.split('_')
    brain_backbone = parts[-2] if len(parts) >= 2 else "UnknownBrain"
    vision_backbone = parts[-1] if len(parts) >= 1 else "UnknownVision"

    eeg_z = embeds['eeg_z']
    if eeg_z.dim() == 3:  # Squeeze reps dim if present
        eeg_z = eeg_z.mean(dim=0)

    img_z = embeds['img_z']

    # Normalize
    eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
    img_z = img_z / img_z.norm(dim=-1, keepdim=True)

    # Compute full 200x200 similarity matrix
    sim_matrix = eeg_z @ img_z.T

    # Evaluate for k=3 and k=7
    for name, labels in [("Standard Categories (test_cat)", cats), 
                         ("Super Categories (test_super_cat)", super_cats)]:

        # Calculate metrics (Top-1 will be the same for both k=3 and k=7 calls)
        (top1_acc, h_acc_3, h_inc_3, w_acc_3, w_inc_3,
         top1_errs, h_errs_3, w_errs_3) = evaluate_retrieval(sim_matrix, labels, k=3, margin=MARGIN)
         
        (_, h_acc_7, h_inc_7, w_acc_7, w_inc_7,
         _, h_errs_7, w_errs_7) = evaluate_retrieval(sim_matrix, labels, k=7, margin=MARGIN)

        # Print outputs
        print("=" * 90)
        print(f" {name} — Title: {brain_backbone} / {vision_backbone}")
        print("=" * 90)

        print(f"--- Top-1 Retrieval ---")
        print(f"  Category Match | Accuracy: {top1_acc*100:>5.2f}%")
        print(f"    -> Top Wrong : {format_wrong_guesses(top1_errs)}")

        print(f"\n--- Top-3 Retrieval ---")
        print(f"  Hard Vote      | Accuracy: {h_acc_3*100:>5.2f}% | Inconclusive: {h_inc_3:>3} / 200")
        print(f"    -> Top Wrong : {format_wrong_guesses(h_errs_3)}")
        print(f"  Weighted       | Accuracy: {w_acc_3*100:>5.2f}% | Inconclusive: {w_inc_3:>3} / 200 (Margin <= {MARGIN})")
        print(f"    -> Top Wrong : {format_wrong_guesses(w_errs_3)}")

        print(f"\n--- Top-7 Retrieval ---")
        print(f"  Hard Vote      | Accuracy: {h_acc_7*100:>5.2f}% | Inconclusive: {h_inc_7:>3} / 200")
        print(f"    -> Top Wrong : {format_wrong_guesses(h_errs_7)}")
        print(f"  Weighted       | Accuracy: {w_acc_7*100:>5.2f}% | Inconclusive: {w_inc_7:>3} / 200 (Margin <= {MARGIN})")
        print(f"    -> Top Wrong : {format_wrong_guesses(w_errs_7)}\n")

if __name__ == "__main__":
    main()