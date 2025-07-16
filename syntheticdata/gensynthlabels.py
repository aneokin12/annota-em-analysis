import numpy as np

def generate_synthetic_labels(tensor, seed=None, p_relevant=0.4, noise_level=0.1, relevance_threshold=0.4):
    """
    Generate a synthetic 'true' label for each task in the tensor.
    Counts a task as relevant if at least 30% of students annotated it as relevant.
    Args:
        tensor: N x J x K tensor (tasks x labels x students)
        seed: Optional random seed for reproducibility
        p_relevant: Probability of a task being relevant (default 0.2 = 20% relevant, 80% not relevant)
        noise_level: Probability of flipping the majority vote (default 0.05 = 5% noise)
        relevance_threshold: Fraction of students that must annotate as relevant to count as majority (default 0.3)
    Returns:
        labels: 1D numpy array of length N, each entry 0 or 1
    """
    N, J, K = tensor.shape
    rng = np.random.default_rng(seed)
    
    # Calculate how many students annotated each task as relevant (label 1)
    relevant_annotations_per_task = np.sum(tensor[:, 1, :], axis=1)  # Sum over students for label 1
    relevance_ratio = relevant_annotations_per_task / K
    
    # Initialize labels based on relevance threshold
    labels = np.zeros(N, dtype=int)
    for task_idx in range(N):
        if relevance_ratio[task_idx] >= relevance_threshold:
            # At least 30% of students found this relevant
            labels[task_idx] = 1
        else:
            # Less than 30% of students found this relevant
            labels[task_idx] = 0
    
    # Add noise: flip some labels randomly (reduced noise level)
    noise_mask = rng.random(N) < noise_level
    labels[noise_mask] = 1 - labels[noise_mask]
    
    # Calculate current ratio and target ratio
    current_relevant_ratio = np.mean(labels)
    target_relevant_ratio = p_relevant
    
    # Only adjust if we're significantly far from target (increased tolerance)
    if abs(current_relevant_ratio - target_relevant_ratio) > 0.08:  # 8% tolerance
        if current_relevant_ratio > target_relevant_ratio:
            # Too many relevant, flip some relevant to not relevant
            relevant_indices = np.where(labels == 1)[0]
            num_to_flip = int((current_relevant_ratio - target_relevant_ratio) * N)
            if len(relevant_indices) > num_to_flip:
                flip_indices = rng.choice(relevant_indices, num_to_flip, replace=False)
                labels[flip_indices] = 0
        else:
            # Too few relevant, flip some not relevant to relevant
            not_relevant_indices = np.where(labels == 0)[0]
            num_to_flip = int((target_relevant_ratio - current_relevant_ratio) * N)
            if len(not_relevant_indices) > num_to_flip:
                flip_indices = rng.choice(not_relevant_indices, num_to_flip, replace=False)
                labels[flip_indices] = 1
    
    return labels

if __name__ == "__main__":
    import sys
    tensor = np.load(sys.argv[1]) if len(sys.argv) > 1 else np.load('hint_tensor.npy')
    labels = generate_synthetic_labels(tensor)
    print("Synthetic labels:", labels)
    print("Label counts:", np.bincount(labels))
    print(f"Percentage relevant: {np.mean(labels)*100:.1f}%")
    
    # Show comparison with old majority vote
    majority_votes = np.argmax(np.sum(tensor, axis=2), axis=1)
    print(f"Old majority vote percentage relevant: {np.mean(majority_votes)*100:.1f}%")
    print(f"Agreement between synthetic and old majority: {np.mean(labels == majority_votes)*100:.1f}%")
    
    # Show new relevance statistics
    relevant_annotations_per_task = np.sum(tensor[:, 1, :], axis=1)
    relevance_ratio = relevant_annotations_per_task / tensor.shape[2]
    print(f"Average relevance ratio per task: {np.mean(relevance_ratio)*100:.1f}%")
    print(f"Tasks with >=30% students finding relevant: {np.sum(relevance_ratio >= 0.3)} out of {len(relevance_ratio)}")
    print(f"Tasks with <30% students finding relevant: {np.sum(relevance_ratio < 0.3)} out of {len(relevance_ratio)}")
    
    # Show participation statistics
    student_annotations_per_task = np.sum(tensor, axis=(1,2))
    participation_ratio = student_annotations_per_task / tensor.shape[2]
    print(f"Average student annotations per task: {np.mean(student_annotations_per_task):.1f}")
    print(f"Average participation ratio: {np.mean(participation_ratio)*100:.1f}%") 