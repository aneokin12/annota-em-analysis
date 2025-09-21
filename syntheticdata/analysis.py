import numpy as np
from metrics import calculate_custom_metrics, print_custom_metrics_summary

def calculate_student_accuracy(tensor, ground_truth_labels, student_idx, binary_schema=False):
    """
    Calculate accuracy for a specific student by comparing their annotations to ground truth.
    
    Args:
        tensor: N x J x K tensor (tasks x labels x students)
        ground_truth_labels: 1D array of length N, true label for each task
        student_idx: index of the student to evaluate
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing accuracy metrics for the student
    """
    N, J, K = tensor.shape
    
    # Get student's annotations (convert from one-hot to class labels)
    student_annotations = np.argmax(tensor[:, :, student_idx], axis=1)
    
    # Prepare ground truth for comparison
    if binary_schema:
        # Treat 2 as 1 for binary schema
        gt_for_comparison = np.where(ground_truth_labels == 2, 1, ground_truth_labels)
    else:
        gt_for_comparison = np.where(ground_truth_labels == 2, 1, ground_truth_labels)
    
    # Calculate basic accuracy
    correct_predictions = (student_annotations == gt_for_comparison).sum()
    total_predictions = len(student_annotations)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Calculate class-specific metrics
    # For binary schema: 0 vs 1
    # For three-class schema: 0 vs (1 or 2)
    if binary_schema:
        # Binary classification: 0 vs 1
        true_positives = ((student_annotations == 1) & (gt_for_comparison == 1)).sum()
        true_negatives = ((student_annotations == 0) & (gt_for_comparison == 0)).sum()
        false_positives = ((student_annotations == 1) & (gt_for_comparison == 0)).sum()
        false_negatives = ((student_annotations == 0) & (gt_for_comparison == 1)).sum()
        
        # Calculate precision and recall for positive class (1)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    else:
        # Three-class classification: 0 vs (1 or 2)
        # Treat 1 and 2 as "relevant" class
        true_positives = ((student_annotations == 1) & ((gt_for_comparison == 1) | (gt_for_comparison == 2))).sum()
        true_negatives = ((student_annotations == 0) & (gt_for_comparison == 0)).sum()
        false_positives = ((student_annotations == 1) & (gt_for_comparison == 0)).sum()
        false_negatives = ((student_annotations == 0) & ((gt_for_comparison == 1) | (gt_for_comparison == 2))).sum()
        
        # Calculate precision and recall for positive class (relevant)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_predictions': total_predictions
    }

def calculate_task_accuracy(tensor, ground_truth_labels, task_idx, binary_schema=False):
    """
    Calculate accuracy for a specific task by comparing all student annotations to ground truth.
    
    Args:
        tensor: N x J x K tensor (tasks x labels x students)
        ground_truth_labels: 1D array of length N, true label for each task
        task_idx: index of the task to evaluate
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing accuracy metrics for the task
    """
    N, J, K = tensor.shape
    
    # Get all student annotations for this task (convert from one-hot to class labels)
    task_annotations = np.argmax(tensor[task_idx, :, :], axis=0)  # Shape: (K,)
    
    # Get ground truth for this task
    true_label = ground_truth_labels[task_idx]
    
    # Prepare ground truth for comparison
    if binary_schema:
        # Treat 2 as 1 for binary schema
        gt_for_comparison = 1 if true_label == 2 else true_label
    else:
        gt_for_comparison = true_label
    
    # Calculate basic accuracy
    correct_predictions = (task_annotations == gt_for_comparison).sum()
    total_predictions = len(task_annotations)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Calculate class-specific metrics
    if binary_schema:
        # Binary classification: 0 vs 1
        true_positives = ((task_annotations == 1) & (gt_for_comparison == 1)).sum()
        true_negatives = ((task_annotations == 0) & (gt_for_comparison == 0)).sum()
        false_positives = ((task_annotations == 1) & (gt_for_comparison == 0)).sum()
        false_negatives = ((task_annotations == 0) & (gt_for_comparison == 1)).sum()
        
        # Calculate precision and recall for positive class (1)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    else:
        # Three-class classification: 0 vs (1 or 2)
        # Treat 1 and 2 as "relevant" class
        true_positives = ((task_annotations == 1) & ((gt_for_comparison == 1) | (gt_for_comparison == 2))).sum()
        true_negatives = ((task_annotations == 0) & (gt_for_comparison == 0)).sum()
        false_positives = ((task_annotations == 1) & (gt_for_comparison == 0)).sum()
        false_negatives = ((task_annotations == 0) & ((gt_for_comparison == 1) | (gt_for_comparison == 2))).sum()
        
        # Calculate precision and recall for positive class (relevant)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'ground_truth': true_label,
        'gt_for_comparison': gt_for_comparison
    }

def analyze_student_improvements(initial_tensor, final_tensor, ground_truth_labels, binary_schema=False):
    """
    Analyze how individual student accuracy improves from initial to final tensor.
    
    Args:
        initial_tensor: Initial N x J x K tensor
        final_tensor: Final N x J x K tensor after synthetic edits
        ground_truth_labels: 1D array of length N, true label for each task
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing improvement analysis for all students
    """
    N, J, K = initial_tensor.shape
    
    student_improvements = {}
    all_initial_accuracies = []
    all_final_accuracies = []
    all_improvements = []
    
    print(f"\nAnalyzing student-level accuracy improvements for {K} students...")
    print(f"Schema: {'Binary (2→1)' if binary_schema else 'Three-class (0,1,2)'}")
    print("-" * 80)
    
    for student_idx in range(K):
        # Calculate initial accuracy
        initial_metrics = calculate_student_accuracy(initial_tensor, ground_truth_labels, student_idx, binary_schema)
        
        # Calculate final accuracy
        final_metrics = calculate_student_accuracy(final_tensor, ground_truth_labels, student_idx, binary_schema)
        
        # Calculate improvements
        accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
        precision_improvement = final_metrics['precision'] - initial_metrics['precision']
        recall_improvement = final_metrics['recall'] - initial_metrics['recall']
        f1_improvement = final_metrics['f1'] - initial_metrics['f1']
        
        student_improvements[student_idx] = {
            'initial': initial_metrics,
            'final': {
                'accuracy': final_metrics['accuracy'],
                'precision': final_metrics['precision'],
                'recall': final_metrics['recall'],
                'f1': final_metrics['f1']
            },
            'improvements': {
                'accuracy': accuracy_improvement,
                'precision': precision_improvement,
                'recall': recall_improvement,
                'f1': f1_improvement
            }
        }
        
        all_initial_accuracies.append(initial_metrics['accuracy'])
        all_final_accuracies.append(final_metrics['accuracy'])
        all_improvements.append(accuracy_improvement)
        
        # Print individual student results
        print(f"Student {student_idx:2d}: "
              f"Initial Acc={initial_metrics['accuracy']:.3f}, "
              f"Final Acc={final_metrics['accuracy']:.3f}, "
              f"Improvement={accuracy_improvement:+.3f}")
        print(f"           "
              f"Precision: {initial_metrics['precision']:.3f}→{final_metrics['precision']:.3f} ({precision_improvement:+.3f}), "
              f"Recall: {initial_metrics['recall']:.3f}→{final_metrics['recall']:.3f} ({recall_improvement:+.3f})")
    
    # Calculate summary statistics
    summary_stats = {
        'mean_initial_accuracy': np.mean(all_initial_accuracies),
        'mean_final_accuracy': np.mean(all_final_accuracies),
        'mean_improvement': np.mean(all_improvements),
        'std_improvement': np.std(all_improvements),
        'students_improved': sum(1 for imp in all_improvements if imp > 0),
        'students_declined': sum(1 for imp in all_improvements if imp < 0),
        'students_unchanged': sum(1 for imp in all_improvements if imp == 0),
        'max_improvement': max(all_improvements),
        'min_improvement': min(all_improvements)
    }
    
    print("\n" + "=" * 80)
    print("STUDENT ACCURACY IMPROVEMENT SUMMARY")
    print("=" * 80)
    print(f"Mean Initial Accuracy: {summary_stats['mean_initial_accuracy']:.3f}")
    print(f"Mean Final Accuracy:   {summary_stats['mean_final_accuracy']:.3f}")
    print(f"Mean Improvement:      {summary_stats['mean_improvement']:+.3f} ± {summary_stats['std_improvement']:.3f}")
    print(f"Students Improved:     {summary_stats['students_improved']}/{K} ({summary_stats['students_improved']/K*100:.1f}%)")
    print(f"Students Declined:     {summary_stats['students_declined']}/{K} ({summary_stats['students_declined']/K*100:.1f}%)")
    print(f"Students Unchanged:    {summary_stats['students_unchanged']}/{K} ({summary_stats['students_unchanged']/K*100:.1f}%)")
    print(f"Max Improvement:       {summary_stats['max_improvement']:+.3f}")
    print(f"Min Improvement:       {summary_stats['min_improvement']:+.3f}")
    
    # Calculate and display precision and recall improvements
    all_precision_improvements = [student_improvements[i]['improvements']['precision'] for i in range(K)]
    all_recall_improvements = [student_improvements[i]['improvements']['recall'] for i in range(K)]
    
    mean_precision_improvement = np.mean(all_precision_improvements)
    std_precision_improvement = np.std(all_precision_improvements)
    mean_recall_improvement = np.mean(all_recall_improvements)
    std_recall_improvement = np.std(all_recall_improvements)
    
    print(f"\nPrecision Improvements:")
    print(f"Mean Precision Improvement: {mean_precision_improvement:+.3f} ± {std_precision_improvement:.3f}")
    print(f"Students Improved Precision: {sum(1 for imp in all_precision_improvements if imp > 0)}/{K} ({sum(1 for imp in all_precision_improvements if imp > 0)/K*100:.1f}%)")
    
    print(f"\nRecall Improvements:")
    print(f"Mean Recall Improvement: {mean_recall_improvement:+.3f} ± {std_recall_improvement:.3f}")
    print(f"Students Improved Recall: {sum(1 for imp in all_recall_improvements if imp > 0)}/{K} ({sum(1 for imp in all_recall_improvements if imp > 0)/K*100:.1f}%)")
    
    return student_improvements, summary_stats

def analyze_student_improvements_multiple_trials(initial_tensor, trial_results, ground_truth_labels, binary_schema=False):
    """
    Analyze student improvements across multiple trials, averaging final accuracies.
    
    Args:
        initial_tensor: Initial N x J x K tensor
        trial_results: List of (final_tensor, round_history, total_rounds, metrics_history, processed_pairs) tuples
        ground_truth_labels: 1D array of length N, true label for each task
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing averaged improvement analysis for all students
    """
    N, J, K = initial_tensor.shape
    num_trials = len(trial_results)
    
    print(f"\nAnalyzing student-level accuracy improvements across {num_trials} trials...")
    print(f"Schema: {'Binary (2→1)' if binary_schema else 'Three-class (0,1,2)'}")
    print("-" * 80)
    
    # Collect final accuracies for each student across all trials
    student_final_accuracies = {i: [] for i in range(K)}
    student_final_precisions = {i: [] for i in range(K)}
    student_final_recalls = {i: [] for i in range(K)}
    student_final_f1s = {i: [] for i in range(K)}
    
    for trial_idx, (final_tensor, _, _, _) in enumerate(trial_results):
        for student_idx in range(K):
            final_metrics = calculate_student_accuracy(final_tensor, ground_truth_labels, student_idx, binary_schema)
            student_final_accuracies[student_idx].append(final_metrics['accuracy'])
            student_final_precisions[student_idx].append(final_metrics['precision'])
            student_final_recalls[student_idx].append(final_metrics['recall'])
            student_final_f1s[student_idx].append(final_metrics['f1'])
    
    # Calculate averaged improvements
    student_improvements = {}
    all_initial_accuracies = []
    all_avg_final_accuracies = []
    all_avg_improvements = []
    
    for student_idx in range(K):
        # Initial accuracy (same for all trials)
        initial_metrics = calculate_student_accuracy(initial_tensor, ground_truth_labels, student_idx, binary_schema)
        
        # Average final metrics across trials
        avg_final_accuracy = np.mean(student_final_accuracies[student_idx])
        avg_final_precision = np.mean(student_final_precisions[student_idx])
        avg_final_recall = np.mean(student_final_recalls[student_idx])
        avg_final_f1 = np.mean(student_final_f1s[student_idx])
        
        # Calculate improvements
        accuracy_improvement = avg_final_accuracy - initial_metrics['accuracy']
        precision_improvement = avg_final_precision - initial_metrics['precision']
        recall_improvement = avg_final_recall - initial_metrics['recall']
        f1_improvement = avg_final_f1 - initial_metrics['f1']
        
        student_improvements[student_idx] = {
            'initial': initial_metrics,
            'final': {
                'accuracy': avg_final_accuracy,
                'precision': avg_final_precision,
                'recall': avg_final_recall,
                'f1': avg_final_f1
            },
            'improvements': {
                'accuracy': accuracy_improvement,
                'precision': precision_improvement,
                'recall': recall_improvement,
                'f1': f1_improvement
            },
            'trial_data': {
                'final_accuracies': student_final_accuracies[student_idx],
                'final_precisions': student_final_precisions[student_idx],
                'final_recalls': student_final_recalls[student_idx],
                'final_f1s': student_final_f1s[student_idx]
            }
        }
        
        all_initial_accuracies.append(initial_metrics['accuracy'])
        all_avg_final_accuracies.append(avg_final_accuracy)
        all_avg_improvements.append(accuracy_improvement)
        
        # Print individual student results
        std_accuracy = np.std(student_final_accuracies[student_idx])
        print(f"Student {student_idx:2d}: "
              f"Initial Acc={initial_metrics['accuracy']:.3f}, "
              f"Final Acc={avg_final_accuracy:.3f}±{std_accuracy:.3f}, "
              f"Improvement={accuracy_improvement:+.3f}")
    
    # Calculate summary statistics
    summary_stats = {
        'mean_initial_accuracy': np.mean(all_initial_accuracies),
        'mean_final_accuracy': np.mean(all_avg_final_accuracies),
        'mean_improvement': np.mean(all_avg_improvements),
        'std_improvement': np.std(all_avg_improvements),
        'students_improved': sum(1 for imp in all_avg_improvements if imp > 0),
        'students_declined': sum(1 for imp in all_avg_improvements if imp < 0),
        'students_unchanged': sum(1 for imp in all_avg_improvements if imp == 0),
        'max_improvement': max(all_avg_improvements),
        'min_improvement': min(all_avg_improvements)
    }
    
    print("\n" + "=" * 80)
    print("STUDENT ACCURACY IMPROVEMENT SUMMARY (AVERAGED ACROSS TRIALS)")
    print("=" * 80)
    print(f"Mean Initial Accuracy: {summary_stats['mean_initial_accuracy']:.3f}")
    print(f"Mean Final Accuracy:   {summary_stats['mean_final_accuracy']:.3f}")
    print(f"Mean Improvement:      {summary_stats['mean_improvement']:+.3f} ± {summary_stats['std_improvement']:.3f}")
    print(f"Students Improved:     {summary_stats['students_improved']}/{K} ({summary_stats['students_improved']/K*100:.1f}%)")
    print(f"Students Declined:     {summary_stats['students_declined']}/{K} ({summary_stats['students_declined']/K*100:.1f}%)")
    print(f"Students Unchanged:    {summary_stats['students_unchanged']}/{K} ({summary_stats['students_unchanged']/K*100:.1f}%)")
    print(f"Max Improvement:       {summary_stats['max_improvement']:+.3f}")
    print(f"Min Improvement:       {summary_stats['min_improvement']:+.3f}")
    
    return student_improvements, summary_stats

def analyze_task_improvements(initial_tensor, final_tensor, ground_truth_labels, binary_schema=False):
    """
    Analyze how task-level accuracy improves from initial to final tensor.
    
    Args:
        initial_tensor: Initial N x J x K tensor
        final_tensor: Final N x J x K tensor after synthetic edits
        ground_truth_labels: 1D array of length N, true label for each task
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing improvement analysis for all tasks
    """
    N, J, K = initial_tensor.shape
    
    task_improvements = {}
    all_initial_accuracies = []
    all_final_accuracies = []
    all_improvements = []
    
    print(f"\nAnalyzing task-level accuracy improvements for {N} tasks...")
    print(f"Schema: {'Binary (2→1)' if binary_schema else 'Three-class (0,1,2)'}")
    print("-" * 80)
    
    for task_idx in range(N):
        # Calculate initial accuracy
        initial_metrics = calculate_task_accuracy(initial_tensor, ground_truth_labels, task_idx, binary_schema)
        
        # Calculate final accuracy
        final_metrics = calculate_task_accuracy(final_tensor, ground_truth_labels, task_idx, binary_schema)
        
        # Calculate improvements
        accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
        precision_improvement = final_metrics['precision'] - initial_metrics['precision']
        recall_improvement = final_metrics['recall'] - initial_metrics['recall']
        f1_improvement = final_metrics['f1'] - initial_metrics['f1']
        
        task_improvements[task_idx] = {
            'initial': initial_metrics,
            'final': final_metrics,
            'improvements': {
                'accuracy': accuracy_improvement,
                'precision': precision_improvement,
                'recall': recall_improvement,
                'f1': f1_improvement
            }
        }
        
        all_initial_accuracies.append(initial_metrics['accuracy'])
        all_final_accuracies.append(final_metrics['accuracy'])
        all_improvements.append(accuracy_improvement)
        
        # Print individual task results (only first 10 and last 10 for brevity)
        if task_idx < 10 or task_idx >= N - 10:
            print(f"Task {task_idx:3d}: "
                  f"Initial Acc={initial_metrics['accuracy']:.3f} "
                  f"({initial_metrics['correct_predictions']:2d}/{K}), "
                  f"Final Acc={final_metrics['accuracy']:.3f} "
                  f"({final_metrics['correct_predictions']:2d}/{K}), "
                  f"Improvement={accuracy_improvement:+.3f}")
        elif task_idx == 10:
            print("...")
    
    # Calculate summary statistics
    summary_stats = {
        'mean_initial_accuracy': np.mean(all_initial_accuracies),
        'mean_final_accuracy': np.mean(all_final_accuracies),
        'mean_improvement': np.mean(all_improvements),
        'std_improvement': np.std(all_improvements),
        'tasks_improved': sum(1 for imp in all_improvements if imp > 0),
        'tasks_declined': sum(1 for imp in all_improvements if imp < 0),
        'tasks_unchanged': sum(1 for imp in all_improvements if imp == 0),
        'max_improvement': max(all_improvements),
        'min_improvement': min(all_improvements)
    }
    
    print("\n" + "=" * 80)
    print("TASK ACCURACY IMPROVEMENT SUMMARY")
    print("=" * 80)
    print(f"Mean Initial Accuracy: {summary_stats['mean_initial_accuracy']:.3f}")
    print(f"Mean Final Accuracy:   {summary_stats['mean_final_accuracy']:.3f}")
    print(f"Mean Improvement:      {summary_stats['mean_improvement']:+.3f} ± {summary_stats['std_improvement']:.3f}")
    print(f"Tasks Improved:        {summary_stats['tasks_improved']}/{N} ({summary_stats['tasks_improved']/N*100:.1f}%)")
    print(f"Tasks Declined:        {summary_stats['tasks_declined']}/{N} ({summary_stats['tasks_declined']/N*100:.1f}%)")
    print(f"Tasks Unchanged:       {summary_stats['tasks_unchanged']}/{N} ({summary_stats['tasks_unchanged']/N*100:.1f}%)")
    print(f"Max Improvement:       {summary_stats['max_improvement']:+.3f}")
    print(f"Min Improvement:       {summary_stats['min_improvement']:+.3f}")
    
    return task_improvements, summary_stats

def analyze_task_improvements_multiple_trials(initial_tensor, trial_results, ground_truth_labels, binary_schema=False):
    """
    Analyze task-level accuracy improvements across multiple trials, averaging final accuracies.
    
    Args:
        initial_tensor: Initial N x J x K tensor
        trial_results: List of (final_tensor, round_history, total_rounds, metrics_history, processed_pairs) tuples
        ground_truth_labels: 1D array of length N, true label for each task
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing averaged improvement analysis for all tasks
    """
    N, J, K = initial_tensor.shape
    num_trials = len(trial_results)
    
    print(f"\nAnalyzing task-level accuracy improvements across {num_trials} trials...")
    print(f"Schema: {'Binary (2→1)' if binary_schema else 'Three-class (0,1,2)'}")
    print("-" * 80)
    
    # Collect final accuracies for each task across all trials
    task_final_accuracies = {i: [] for i in range(N)}
    task_final_precisions = {i: [] for i in range(N)}
    task_final_recalls = {i: [] for i in range(N)}
    
    for trial_idx, (final_tensor, _, _, _) in enumerate(trial_results):
        for task_idx in range(N):
            final_metrics = calculate_task_accuracy(final_tensor, ground_truth_labels, task_idx, binary_schema)
            task_final_accuracies[task_idx].append(final_metrics['accuracy'])
            task_final_precisions[task_idx].append(final_metrics['precision'])
            task_final_recalls[task_idx].append(final_metrics['recall'])
    
    # Calculate averaged improvements
    task_improvements = {}
    all_initial_accuracies = []
    all_avg_final_accuracies = []
    all_avg_improvements = []
    
    for task_idx in range(N):
        # Initial accuracy (same for all trials)
        initial_metrics = calculate_task_accuracy(initial_tensor, ground_truth_labels, task_idx, binary_schema)
        
        # Average final metrics across trials
        avg_final_accuracy = np.mean(task_final_accuracies[task_idx])
        avg_final_precision = np.mean(task_final_precisions[task_idx])
        avg_final_recall = np.mean(task_final_recalls[task_idx])
        
        # Calculate improvements
        accuracy_improvement = avg_final_accuracy - initial_metrics['accuracy']
        precision_improvement = avg_final_precision - initial_metrics['precision']
        recall_improvement = avg_final_recall - initial_metrics['recall']
        
        task_improvements[task_idx] = {
            'initial': initial_metrics,
            'final': {
                'accuracy': avg_final_accuracy,
                'precision': avg_final_precision,
                'recall': avg_final_recall
            },
            'improvements': {
                'accuracy': accuracy_improvement,
                'precision': precision_improvement,
                'recall': recall_improvement
            },
            'trial_data': {
                'final_accuracies': task_final_accuracies[task_idx],
                'final_precisions': task_final_precisions[task_idx],
                'final_recalls': task_final_recalls[task_idx]
            }
        }
        
        all_initial_accuracies.append(initial_metrics['accuracy'])
        all_avg_final_accuracies.append(avg_final_accuracy)
        all_avg_improvements.append(accuracy_improvement)
        
        # Print individual task results (only first 10 and last 10 for brevity)
        if task_idx < 10 or task_idx >= N - 10:
            std_accuracy = np.std(task_final_accuracies[task_idx])
            print(f"Task {task_idx:3d}: "
                  f"Initial Acc={initial_metrics['accuracy']:.3f} "
                  f"({initial_metrics['correct_predictions']:2d}/{K}), "
                  f"Final Acc={avg_final_accuracy:.3f}±{std_accuracy:.3f} "
                  f"({final_metrics['correct_predictions']:2d}/{K}), "
                  f"Improvement={accuracy_improvement:+.3f}")
        elif task_idx == 10:
            print("...")
    
    # Calculate summary statistics
    summary_stats = {
        'mean_initial_accuracy': np.mean(all_initial_accuracies),
        'mean_final_accuracy': np.mean(all_avg_final_accuracies),
        'mean_improvement': np.mean(all_avg_improvements),
        'std_improvement': np.std(all_avg_improvements),
        'tasks_improved': sum(1 for imp in all_avg_improvements if imp > 0),
        'tasks_declined': sum(1 for imp in all_avg_improvements if imp < 0),
        'tasks_unchanged': sum(1 for imp in all_avg_improvements if imp == 0),
        'max_improvement': max(all_avg_improvements),
        'min_improvement': min(all_avg_improvements)
    }
    
    print("\n" + "=" * 80)
    print("TASK ACCURACY IMPROVEMENT SUMMARY (AVERAGED ACROSS TRIALS)")
    print("=" * 80)
    print(f"Mean Initial Accuracy: {summary_stats['mean_initial_accuracy']:.3f}")
    print(f"Mean Final Accuracy:   {summary_stats['mean_final_accuracy']:.3f}")
    print(f"Mean Improvement:      {summary_stats['mean_improvement']:+.3f} ± {summary_stats['std_improvement']:.3f}")
    print(f"Tasks Improved:        {summary_stats['tasks_improved']}/{N} ({summary_stats['tasks_improved']/N*100:.1f}%)")
    print(f"Tasks Declined:        {summary_stats['tasks_declined']}/{N} ({summary_stats['tasks_declined']/N*100:.1f}%)")
    print(f"Tasks Unchanged:       {summary_stats['tasks_unchanged']}/{N} ({summary_stats['tasks_unchanged']/N*100:.1f}%)")
    print(f"Max Improvement:       {summary_stats['max_improvement']:+.3f}")
    print(f"Min Improvement:       {summary_stats['min_improvement']:+.3f}")
    
    return task_improvements, summary_stats
