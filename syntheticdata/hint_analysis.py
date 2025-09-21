import numpy as np
from model import dawid_skene

def get_em_predictions(tensor):
    """Run Dawid-Skene EM algorithm to get predictions for each task."""
    predictions = dawid_skene(tensor)
    return predictions

def analyze_hint_quality(tensor, ground_truth_labels, binary_schema=False):
    """
    Analyze the quality of EM-generated hints by checking how many were correct.
    
    Args:
        tensor: N x J x K tensor (tasks x labels x students)
        ground_truth_labels: 1D array of length N, true label for each task
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing hint quality analysis
    """
    N, J, K = tensor.shape
    
    # Get EM predictions
    em_predictions = get_em_predictions(tensor)
    
    # Prepare ground truth for comparison
    if binary_schema:
        gt_for_comparison = np.where(ground_truth_labels == 2, 1, ground_truth_labels)
    else:
        gt_for_comparison = ground_truth_labels
    
    # Categorize disagreements and check hint quality
    missed_relevant_hints = []
    irrelevant_disagreement_hints = []
    other_disagreements = []
    
    for student_idx in range(K):
        for task_idx in range(N):
            hint = em_predictions[task_idx]
            true_label = gt_for_comparison[task_idx]
            student_annotation = np.argmax(tensor[task_idx, :, student_idx])
            
            # Check if there's a disagreement
            if student_annotation != hint:
                hint_correct = (hint == true_label)
                
                if student_annotation == 0 and hint == 1:
                    # Missed relevant: student said 0, hint says 1
                    missed_relevant_hints.append({
                        'student_idx': student_idx,
                        'task_idx': task_idx,
                        'hint': hint,
                        'true_label': true_label,
                        'student_annotation': student_annotation,
                        'hint_correct': hint_correct
                    })
                elif student_annotation == 1 and hint == 0:
                    # Irrelevant disagreement: student said 1, hint says 0
                    irrelevant_disagreement_hints.append({
                        'student_idx': student_idx,
                        'task_idx': task_idx,
                        'hint': hint,
                        'true_label': true_label,
                        'student_annotation': student_annotation,
                        'hint_correct': hint_correct
                    })
                else:
                    # Other disagreements
                    other_disagreements.append({
                        'student_idx': student_idx,
                        'task_idx': task_idx,
                        'hint': hint,
                        'true_label': true_label,
                        'student_annotation': student_annotation,
                        'hint_correct': hint_correct
                    })
    
    # Analyze hint quality
    def analyze_hint_category(hints, category_name):
        if not hints:
            return {
                'count': 0,
                'correct_count': 0,
                'accuracy': 0.0,
                'hints': []
            }
        
        correct_count = sum(1 for h in hints if h['hint_correct'])
        accuracy = correct_count / len(hints) if hints else 0.0
        
        return {
            'count': len(hints),
            'correct_count': correct_count,
            'accuracy': accuracy,
            'hints': hints
        }
    
    missed_relevant_analysis = analyze_hint_category(missed_relevant_hints, 'missed_relevant')
    irrelevant_analysis = analyze_hint_category(irrelevant_disagreement_hints, 'irrelevant_disagreement')
    other_analysis = analyze_hint_category(other_disagreements, 'other')
    
    # Overall analysis
    total_disagreements = len(missed_relevant_hints) + len(irrelevant_disagreement_hints) + len(other_disagreements)
    total_correct_hints = missed_relevant_analysis['correct_count'] + irrelevant_analysis['correct_count'] + other_analysis['correct_count']
    overall_accuracy = total_correct_hints / total_disagreements if total_disagreements > 0 else 0.0
    
    analysis = {
        'missed_relevant': missed_relevant_analysis,
        'irrelevant_disagreement': irrelevant_analysis,
        'other': other_analysis,
        'overall': {
            'total_disagreements': total_disagreements,
            'total_correct_hints': total_correct_hints,
            'overall_accuracy': overall_accuracy
        }
    }
    
    # Print results
    print(f"\n=== HINT QUALITY ANALYSIS ===")
    print(f"Schema: {'Binary (2→1)' if binary_schema else 'Three-class (0,1,2)'}")
    print(f"Total disagreements: {total_disagreements}")
    print(f"Overall hint accuracy: {overall_accuracy:.3f} ({total_correct_hints}/{total_disagreements})")
    print()
    
    print(f"Missed Relevant Hints:")
    print(f"  Count: {missed_relevant_analysis['count']}")
    print(f"  Correct: {missed_relevant_analysis['correct_count']}")
    print(f"  Accuracy: {missed_relevant_analysis['accuracy']:.3f}")
    
    print(f"\nIrrelevant Disagreement Hints:")
    print(f"  Count: {irrelevant_analysis['count']}")
    print(f"  Correct: {irrelevant_analysis['correct_count']}")
    print(f"  Accuracy: {irrelevant_analysis['accuracy']:.3f}")
    
    if other_analysis['count'] > 0:
        print(f"\nOther Disagreements:")
        print(f"  Count: {other_analysis['count']}")
        print(f"  Correct: {other_analysis['correct_count']}")
        print(f"  Accuracy: {other_analysis['accuracy']:.3f}")
    
    return analysis

def analyze_task_improvement_breakdown(initial_tensor, final_tensor, ground_truth_labels, binary_schema=False):
    """
    Analyze the breakdown of tasks by their improvement status, hint types, and ground truth labels.
    
    Args:
        initial_tensor: Initial N x J x K tensor
        final_tensor: Final N x J x K tensor after synthetic edits
        ground_truth_labels: 1D array of length N, true label for each task
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing detailed breakdown analysis
    """
    from analysis import calculate_task_accuracy
    
    N, J, K = initial_tensor.shape
    
    # Get EM predictions for initial state
    initial_predictions = get_em_predictions(initial_tensor)
    
    # Prepare ground truth for comparison
    if binary_schema:
        gt_for_comparison = np.where(ground_truth_labels == 2, 1, ground_truth_labels)
    else:
        gt_for_comparison = ground_truth_labels
    
    # Calculate task improvements
    task_improvements = {}
    for task_idx in range(N):
        initial_metrics = calculate_task_accuracy(initial_tensor, ground_truth_labels, task_idx, binary_schema)
        final_metrics = calculate_task_accuracy(final_tensor, ground_truth_labels, task_idx, binary_schema)
        accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
        
        task_improvements[task_idx] = {
            'initial_accuracy': initial_metrics['accuracy'],
            'final_accuracy': final_metrics['accuracy'],
            'improvement': accuracy_improvement,
            'ground_truth': ground_truth_labels[task_idx],
            'gt_for_comparison': gt_for_comparison[task_idx],
            'em_prediction': initial_predictions[task_idx]
        }
    
    # Categorize tasks by improvement status
    improved_tasks = []
    declined_tasks = []
    unchanged_tasks = []
    
    for task_idx, data in task_improvements.items():
        if data['improvement'] > 0:
            improved_tasks.append(task_idx)
        elif data['improvement'] < 0:
            declined_tasks.append(task_idx)
        else:
            unchanged_tasks.append(task_idx)
    
    # Analyze hint types for each category
    def analyze_hint_types_for_tasks(task_indices):
        """Analyze hint types for a given set of tasks."""
        hint_analysis = {
            'missed_relevant': 0,
            'irrelevant_disagreement': 0,
            'other': 0,
            'no_disagreements': 0,
            'correct_hints': 0,
            'incorrect_hints': 0,
            'total_hints': 0
        }
        
        for task_idx in task_indices:
            task_hints = {
                'missed_relevant': 0,
                'irrelevant_disagreement': 0,
                'other': 0,
                'correct_hints': 0,
                'incorrect_hints': 0
            }
            
            em_prediction = initial_predictions[task_idx]
            true_label = gt_for_comparison[task_idx]
            
            # Check each student's annotation for this task
            for student_idx in range(K):
                student_annotation = np.argmax(initial_tensor[task_idx, :, student_idx])
                
                # Check if there's a disagreement
                if student_annotation != em_prediction:
                    hint_correct = (em_prediction == true_label)
                    
                    if student_annotation == 0 and em_prediction == 1:
                        task_hints['missed_relevant'] += 1
                    elif student_annotation == 1 and em_prediction == 0:
                        task_hints['irrelevant_disagreement'] += 1
                    else:
                        task_hints['other'] += 1
                    
                    if hint_correct:
                        task_hints['correct_hints'] += 1
                    else:
                        task_hints['incorrect_hints'] += 1
            
            # Aggregate for this task
            if (task_hints['missed_relevant'] + task_hints['irrelevant_disagreement'] + task_hints['other']) == 0:
                hint_analysis['no_disagreements'] += 1
            else:
                hint_analysis['missed_relevant'] += task_hints['missed_relevant']
                hint_analysis['irrelevant_disagreement'] += task_hints['irrelevant_disagreement']
                hint_analysis['other'] += task_hints['other']
                hint_analysis['correct_hints'] += task_hints['correct_hints']
                hint_analysis['incorrect_hints'] += task_hints['incorrect_hints']
                hint_analysis['total_hints'] += (task_hints['missed_relevant'] + task_hints['irrelevant_disagreement'] + task_hints['other'])
        
        return hint_analysis
    
    # Analyze each category
    improved_analysis = analyze_hint_types_for_tasks(improved_tasks)
    declined_analysis = analyze_hint_types_for_tasks(declined_tasks)
    unchanged_analysis = analyze_hint_types_for_tasks(unchanged_tasks)
    
    # Analyze ground truth label distribution
    def analyze_ground_truth_distribution(task_indices):
        """Analyze ground truth label distribution for a given set of tasks."""
        label_counts = {}
        for task_idx in task_indices:
            label = ground_truth_labels[task_idx]
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    improved_labels = analyze_ground_truth_distribution(improved_tasks)
    declined_labels = analyze_ground_truth_distribution(declined_tasks)
    unchanged_labels = analyze_ground_truth_distribution(unchanged_tasks)
    
    # Compile results
    breakdown = {
        'improved': {
            'count': len(improved_tasks),
            'tasks': improved_tasks,
            'hint_analysis': improved_analysis,
            'ground_truth_distribution': improved_labels
        },
        'declined': {
            'count': len(declined_tasks),
            'tasks': declined_tasks,
            'hint_analysis': declined_analysis,
            'ground_truth_distribution': declined_labels
        },
        'unchanged': {
            'count': len(unchanged_tasks),
            'tasks': unchanged_tasks,
            'hint_analysis': unchanged_analysis,
            'ground_truth_distribution': unchanged_labels
        }
    }
    
    # Print detailed breakdown
    print(f"\n=== TASK IMPROVEMENT BREAKDOWN ANALYSIS ===")
    print(f"Schema: {'Binary (2→1)' if binary_schema else 'Three-class (0,1,2)'}")
    print(f"Total tasks: {N}")
    print()
    
    for category in ['improved', 'declined', 'unchanged']:
        data = breakdown[category]
        print(f"{category.upper()} TASKS ({data['count']} tasks, {data['count']/N*100:.1f}%):")
        
        # Ground truth distribution
        print(f"  Ground Truth Distribution:")
        for label, count in sorted(data['ground_truth_distribution'].items()):
            print(f"    Label {label}: {count} tasks ({count/data['count']*100:.1f}%)")
        
        # Hint analysis
        hint_data = data['hint_analysis']
        print(f"  Hint Analysis:")
        print(f"    No disagreements: {hint_data['no_disagreements']} tasks ({hint_data['no_disagreements']/data['count']*100:.1f}%)")
        print(f"    Missed relevant hints: {hint_data['missed_relevant']} total")
        print(f"    Irrelevant disagreement hints: {hint_data['irrelevant_disagreement']} total")
        print(f"    Other hints: {hint_data['other']} total")
        
        if hint_data['total_hints'] > 0:
            print(f"    Correct hints: {hint_data['correct_hints']} ({hint_data['correct_hints']/hint_data['total_hints']*100:.1f}%)")
            print(f"    Incorrect hints: {hint_data['incorrect_hints']} ({hint_data['incorrect_hints']/hint_data['total_hints']*100:.1f}%)")
        
        print()
    
    return breakdown

def analyze_task_improvement_breakdown_multiple_trials(initial_tensor, trial_results, ground_truth_labels, binary_schema=False):
    """
    Analyze the breakdown of tasks by their improvement status across multiple trials.
    
    Args:
        initial_tensor: Initial N x J x K tensor
        trial_results: List of (final_tensor, round_history, total_rounds, metrics_history, processed_pairs) tuples
        ground_truth_labels: 1D array of length N, true label for each task
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing detailed breakdown analysis averaged across trials
    """
    from analysis import calculate_task_accuracy
    
    N, J, K = initial_tensor.shape
    num_trials = len(trial_results)
    
    print(f"\nAnalyzing task improvement breakdown across {num_trials} trials...")
    print(f"Schema: {'Binary (2→1)' if binary_schema else 'Three-class (0,1,2)'}")
    print("-" * 80)
    
    # Get EM predictions for initial state
    initial_predictions = get_em_predictions(initial_tensor)
    
    # Prepare ground truth for comparison
    if binary_schema:
        gt_for_comparison = np.where(ground_truth_labels == 2, 1, ground_truth_labels)
    else:
        gt_for_comparison = ground_truth_labels
    
    # Calculate task improvements across all trials
    task_improvements = {}
    for task_idx in range(N):
        initial_metrics = calculate_task_accuracy(initial_tensor, ground_truth_labels, task_idx, binary_schema)
        
        # Collect final accuracies across trials
        final_accuracies = []
        for final_tensor, _, _, _ in trial_results:
            final_metrics = calculate_task_accuracy(final_tensor, ground_truth_labels, task_idx, binary_schema)
            final_accuracies.append(final_metrics['accuracy'])
        
        # Average final accuracy across trials
        avg_final_accuracy = np.mean(final_accuracies)
        accuracy_improvement = avg_final_accuracy - initial_metrics['accuracy']
        
        task_improvements[task_idx] = {
            'initial_accuracy': initial_metrics['accuracy'],
            'final_accuracy': avg_final_accuracy,
            'improvement': accuracy_improvement,
            'ground_truth': ground_truth_labels[task_idx],
            'gt_for_comparison': gt_for_comparison[task_idx],
            'em_prediction': initial_predictions[task_idx],
            'std_final_accuracy': np.std(final_accuracies)
        }
    
    # Categorize tasks by improvement status
    improved_tasks = []
    declined_tasks = []
    unchanged_tasks = []
    
    for task_idx, data in task_improvements.items():
        if data['improvement'] > 0:
            improved_tasks.append(task_idx)
        elif data['improvement'] < 0:
            declined_tasks.append(task_idx)
        else:
            unchanged_tasks.append(task_idx)
    
    # Analyze hint types for each category (using initial state)
    def analyze_hint_types_for_tasks(task_indices):
        """Analyze hint types for a given set of tasks."""
        hint_analysis = {
            'missed_relevant': 0,
            'irrelevant_disagreement': 0,
            'other': 0,
            'no_disagreements': 0,
            'correct_hints': 0,
            'incorrect_hints': 0,
            'total_hints': 0
        }
        
        for task_idx in task_indices:
            task_hints = {
                'missed_relevant': 0,
                'irrelevant_disagreement': 0,
                'other': 0,
                'correct_hints': 0,
                'incorrect_hints': 0
            }
            
            em_prediction = initial_predictions[task_idx]
            true_label = gt_for_comparison[task_idx]
            
            # Check each student's annotation for this task
            for student_idx in range(K):
                student_annotation = np.argmax(initial_tensor[task_idx, :, student_idx])
                
                # Check if there's a disagreement
                if student_annotation != em_prediction:
                    hint_correct = (em_prediction == true_label)
                    
                    if student_annotation == 0 and em_prediction == 1:
                        task_hints['missed_relevant'] += 1
                    elif student_annotation == 1 and em_prediction == 0:
                        task_hints['irrelevant_disagreement'] += 1
                    else:
                        task_hints['other'] += 1
                    
                    if hint_correct:
                        task_hints['correct_hints'] += 1
                    else:
                        task_hints['incorrect_hints'] += 1
            
            # Aggregate for this task
            if (task_hints['missed_relevant'] + task_hints['irrelevant_disagreement'] + task_hints['other']) == 0:
                hint_analysis['no_disagreements'] += 1
            else:
                hint_analysis['missed_relevant'] += task_hints['missed_relevant']
                hint_analysis['irrelevant_disagreement'] += task_hints['irrelevant_disagreement']
                hint_analysis['other'] += task_hints['other']
                hint_analysis['correct_hints'] += task_hints['correct_hints']
                hint_analysis['incorrect_hints'] += task_hints['incorrect_hints']
                hint_analysis['total_hints'] += (task_hints['missed_relevant'] + task_hints['irrelevant_disagreement'] + task_hints['other'])
        
        return hint_analysis
    
    # Analyze each category
    improved_analysis = analyze_hint_types_for_tasks(improved_tasks)
    declined_analysis = analyze_hint_types_for_tasks(declined_tasks)
    unchanged_analysis = analyze_hint_types_for_tasks(unchanged_tasks)
    
    # Analyze ground truth label distribution
    def analyze_ground_truth_distribution(task_indices):
        """Analyze ground truth label distribution for a given set of tasks."""
        label_counts = {}
        for task_idx in task_indices:
            label = ground_truth_labels[task_idx]
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    improved_labels = analyze_ground_truth_distribution(improved_tasks)
    declined_labels = analyze_ground_truth_distribution(declined_tasks)
    unchanged_labels = analyze_ground_truth_distribution(unchanged_tasks)
    
    # Compile results
    breakdown = {
        'improved': {
            'count': len(improved_tasks),
            'tasks': improved_tasks,
            'hint_analysis': improved_analysis,
            'ground_truth_distribution': improved_labels
        },
        'declined': {
            'count': len(declined_tasks),
            'tasks': declined_tasks,
            'hint_analysis': declined_analysis,
            'ground_truth_distribution': declined_labels
        },
        'unchanged': {
            'count': len(unchanged_tasks),
            'tasks': unchanged_tasks,
            'hint_analysis': unchanged_analysis,
            'ground_truth_distribution': unchanged_labels
        }
    }
    
    # Print detailed breakdown
    print(f"\n=== TASK IMPROVEMENT BREAKDOWN ANALYSIS (AVERAGED ACROSS TRIALS) ===")
    print(f"Schema: {'Binary (2→1)' if binary_schema else 'Three-class (0,1,2)'}")
    print(f"Total tasks: {N}")
    print()
    
    for category in ['improved', 'declined', 'unchanged']:
        data = breakdown[category]
        print(f"{category.upper()} TASKS ({data['count']} tasks, {data['count']/N*100:.1f}%):")
        
        # Ground truth distribution
        print(f"  Ground Truth Distribution:")
        for label, count in sorted(data['ground_truth_distribution'].items()):
            print(f"    Label {label}: {count} tasks ({count/data['count']*100:.1f}%)")
        
        # Hint analysis
        hint_data = data['hint_analysis']
        print(f"  Hint Analysis:")
        print(f"    No disagreements: {hint_data['no_disagreements']} tasks ({hint_data['no_disagreements']/data['count']*100:.1f}%)")
        print(f"    Missed relevant hints: {hint_data['missed_relevant']} total")
        print(f"    Irrelevant disagreement hints: {hint_data['irrelevant_disagreement']} total")
        print(f"    Other hints: {hint_data['other']} total")
        
        if hint_data['total_hints'] > 0:
            print(f"    Correct hints: {hint_data['correct_hints']} ({hint_data['correct_hints']/hint_data['total_hints']*100:.1f}%)")
            print(f"    Incorrect hints: {hint_data['incorrect_hints']} ({hint_data['incorrect_hints']/hint_data['total_hints']*100:.1f}%)")
        
        print()
    
    return breakdown
