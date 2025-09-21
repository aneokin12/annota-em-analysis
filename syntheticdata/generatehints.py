import numpy as np
from model import dawid_skene
from sampler import (
    sample_student_global, 
    sample_student_bootstrap, 
    sample_student_hier,
    simulate_response,
    alpha_global
)
from analysis import (
    calculate_student_accuracy,
    calculate_task_accuracy,
    analyze_student_improvements,
    analyze_student_improvements_multiple_trials,
    analyze_task_improvements,
    analyze_task_improvements_multiple_trials
)
from plotting import (
    plot_student_improvements,
    plot_task_improvements,
    plot_hint_quality,
    plot_task_improvement_breakdown
)
from hint_analysis import (
    get_em_predictions,
    analyze_hint_quality,
    analyze_task_improvement_breakdown,
    analyze_task_improvement_breakdown_multiple_trials
)

def cluster_students_by_performance(tensor, ground_truth_labels, n_clusters=3, binary_schema=False):
    """
    Cluster students based on their initial performance using K-means.
    
    Args:
        tensor: Initial N x J x K tensor
        ground_truth_labels: 1D array of length N, true label for each task
        n_clusters: Number of clusters to create
        binary_schema: If True, treat 2 as 1 for accuracy calculations
    
    Returns:
        dict: Dictionary containing cluster assignments and performance metrics
    """
    from sklearn.cluster import KMeans
    
    N, J, K = tensor.shape
    
    # Calculate initial performance metrics for each student
    student_features = []
    student_performance = {}
    
    for student_idx in range(K):
        # Calculate accuracy, precision, recall, F1
        metrics = calculate_student_accuracy(tensor, ground_truth_labels, student_idx, binary_schema)
        
        # Create feature vector for clustering
        features = [
            metrics['accuracy'],
            metrics['precision'], 
            metrics['recall'],
            metrics['f1'],
            metrics['true_positives'] / metrics['total_predictions'] if metrics['total_predictions'] > 0 else 0,
            metrics['true_negatives'] / metrics['total_predictions'] if metrics['total_predictions'] > 0 else 0
        ]
        
        student_features.append(features)
        student_performance[student_idx] = metrics
    
    # Convert to numpy array for clustering
    X = np.array(student_features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Analyze clusters
    cluster_analysis = {}
    for cluster_id in range(n_clusters):
        cluster_students = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        # Calculate cluster statistics
        cluster_accuracies = [student_performance[i]['accuracy'] for i in cluster_students]
        cluster_precisions = [student_performance[i]['precision'] for i in cluster_students]
        cluster_recalls = [student_performance[i]['recall'] for i in cluster_students]
        cluster_f1s = [student_performance[i]['f1'] for i in cluster_students]
        
        cluster_analysis[cluster_id] = {
            'students': cluster_students,
            'size': len(cluster_students),
            'mean_accuracy': np.mean(cluster_accuracies),
            'std_accuracy': np.std(cluster_accuracies),
            'mean_precision': np.mean(cluster_precisions),
            'mean_recall': np.mean(cluster_recalls),
            'mean_f1': np.mean(cluster_f1s),
            'performance_range': (min(cluster_accuracies), max(cluster_accuracies))
        }
    
    # Sort clusters by mean accuracy (best to worst)
    sorted_clusters = sorted(cluster_analysis.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
    
    # Create student-to-cluster mapping
    student_clusters = {}
    for student_idx, cluster_id in enumerate(cluster_labels):
        student_clusters[student_idx] = cluster_id
    
    # Print cluster analysis
    print(f"\n=== STUDENT CLUSTERING ANALYSIS ===")
    print(f"Number of clusters: {n_clusters}")
    print(f"Total students: {K}")
    print()
    
    for i, (cluster_id, analysis) in enumerate(sorted_clusters):
        print(f"Cluster {i+1} (ID: {cluster_id}): {analysis['size']} students")
        print(f"  Mean Accuracy: {analysis['mean_accuracy']:.3f} ± {analysis['std_accuracy']:.3f}")
        print(f"  Mean Precision: {analysis['mean_precision']:.3f}")
        print(f"  Mean Recall: {analysis['mean_recall']:.3f}")
        print(f"  Mean F1: {analysis['mean_f1']:.3f}")
        print(f"  Accuracy Range: {analysis['performance_range'][0]:.3f} - {analysis['performance_range'][1]:.3f}")
        print(f"  Students: {analysis['students']}")
        print()
    
    return {
        'cluster_labels': cluster_labels,
        'student_clusters': student_clusters,
        'cluster_analysis': cluster_analysis,
        'sorted_clusters': sorted_clusters,
        'student_performance': student_performance,
        'n_clusters': n_clusters
    }
from gensynthlabels import generate_synthetic_labels
from metrics import calculate_custom_metrics, print_custom_metrics_summary
from load_real_labels import load_real_ground_truth, verify_tensor_labels_match, load_concatenated_labels

def load_initial_tensor():
    """Load the initial concatenated tensor from disk."""
    return np.load('concatenated_tensor.npy')











# def apply_one_synthetic_edit(tensor, ground_truth_labels, student_idx, sampling_method='hierarchical', processed_pairs=None):
#     """
#     Perform one synthetic edit: one student responds to a hint on a task where they disagree with the EM prediction.
#     Only process student-task pairs that haven't been processed before.
#     Randomly choose between annotating missed relevant (0→1) or irrelevant (1→0) annotations, with a 70% bias towards missed relevant ones.
#     Args:
#         tensor: N x J x K tensor (tasks x labels x students)
#         ground_truth_labels: 1D array of length N, true label for each task
#         student_idx: index of the student to make the edit
#         sampling_method: 'global', 'bootstrap', or 'hierarchical'
#         processed_pairs: set of (student_idx, task_idx) tuples that have already been processed
#     Returns:
#         updated_tensor: tensor after the edit
#         edited_task_idx: index of the task edited (or None if no edit was made)
#         processed_pairs: updated set of processed pairs
#     """
#     N, J, K = tensor.shape
#     em_predictions = get_em_predictions(tensor)
#     updated_tensor = tensor.copy()
    
#     if processed_pairs is None:
#         processed_pairs = set()
    
#     # Collect indices of available disagreements (not yet processed)
#     missed_relevant_tasks = []  # student said 0, ground truth 1
#     irrelevant_disagreement_tasks = []  # student said 1, ground truth 0
#     fallback_disagreement_tasks = []
    
#     for task_idx in range(N):
#         # Skip if this student-task pair has already been processed
#         if (student_idx, task_idx) in processed_pairs:
#             continue
            
#         hint = em_predictions[task_idx]
#         true_label = ground_truth_labels[task_idx]
#         if true_label == 2:
#             true_label = 1
#         student_annotation = np.argmax(tensor[task_idx, :, student_idx])
        
#         if student_annotation == 0 and true_label == 1 and student_annotation != hint:
#             missed_relevant_tasks.append(task_idx)
#         elif student_annotation == 1 and true_label == 0 and student_annotation != hint:
#             irrelevant_disagreement_tasks.append(task_idx)
#         elif student_annotation != hint:
#             fallback_disagreement_tasks.append(task_idx)
    
#     # Choose which set to sample from
#     chosen_task = None
#     rand_val = np.random.rand()
#     if missed_relevant_tasks and rand_val < 0.57:
#         chosen_task = np.random.choice(missed_relevant_tasks)
#     elif irrelevant_disagreement_tasks:
#         chosen_task = np.random.choice(irrelevant_disagreement_tasks)
#     elif fallback_disagreement_tasks:
#         chosen_task = np.random.choice(fallback_disagreement_tasks)
#     else:
#         # print(f"Student {student_idx} has no unprocessed disagreements with EM predictions.")
#         return updated_tensor, None, processed_pairs
    
#     # Now perform the edit on the chosen task
#     hint = em_predictions[chosen_task]
#     true_label = ground_truth_labels[chosen_task]
#     if true_label == 2:
#         true_label = 1
#     if sampling_method == 'global':
#         q_vec = sample_student_global(alpha_global)
#     elif sampling_method == 'bootstrap':
#         from sampler import df_probs
#         q_vec = sample_student_bootstrap(df_probs)
#     elif sampling_method == 'hierarchical':
#         q_vec = sample_student_hier(alpha_global, kappa=1.0)
#     else:
#         raise ValueError(f"Unknown sampling method: {sampling_method}")
#     correct_to_agree = (hint == true_label)
#     response = simulate_response(correct_to_agree, q_vec)
#     if response == 'AGREE':
#         updated_tensor[chosen_task, :, student_idx] = 0
#         updated_tensor[chosen_task, hint, student_idx] = 1
#     else:
#         updated_tensor[chosen_task, :, student_idx] = 0
#         updated_tensor[chosen_task, 1 - hint, student_idx] = 1
    
#     # Mark this student-task pair as processed
#     processed_pairs.add((student_idx, chosen_task))
    
#     return updated_tensor, chosen_task, processed_pairs

def run_until_hint_exhaustion(tensor, ground_truth_labels, max_rounds=10000, sampling_method='hierarchical', binary_schema=False, use_proficiency_correlation=False):
    """
    Run synthetic edit rounds until all available hints have been responded to.
    
    Args:
        tensor: Initial N x J x K tensor
        ground_truth_labels: 1D array of length N, true label for each task
        max_rounds: Maximum number of rounds to run (safety limit)
        sampling_method: 'global', 'bootstrap', or 'hierarchical'
        binary_schema: If True, treat 2 as 1 for metrics calculations
        use_proficiency_correlation: If True, use proficiency-correlated student parameters
    
    Returns:
        final_tensor: Tensor after all hints exhausted
        round_history: List of (round_num, student_idx, task_idx, predictions_changed) tuples
        total_rounds: Total number of rounds run
        processed_pairs: Set of (student_idx, task_idx) tuples that were processed
    """
    N, J, K = tensor.shape
    current_tensor = tensor.copy()
    round_history = []
    processed_pairs = set()
    previous_predictions = None
    
    # Create proficiency-correlated student parameters if requested
    if use_proficiency_correlation:
        student_params = create_proficiency_correlated_students(K)
        print(f"Using proficiency-correlated student parameters")
        print(f"Student 0 (least proficient): expected accuracy {student_params[0]['expected_accuracy']:.3f}")
        print(f"Student {K-1} (most proficient): expected accuracy {student_params[K-1]['expected_accuracy']:.3f}")
    else:
        student_params = None
    
    print(f"Starting hint exhaustion run: max_rounds={max_rounds}")
    print(f"Total possible student-task pairs: {N * K}")
    
    # Pre-compute all initial disagreements once
    def get_all_disagreements(tensor, predictions):
        """Get all disagreements between student annotations and EM predictions."""
        all_disagreements = []
        
        for student_idx in range(K):
            for task_idx in range(N):
                hint = predictions[task_idx]
                true_label = ground_truth_labels[task_idx]
                if true_label == 2:
                    true_label = 1
                student_annotation = np.argmax(tensor[task_idx, :, student_idx])
                
                # Check if there's a disagreement
                if student_annotation != hint:
                    # Categorize the disagreement
                    if student_annotation == 0 and hint == 1:
                        category = 'missed_relevant'

                    elif student_annotation == 1 and hint == 0:
                        category = 'irrelevant_disagreement'
                    # else:
                    #     category = 'fallback_disagreement'
                    
                    all_disagreements.append((student_idx, task_idx, category))
        # print(f"All disagreements: {all_disagreements}")
        return all_disagreements
    
    # Get initial disagreements
        current_predictions = get_em_predictions(current_tensor)
    all_disagreements = get_all_disagreements(current_tensor, current_predictions)
    available_disagreements = all_disagreements.copy()  # Start with all disagreements available
        
    # print(f"Initial disagreements found: {len(all_disagreements)}")
        
    for round_num in range(max_rounds):

        
        # Check if predictions changed from previous round
        if previous_predictions is not None:
            predictions_changed = np.sum(current_predictions != previous_predictions)
            # if predictions_changed > 0:
                # print(f"Round {round_num}: {predictions_changed} EM predictions changed")
        else:
            predictions_changed = 0
            # print(f"Round {round_num}: Initial EM predictions")
        
        # If no available disagreements, we're done
        if not available_disagreements:
            print(f"Round {round_num}: No more available hints! All disagreements processed.")
            break
        
        # Categorize available disagreements for biased sampling
        missed_relevant = [(s, t) for s, t, c in available_disagreements if c == 'missed_relevant']
        irrelevant_disagreement = [(s, t) for s, t, c in available_disagreements if c == 'irrelevant_disagreement']
        # fallback_disagreement = [(s, t) for s, t, c in available_disagreements if c == 'fallback_disagreement']
        
        # Choose which category to sample from (biased towards missed relevant)
        chosen_pair = None
        rand_val = np.random.rand()
        
        if irrelevant_disagreement: 
            # print(f"Irrel")
            chosen_pair = np.random.choice(len(irrelevant_disagreement))
            student_idx, task_idx = irrelevant_disagreement[chosen_pair]
        elif missed_relevant: 
            # print(f"Irrel")
            chosen_pair = np.random.choice(len(missed_relevant))
            student_idx, task_idx = missed_relevant[chosen_pair]
        elif not irrelevant_disagreement:
            chosen_pair = np.random.choice(len(missed_relevant))
            student_idx, task_idx = missed_relevant[chosen_pair]
        elif not missed_relevant:
            chosen_pair = np.random.choice(len(irrelevant_disagreement))
            student_idx, task_idx = irrelevant_disagreement[chosen_pair]
        else:
            print(f"Else")
        # elif fallback_disagreement:
        #     chosen_pair = np.random.choice(len(fallback_disagreement))
        #     student_idx, task_idx = fallback_disagreement[chosen_pair]
        # else:
        #     # This shouldn't happen since we checked available_disagreements is not empty
        #     print(f"Error: No disagreements found despite having {len(available_disagreements)} available")
        #     break
        
        # Apply the synthetic edit
        hint = current_predictions[task_idx]
        true_label = ground_truth_labels[task_idx]
        if true_label == 2:
            true_label = 1
        
        if use_proficiency_correlation and student_params is not None:
            # Use proficiency-correlated parameters for this specific student
            student_alpha = student_params[student_idx]['alpha']
            q_vec = student_alpha / student_alpha.sum()  # Normalize to probabilities
        elif sampling_method == 'global':
            q_vec = sample_student_global(alpha_global)
        elif sampling_method == 'bootstrap':
            from sampler import df_probs
            q_vec = sample_student_bootstrap(df_probs)
        elif sampling_method == 'hierarchical':
            q_vec = sample_student_hier(alpha_global, kappa=1.0)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        correct_to_agree = (hint == true_label)
        response = simulate_response(correct_to_agree, q_vec)
        
        if response == 'AGREE':
            current_tensor[task_idx, :, student_idx] = 0
            current_tensor[task_idx, hint, student_idx] = 1
        else:
            current_tensor[task_idx, :, student_idx] = 0
            current_tensor[task_idx, 1 - hint, student_idx] = 1
        
        # Mark this pair as processed
        processed_pairs.add((student_idx, task_idx))
        
        # Remove this disagreement from available_disagreements
        available_disagreements = [(s, t, c) for s, t, c in available_disagreements if (s, t) != (student_idx, task_idx)]
        
        # Record the round
        round_history.append((round_num, student_idx, task_idx, predictions_changed))
        
        # Update for next round
        previous_predictions = current_predictions
        
        # Check if EM predictions changed - if so, we need to recompute disagreements
        new_predictions = get_em_predictions(current_tensor)
        if not np.array_equal(current_predictions, new_predictions):
            print(f"Round {round_num}: EM predictions changed, recomputing disagreements...")
            current_predictions = new_predictions
            # Recompute all disagreements and update available ones
            all_disagreements = get_all_disagreements(current_tensor, current_predictions)
            # Only keep disagreements that haven't been processed yet
            available_disagreements = [(s, t, c) for s, t, c in all_disagreements if (s, t) not in processed_pairs]
            # print(f"All disagreements: {available_disagreements}")
            print(f"New disagreements after EM change: {len(available_disagreements)} available")
        
        # Print progress every 100 rounds
        if round_num % 100 == 0:
            print(f"Completed {round_num} rounds, processed {len(processed_pairs)} pairs, {len(available_disagreements)} remaining...")
    
    total_rounds = round_num + 1
    
    print(f"\nHint exhaustion completed!")
    print(f"Total rounds: {total_rounds}")
    print(f"Total processed pairs: {len(processed_pairs)}/{N * K} ({len(processed_pairs)/(N*K)*100:.1f}%)")
    print(f"Average rounds per student: {total_rounds/K:.1f}")
    print(f"Average rounds per task: {total_rounds/N:.1f}")
    
    return current_tensor, round_history, total_rounds, processed_pairs

def run_multiple_trials_hint_exhaustion(tensor, ground_truth_labels, num_trials=10, max_rounds=10000, sampling_method='global', binary_schema=False, use_proficiency_correlation=False):
    """
    Run multiple trials of the hint exhaustion process and average the results.
    
    Args:
        tensor: Initial N x J x K tensor
        ground_truth_labels: 1D array of length N, true label for each task
        num_trials: Number of trials to run
        max_rounds: Maximum number of rounds per trial
        sampling_method: 'global', 'bootstrap', or 'hierarchical'
        binary_schema: If True, treat 2 as 1 for metrics calculations
        use_proficiency_correlation: If True, use proficiency-correlated student parameters
    
    Returns:
        averaged_metrics_history: List of averaged metric dictionaries for each round
        trial_results: List of (final_tensor, round_history, total_rounds, metrics_history, processed_pairs) for each trial
        max_rounds_across_trials: Maximum number of rounds across all trials
    """
    print(f"Running {num_trials} hint exhaustion trials...")
    
    trial_results = []
    max_rounds_across_trials = 0
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
        # Set different random seed for each trial
        np.random.seed(np.random.randint(0, 10000))  # Random seed each time
        
        # Run single trial
        final_tensor, round_history, total_rounds, processed_pairs = run_until_hint_exhaustion(
            tensor, ground_truth_labels, max_rounds, sampling_method, binary_schema, use_proficiency_correlation
        )
        
        trial_results.append((final_tensor, round_history, total_rounds, processed_pairs))
        max_rounds_across_trials = max(max_rounds_across_trials, total_rounds)
        
        print(f"Trial {trial + 1} completed: {len(round_history)} rounds, {len(processed_pairs)} pairs processed")
    
    return trial_results, max_rounds_across_trials



def main(binary_schema=False):
    print("Loading initial concatenated tensor...")
    initial_tensor = load_initial_tensor()
    print(f"Initial tensor shape: {initial_tensor.shape}")
    print(f"Initial total annotations: {np.sum(initial_tensor)}")
    
    # Load concatenated real ground truth labels
    print("\nLoading concatenated real ground truth labels...")
    
    try:
        ground_truth_labels, dataset_info = load_concatenated_labels()
        
        # Verify that labels match tensor dimensions
        if not verify_tensor_labels_match(initial_tensor, ground_truth_labels):
            print("Warning: Tensor and labels don't match. Check your data!")
            return
        
        print(f"Successfully loaded {len(ground_truth_labels)} concatenated ground truth labels")
        print(f"Dataset breakdown:")
        for i, (transcript_id, research_question_id, num_labels) in enumerate(dataset_info):
            print(f"  Dataset {i+1}: {transcript_id} + {research_question_id} = {num_labels} labels")
        
        # Show initial custom metrics
        initial_predictions = get_em_predictions(initial_tensor)
        print(f"\nInitial Custom Metrics:")
        print_custom_metrics_summary(initial_predictions, ground_truth_labels, "Initial", binary_schema)
        
    except Exception as e:
        print(f"Error loading concatenated labels: {e}")
        print("Falling back to synthetic labels...")
        ground_truth_labels = generate_synthetic_labels(initial_tensor)
        print(f"Generated synthetic labels with distribution: {np.bincount(ground_truth_labels, length=3)}")

    # Analyze hint quality before running trials
    print("\n" + "="*60)
    print("ANALYZING INITIAL HINT QUALITY")
    print("="*60)
    hint_quality_analysis = analyze_hint_quality(initial_tensor, ground_truth_labels, binary_schema)
    
    # Plot hint quality
    schema_suffix = "_binary" if binary_schema else ""
    plot_hint_quality(hint_quality_analysis, save_path=f'hint_quality_analysis{schema_suffix}.png', binary_schema=binary_schema)

    # Run multiple trials
    print("\nRunning multiple trials...")
    trial_results, max_rounds = run_multiple_trials_hint_exhaustion(
        initial_tensor, ground_truth_labels, num_trials=5, max_rounds=100000, 
        sampling_method='hierarchical', binary_schema=binary_schema, use_proficiency_correlation=True
    )
    
    # Show summary statistics
    print(f"\nTrial Summary:")
    print(f"Number of trials: {len(trial_results)}")
    print(f"Max rounds across trials: {max_rounds}")
    
    # Calculate average total rounds and processed pairs
    total_rounds = [trial_result[2] for trial_result in trial_results]
    processed_pairs_counts = [len(trial_result[3]) for trial_result in trial_results]
    avg_total_rounds = np.mean(total_rounds)
    std_total_rounds = np.std(total_rounds)
    avg_processed_pairs = np.mean(processed_pairs_counts)
    std_processed_pairs = np.std(processed_pairs_counts)
    
    print(f"Average total rounds: {avg_total_rounds:.1f} ± {std_total_rounds:.1f}")
    print(f"Average processed pairs: {avg_processed_pairs:.1f} ± {std_processed_pairs:.1f}")
    print(f"Average pairs per trial: {avg_processed_pairs/(initial_tensor.shape[0] * initial_tensor.shape[2])*100:.1f}% of all possible pairs")
    
        # Analyze student-level accuracy improvements across multiple trials
    print("\nAnalyzing student-level accuracy improvements across multiple trials...")
    student_improvements, summary_stats = analyze_student_improvements_multiple_trials(
        initial_tensor, trial_results, ground_truth_labels, binary_schema
    )
    
    # Plot student-level improvements
    print("\nPlotting student-level accuracy improvements...")
    plot_student_improvements(
        student_improvements, summary_stats, 
        save_path=f'student_improvements{schema_suffix}.png', 
        binary_schema=binary_schema
    )

    # Analyze task-level accuracy improvements across multiple trials
    print("\nAnalyzing task-level accuracy improvements across multiple trials...")
    task_improvements, task_summary_stats = analyze_task_improvements_multiple_trials(
        initial_tensor, trial_results, ground_truth_labels, binary_schema
    )
    
    # Analyze task improvement breakdown across multiple trials
    print("\nAnalyzing task improvement breakdown across multiple trials...")
    task_breakdown = analyze_task_improvement_breakdown_multiple_trials(
        initial_tensor, trial_results, ground_truth_labels, binary_schema
    )
    
    # Plot task-level improvements
    print("\nPlotting task-level accuracy improvements...")
    plot_task_improvements(
        task_improvements, task_summary_stats,
        save_path=f'task_improvements{schema_suffix}.png',
        binary_schema=binary_schema
    )
    
    # Plot task improvement breakdown
    print("\nPlotting task improvement breakdown...")
    plot_task_improvement_breakdown(
        task_breakdown,
        save_path=f'task_improvement_breakdown{schema_suffix}.png',
        binary_schema=binary_schema
    )
    
    return trial_results

def trace_em_convergence(tensor, ground_truth_labels, max_rounds=10000, sampling_method='hierarchical', binary_schema=False):
    """
    Diagnostic function to trace EM convergence and understand why it gets stuck.
    
    Args:
        tensor: Initial N x J x K tensor
        ground_truth_labels: 1D array of length N, true label for each task
        max_rounds: Maximum number of rounds to run
        sampling_method: 'global', 'bootstrap', or 'hierarchical'
        binary_schema: If True, treat 2 as 1 for metrics calculations
    
    Returns:
        convergence_trace: Dictionary containing detailed convergence information
    """
    N, J, K = tensor.shape
    current_tensor = tensor.copy()
    convergence_trace = {
        'rounds': [],
        'predictions_changed': [],
        'prediction_changes_by_task': [],
        'confusion_matrices': [],
        'class_priors': [],
        'tensor_changes': [],
        'disagreement_counts': []
    }
    
    print(f"Starting EM convergence trace...")
    print(f"Tensor shape: {N} tasks x {J} labels x {K} students")
    
    # Pre-compute all initial disagreements once
    def get_all_disagreements(tensor, predictions):
        """Get all disagreements between student annotations and EM predictions."""
        all_disagreements = []
        
        for student_idx in range(K):
            for task_idx in range(N):
                hint = predictions[task_idx]
                true_label = ground_truth_labels[task_idx]
                if true_label == 2:
                    true_label = 1
                student_annotation = np.argmax(tensor[task_idx, :, student_idx])
                
                # Check if there's a disagreement
                if student_annotation != hint:
                    if student_annotation == 0 and hint == 1:
                        category = 'missed_relevant'
                    elif student_annotation == 1 and hint == 0:
                        category = 'irrelevant_disagreement'
                    else:
                        category = 'other'
                    
                    all_disagreements.append((student_idx, task_idx, category))
        return all_disagreements
    
    # Get initial state
    current_predictions = get_em_predictions(current_tensor)
    all_disagreements = get_all_disagreements(current_tensor, current_predictions)
    available_disagreements = all_disagreements.copy()
    processed_pairs = set()
    
    print(f"Initial disagreements: {len(all_disagreements)}")
    print(f"Initial predictions: {np.bincount(current_predictions, minlength=J)}")
    
    previous_predictions = None
    previous_tensor = current_tensor.copy()
    
    for round_num in range(max_rounds):
        # Track current state
        convergence_trace['rounds'].append(round_num)
        convergence_trace['disagreement_counts'].append(len(available_disagreements))
        
        # Check if predictions changed
        if previous_predictions is not None:
            predictions_changed = np.sum(current_predictions != previous_predictions)
            convergence_trace['predictions_changed'].append(predictions_changed)
            
            if predictions_changed > 0:
                # Track which tasks changed
                changed_tasks = np.where(current_predictions != previous_predictions)[0]
                convergence_trace['prediction_changes_by_task'].append({
                    'round': round_num,
                    'changed_tasks': changed_tasks.tolist(),
                    'old_predictions': previous_predictions[changed_tasks].tolist(),
                    'new_predictions': current_predictions[changed_tasks].tolist()
                })
                print(f"Round {round_num}: {predictions_changed} predictions changed at tasks {changed_tasks}")
        else:
            convergence_trace['predictions_changed'].append(0)
        
        # If no available disagreements, we're done
        if not available_disagreements:
            print(f"Round {round_num}: No more available hints!")
            break
        
        # Sample a disagreement
        missed_relevant = [(s, t) for s, t, c in available_disagreements if c == 'missed_relevant']
        irrelevant_disagreement = [(s, t) for s, t, c in available_disagreements if c == 'irrelevant_disagreement']
        
        if irrelevant_disagreement:
            chosen_pair_idx = np.random.choice(len(irrelevant_disagreement))
            student_idx, task_idx = irrelevant_disagreement[chosen_pair_idx]
        elif missed_relevant:
            chosen_pair_idx = np.random.choice(len(missed_relevant))
            student_idx, task_idx = missed_relevant[chosen_pair_idx]
        else:
            # This shouldn't happen
            break
        
        # Apply the synthetic edit
        hint = current_predictions[task_idx]
        true_label = ground_truth_labels[task_idx]
        if true_label == 2:
            true_label = 1
        
        if sampling_method == 'global':
            q_vec = sample_student_global(alpha_global)
        elif sampling_method == 'bootstrap':
            from sampler import df_probs
            q_vec = sample_student_bootstrap(df_probs)
        elif sampling_method == 'hierarchical':
            q_vec = sample_student_hier(alpha_global, kappa=1.0)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        correct_to_agree = (hint == true_label)
        response = simulate_response(correct_to_agree, q_vec)
        
        # Track tensor change
        old_annotation = np.argmax(current_tensor[task_idx, :, student_idx])
        
        if response == 'AGREE':
            current_tensor[task_idx, :, student_idx] = 0
            current_tensor[task_idx, hint, student_idx] = 1
            new_annotation = hint
        else:
            current_tensor[task_idx, :, student_idx] = 0
            current_tensor[task_idx, 1 - hint, student_idx] = 1
            new_annotation = 1 - hint
        
        # Record the change
        convergence_trace['tensor_changes'].append({
            'round': round_num,
            'student_idx': student_idx,
            'task_idx': task_idx,
            'old_annotation': old_annotation,
            'new_annotation': new_annotation,
            'hint': hint,
            'response': response,
            'correct_to_agree': correct_to_agree
        })
        
        # Mark this pair as processed
        processed_pairs.add((student_idx, task_idx))
        
        # Remove this disagreement from available_disagreements
        available_disagreements = [(s, t, c) for s, t, c in available_disagreements if (s, t) != (student_idx, task_idx)]
        
        # Update for next round
        previous_predictions = current_predictions
        previous_tensor = current_tensor.copy()
        
        # Check if EM predictions changed
        new_predictions = get_em_predictions(current_tensor)
        if not np.array_equal(current_predictions, new_predictions):
            print(f"Round {round_num}: EM predictions changed, recomputing disagreements...")
            current_predictions = new_predictions
            all_disagreements = get_all_disagreements(current_tensor, current_predictions)
            available_disagreements = [(s, t, c) for s, t, c in all_disagreements if (s, t) not in processed_pairs]
            print(f"New disagreements after EM change: {len(available_disagreements)} available")
        
        # Print progress every 500 rounds
        if round_num % 500 == 0:
            print(f"Round {round_num}: {len(available_disagreements)} disagreements remaining")
    
    print(f"\nEM convergence trace completed!")
    print(f"Total rounds: {len(convergence_trace['rounds'])}")
    print(f"Total prediction changes: {sum(convergence_trace['predictions_changed'])}")
    
    return convergence_trace

def analyze_convergence_trace(convergence_trace):
    """
    Analyze the convergence trace to understand why EM gets stuck.
    
    Args:
        convergence_trace: Dictionary containing convergence trace data
    
    Returns:
        analysis: Dictionary containing analysis results
    """
    rounds = convergence_trace['rounds']
    predictions_changed = convergence_trace['predictions_changed']
    tensor_changes = convergence_trace['tensor_changes']
    
    # Find when predictions stop changing
    last_change_round = None
    for i, changed in enumerate(predictions_changed):
        if changed > 0:
            last_change_round = i
    
    # Analyze the pattern
    analysis = {
        'total_rounds': len(rounds),
        'total_prediction_changes': sum(predictions_changed),
        'last_change_round': last_change_round,
        'rounds_after_last_change': len(rounds) - last_change_round if last_change_round is not None else 0,
        'prediction_change_frequency': sum(predictions_changed) / len(rounds) if rounds else 0,
        'early_vs_late_changes': {
            'first_quarter': sum(predictions_changed[:len(rounds)//4]),
            'second_quarter': sum(predictions_changed[len(rounds)//4:len(rounds)//2]),
            'third_quarter': sum(predictions_changed[len(rounds)//2:3*len(rounds)//4]),
            'fourth_quarter': sum(predictions_changed[3*len(rounds)//4:])
        }
    }
    
    print(f"\n=== EM Convergence Analysis ===")
    print(f"Total rounds: {analysis['total_rounds']}")
    print(f"Total prediction changes: {analysis['total_prediction_changes']}")
    print(f"Last change at round: {analysis['last_change_round']}")
    print(f"Rounds after last change: {analysis['rounds_after_last_change']}")
    print(f"Prediction change frequency: {analysis['prediction_change_frequency']:.3f}")
    print(f"\nPrediction changes by quarter:")
    for quarter, count in analysis['early_vs_late_changes'].items():
        print(f"  {quarter}: {count} changes")
    
    return analysis






if __name__ == "__main__":
    import sys
    
    # Check if binary schema flag is provided
    binary_schema = "--binary" in sys.argv
    
    if binary_schema:
        print("Running with binary schema (treating 2 as 1 for metrics)")
    else:
        print("Running with three-class schema (0=not relevant, 1=relevant, 2=required)")
    
    main(binary_schema=binary_schema)
