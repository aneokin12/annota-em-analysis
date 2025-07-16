import numpy as np
import matplotlib.pyplot as plt
from model import dawid_skene
from sampler import (
    sample_student_global, 
    sample_student_bootstrap, 
    sample_student_hier,
    simulate_response,
    alpha_global
)
from gensynthlabels import generate_synthetic_labels
from metrics import calculate_all_metrics, print_metrics_summary

def load_initial_tensor():
    """Load the initial hint tensor from disk."""
    return np.load('hint_tensor.npy')

def get_em_predictions(tensor):
    """Run Dawid-Skene EM algorithm to get predictions for each task."""
    predictions = dawid_skene(tensor)
    return predictions

def generate_hint_for_task(prediction):
    """Convert EM prediction to a hint (0 = not relevant, 1 = relevant)."""
    return prediction

def apply_synthetic_hints_once(tensor, ground_truth_labels, sampling_method='global'):
    """
    Apply synthetic hints to the tensor once.
    Args:
        tensor: N x J x K tensor (tasks x labels x students)
        ground_truth_labels: 1D array of length N, true label for each task
        sampling_method: 'global', 'bootstrap', or 'hierarchical'
    Returns:
        Updated tensor with synthetic hint responses
    """
    N, J, K = tensor.shape
    em_predictions = get_em_predictions(tensor)
    print(f"EM predictions shape: {em_predictions.shape}")
    print(f"Sample predictions: {em_predictions[:10]}")
    updated_tensor = tensor.copy()
    for task_idx in range(N):
        hint = generate_hint_for_task(em_predictions[task_idx])
        true_label = ground_truth_labels[task_idx]
        print(f"Task {task_idx}: EM prediction = {hint}, True label = {true_label} (0=not relevant, 1=relevant)")
        for student_idx in range(K):
            if sampling_method == 'global':
                q_vec = sample_student_global(alpha_global)
            elif sampling_method == 'bootstrap':
                from sampler import df_probs
                q_vec = sample_student_bootstrap(df_probs)
            elif sampling_method == 'hierarchical':
                q_vec = sample_student_hier(alpha_global, kappa=1.0)
            else:
                raise ValueError(f"Unknown sampling method: {sampling_method}")
            # Determine if it is correct to agree (prediction matches ground truth)
            correct_to_agree = (hint == true_label)
            response = simulate_response(correct_to_agree, q_vec)
            if response == 'AGREE':
                updated_tensor[task_idx, :, student_idx] = 0
                updated_tensor[task_idx, hint, student_idx] = 1
            else:
                updated_tensor[task_idx, :, student_idx] = 0
                updated_tensor[task_idx, 1 - hint, student_idx] = 1
    return updated_tensor

def apply_one_synthetic_edit(tensor, ground_truth_labels, student_idx, sampling_method='global'):
    """
    Perform one synthetic edit: one student responds to a hint on a task where they disagree with the EM prediction.
    Args:
        tensor: N x J x K tensor (tasks x labels x students)
        ground_truth_labels: 1D array of length N, true label for each task
        student_idx: index of the student to make the edit
        sampling_method: 'global', 'bootstrap', or 'hierarchical'
    Returns:
        updated_tensor: tensor after the edit
        edited_task_idx: index of the task edited (or None if no edit was made)
    """
    N, J, K = tensor.shape
    em_predictions = get_em_predictions(tensor)
    updated_tensor = tensor.copy()
    for task_idx in range(N):
        hint = generate_hint_for_task(em_predictions[task_idx])
        true_label = ground_truth_labels[task_idx]
        # Find student's current annotation (argmax over labels)
        student_annotation = np.argmax(tensor[task_idx, :, student_idx])
        # Only respond if annotation disagrees with EM prediction
        if student_annotation != hint:
            # Sample confusion matrix for this student
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
            if response == 'AGREE':
                updated_tensor[task_idx, :, student_idx] = 0
                updated_tensor[task_idx, hint, student_idx] = 1
            else:
                updated_tensor[task_idx, :, student_idx] = 0
                updated_tensor[task_idx, 1 - hint, student_idx] = 1
            print(f"Student {student_idx} edited task {task_idx}: disagreed with EM, responded '{response}' (hint={hint}, true_label={true_label})")
            return updated_tensor, task_idx  # Only one edit per round
    print(f"Student {student_idx} had no disagreements with EM predictions; no edit made.")
    return updated_tensor, None

def run_until_convergence(tensor, ground_truth_labels, max_rounds=1000, convergence_threshold=5, sampling_method='global'):
    """
    Run synthetic edit rounds until convergence is detected.
    
    Args:
        tensor: Initial N x J x K tensor
        ground_truth_labels: 1D array of length N, true label for each task
        max_rounds: Maximum number of rounds to run
        convergence_threshold: Number of consecutive rounds with no EM prediction changes to consider converged
        sampling_method: 'global', 'bootstrap', or 'hierarchical'
    
    Returns:
        final_tensor: Tensor after convergence
        round_history: List of (round_num, student_idx, task_idx, predictions_changed) tuples
        converged_round: Round number when convergence was detected (or max_rounds if not converged)
        metrics_history: List of metric dictionaries for each round
    """
    N, J, K = tensor.shape
    current_tensor = tensor.copy()
    round_history = []
    metrics_history = []
    unchanged_rounds = 0
    previous_predictions = None
    
    print(f"Starting convergence run: max_rounds={max_rounds}, convergence_threshold={convergence_threshold}")
    
    for round_num in range(max_rounds):
        # Get current EM predictions
        current_predictions = get_em_predictions(current_tensor)
        
        # Calculate metrics for this round
        metrics = calculate_all_metrics(current_predictions, ground_truth_labels)
        metrics_history.append(metrics)
        
        # Print metrics every 10 rounds or when predictions change significantly
        if round_num % 10 == 0 or round_num < 5:
            print_metrics_summary(current_predictions, ground_truth_labels, round_num)
        
        # Check if predictions changed from previous round
        if previous_predictions is not None:
            predictions_changed = np.sum(current_predictions != previous_predictions)
            if predictions_changed == 0:
                unchanged_rounds += 1
                print(f"Round {round_num}: No EM prediction changes (unchanged_rounds={unchanged_rounds})")
            else:
                unchanged_rounds = 0
                print(f"Round {round_num}: {predictions_changed} EM predictions changed")
        else:
            predictions_changed = 0
            print(f"Round {round_num}: Initial EM predictions")
        
        # Check for convergence
        if unchanged_rounds >= convergence_threshold:
            print(f"Convergence detected after {round_num} rounds!")
            break
        
        # Pick a random student for this round
        student_idx = np.random.randint(0, K)
        
        # Apply one synthetic edit
        updated_tensor, edited_task_idx = apply_one_synthetic_edit(
            current_tensor, ground_truth_labels, student_idx, sampling_method
        )
        
        # Record the round
        round_history.append((round_num, student_idx, edited_task_idx, predictions_changed))
        
        # Update for next round
        current_tensor = updated_tensor
        previous_predictions = current_predictions
        
        # Print progress every 10 rounds
        if round_num % 10 == 0:
            print(f"Completed {round_num} rounds...")
    
    converged_round = round_num if unchanged_rounds >= convergence_threshold else max_rounds
    
    return current_tensor, round_history, converged_round, metrics_history

def plot_metrics_over_rounds(metrics_history, save_path=None):
    """
    Plot F1 score, precision, recall, and accuracy over rounds.
    
    Args:
        metrics_history: List of metric dictionaries for each round
        save_path: Optional path to save the plot
    """
    rounds = list(range(len(metrics_history)))
    
    # Extract metrics
    f1_scores = [m['f1'] for m in metrics_history]
    precision_scores = [m['precision'] for m in metrics_history]
    recall_scores = [m['recall'] for m in metrics_history]
    accuracy_scores = [m['accuracy'] for m in metrics_history]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(rounds, f1_scores, 'b-', linewidth=2, label='F1 Score')
    plt.xlabel('Round')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(rounds, precision_scores, 'g-', linewidth=2, label='Precision')
    plt.xlabel('Round')
    plt.ylabel('Precision')
    plt.title('Precision Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(rounds, recall_scores, 'r-', linewidth=2, label='Recall')
    plt.xlabel('Round')
    plt.ylabel('Recall')
    plt.title('Recall Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(rounds, accuracy_scores, 'purple', linewidth=2, label='Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def run_multiple_trials(tensor, ground_truth_labels, num_trials=10, max_rounds=1000, convergence_threshold=20, sampling_method='global'):
    """
    Run multiple trials of the synthetic hint convergence process and average the results.
    
    Args:
        tensor: Initial N x J x K tensor
        ground_truth_labels: 1D array of length N, true label for each task
        num_trials: Number of trials to run
        max_rounds: Maximum number of rounds per trial
        convergence_threshold: Number of consecutive rounds with no EM prediction changes to consider converged
        sampling_method: 'global', 'bootstrap', or 'hierarchical'
    
    Returns:
        averaged_metrics_history: List of averaged metric dictionaries for each round
        trial_results: List of (final_tensor, round_history, converged_round, metrics_history) for each trial
        max_rounds_across_trials: Maximum number of rounds across all trials
    """
    print(f"Running {num_trials} trials...")
    
    trial_results = []
    max_rounds_across_trials = 0
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
        # Set different random seed for each trial
        np.random.seed(trial)
        
        # Run single trial
        final_tensor, round_history, converged_round, metrics_history = run_until_convergence(
            tensor, ground_truth_labels, max_rounds, convergence_threshold, sampling_method
        )
        
        trial_results.append((final_tensor, round_history, converged_round, metrics_history))
        max_rounds_across_trials = max(max_rounds_across_trials, len(metrics_history))
        
        print(f"Trial {trial + 1} completed: {len(round_history)} rounds, converged at {converged_round}")
    
    # Calculate averaged metrics across trials
    print(f"\nCalculating averaged metrics across {num_trials} trials...")
    averaged_metrics_history = []
    
    for round_num in range(max_rounds_across_trials):
        round_metrics = []
        
        for trial_result in trial_results:
            _, _, _, metrics_history = trial_result
            
            # If this trial has data for this round, include it
            if round_num < len(metrics_history):
                round_metrics.append(metrics_history[round_num])
        
        # Average the metrics for this round across all trials that reached this round
        if round_metrics:
            avg_metrics = {}
            for metric_name in ['f1', 'precision', 'recall', 'accuracy']:
                values = [m[metric_name] for m in round_metrics]
                avg_metrics[metric_name] = np.mean(values)
                avg_metrics[f'{metric_name}_std'] = np.std(values)
            
            averaged_metrics_history.append(avg_metrics)
        else:
            break
    
    print(f"Averaged metrics calculated for {len(averaged_metrics_history)} rounds")
    
    return averaged_metrics_history, trial_results, max_rounds_across_trials

def plot_averaged_metrics_over_rounds(averaged_metrics_history, save_path=None, show_std=True):
    """
    Plot averaged F1 score, precision, recall, and accuracy over rounds with standard deviation.
    
    Args:
        averaged_metrics_history: List of averaged metric dictionaries for each round
        save_path: Optional path to save the plot
        show_std: Whether to show standard deviation bands
    """
    rounds = list(range(len(averaged_metrics_history)))
    
    # Extract averaged metrics
    f1_scores = [m['f1'] for m in averaged_metrics_history]
    precision_scores = [m['precision'] for m in averaged_metrics_history]
    recall_scores = [m['recall'] for m in averaged_metrics_history]
    accuracy_scores = [m['accuracy'] for m in averaged_metrics_history]
    
    # Extract standard deviations if available
    f1_stds = [m.get('f1_std', 0) for m in averaged_metrics_history]
    precision_stds = [m.get('precision_std', 0) for m in averaged_metrics_history]
    recall_stds = [m.get('recall_std', 0) for m in averaged_metrics_history]
    accuracy_stds = [m.get('accuracy_std', 0) for m in averaged_metrics_history]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(rounds, f1_scores, 'b-', linewidth=2, label='F1 Score')
    if show_std and any(f1_stds):
        plt.fill_between(rounds, 
                        [f1 - std for f1, std in zip(f1_scores, f1_stds)],
                        [f1 + std for f1, std in zip(f1_scores, f1_stds)],
                        alpha=0.3, color='blue')
    plt.xlabel('Round')
    plt.ylabel('F1 Score')
    plt.title('Average F1 Score Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(rounds, precision_scores, 'g-', linewidth=2, label='Precision')
    if show_std and any(precision_stds):
        plt.fill_between(rounds, 
                        [p - std for p, std in zip(precision_scores, precision_stds)],
                        [p + std for p, std in zip(precision_scores, precision_stds)],
                        alpha=0.3, color='green')
    plt.xlabel('Round')
    plt.ylabel('Precision')
    plt.title('Average Precision Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(rounds, recall_scores, 'r-', linewidth=2, label='Recall')
    if show_std and any(recall_stds):
        plt.fill_between(rounds, 
                        [r - std for r, std in zip(recall_scores, recall_stds)],
                        [r + std for r, std in zip(recall_scores, recall_stds)],
                        alpha=0.3, color='red')
    plt.xlabel('Round')
    plt.ylabel('Recall')
    plt.title('Average Recall Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(rounds, accuracy_scores, 'purple', linewidth=2, label='Accuracy')
    if show_std and any(accuracy_stds):
        plt.fill_between(rounds, 
                        [a - std for a, std in zip(accuracy_scores, accuracy_stds)],
                        [a + std for a, std in zip(accuracy_scores, accuracy_stds)],
                        alpha=0.3, color='purple')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Average Accuracy Over Rounds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Averaged metrics plot saved to {save_path}")
    
    plt.show()

def main():
    print("Loading initial tensor...")
    initial_tensor = load_initial_tensor()
    print(f"Initial tensor shape: {initial_tensor.shape}")
    print(f"Initial total annotations: {np.sum(initial_tensor)}")
    
    print("\nGenerating synthetic ground truth labels...")
    ground_truth_labels = generate_synthetic_labels(initial_tensor)
    print(f"Ground truth label counts: {np.bincount(ground_truth_labels)}")
    print(f"Percentage relevant: {np.mean(ground_truth_labels)*100:.1f}%")

    # Run multiple trials
    print("\nRunning multiple trials...")
    averaged_metrics_history, trial_results, max_rounds = run_multiple_trials(
        initial_tensor, ground_truth_labels, num_trials=5, max_rounds=1000, convergence_threshold=20
    )
    
    # Show summary statistics
    print(f"\nTrial Summary:")
    print(f"Number of trials: {len(trial_results)}")
    print(f"Max rounds across trials: {max_rounds}")
    
    # Calculate average convergence round
    convergence_rounds = [trial_result[2] for trial_result in trial_results]
    avg_convergence = np.mean(convergence_rounds)
    std_convergence = np.std(convergence_rounds)
    print(f"Average convergence round: {avg_convergence:.1f} ± {std_convergence:.1f}")
    
    # Show final averaged metrics
    if averaged_metrics_history:
        final_avg_metrics = averaged_metrics_history[-1]
        print(f"\nFinal Averaged Metrics:")
        print(f"F1: {final_avg_metrics['f1']:.3f} ± {final_avg_metrics.get('f1_std', 0):.3f}")
        print(f"Precision: {final_avg_metrics['precision']:.3f} ± {final_avg_metrics.get('precision_std', 0):.3f}")
        print(f"Recall: {final_avg_metrics['recall']:.3f} ± {final_avg_metrics.get('recall_std', 0):.3f}")
        print(f"Accuracy: {final_avg_metrics['accuracy']:.3f} ± {final_avg_metrics.get('accuracy_std', 0):.3f}")
    
    # Plot averaged metrics over rounds
    print(f"\nPlotting averaged metrics over {len(averaged_metrics_history)} rounds...")
    plot_averaged_metrics_over_rounds(averaged_metrics_history, save_path='averaged_metrics_over_rounds.png')
    
    return averaged_metrics_history, trial_results

if __name__ == "__main__":
    main()
