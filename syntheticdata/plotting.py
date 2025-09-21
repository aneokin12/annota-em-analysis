import numpy as np
import matplotlib.pyplot as plt

def plot_student_improvements(student_improvements, summary_stats, save_path=None, binary_schema=False):
    """
    Plot student-level accuracy improvements.
    
    Args:
        student_improvements: Dictionary containing improvement data for each student
        summary_stats: Summary statistics dictionary
        save_path: Optional path to save the plot
        binary_schema: If True, treat 2 as 1 for metrics calculations
    """
    # Set publication-quality style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    
    K = len(student_improvements)
    student_indices = list(range(K))
    
    # Extract data for plotting
    initial_accuracies = [student_improvements[i]['initial']['accuracy'] for i in student_indices]
    final_accuracies = [student_improvements[i]['final']['accuracy'] for i in student_indices]
    improvements = [student_improvements[i]['improvements']['accuracy'] for i in student_indices]
    
    # Extract precision and recall data
    initial_precisions = [student_improvements[i]['initial']['precision'] for i in student_indices]
    final_precisions = [student_improvements[i]['final']['precision'] for i in student_indices]
    precision_improvements = [student_improvements[i]['improvements']['precision'] for i in student_indices]
    
    initial_recalls = [student_improvements[i]['initial']['recall'] for i in student_indices]
    final_recalls = [student_improvements[i]['final']['recall'] for i in student_indices]
    recall_improvements = [student_improvements[i]['improvements']['recall'] for i in student_indices]
    
    # Extract F1 data
    initial_f1s = [student_improvements[i]['initial']['f1'] for i in student_indices]
    final_f1s = [student_improvements[i]['final']['f1'] for i in student_indices]
    f1_improvements = [student_improvements[i]['improvements']['f1'] for i in student_indices]
    
    # Create the plot with horizontal layout
    schema_type = "Binary Schema (2→1)" if binary_schema else "Three-Class Schema (0,1,2)"
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Student Performance (Before Hints vs After)', fontsize=14, fontweight='bold', y=0.95)
    
    # Color palette for publication
    colors = {
        'scatter': '#2E86AB',
        'diagonal': '#A23B72',
        'improvement': '#F18F01',
        'decline': '#C73E1D',
        'no_change': '#6B5B95',
        'best_fit': '#2E8B57',
        'grid': '#E0E0E0'
    }
    
    # Plot 1: Before vs After accuracy
    ax1.scatter(initial_accuracies, final_accuracies, alpha=0.8, s=60, color=colors['scatter'], edgecolors='white', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], color=colors['diagonal'], linestyle='--', alpha=0.7, linewidth=1.5, label='No Change')
    
    # Add line of best fit
    if len(initial_accuracies) > 1:
        z = np.polyfit(initial_accuracies, final_accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(initial_accuracies), max(initial_accuracies), 100)
        y_line = p(x_line)
        slope = z[0]
        
        # Color code the line based on slope
        if slope < 1:
            line_color = colors['best_fit']
            line_label = f'Best Fit (slope={slope:.3f})'
        elif slope > 1:
            line_color = colors['decline']
            line_label = f'Best Fit (slope={slope:.3f})'
        else:
            line_color = colors['no_change']
            line_label = f'Best Fit (slope={slope:.3f})'
        
        ax1.plot(x_line, y_line, color=line_color, linewidth=2.5, alpha=0.9, label=line_label)
        
        # Add slope interpretation text
        if slope < 1:
            interpretation = "Less proficient students\nimprove most"
        elif slope > 1:
            interpretation = "More proficient students\nimprove most"
        else:
            interpretation = "Equal improvement\nacross proficiency levels"
        
        ax1.text(0.02, 0.98, f"Slope: {slope:.3f}\n{interpretation}", 
                 transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F8FF', alpha=0.9, edgecolor='#4682B4'))
    
    ax1.set_xlabel('Initial Accuracy', fontweight='bold')
    ax1.set_ylabel('Final Accuracy', fontweight='bold')
    ax1.set_title('Student Accuracy: Before vs After', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Add diagonal line for reference
    min_acc = min(min(initial_accuracies), min(final_accuracies))
    max_acc = max(max(initial_accuracies), max(final_accuracies))
    ax1.set_xlim(min_acc - 0.05, max_acc + 0.05)
    ax1.set_ylim(min_acc - 0.05, max_acc + 0.05)
    
    # Plot 2: Before vs After F1 score
    ax2.scatter(initial_f1s, final_f1s, alpha=0.8, s=60, color=colors['improvement'], edgecolors='white', linewidth=0.5)
    ax2.plot([0, 1], [0, 1], color=colors['diagonal'], linestyle='--', alpha=0.7, linewidth=1.5, label='No Change')
    
    # Add line of best fit for F1
    if len(initial_f1s) > 1:
        z_f1 = np.polyfit(initial_f1s, final_f1s, 1)
        p_f1 = np.poly1d(z_f1)
        x_line_f1 = np.linspace(min(initial_f1s), max(initial_f1s), 100)
        y_line_f1 = p_f1(x_line_f1)
        slope_f1 = z_f1[0]
        ax2.plot(x_line_f1, y_line_f1, color=colors['improvement'], linewidth=2.5, alpha=0.9, 
                label=f'Best Fit (slope={slope_f1:.3f})')
    
    ax2.set_xlabel('Initial F1 Score', fontweight='bold')
    ax2.set_ylabel('Final F1 Score', fontweight='bold')
    ax2.set_title('Student F1 Score: Before vs After', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Add diagonal line for reference
    min_f1 = min(min(initial_f1s), min(final_f1s))
    max_f1 = max(max(initial_f1s), max(final_f1s))
    ax2.set_xlim(min_f1 - 0.05, max_f1 + 0.05)
    ax2.set_ylim(min_f1 - 0.05, max_f1 + 0.05)
    
    # Plot 3: Before vs After precision
    ax3.scatter(initial_precisions, final_precisions, alpha=0.8, s=60, color=colors['best_fit'], edgecolors='white', linewidth=0.5)
    ax3.plot([0, 1], [0, 1], color=colors['diagonal'], linestyle='--', alpha=0.7, linewidth=1.5, label='No Change')
    
    # Add line of best fit for precision
    if len(initial_precisions) > 1:
        z_prec = np.polyfit(initial_precisions, final_precisions, 1)
        p_prec = np.poly1d(z_prec)
        x_line_prec = np.linspace(min(initial_precisions), max(initial_precisions), 100)
        y_line_prec = p_prec(x_line_prec)
        slope_prec = z_prec[0]
        ax3.plot(x_line_prec, y_line_prec, color=colors['best_fit'], linewidth=2.5, alpha=0.9, 
                label=f'Best Fit (slope={slope_prec:.3f})')
    
    ax3.set_xlabel('Initial Precision', fontweight='bold')
    ax3.set_ylabel('Final Precision', fontweight='bold')
    ax3.set_title('Student Precision: Before vs After', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, color=colors['grid'])
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Add diagonal line for reference
    min_prec = min(min(initial_precisions), min(final_precisions))
    max_prec = max(max(initial_precisions), max(final_precisions))
    ax3.set_xlim(min_prec - 0.05, max_prec + 0.05)
    ax3.set_ylim(min_prec - 0.05, max_prec + 0.05)
    
    # Plot 4: Before vs After recall
    ax4.scatter(initial_recalls, final_recalls, alpha=0.8, s=60, color=colors['improvement'], edgecolors='white', linewidth=0.5)
    ax4.plot([0, 1], [0, 1], color=colors['diagonal'], linestyle='--', alpha=0.7, linewidth=1.5, label='No Change')
    
    # Add line of best fit for recall
    if len(initial_recalls) > 1:
        z_rec = np.polyfit(initial_recalls, final_recalls, 1)
        p_rec = np.poly1d(z_rec)
        x_line_rec = np.linspace(min(initial_recalls), max(initial_recalls), 100)
        y_line_rec = p_rec(x_line_rec)
        slope_rec = z_rec[0]
        ax4.plot(x_line_rec, y_line_rec, color=colors['improvement'], linewidth=2.5, alpha=0.9, 
                label=f'Best Fit (slope={slope_rec:.3f})')
    
    ax4.set_xlabel('Initial Recall', fontweight='bold')
    ax4.set_ylabel('Final Recall', fontweight='bold')
    ax4.set_title('Student Recall: Before vs After', fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, color=colors['grid'])
    ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Add diagonal line for reference
    min_rec = min(min(initial_recalls), min(final_recalls))
    max_rec = max(max(initial_recalls), max(final_recalls))
    ax4.set_xlim(min_rec - 0.05, max_rec + 0.05)
    ax4.set_ylim(min_rec - 0.05, max_rec + 0.05)
    
    # Improve overall layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Student improvements plot saved to {save_path}")
    
    plt.show()

def plot_task_improvements(task_improvements, summary_stats, save_path=None, binary_schema=False):
    """
    Plot task-level accuracy improvements.
    
    Args:
        task_improvements: Dictionary containing improvement data for each task
        summary_stats: Summary statistics dictionary
        save_path: Optional path to save the plot
        binary_schema: If True, treat 2 as 1 for metrics calculations
    """
    # Set publication-quality style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    
    N = len(task_improvements)
    task_indices = list(range(N))
    
    # Extract data for plotting
    initial_accuracies = [task_improvements[i]['initial']['accuracy'] for i in task_indices]
    final_accuracies = [task_improvements[i]['final']['accuracy'] for i in task_indices]
    improvements = [task_improvements[i]['improvements']['accuracy'] for i in task_indices]
    
    # Create the plot with better spacing
    schema_type = "Binary Schema (2→1)" if binary_schema else "Three-Class Schema (0,1,2)"
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Task-Level Accuracy Improvements\n{schema_type}', fontsize=14, fontweight='bold', y=0.95)
    
    # Color palette for publication
    colors = {
        'scatter': '#2E86AB',
        'diagonal': '#A23B72',
        'improvement': '#F18F01',
        'decline': '#C73E1D',
        'no_change': '#6B5B95',
        'initial': '#4682B4',
        'final': '#32CD32',
        'histogram': '#87CEEB',
        'grid': '#E0E0E0'
    }
    
    # Plot 1: Before vs After accuracy
    ax1.scatter(initial_accuracies, final_accuracies, alpha=0.8, s=40, color=colors['scatter'], edgecolors='white', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], color=colors['diagonal'], linestyle='--', alpha=0.7, linewidth=1.5, label='No Change')
    ax1.set_xlabel('Initial Task Accuracy', fontweight='bold')
    ax1.set_ylabel('Final Task Accuracy', fontweight='bold')
    ax1.set_title('Task Accuracy: Before vs After', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Add diagonal line for reference
    min_acc = min(min(initial_accuracies), min(final_accuracies))
    max_acc = max(max(initial_accuracies), max(final_accuracies))
    ax1.set_xlim(min_acc - 0.05, max_acc + 0.05)
    ax1.set_ylim(min_acc - 0.05, max_acc + 0.05)
    
    # Plot 2: Improvement distribution
    bar_colors = [colors['improvement'] if imp > 0 else colors['decline'] if imp < 0 else colors['no_change'] for imp in improvements]
    bars = ax2.bar(task_indices, improvements, color=bar_colors, alpha=0.8, width=1.0, edgecolor='white', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Task Index', fontweight='bold')
    ax2.set_ylabel('Accuracy Improvement', fontweight='bold')
    ax2.set_title('Individual Task Improvements', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    
    # Plot 3: Task accuracy over time (line plot)
    ax3.plot(task_indices, initial_accuracies, color=colors['initial'], alpha=0.8, label='Initial', linewidth=2)
    ax3.plot(task_indices, final_accuracies, color=colors['final'], alpha=0.8, label='Final', linewidth=2)
    ax3.fill_between(task_indices, initial_accuracies, final_accuracies, 
                     where=(np.array(final_accuracies) > np.array(initial_accuracies)), 
                     alpha=0.3, color=colors['improvement'], label='Improvement')
    ax3.fill_between(task_indices, initial_accuracies, final_accuracies, 
                     where=(np.array(final_accuracies) < np.array(initial_accuracies)), 
                     alpha=0.3, color=colors['decline'], label='Decline')
    ax3.set_xlabel('Task Index', fontweight='bold')
    ax3.set_ylabel('Task Accuracy', fontweight='bold')
    ax3.set_title('Task Accuracy Evolution', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, color=colors['grid'])
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Plot 4: Improvement magnitude distribution (histogram)
    ax4.hist(improvements, bins=20, alpha=0.8, color=colors['histogram'], edgecolor='white', linewidth=0.8)
    ax4.axvline(x=0, color=colors['diagonal'], linestyle='--', alpha=0.8, linewidth=2, label='No Change')
    ax4.axvline(x=np.mean(improvements), color=colors['improvement'], linestyle='-', alpha=0.8, linewidth=2.5, 
                label=f'Mean: {np.mean(improvements):+.3f}')
    ax4.set_xlabel('Accuracy Improvement', fontweight='bold')
    ax4.set_ylabel('Number of Tasks', fontweight='bold')
    ax4.set_title('Distribution of Task Improvements', fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, color=colors['grid'])
    ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Add summary text to the first plot
    summary_text = f"Mean Improvement: {summary_stats['mean_improvement']:+.3f}\n"
    summary_text += f"Tasks Improved: {summary_stats['tasks_improved']}/{N}\n"
    summary_text += f"Tasks Declined: {summary_stats['tasks_declined']}/{N}"
    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8DC', alpha=0.9, edgecolor='#DAA520'))
    
    # Improve overall layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Task improvements plot saved to {save_path}")
    
    plt.show()

def plot_hint_quality(analysis, save_path=None, binary_schema=False):
    """
    Plot the hint quality analysis results.
    
    Args:
        analysis: Dictionary containing hint quality analysis
        save_path: Optional path to save the plot
        binary_schema: If True, treat 2 as 1 for metrics calculations
    """
    # Set publication-quality style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    
    categories = ['missed_relevant', 'irrelevant_disagreement']
    counts = [analysis[cat]['count'] for cat in categories]
    correct_counts = [analysis[cat]['correct_count'] for cat in categories]
    accuracies = [analysis[cat]['accuracy'] for cat in categories]
    
    # Create the plot with better spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    schema_type = "Binary Schema (2→1)" if binary_schema else "Three-Class Schema (0,1,2)"
    fig.suptitle(f'Hint Quality Analysis\n{schema_type}', fontsize=14, fontweight='bold', y=0.95)
    
    # Color palette for publication
    colors = {
        'counts': ['#2E86AB', '#F18F01'],
        'accuracies': ['#2E8B57', '#C73E1D'],
        'grid': '#E0E0E0'
    }
    
    # Plot 1: Count of hints by category
    bars1 = ax1.bar(categories, counts, color=colors['counts'], alpha=0.8, edgecolor='white', linewidth=1)
    ax1.set_ylabel('Number of Hints', fontweight='bold')
    ax1.set_title('Total Hints by Category', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 2: Accuracy of hints by category
    bars2 = ax2.bar(categories, accuracies, color=colors['accuracies'], alpha=0.8, edgecolor='white', linewidth=1)
    ax2.set_ylabel('Hint Accuracy', fontweight='bold')
    ax2.set_title('Hint Accuracy by Category', fontweight='bold', pad=15)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    
    # Add value labels on bars
    for bar, acc in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Improve layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Hint quality plot saved to {save_path}")
    
    plt.show()

def plot_task_improvement_breakdown(breakdown, save_path=None, binary_schema=False):
    """
    Plot the task improvement breakdown analysis.
    
    Args:
        breakdown: Dictionary containing breakdown analysis
        save_path: Optional path to save the plot
        binary_schema: If True, treat 2 as 1 for metrics calculations
    """
    # Set publication-quality style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    
    categories = ['improved', 'declined', 'unchanged']
    counts = [breakdown[cat]['count'] for cat in categories]
    
    # Create the plot with better spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
    schema_type = "Binary Schema (2→1)" if binary_schema else "Three-Class Schema (0,1,2)"
    
    # Color palette for publication
    colors = {
        'improved': '#2E8B57',
        'declined': '#C73E1D',
        'unchanged': '#6B5B95',
        'grid': '#E0E0E0',
        'labels': ['#2E86AB', '#F18F01', '#A23B72', '#32CD32', '#FF6B6B']
    }
    
    # Plot 1: Task counts by improvement status
    bar_colors = [colors['improved'], colors['declined'], colors['unchanged']]
    bars1 = ax1.bar(categories, counts, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax1.set_ylabel('Number of Tasks', fontweight='bold')
    ax1.set_title('Task Counts by Improvement Status', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 2: Ground truth distribution by category
    all_labels = set()
    for category in categories:
        all_labels.update(breakdown[category]['ground_truth_distribution'].keys())
    
    all_labels = sorted(all_labels)
    x = np.arange(len(categories))
    width = 0.8 / len(all_labels)
    
    for i, label in enumerate(all_labels):
        values = [breakdown[cat]['ground_truth_distribution'].get(label, 0) for cat in categories]
        ax2.bar(x + i*width, values, width, label=f'Label {label}', alpha=0.8, 
                color=colors['labels'][i % len(colors['labels'])], edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('Improvement Status', fontweight='bold')
    ax2.set_ylabel('Number of Tasks', fontweight='bold')
    ax2.set_title('Ground Truth Label Distribution by Category', fontweight='bold', pad=15)
    ax2.set_xticks(x + width * (len(all_labels) - 1) / 2)
    ax2.set_xticklabels(categories)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    
    # Plot 3: Hint types by category
    hint_types = ['missed_relevant', 'irrelevant_disagreement', 'other']
    hint_data = {}
    for hint_type in hint_types:
        hint_data[hint_type] = [breakdown[cat]['hint_analysis'][hint_type] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.8 / len(hint_types)
    
    for i, hint_type in enumerate(hint_types):
        ax3.bar(x + i*width, hint_data[hint_type], width, 
                label=hint_type.replace('_', ' ').title(), alpha=0.8,
                color=colors['labels'][i % len(colors['labels'])], edgecolor='white', linewidth=0.5)
    
    ax3.set_xlabel('Improvement Status', fontweight='bold')
    ax3.set_ylabel('Number of Hints', fontweight='bold')
    ax3.set_title('Hint Types by Category', fontweight='bold', pad=15)
    ax3.set_xticks(x + width * (len(hint_types) - 1) / 2)
    ax3.set_xticklabels(categories)
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax3.grid(True, alpha=0.3, color=colors['grid'])
    
    # Plot 4: Hint accuracy by category
    accuracies = []
    for category in categories:
        hint_data = breakdown[category]['hint_analysis']
        if hint_data['total_hints'] > 0:
            accuracy = hint_data['correct_hints'] / hint_data['total_hints']
        else:
            accuracy = 0
        accuracies.append(accuracy)
    
    bars4 = ax4.bar(categories, accuracies, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax4.set_ylabel('Hint Accuracy', fontweight='bold')
    ax4.set_title('Hint Accuracy by Category', fontweight='bold', pad=15)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3, color=colors['grid'])
    
    # Add value labels on bars
    for bar, acc in zip(bars4, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Improve overall layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Task improvement breakdown plot saved to {save_path}")
    
    plt.show()
