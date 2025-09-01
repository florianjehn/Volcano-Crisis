"""
Plotting module for Volcano Crisis analysis.
This module creates visualizations from the prepared data to explore
the relationship between volcanic eruptions and global crises.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


def plot_crisis_timeline(crisis_matrix_path: str = '../results/crisis_overview.csv',
                        save_path: str = '../results/crisis_timeline.png',
                        figsize: tuple = (20, 12)):
    """
    Create a timeline plot showing all crises and their durations.
    
    Args:
        crisis_matrix_path: Path to the crisis overview CSV
        save_path: Path to save the plot
        figsize: Figure size for the plot
    """
    print("Creating crisis timeline plot...")
    
    # Read the crisis matrix
    crisis_matrix = pd.read_csv(crisis_matrix_path, index_col=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get crisis periods (start and end years for each crisis)
    crisis_periods = []
    for crisis in crisis_matrix.columns:
        active_years = crisis_matrix[crisis_matrix[crisis] == 1].index
        if len(active_years) > 0:
            # Find continuous periods
            start_year = active_years[0]
            end_year = active_years[0]
            
            for i in range(1, len(active_years)):
                if active_years[i] - active_years[i-1] > 1:
                    # Gap found, save current period
                    crisis_periods.append({
                        'crisis': crisis,
                        'start': start_year,
                        'end': end_year
                    })
                    start_year = active_years[i]
                end_year = active_years[i]
            
            # Save last period
            crisis_periods.append({
                'crisis': crisis,
                'start': start_year,
                'end': end_year
            })
    
    # Sort by start date
    crisis_periods.sort(key=lambda x: x['start'])
    
    # Plot each crisis as a horizontal line
    y_positions = {}
    current_y = 0
    
    for period in crisis_periods:
        crisis_name = period['crisis']
        if crisis_name not in y_positions:
            y_positions[crisis_name] = current_y
            current_y += 1
        
        # Plot line for this period
        ax.plot([period['start'], period['end']], 
               [y_positions[crisis_name], y_positions[crisis_name]],
               linewidth=2, alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Crisis', fontsize=12)
    ax.set_title('Timeline of All Crises (Sorted by Start Date)', fontsize=14, fontweight='bold')
    
    # Set y-axis labels (show every nth crisis name to avoid overcrowding)
    n_crises = len(y_positions)
    step = max(1, n_crises // 30)  # Show max 30 labels
    
    selected_crises = list(y_positions.keys())[::step]
    selected_positions = [y_positions[c] for c in selected_crises]
    
    ax.set_yticks(selected_positions)
    ax.set_yticklabels(selected_crises, fontsize=8)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for major volcanic eruptions
    volcano_df = pd.read_csv('../data/volcano_list.csv')
    volcano_df = volcano_df[volcano_df['year'] <= 1920]
    
    # Add VEI 7 eruptions as red lines
    vei7 = volcano_df[volcano_df['VEI'] == 7]
    for _, eruption in vei7.iterrows():
        ax.axvline(x=eruption['year'], color='red', alpha=0.3, linestyle='--', linewidth=1)
    
    # Add VEI 6 eruptions as orange lines
    vei6 = volcano_df[volcano_df['VEI'] == 6]
    for _, eruption in vei6.iterrows():
        ax.axvline(x=eruption['year'], color='orange', alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Add legend for volcanic eruptions
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='VEI 7 Eruption'),
        Line2D([0], [0], color='orange', linestyle='--', alpha=0.5, label='VEI 6 Eruption'),
        Line2D([0], [0], color='blue', linewidth=2, alpha=0.7, label='Crisis Period')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Crisis timeline saved to {save_path}")
    plt.show()


def plot_comparison_all_windows(results_path: str = '../results/volcano_crisis_analysis.pkl',
                               save_path: str = '../results/comparison_all_windows.png'):
    """
    Create comprehensive comparison plots for all time windows.
    
    Args:
        results_path: Path to the analysis results pickle file
        save_path: Path to save the plot
    """
    print("Creating comprehensive comparison plots...")
    
    # Load results
    with open(results_path, 'rb') as f:
        all_results = pickle.load(f)
    
    # Create subplots for each time window
    time_windows = sorted(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, window in enumerate(time_windows):
        ax = axes[idx]
        results = all_results[window]
        
        # Prepare data for plotting
        plot_data = []
        
        # Random sampling distribution
        plot_data.extend([('Random\nSampling', val) for val in results['random']['crisis_counts']])
        
        # VEI 6 post-eruption
        if results['vei6']['post_eruption_crisis_counts']:
            plot_data.extend([('VEI 6\nPost', val) for val in results['vei6']['post_eruption_crisis_counts']])
        
        # VEI 6 pre-eruption
        if 'pre_eruption_crisis_counts' in results['vei6'] and results['vei6']['pre_eruption_crisis_counts']:
            plot_data.extend([('VEI 6\nPre', val) for val in results['vei6']['pre_eruption_crisis_counts']])
        
        # VEI 7 post-eruption
        if results['vei7']['post_eruption_crisis_counts']:
            plot_data.extend([('VEI 7\nPost', val) for val in results['vei7']['post_eruption_crisis_counts']])
        
        # VEI 7 pre-eruption
        if 'pre_eruption_crisis_counts' in results['vei7'] and results['vei7']['pre_eruption_crisis_counts']:
            plot_data.extend([('VEI 7\nPre', val) for val in results['vei7']['pre_eruption_crisis_counts']])
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame(plot_data, columns=['Category', 'Crisis Count'])
        
        # Create box plot
        sns.boxplot(data=df_plot, x='Category', y='Crisis Count', ax=ax)
        
        # Add mean markers
        categories = df_plot['Category'].unique()
        for i, cat in enumerate(categories):
            mean_val = df_plot[df_plot['Category'] == cat]['Crisis Count'].mean()
            ax.plot(i, mean_val, marker='D', color='red', markersize=8, label='Mean' if i == 0 else '')
        
        # Add title and labels
        ax.set_title(f'{window}-Year Window Analysis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Analysis Type', fontsize=10)
        ax.set_ylabel('Number of Active Crises', fontsize=10)
        
        # Add significance indicators if available
        if 'vei6_stats' in results and results['vei6_stats']:
            p_val = results['vei6_stats'].get('permutation', {}).get('p_value', 1)
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            ax.text(1, ax.get_ylim()[1] * 0.95, f'VEI6: {sig_text}', fontsize=9, ha='center')
        
        if 'vei7_stats' in results and results['vei7_stats']:
            p_val = results['vei7_stats'].get('permutation', {}).get('p_value', 1)
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            ax.text(3, ax.get_ylim()[1] * 0.95, f'VEI7: {sig_text}', fontsize=9, ha='center')
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Crisis Frequency Analysis: Volcanic Eruptions vs Random Sampling', 
                fontsize=14, fontweight='bold', y=1.02)
    
    # Add legend for significance
    fig.text(0.5, -0.02, 'Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant', 
            ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.show()


def plot_detailed_distributions(results_path: str = '../results/volcano_crisis_analysis.pkl',
                              save_path: str = '../results/detailed_distributions.png'):
    """
    Create detailed distribution plots with multiple visualization types.
    
    Args:
        results_path: Path to the analysis results pickle file
        save_path: Path to save the plot
    """
    print("Creating detailed distribution plots...")
    
    # Load results
    with open(results_path, 'rb') as f:
        all_results = pickle.load(f)
    
    # Focus on 10-year window for detailed analysis
    results = all_results[10]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Box plot
    ax1 = plt.subplot(2, 3, 1)
    data_boxplot = [
        results['random']['crisis_counts'],
        results['vei6']['post_eruption_crisis_counts'] if results['vei6']['post_eruption_crisis_counts'] else [0],
        results['vei7']['post_eruption_crisis_counts'] if results['vei7']['post_eruption_crisis_counts'] else [0]
    ]
    bp = ax1.boxplot(data_boxplot, labels=['Random', 'VEI 6', 'VEI 7'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'orange', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax1.set_ylabel('Number of Crises')
    ax1.set_title('Box Plot Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 2. Violin plot
    ax2 = plt.subplot(2, 3, 2)
    parts = ax2.violinplot(data_boxplot, positions=[1, 2, 3], showmeans=True, showmedians=True)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['Random', 'VEI 6', 'VEI 7'])
    ax2.set_ylabel('Number of Crises')
    ax2.set_title('Violin Plot Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram/Density plot
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(results['random']['crisis_counts'], bins=30, alpha=0.5, label='Random', density=True, color='blue')
    if results['vei6']['post_eruption_crisis_counts']:
        ax3.hist(results['vei6']['post_eruption_crisis_counts'], bins=15, alpha=0.5, 
                label='VEI 6', density=True, color='orange')
    if results['vei7']['post_eruption_crisis_counts']:
        ax3.hist(results['vei7']['post_eruption_crisis_counts'], bins=10, alpha=0.5,
                label='VEI 7', density=True, color='red')
    ax3.set_xlabel('Number of Crises')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Bar chart with error bars
    ax4 = plt.subplot(2, 3, 4)
    means = [
        np.mean(results['random']['crisis_counts']),
        np.mean(results['vei6']['post_eruption_crisis_counts']) if results['vei6']['post_eruption_crisis_counts'] else 0,
        np.mean(results['vei7']['post_eruption_crisis_counts']) if results['vei7']['post_eruption_crisis_counts'] else 0
    ]
    stds = [
        np.std(results['random']['crisis_counts']),
        np.std(results['vei6']['post_eruption_crisis_counts']) if results['vei6']['post_eruption_crisis_counts'] else 0,
        np.std(results['vei7']['post_eruption_crisis_counts']) if results['vei7']['post_eruption_crisis_counts'] else 0
    ]
    bars = ax4.bar(['Random', 'VEI 6', 'VEI 7'], means, yerr=stds, 
                   capsize=5, color=['lightblue', 'orange', 'red'], alpha=0.7)
    ax4.set_ylabel('Mean Number of Crises')
    ax4.set_title('Mean Comparison with Standard Deviation')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Cumulative distribution
    ax5 = plt.subplot(2, 3, 5)
    
    # Random sampling
    sorted_random = np.sort(results['random']['crisis_counts'])
    cumulative_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)
    ax5.plot(sorted_random, cumulative_random, label='Random', linewidth=2, color='blue')
    
    # VEI 6
    if results['vei6']['post_eruption_crisis_counts']:
        sorted_vei6 = np.sort(results['vei6']['post_eruption_crisis_counts'])
        cumulative_vei6 = np.arange(1, len(sorted_vei6) + 1) / len(sorted_vei6)
        ax5.plot(sorted_vei6, cumulative_vei6, label='VEI 6', linewidth=2, color='orange')
    
    # VEI 7
    if results['vei7']['post_eruption_crisis_counts']:
        sorted_vei7 = np.sort(results['vei7']['post_eruption_crisis_counts'])
        cumulative_vei7 = np.arange(1, len(sorted_vei7) + 1) / len(sorted_vei7)
        ax5.plot(sorted_vei7, cumulative_vei7, label='VEI 7', linewidth=2, color='red')
    
    ax5.set_xlabel('Number of Crises')
    ax5.set_ylabel('Cumulative Probability')
    ax5.set_title('Cumulative Distribution Function')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistical summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Prepare summary data
    summary_data = []
    
    # Random sampling
    summary_data.append(['Random Sampling', 
                        f"{results['random']['mean']:.2f}",
                        f"{results['random']['median']:.1f}",
                        f"{results['random']['std']:.2f}",
                        f"{results['random']['n_samples']}"])
    
    # VEI 6
    if 'post_mean' in results['vei6']:
        vei6_p = results.get('vei6_stats', {}).get('permutation', {}).get('p_value', None)
        vei6_d = results.get('vei6_stats', {}).get('cohen_d', None)
        summary_data.append(['VEI 6 Post-eruption',
                           f"{results['vei6']['post_mean']:.2f}",
                           f"{results['vei6']['post_median']:.1f}",
                           f"{results['vei6']['post_std']:.2f}",
                           f"{len(results['vei6']['post_eruption_crisis_counts'])}"])
        if vei6_p is not None:
            summary_data.append(['VEI 6 Statistics',
                               f"p={vei6_p:.4f}",
                               f"d={vei6_d:.3f}" if vei6_d else "N/A",
                               "", ""])
    
    # VEI 7
    if 'post_mean' in results['vei7']:
        vei7_p = results.get('vei7_stats', {}).get('permutation', {}).get('p_value', None)
        vei7_d = results.get('vei7_stats', {}).get('cohen_d', None)
        summary_data.append(['VEI 7 Post-eruption',
                           f"{results['vei7']['post_mean']:.2f}",
                           f"{results['vei7']['post_median']:.1f}",
                           f"{results['vei7']['post_std']:.2f}",
                           f"{len(results['vei7']['post_eruption_crisis_counts'])}"])
        if vei7_p is not None:
            summary_data.append(['VEI 7 Statistics',
                               f"p={vei7_p:.4f}",
                               f"d={vei7_d:.3f}" if vei7_d else "N/A",
                               "", ""])
    
    # Create table
    table = ax6.table(cellText=summary_data,
                     colLabels=['Category', 'Mean', 'Median', 'Std Dev', 'N'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if 'Statistics' in str(summary_data[i-1][0]) if i <= len(summary_data) else False:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('#ffffff')
    
    ax6.set_title('Statistical Summary (10-year window)', fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('Detailed Distribution Analysis: Crisis Frequency Following Volcanic Eruptions',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed distributions saved to {save_path}")
    plt.show()


def plot_pre_post_comparison(results_path: str = '../results/volcano_crisis_analysis.pkl',
                            save_path: str = '../results/pre_post_comparison.png'):
    """
    Create plots comparing pre- and post-eruption crisis frequencies.
    
    Args:
        results_path: Path to the analysis results pickle file
        save_path: Path to save the plot
    """
    print("Creating pre- vs post-eruption comparison plots...")
    
    # Load results
    with open(results_path, 'rb') as f:
        all_results = pickle.load(f)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for both VEI levels
    for idx, vei in enumerate([6, 7]):
        ax = axes[idx]
        
        pre_data = []
        post_data = []
        windows = []
        
        for window in sorted(all_results.keys()):
            results = all_results[window][f'vei{vei}']
            
            if 'pre_mean' in results and 'post_mean' in results:
                pre_data.append(results['pre_mean'])
                post_data.append(results['post_mean'])
                windows.append(window)
        
        if pre_data and post_data:
            # Create grouped bar chart
            x = np.arange(len(windows))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pre_data, width, label='Pre-eruption', color='skyblue', alpha=0.7)
            bars2 = ax.bar(x + width/2, post_data, width, label='Post-eruption', color='coral', alpha=0.7)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            
            # Customize plot
            ax.set_xlabel('Time Window (years)', fontsize=11)
            ax.set_ylabel('Mean Number of Crises', fontsize=11)
            ax.set_title(f'VEI {vei} Eruptions: Pre vs Post Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(windows)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add baseline from random sampling
            for i, window in enumerate(windows):
                random_mean = all_results[window]['random']['mean']
                ax.axhline(y=random_mean, color='gray', linestyle='--', alpha=0.3)
                if i == 0:
                    ax.text(ax.get_xlim()[1] * 0.98, random_mean, 'Random baseline',
                           ha='right', va='bottom', fontsize=9, color='gray')
    
    fig.suptitle('Pre- vs Post-Eruption Crisis Frequency Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Pre-post comparison saved to {save_path}")
    plt.show()


def plot_effect_size_summary(results_path: str = '../results/volcano_crisis_analysis.pkl',
                            save_path: str = '../results/effect_sizes.png'):
    """
    Create a plot summarizing effect sizes across different time windows.
    
    Args:
        results_path: Path to the analysis results pickle file
        save_path: Path to save the plot
    """
    print("Creating effect size summary plot...")
    
    # Load results
    with open(results_path, 'rb') as f:
        all_results = pickle.load(f)
    
    # Prepare data
    windows = sorted(all_results.keys())
    vei6_effects = []
    vei7_effects = []
    vei6_pvals = []
    vei7_pvals = []
    
    for window in windows:
        # VEI 6
        if 'vei6_stats' in all_results[window] and all_results[window]['vei6_stats']:
            vei6_effects.append(all_results[window]['vei6_stats'].get('cohen_d', 0))
            vei6_pvals.append(all_results[window]['vei6_stats'].get('permutation', {}).get('p_value', 1))
        else:
            vei6_effects.append(0)
            vei6_pvals.append(1)
        
        # VEI 7
        if 'vei7_stats' in all_results[window] and all_results[window]['vei7_stats']:
            vei7_effects.append(all_results[window]['vei7_stats'].get('cohen_d', 0))
            vei7_pvals.append(all_results[window]['vei7_stats'].get('permutation', {}).get('p_value', 1))
        else:
            vei7_effects.append(0)
            vei7_pvals.append(1)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Effect sizes
    x = np.arange(len(windows))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, vei6_effects, width, label='VEI 6', color='orange', alpha=0.7)
    bars2 = ax1.bar(x + width/2, vei7_effects, width, label='VEI 7', color='red', alpha=0.7)
    
    # Add significance stars
    for i, (bar1, bar2, p6, p7) in enumerate(zip(bars1, bars2, vei6_pvals, vei7_pvals)):
        # VEI 6
        if p6 < 0.001:
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02, '***', 
                    ha='center', fontsize=10)
        elif p6 < 0.01:
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02, '**',
                    ha='center', fontsize=10)
        elif p6 < 0.05:
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02, '*',
                    ha='center', fontsize=10)
        
        # VEI 7
        if p7 < 0.001:
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02, '***',
                    ha='center', fontsize=10)
        elif p7 < 0.01:
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02, '**',
                    ha='center', fontsize=10)
        elif p7 < 0.05:
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02, '*',
                    ha='center', fontsize=10)
    
    # Reference lines for effect size interpretation
    ax1.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.8, color='gray', linestyle='-', alpha=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Labels for effect size thresholds
    ax1.text(ax1.get_xlim()[0] - 0.5, 0.2, 'Small', ha='right', va='center', fontsize=9, color='gray')
    ax1.text(ax1.get_xlim()[0] - 0.5, 0.5, 'Medium', ha='right', va='center', fontsize=9, color='gray')
    ax1.text(ax1.get_xlim()[0] - 0.5, 0.8, 'Large', ha='right', va='center', fontsize=9, color='gray')
    
    ax1.set_ylabel("Cohen's d (Effect Size)", fontsize=11)
    ax1.set_title('Effect Sizes: Volcanic Eruptions vs Random Baseline', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x)
    ax1.set_xticklabels(windows)
    
    # Plot 2: P-values
    ax2.plot(x, vei6_pvals, 'o-', label='VEI 6', color='orange', linewidth=2, markersize=8)
    ax2.plot(x, vei7_pvals, 's-', label='VEI 7', color='red', linewidth=2, markersize=8)
    
    # Significance thresholds
    ax2.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.001, color='gray', linestyle='--', alpha=0.5)
    
    # Labels for significance levels
    ax2.text(ax2.get_xlim()[0] - 0.5, 0.05, 'p=0.05', ha='right', va='center', fontsize=9, color='gray')
    ax2.text(ax2.get_xlim()[0] - 0.5, 0.01, 'p=0.01', ha='right', va='center', fontsize=9, color='gray')
    ax2.text(ax2.get_xlim()[0] - 0.5, 0.001, 'p=0.001', ha='right', va='center', fontsize=9, color='gray')
    
    ax2.set_xlabel('Time Window (years)', fontsize=11)
    ax2.set_ylabel('P-value (Permutation Test)', fontsize=11)
    ax2.set_title('Statistical Significance Across Time Windows', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(windows)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Effect size summary saved to {save_path}")
    plt.show()


def create_summary_report(results_path: str = '../results/volcano_crisis_analysis.pkl',
                         summary_path: str = '../results/analysis_summary.csv'):
    """
    Create a text summary report of the findings.
    
    Args:
        results_path: Path to the analysis results pickle file
        summary_path: Path to the summary CSV file
    """
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    # Load results
    with open(results_path, 'rb') as f:
        all_results = pickle.load(f)
    
    # Load summary CSV
    summary_df = pd.read_csv(summary_path)
    
    print("\n1. OVERALL FINDINGS")
    print("-" * 40)
    
    # Check 10-year window results (most commonly used)
    results_10 = all_results[10]
    
    print(f"Random baseline (10-year window):")
    print(f"  Mean crises: {results_10['random']['mean']:.2f}")
    print(f"  Std deviation: {results_10['random']['std']:.2f}")
    
    if 'post_mean' in results_10['vei6']:
        print(f"\nVEI 6 eruptions (10-year window):")
        print(f"  Mean crises post-eruption: {results_10['vei6']['post_mean']:.2f}")
        print(f"  Mean crises pre-eruption: {results_10['vei6'].get('pre_mean', 'N/A'):.2f}")
        
        if 'vei6_stats' in results_10:
            cohen_d = results_10['vei6_stats'].get('cohen_d', None)
            p_value = results_10['vei6_stats'].get('permutation', {}).get('p_value', None)
            if cohen_d and p_value:
                print(f"  Effect size (Cohen's d): {cohen_d:.3f}")
                print(f"  P-value: {p_value:.4f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    if 'post_mean' in results_10['vei7']:
        print(f"\nVEI 7 eruptions (10-year window):")
        print(f"  Mean crises post-eruption: {results_10['vei7']['post_mean']:.2f}")
        print(f"  Mean crises pre-eruption: {results_10['vei7'].get('pre_mean', 'N/A'):.2f}")
        
        if 'vei7_stats' in results_10:
            cohen_d = results_10['vei7_stats'].get('cohen_d', None)
            p_value = results_10['vei7_stats'].get('permutation', {}).get('p_value', None)
            if cohen_d and p_value:
                print(f"  Effect size (Cohen's d): {cohen_d:.3f}")
                print(f"  P-value: {p_value:.4f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    print("\n2. EFFECT SIZE INTERPRETATION")
    print("-" * 40)
    print("Cohen's d interpretation guide:")
    print("  0.2 = Small effect")
    print("  0.5 = Medium effect")
    print("  0.8 = Large effect")
    
    print("\n3. FINDINGS ACROSS TIME WINDOWS")
    print("-" * 40)
    
    for window in sorted(all_results.keys()):
        print(f"\n{window}-year window:")
        
        if 'vei6_stats' in all_results[window] and all_results[window]['vei6_stats']:
            vei6_d = all_results[window]['vei6_stats'].get('cohen_d', 0)
            vei6_p = all_results[window]['vei6_stats'].get('permutation', {}).get('p_value', 1)
            print(f"  VEI 6: d={vei6_d:.3f}, p={vei6_p:.4f}")
        
        if 'vei7_stats' in all_results[window] and all_results[window]['vei7_stats']:
            vei7_d = all_results[window]['vei7_stats'].get('cohen_d', 0)
            vei7_p = all_results[window]['vei7_stats'].get('permutation', {}).get('p_value', 1)
            print(f"  VEI 7: d={vei7_d:.3f}, p={vei7_p:.4f}")
    
    print("\n" + "="*60)
    print("END OF SUMMARY REPORT")
    print("="*60)


def main():
    """Main execution function for plotting."""
    
    # Check if results exist
    results_path = Path('../results/volcano_crisis_analysis.pkl')
    if not results_path.exists():
        print("ERROR: Results file not found!")
        print("Please run data_preparation.py first to generate the analysis results.")
        return
    
    print("Starting visualization generation...")
    print("="*60)
    
    # Create all plots
    try:
        # 1. Crisis timeline
        plot_crisis_timeline()
        
        # 2. Comparison across all windows
        plot_comparison_all_windows()
        
        # 3. Detailed distributions
        plot_detailed_distributions()
        
        # 4. Pre vs Post comparison
        plot_pre_post_comparison()
        
        # 5. Effect size summary
        plot_effect_size_summary()
        
        # 6. Generate text summary
        create_summary_report()
        
        print("\n" + "="*60)
        print("All visualizations completed successfully!")
        print("Check the ../results/ folder for all generated plots.")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR during visualization: {str(e)}")
        print("Please ensure all required data files are present.")
        raise


if __name__ == "__main__":
    main()