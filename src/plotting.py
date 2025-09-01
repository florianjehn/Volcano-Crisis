"""
Plotting module for Volcano Crisis analysis.
Creates visualizations from prepared data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Use ALLFED style sheet
plt.style.use("https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle")
warnings.filterwarnings('ignore')


def plot_crisis_timeline(save_path='results/crisis_timeline.png'):
    """Create timeline plot showing all crises and their durations."""
    print("Creating crisis timeline plot...")
    
    # Read data
    crisis_matrix = pd.read_csv('results/crisis_overview.csv', index_col=0)
    volcano_df = pd.read_csv('data/volcano_list.csv')
    volcano_df = volcano_df[volcano_df['Year'] <= 1920]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Get crisis periods
    crisis_periods = []
    for crisis in crisis_matrix.columns:
        active_Years = crisis_matrix[crisis_matrix[crisis] == 1].index
        if len(active_Years) > 0:
            start_Year = active_Years[0]
            end_Year = active_Years[0]
            
            for i in range(1, len(active_Years)):
                if active_Years[i] - active_Years[i-1] > 1:
                    crisis_periods.append({
                        'crisis': crisis,
                        'start': start_Year,
                        'end': end_Year
                    })
                    start_Year = active_Years[i]
                end_Year = active_Years[i]
            
            crisis_periods.append({
                'crisis': crisis,
                'start': start_Year,
                'end': end_Year
            })
    
    # Sort by start date
    crisis_periods.sort(key=lambda x: x['start'])
    
    # Plot each crisis
    y_positions = {}
    current_y = 0
    
    for period in crisis_periods:
        crisis_name = period['crisis']
        if crisis_name not in y_positions:
            y_positions[crisis_name] = current_y
            current_y += 1
        
        ax.plot([period['start'], period['end']], 
               [y_positions[crisis_name], y_positions[crisis_name]],
               linewidth=2, alpha=0.7, color="grey")
    
    # Add volcanic eruptions
    # VEI 7 eruptions - red lines
    vei7 = volcano_df[volcano_df['VEI'] == 7]
    for _, eruption in vei7.iterrows():
        ax.axvline(x=eruption['Year'], color='red', alpha=1, linestyle='--', linewidth=1.5)
    
    # VEI 6 eruptions - orange lines
    vei6 = volcano_df[volcano_df['VEI'] == 6]
    for _, eruption in vei6.iterrows():
        ax.axvline(x=eruption['Year'], color='orange', alpha=1, linestyle=':', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Crisis', fontsize=12)
    ax.set_title('Timeline of All Crises with Major Volcanic Eruptions', fontsize=14, fontweight='bold')
    
    # Show subset of crisis labels
    n_crises = len(y_positions)
    step = max(1, n_crises // 30)
    selected_crises = list(y_positions.keys())[::step]
    selected_positions = [y_positions[c] for c in selected_crises]
    
    ax.set_yticks(selected_positions)
    ax.set_yticklabels(selected_crises, fontsize=8)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='VEI 7 Eruption'),
        Line2D([0], [0], color='orange', linestyle=':', alpha=0.5, label='VEI 6 Eruption'),
        Line2D([0], [0], color='grey', linewidth=2, alpha=0.7, label='Crisis Period')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_all_windows(save_path='results/comparison_all_windows.png'):
    """Create comparison plots for different time windows."""
    print("Creating comparison plots...")
    
    # Read data
    results_df = pd.read_csv('results/volcano_crisis_analysis.csv')
    detailed_df = pd.read_csv('results/detailed_crisis_counts.csv')
    
    # Select specific windows for visualization (0, 10, 20, 50 Years)
    windows_to_plot = [0, 10, 20, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, window in enumerate(windows_to_plot):
        ax = axes[idx]
        
        # Get data for this window
        volcano_data = detailed_df[(detailed_df['window'] == window) & (detailed_df['type'] == 'volcano')]['crisis_count'].values
        random_data = detailed_df[(detailed_df['window'] == window) & (detailed_df['type'] == 'random')]['crisis_count'].values
        
        # Create box plot
        data_to_plot = [random_data, volcano_data]
        bp = ax.boxplot(data_to_plot, labels=['Random\nBaseline', 'Volcanic\nEruptions'], patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'coral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add mean markers
        for i, data in enumerate(data_to_plot):
            if len(data) > 0:
                mean_val = np.mean(data)
                ax.plot(i+1, mean_val, marker='D', color='red', markersize=8)
        
        # Get statistics for this window
        window_stats = results_df[results_df['window_years'] == window].iloc[0]
        
        # Add significance indicator
        p_val = window_stats['permutation_pvalue']
        if pd.notna(p_val):
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            cohen_d = window_stats['cohen_d']
            ax.text(0.98, 0.98, f'p={p_val:.3f} ({sig_text})\nd={cohen_d:.2f}', 
                   transform=ax.transAxes, ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Labels
        ax.set_ylabel('Number of Active Crises')
        ax.set_title(f'{window}-Year Window', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Crisis Frequency: Volcanic Eruptions vs Random Baseline', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_effect_size_summary(save_path='results/effect_sizes.png'):
    """Create plot showing effect sizes and p-values across all time windows."""
    print("Creating effect size summary plot...")
    
    # Read data
    results_df = pd.read_csv('results/volcano_crisis_analysis.csv')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Effect sizes (Cohen's d)
    windows = results_df['window_years'].values
    cohen_d = results_df['cohen_d'].values
    
    bars = ax1.bar(windows, cohen_d, width=8, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Color bars by significance
    p_values = results_df['permutation_pvalue'].values
    for bar, p in zip(bars, p_values):
        if pd.notna(p):
            if p < 0.001:
                bar.set_facecolor('darkgreen')
            elif p < 0.01:
                bar.set_facecolor('green')
            elif p < 0.05:
                bar.set_facecolor('lightgreen')
    
    # Reference lines for effect size
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Small effect')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
    ax1.axhline(y=0.8, color='gray', linestyle='-', alpha=0.5, label='Large effect')
    
    ax1.set_ylabel("Cohen's d (Effect Size)")
    ax1.set_title('Effect Size of Volcanic Eruptions on Crisis Frequency', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: P-values
    ax2.scatter(windows, p_values, s=50, color='darkred', alpha=0.7)
    ax2.plot(windows, p_values, 'r-', alpha=0.3)
    
    # Significance thresholds
    ax2.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='p=0.05')
    ax2.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='p=0.01')
    ax2.axhline(y=0.001, color='gray', linestyle='--', alpha=0.5, label='p=0.001')
    
    ax2.set_xlabel('Time Window (Years)')
    ax2.set_ylabel('P-value (Permutation Test)')
    ax2.set_title('Statistical Significance Across Time Windows', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_ylim([0.0001, 1])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mean_comparison(save_path='results/mean_comparison.png'):
    """Create plot comparing means across all time windows."""
    print("Creating mean comparison plot...")
    
    # Read data
    results_df = pd.read_csv('results/volcano_crisis_analysis.csv')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    windows = results_df['window_years'].values
    volcano_means = results_df['volcano_mean'].values
    volcano_stds = results_df['volcano_std'].values
    random_means = results_df['random_mean'].values
    random_stds = results_df['random_std'].values
    
    # Plot lines with error bands
    ax.plot(windows, volcano_means, 'o-', label='Volcanic Eruptions', color='red', linewidth=2, markersize=6)
    ax.fill_between(windows, 
                    volcano_means - volcano_stds, 
                    volcano_means + volcano_stds, 
                    alpha=0.2, color='red')
    
    ax.plot(windows, random_means, 's-', label='Random Baseline', color='blue', linewidth=2, markersize=6)
    ax.fill_between(windows, 
                    random_means - random_stds, 
                    random_means + random_stds, 
                    alpha=0.2, color='blue')
    
    # Add significance markers
    p_values = results_df['permutation_pvalue'].values
    for window, volcano_mean, p in zip(windows, volcano_means, p_values):
        if pd.notna(p) and p < 0.05:
            ax.plot(window, volcano_mean + volcano_stds[list(windows).index(window)] + 2, 
                   '*', color='black', markersize=10)
    
    ax.set_xlabel('Time Window (Years)', fontsize=12)
    ax.set_ylabel('Mean Number of Active Crises', fontsize=12)
    ax.set_title('Mean Crisis Frequency: Volcanic Eruptions vs Random Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def create_summary_report():
    """Print summary of findings."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Read results
    results_df = pd.read_csv('results/volcano_crisis_analysis.csv')
    
    # Key findings for 10-Year window
    window_10 = results_df[results_df['window_years'] == 10].iloc[0]
    
    print(f"\n10-Year Window Results:")
    print(f"  Random baseline: {window_10['random_mean']:.2f} ± {window_10['random_std']:.2f} crises")
    print(f"  Volcanic eruptions: {window_10['volcano_mean']:.2f} ± {window_10['volcano_std']:.2f} crises")
    print(f"  Effect size (Cohen's d): {window_10['cohen_d']:.3f}")
    print(f"  P-value: {window_10['permutation_pvalue']:.4f}")
    
    # Find windows with significant effects
    significant = results_df[results_df['permutation_pvalue'] < 0.05]
    
    if len(significant) > 0:
        print(f"\nSignificant effects (p < 0.05) found at windows:")
        for _, row in significant.iterrows():
            print(f"  {row['window_years']} Years: d={row['cohen_d']:.3f}, p={row['permutation_pvalue']:.4f}")
    else:
        print("\nNo statistically significant effects found at p < 0.05 level")
    
    # Maximum effect size
    max_effect_idx = results_df['cohen_d'].abs().idxmax()
    max_effect = results_df.iloc[max_effect_idx]
    print(f"\nMaximum effect size:")
    print(f"  Window: {max_effect['window_years']} Years")
    print(f"  Cohen's d: {max_effect['cohen_d']:.3f}")
    print(f"  P-value: {max_effect['permutation_pvalue']:.4f}")
    
    print("\n" + "="*60)


def main():
    """Main execution function."""
    
    # Check if results exist
    if not Path('results/volcano_crisis_analysis.csv').exists():
        print("ERROR: Results file not found!")
        print("Please run data_preparation.py first.")
        return
    
    print("Generating visualizations...")
    print("="*60)
    
    # Create all plots
    plot_crisis_timeline()
    plot_comparison_all_windows()
    plot_effect_size_summary()
    plot_mean_comparison()
    
    # Generate summary
    create_summary_report()
    
    print("\nAll visualizations saved to results/")
    print("="*60)


if __name__ == "__main__":
    main()