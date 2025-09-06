"""Plotting functions for volcano-crisis analysis including onset analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.style.use('default')
sns.set_palette("husl")


def create_swarm_plot_data(data_df):
    """Convert before/after data to swarm plot format."""
    plot_data = []
    for _, row in data_df.iterrows():
        window = row['window']
        plot_data.extend([
            {
                'group': f"{window}yr\nBefore",
                'period': 'Before',
                'crisis_count': row['before_count'],
                'window': window,
                'eruption': row['eruption_name']
            },
            {
                'group': f"{window}yr\nAfter",
                'period': 'After',
                'crisis_count': row['after_count'],
                'window': window,
                'eruption': row['eruption_name']
            }
        ])
    return pd.DataFrame(plot_data)


def create_onset_swarm_plot_data(data_df):
    """Convert onset before/after data to swarm plot format."""
    plot_data = []
    for _, row in data_df.iterrows():
        window = row['window']
        plot_data.extend([
            {
                'group': f"{window}yr\nBefore",
                'period': 'Before',
                'onset_count': row['before_onset_count'],
                'window': window,
                'eruption': row['eruption_name']
            },
            {
                'group': f"{window}yr\nAfter",
                'period': 'After',
                'onset_count': row['after_onset_count'],
                'window': window,
                'eruption': row['eruption_name']
            }
        ])
    return pd.DataFrame(plot_data)


def plot_swarm_comparison(data_df, title, save_path):
    """Create swarm plot for before/after comparison."""
    if len(data_df) == 0:
        return
        
    plot_df = create_swarm_plot_data(data_df)
    windows = sorted(data_df['window'].unique())
    
    group_order = []
    for window in windows:
        group_order.extend([f"{window}yr\nBefore", f"{window}yr\nAfter"])
    
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.swarmplot(data=plot_df, x='group', y='crisis_count', hue='period',
                  size=5, alpha=0.7, ax=ax, order=group_order,
                  palette={'Before': '#1f77b4', 'After': '#ff7f0e'})
    
    # Add mean and median lines
    for i, window in enumerate(windows):
        before_data = data_df[data_df['window'] == window]['before_count']
        after_data = data_df[data_df['window'] == window]['after_count']
        
        before_x, after_x = i * 2, i * 2 + 1
        
        # Means (thick lines)
        ax.hlines(before_data.mean(), before_x - 0.3, before_x + 0.3, 
                 colors='darkblue', linewidth=4, alpha=0.9,
                 label='Before Mean' if i == 0 else "")
        ax.hlines(after_data.mean(), after_x - 0.3, after_x + 0.3,
                 colors='darkred', linewidth=4, alpha=0.9,
                 label='After Mean' if i == 0 else "")
        
        # Medians (dashed lines)
        ax.hlines(before_data.median(), before_x - 0.3, before_x + 0.3,
                 colors='darkblue', linewidth=2, linestyles='dashed', alpha=0.8,
                 label='Before Median' if i == 0 else "")
        ax.hlines(after_data.median(), after_x - 0.3, after_x + 0.3,
                 colors='darkred', linewidth=2, linestyles='dashed', alpha=0.8,
                 label='After Median' if i == 0 else "")
    
    # Add date range to title
    year_range = f"{data_df['eruption_year'].min()}-{data_df['eruption_year'].max()} CE"
    full_title = f"{title} ({year_range})"
    
    ax.set_xlabel('Time Window', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Active Crises', fontsize=14, fontweight='bold')
    ax.set_title(full_title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add separators between time windows
    for i in range(1, len(windows)):
        ax.axvline(x=i*2 - 0.5, color='gray', alpha=0.3, linestyle=':', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_onset_swarm_comparison(data_df, title, save_path):
    """
    Create swarm plot specifically for crisis ONSET comparison.
    
    This shows the number of crises that BEGIN in each time window,
    rather than the total number of active crises.
    """
    if len(data_df) == 0:
        return
        
    plot_df = create_onset_swarm_plot_data(data_df)
    windows = sorted(data_df['window'].unique())
    
    group_order = []
    for window in windows:
        group_order.extend([f"{window}yr\nBefore", f"{window}yr\nAfter"])
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Use different colors to distinguish from active crisis plots
    sns.swarmplot(data=plot_df, x='group', y='onset_count', hue='period',
                  size=5, alpha=0.7, ax=ax, order=group_order,
                  palette={'Before': '#2ca02c', 'After': '#d62728'})  # Green and red
    
    # Add mean and median lines with statistical annotations
    for i, window in enumerate(windows):
        before_data = data_df[data_df['window'] == window]['before_onset_count']
        after_data = data_df[data_df['window'] == window]['after_onset_count']
        
        before_x, after_x = i * 2, i * 2 + 1
        
        # Calculate statistics
        mean_before = before_data.mean()
        mean_after = after_data.mean()
        
        # Means (thick lines)
        ax.hlines(mean_before, before_x - 0.3, before_x + 0.3, 
                 colors='darkgreen', linewidth=4, alpha=0.9,
                 label='Before Mean' if i == 0 else "")
        ax.hlines(mean_after, after_x - 0.3, after_x + 0.3,
                 colors='darkred', linewidth=4, alpha=0.9,
                 label='After Mean' if i == 0 else "")
        
        # Medians (dashed lines)
        ax.hlines(before_data.median(), before_x - 0.3, before_x + 0.3,
                 colors='darkgreen', linewidth=2, linestyles='dashed', alpha=0.8,
                 label='Before Median' if i == 0 else "")
        ax.hlines(after_data.median(), after_x - 0.3, after_x + 0.3,
                 colors='darkred', linewidth=2, linestyles='dashed', alpha=0.8,
                 label='After Median' if i == 0 else "")
        
        # Add percentage change annotation
        if mean_before > 0:
            pct_change = ((mean_after - mean_before) / mean_before) * 100
            ax.text((before_x + after_x) / 2, ax.get_ylim()[1] * 0.95,
                   f"{pct_change:+.1f}%", ha='center', fontsize=10,
                   color='darkred' if pct_change > 0 else 'darkgreen',
                   fontweight='bold')
    
    # Add date range to title
    year_range = f"{data_df['eruption_year'].min()}-{data_df['eruption_year'].max()} CE"
    full_title = f"{title} ({year_range})"
    
    ax.set_xlabel('Time Window', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Crisis Onsets', fontsize=14, fontweight='bold')
    ax.set_title(full_title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add separators between time windows
    for i in range(1, len(windows)):
        ax.axvline(x=i*2 - 0.5, color='gray', alpha=0.3, linestyle=':', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_temporal_onset_distribution(distribution_df, save_path, window_size=5):
    """
    Create a density plot showing when crisis onsets occur relative to eruptions.
    
    This visualization reveals if there's a temporal clustering of crisis onsets
    around volcanic eruptions (year 0).
    
    Args:
        distribution_df: DataFrame from compute_temporal_onset_distribution
        save_path: Path to save the figure
        window_size: Size of rolling window for smoothing (years)
    """
    if len(distribution_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Aggregate onset counts by relative year
    onset_by_year = distribution_df.groupby('relative_year')['onset_count'].agg(['mean', 'sum', 'std'])
    
    # Plot 1: Mean onset rate with confidence interval
    ax1 = axes[0]
    x = onset_by_year.index
    y_mean = onset_by_year['mean']
    y_std = onset_by_year['std']
    
    # Apply smoothing
    y_smooth = y_mean.rolling(window=window_size, center=True).mean()
    
    # Plot raw data as light line
    ax1.plot(x, y_mean, alpha=0.3, color='gray', label='Raw mean')
    
    # Plot smoothed line
    ax1.plot(x, y_smooth, color='darkblue', linewidth=2, label=f'{window_size}-year smoothed')
    
    # Add confidence interval
    ax1.fill_between(x, y_mean - y_std/2, y_mean + y_std/2, 
                     alpha=0.2, color='blue', label='±0.5 std')
    
    # Mark eruption year
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Eruption')
    ax1.axvspan(0, 10, alpha=0.1, color='red', label='First decade after')
    
    ax1.set_xlabel('Years Relative to Eruption', fontsize=12)
    ax1.set_ylabel('Mean Crisis Onsets per Year', fontsize=12)
    ax1.set_title('Average Crisis Onset Rate Relative to Major Volcanic Eruptions', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: Cumulative onset difference
    ax2 = axes[1]
    
    # Calculate cumulative difference from baseline
    baseline = y_mean.mean()
    cumulative_diff = (y_mean - baseline).cumsum()
    
    ax2.plot(x, cumulative_diff, color='darkgreen', linewidth=2)
    ax2.fill_between(x, 0, cumulative_diff, 
                     where=(cumulative_diff > 0), color='red', alpha=0.3, label='Above baseline')
    ax2.fill_between(x, 0, cumulative_diff, 
                     where=(cumulative_diff <= 0), color='green', alpha=0.3, label='Below baseline')
    
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax2.set_xlabel('Years Relative to Eruption', fontsize=12)
    ax2.set_ylabel('Cumulative Deviation from Baseline', fontsize=12)
    ax2.set_title('Cumulative Crisis Onset Anomaly', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_onset_difference_heatmap(onset_df, stats_df, save_path):
    """
    Create a heatmap showing the difference in crisis onsets (after - before) for each eruption.
    
    This visualization makes it easy to see patterns across eruptions and time windows,
    with statistical significance indicated.
    """
    if len(onset_df) == 0:
        return
    
    # Pivot data to create matrix for heatmap
    pivot_data = onset_df.pivot_table(
        index='eruption_name',
        columns='window',
        values='onset_difference',
        aggfunc='mean'
    )
    
    # Sort by eruption year
    eruption_years = onset_df.groupby('eruption_name')['eruption_year'].first()
    pivot_data = pivot_data.loc[eruption_years.sort_values().index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_data) * 0.3)))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.1f', center=0,
                cmap='RdBu_r', vmin=-5, vmax=5,
                cbar_kws={'label': 'Crisis Onset Difference (After - Before)'},
                ax=ax, linewidths=0.5)
    
    # Add significance markers
    if len(stats_df) > 0:
        for j, window in enumerate(pivot_data.columns):
            window_stats = stats_df[stats_df['window'] == window]
            if len(window_stats) > 0 and window_stats.iloc[0]['significant_05']:
                # Add asterisk for significant windows
                significance_marker = '**' if window_stats.iloc[0]['significant_01'] else '*'
                ax.text(j + 0.5, -0.5, significance_marker, 
                       ha='center', va='top', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Time Window (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volcanic Eruption', fontsize=12, fontweight='bold')
    ax.set_title('Crisis Onset Differences: After minus Before Eruption\n(Red = More after, Blue = More before)',
                fontsize=14, fontweight='bold')
    
    # Add year labels to eruption names
    y_labels = []
    for name in pivot_data.index:
        year = eruption_years[name]
        vei = onset_df[onset_df['eruption_name'] == name]['vei'].iloc[0]
        y_labels.append(f"{name} ({year} CE, VEI {vei})")
    ax.set_yticklabels(y_labels, rotation=0)
    
    # Add legend for significance
    ax.text(1.02, 0.98, '* p < 0.05\n** p < 0.01', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_statistical_summary(stats_df, save_path):
    """
    Create a summary plot showing statistical test results across time windows.
    
    This helps assess whether observed differences are statistically significant
    and how effect sizes vary with time window size.
    """
    if len(stats_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Mean differences with error bars
    ax1 = axes[0, 0]
    ax1.errorbar(stats_df['window'], stats_df['mean_difference'], 
                yerr=stats_df['std_difference'], 
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time Window (years)', fontsize=11)
    ax1.set_ylabel('Mean Difference (After - Before)', fontsize=11)
    ax1.set_title('Mean Crisis Onset Difference by Time Window', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Color points by significance
    colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'gray' 
             for p in stats_df['p_value']]
    ax1.scatter(stats_df['window'], stats_df['mean_difference'], 
               c=colors, s=100, zorder=5)
    
    # Plot 2: P-values
    ax2 = axes[0, 1]
    ax2.plot(stats_df['window'], stats_df['p_value'], 
            marker='o', linewidth=2, markersize=8)
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='p = 0.05')
    ax2.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='p = 0.01')
    ax2.set_xlabel('Time Window (years)', fontsize=11)
    ax2.set_ylabel('P-value (paired t-test)', fontsize=11)
    ax2.set_title('Statistical Significance by Time Window', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Effect size (Cohen's d)
    ax3 = axes[1, 0]
    ax3.plot(stats_df['window'], stats_df['cohens_d'], 
            marker='s', linewidth=2, markersize=8, color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Small effect')
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
    ax3.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large effect')
    ax3.set_xlabel('Time Window (years)', fontsize=11)
    ax3.set_ylabel("Cohen's d", fontsize=11)
    ax3.set_title('Effect Size by Time Window', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Plot 4: Before vs After means
    ax4 = axes[1, 1]
    ax4.plot(stats_df['window'], stats_df['mean_before'], 
            marker='o', label='Before', linewidth=2, markersize=8, color='green')
    ax4.plot(stats_df['window'], stats_df['mean_after'], 
            marker='o', label='After', linewidth=2, markersize=8, color='red')
    ax4.set_xlabel('Time Window (years)', fontsize=11)
    ax4.set_ylabel('Mean Crisis Onsets', fontsize=11)
    ax4.set_title('Mean Crisis Onsets Before vs After', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle('Statistical Analysis of Crisis Onsets Around Volcanic Eruptions', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_crisis_timeline(crisis_matrix, volcano_df, save_path):
    """Create timeline showing crises and eruptions."""
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot crisis periods
    crisis_periods = []
    for crisis in crisis_matrix.columns:
        active_years = crisis_matrix[crisis_matrix[crisis] == 1].index.tolist()
        if not active_years:
            continue
            
        periods = []
        start = end = active_years[0]
        
        for i in range(1, len(active_years)):
            if active_years[i] == end + 1:
                end = active_years[i]
            else:
                periods.append((start, end))
                start = end = active_years[i]
        periods.append((start, end))
        
        crisis_periods.extend([{'crisis': crisis, 'start': s, 'end': e} for s, e in periods])
    
    crisis_periods.sort(key=lambda x: x['start'])
    
    # Plot crisis lines
    crisis_positions = {}
    y_pos = 0
    for period in crisis_periods:
        crisis = period['crisis']
        if crisis not in crisis_positions:
            crisis_positions[crisis] = y_pos
            y_pos += 1
        ax.plot([period['start'], period['end']], 
               [crisis_positions[crisis], crisis_positions[crisis]], 
               linewidth=2, alpha=0.6, color='gray')
    
    # Plot eruptions
    for _, eruption in volcano_df.iterrows():
        color = 'red' if eruption['VEI'] == 7 else 'orange'
        linestyle = '-' if eruption['VEI'] == 7 else '--'
        ax.axvline(x=eruption['Year'], color=color, alpha=0.8, 
                  linestyle=linestyle, linewidth=2)
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Crisis Events', fontsize=14, fontweight='bold')
    ax.set_title('Timeline of Global Crises and Major Volcanic Eruptions', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='VEI 7 Eruption'),
        Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='VEI 6 Eruption'),
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.6, label='Crisis Period')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_plots(crisis_matrix, volcano_df, before_after_df, 
                    onset_before_after_df, onset_distribution_df, onset_stats_df,
                    output_subfolder='World'):
    """Create all visualizations including onset analysis in the specified subfolder."""
    # Create output directory structure
    output_dir = Path('results') / output_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Adjust title to include region
    region_label = output_subfolder if output_subfolder != 'World' else 'Global'
    
    # === ORIGINAL PLOTS (Active Crises) ===
    
    # Main comparison plot
    plot_swarm_comparison(before_after_df, 
                         f'Crisis Frequency Before vs After Major Volcanic Eruptions (VEI 6-7) - {region_label}',
                         output_dir / 'before_after_comparison.png')
    
    # VEI 6 specific plot
    vei6_data = before_after_df[before_after_df['vei'] == 6]
    if len(vei6_data) > 0:
        plot_swarm_comparison(vei6_data,
                             f'Crisis Frequency Before vs After VEI 6 Volcanic Eruptions - {region_label}',
                             output_dir / 'vei6_before_after_comparison.png')
    
    # VEI 7 specific plot
    vei7_data = before_after_df[before_after_df['vei'] == 7]
    if len(vei7_data) > 0:
        plot_swarm_comparison(vei7_data,
                             f'Crisis Frequency Before vs After VEI 7 Volcanic Eruptions - {region_label}',
                             output_dir / 'vei7_before_after_comparison.png')
    
    # Timeline plot
    plot_crisis_timeline(crisis_matrix, volcano_df, output_dir / 'crisis_timeline.png')
    
    # === NEW PLOTS (Crisis Onsets) ===
    
    # Main onset comparison plot
    plot_onset_swarm_comparison(onset_before_after_df,
                               f'Crisis ONSET Frequency Before vs After Major Volcanic Eruptions - {region_label}',
                               output_dir / 'onset_before_after_comparison.png')
    
    # VEI 6 onset specific
    vei6_onset_data = onset_before_after_df[onset_before_after_df['vei'] == 6]
    if len(vei6_onset_data) > 0:
        plot_onset_swarm_comparison(vei6_onset_data,
                                   f'Crisis ONSET Frequency Before vs After VEI 6 Eruptions - {region_label}',
                                   output_dir / 'vei6_onset_comparison.png')
    
    # VEI 7 onset specific
    vei7_onset_data = onset_before_after_df[onset_before_after_df['vei'] == 7]
    if len(vei7_onset_data) > 0:
        plot_onset_swarm_comparison(vei7_onset_data,
                                   f'Crisis ONSET Frequency Before vs After VEI 7 Eruptions - {region_label}',
                                   output_dir / 'vei7_onset_comparison.png')
    
    # Temporal distribution plot
    plot_temporal_onset_distribution(onset_distribution_df,
                                    output_dir / 'onset_temporal_distribution.png')
    
    # Heatmap of onset differences
    plot_onset_difference_heatmap(onset_before_after_df, onset_stats_df,
                                 output_dir / 'onset_difference_heatmap.png')
    
    # Statistical summary plot
    plot_statistical_summary(onset_stats_df,
                           output_dir / 'onset_statistical_summary.png')
    
    # Print summary statistics for this region
    if len(onset_stats_df) > 0:
        print(f"\n  Onset Analysis Summary for {region_label}:")
        sig_windows = onset_stats_df[onset_stats_df['significant_05']]
        if len(sig_windows) > 0:
            print(f"    Significant differences (p<0.05) in windows: {list(sig_windows['window'].values)}")
            mean_effect = sig_windows['cohens_d'].mean()
            print(f"    Average effect size (Cohen's d): {mean_effect:.3f}")
        else:
            print("    No statistically significant differences found")