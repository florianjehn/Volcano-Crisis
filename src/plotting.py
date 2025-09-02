"""
Clean plotting functions for volcano-crisis analysis.
Creates the main comparison plot and timeline visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use clean style
plt.style.use('default')
sns.set_palette("husl")


def plot_before_after_comparison(before_after_df, save_path='results/before_after_comparison.png'):
    """
    Create comprehensive swarm plot comparing crisis counts before vs after eruptions.
    Shows all time windows (10-100 years) in a single plot with clear before/after groupings.
    """
    print("Creating before/after comparison plot...")
    
    # Prepare data for plotting with proper categorical x-positions
    plot_data = []
    windows = sorted(before_after_df['window'].unique())
    
    for _, row in before_after_df.iterrows():
        window = row['window']
        
        # Create categorical labels for clear grouping
        before_label = f"{window}yr\nBefore"
        after_label = f"{window}yr\nAfter"
        
        # Add "before" data point
        plot_data.append({
            'group': before_label,
            'period': 'Before',
            'crisis_count': row['before_count'],
            'window': window,
            'eruption': row['eruption_name']
        })
        
        # Add "after" data point  
        plot_data.append({
            'group': after_label,
            'period': 'After', 
            'crisis_count': row['after_count'],
            'window': window,
            'eruption': row['eruption_name']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create ordered list of groups for proper x-axis ordering
    group_order = []
    for window in windows:
        group_order.extend([f"{window}yr\nBefore", f"{window}yr\nAfter"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Create swarm plots with distinct colors
    sns.swarmplot(data=plot_df, x='group', y='crisis_count', hue='period', 
                  size=5, alpha=0.7, ax=ax, order=group_order,
                  palette={'Before': '#1f77b4', 'After': '#ff7f0e'})
    
    # Calculate and plot means and medians for each window/period combination
    for i, window in enumerate(windows):
        # Before data
        before_data = before_after_df[before_after_df['window'] == window]['before_count']
        before_mean = before_data.mean()
        before_median = before_data.median()
        
        # After data
        after_data = before_after_df[before_after_df['window'] == window]['after_count']
        after_mean = after_data.mean()
        after_median = after_data.median()
        
        # X positions for this window (before and after)
        before_x = i * 2  # Even positions for "before"
        after_x = i * 2 + 1  # Odd positions for "after"
        
        # Plot means (thick horizontal lines)
        ax.hlines(before_mean, before_x - 0.3, before_x + 0.3, colors='darkblue', 
                 linewidth=4, alpha=0.9, label='Before Mean' if i == 0 else "")
        ax.hlines(after_mean, after_x - 0.3, after_x + 0.3, colors='darkred', 
                 linewidth=4, alpha=0.9, label='After Mean' if i == 0 else "")
        
        # Plot medians (thin dashed lines)
        ax.hlines(before_median, before_x - 0.3, before_x + 0.3, colors='darkblue', 
                 linewidth=2, linestyles='dashed', alpha=0.8, label='Before Median' if i == 0 else "")
        ax.hlines(after_median, after_x - 0.3, after_x + 0.3, colors='darkred', 
                 linewidth=2, linestyles='dashed', alpha=0.8, label='After Median' if i == 0 else "")
    
    # Customize plot
    ax.set_xlabel('Time Window', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Active Crises', fontsize=14, fontweight='bold') 
    ax.set_title('Crisis Frequency Before vs After Major Volcanic Eruptions (VEI 6-7)', 
                fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels for readability
    ax.tick_params(axis='x', rotation=45)
    
    # Add vertical lines to separate time windows
    for i in range(1, len(windows)):
        ax.axvline(x=i*2 - 0.5, color='gray', alpha=0.3, linestyle=':', linewidth=1)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add subtitle with methodology
    fig.text(0.5, 0.02, 'Comparing X years before (year -X to -1) vs X years after (year +1 to +X) each eruption', 
             ha='center', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)
    
    for window in windows:
        before_data = before_after_df[before_after_df['window'] == window]['before_count']
        after_data = before_after_df[before_after_df['window'] == window]['after_count']
        
        print(f"{window:3d}-year window:")
        print(f"  Before: mean={before_data.mean():.1f}, median={before_data.median():.1f}")
        print(f"  After:  mean={after_data.mean():.1f}, median={after_data.median():.1f}")
        print(f"  Diff:   mean={after_data.mean() - before_data.mean():.1f}, median={after_data.median() - before_data.median():.1f}")
        print()


def plot_crisis_timeline(crisis_matrix, volcano_df, save_path='results/crisis_timeline.png'):
    """
    Create timeline showing all crises and volcanic eruption dates.
    """
    print("Creating crisis timeline plot...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Find all crisis periods
    crisis_periods = []
    for crisis in crisis_matrix.columns:
        active_years = crisis_matrix[crisis_matrix[crisis] == 1].index.tolist()
        
        if len(active_years) > 0:
            # Group consecutive years into periods
            periods = []
            start = active_years[0]
            end = active_years[0]
            
            for i in range(1, len(active_years)):
                if active_years[i] == end + 1:  # Consecutive year
                    end = active_years[i]
                else:  # Gap found, save current period and start new one
                    periods.append((start, end))
                    start = active_years[i]
                    end = active_years[i]
            
            # Add final period
            periods.append((start, end))
            
            for start, end in periods:
                crisis_periods.append({
                    'crisis': crisis,
                    'start': start,
                    'end': end,
                    'duration': end - start + 1
                })
    
    # Sort by start year
    crisis_periods.sort(key=lambda x: x['start'])
    
    # Plot crisis periods
    y_pos = 0
    crisis_positions = {}
    
    for period in crisis_periods:
        crisis = period['crisis']
        
        # Assign consistent y-position for each crisis
        if crisis not in crisis_positions:
            crisis_positions[crisis] = y_pos
            y_pos += 1
        
        # Plot the crisis period
        ax.plot([period['start'], period['end']], 
               [crisis_positions[crisis], crisis_positions[crisis]], 
               linewidth=2, alpha=0.6, color='gray')
    
    # Add volcanic eruptions as vertical lines
    for _, eruption in volcano_df.iterrows():
        color = 'red' if eruption['VEI'] == 7 else 'orange'
        linestyle = '-' if eruption['VEI'] == 7 else '--'
        ax.axvline(x=eruption['Year'], color=color, alpha=0.8, 
                  linestyle=linestyle, linewidth=2)
    
    # Customize plot
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Crisis Events', fontsize=14, fontweight='bold')
    ax.set_title('Timeline of Global Crises and Major Volcanic Eruptions', fontsize=16, fontweight='bold')
    
    # Show only subset of crisis labels (too many to show all)
    n_labels = min(20, len(crisis_positions))
    selected_crises = list(crisis_positions.keys())[::len(crisis_positions)//n_labels][:n_labels]
    selected_positions = [crisis_positions[c] for c in selected_crises]
    
    ax.set_yticks(selected_positions)
    ax.set_yticklabels(selected_crises, fontsize=8)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='VEI 7 Eruption'),
        Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='VEI 6 Eruption'), 
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.6, label='Crisis Period')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def plot_vei6_comparison(before_after_df, volcano_df, save_path='results/vei6_before_after_comparison.png'):
    """
    Create swarm plot for VEI 6 eruptions only.
    """
    print("Creating VEI 6 comparison plot...")
    
    # Filter for VEI 6 eruptions only
    vei6_data = before_after_df[before_after_df['vei'] == 6]
    
    if len(vei6_data) == 0:
        print("  No VEI 6 data found, skipping plot")
        return
    
    # Prepare data for plotting with proper categorical x-positions
    plot_data = []
    windows = sorted(vei6_data['window'].unique())
    
    for _, row in vei6_data.iterrows():
        window = row['window']
        
        # Create categorical labels for clear grouping
        before_label = f"{window}yr\nBefore"
        after_label = f"{window}yr\nAfter"
        
        # Add "before" data point
        plot_data.append({
            'group': before_label,
            'period': 'Before',
            'crisis_count': row['before_count'],
            'window': window,
            'eruption': row['eruption_name']
        })
        
        # Add "after" data point  
        plot_data.append({
            'group': after_label,
            'period': 'After', 
            'crisis_count': row['after_count'],
            'window': window,
            'eruption': row['eruption_name']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create ordered list of groups for proper x-axis ordering
    group_order = []
    for window in windows:
        group_order.extend([f"{window}yr\nBefore", f"{window}yr\nAfter"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Create swarm plots with distinct colors
    sns.swarmplot(data=plot_df, x='group', y='crisis_count', hue='period', 
                  size=5, alpha=0.7, ax=ax, order=group_order,
                  palette={'Before': '#1f77b4', 'After': '#ff7f0e'})
    
    # Calculate and plot means and medians for each window/period combination
    for i, window in enumerate(windows):
        # Before data
        before_data = vei6_data[vei6_data['window'] == window]['before_count']
        before_mean = before_data.mean()
        before_median = before_data.median()
        
        # After data
        after_data = vei6_data[vei6_data['window'] == window]['after_count']
        after_mean = after_data.mean()
        after_median = after_data.median()
        
        # X positions for this window (before and after)
        before_x = i * 2  # Even positions for "before"
        after_x = i * 2 + 1  # Odd positions for "after"
        
        # Plot means (thick horizontal lines)
        ax.hlines(before_mean, before_x - 0.3, before_x + 0.3, colors='darkblue', 
                 linewidth=4, alpha=0.9, label='Before Mean' if i == 0 else "")
        ax.hlines(after_mean, after_x - 0.3, after_x + 0.3, colors='darkred', 
                 linewidth=4, alpha=0.9, label='After Mean' if i == 0 else "")
        
        # Plot medians (thin dashed lines)
        ax.hlines(before_median, before_x - 0.3, before_x + 0.3, colors='darkblue', 
                 linewidth=2, linestyles='dashed', alpha=0.8, label='Before Median' if i == 0 else "")
        ax.hlines(after_median, after_x - 0.3, after_x + 0.3, colors='darkred', 
                 linewidth=2, linestyles='dashed', alpha=0.8, label='After Median' if i == 0 else "")
    
    # Customize plot
    ax.set_xlabel('Time Window', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Active Crises', fontsize=14, fontweight='bold') 
    ax.set_title('Crisis Frequency Before vs After VEI 6 Volcanic Eruptions', 
                fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels for readability
    ax.tick_params(axis='x', rotation=45)
    
    # Add vertical lines to separate time windows
    for i in range(1, len(windows)):
        ax.axvline(x=i*2 - 0.5, color='gray', alpha=0.3, linestyle=':', linewidth=1)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Count unique eruptions
    n_eruptions = len(vei6_data['eruption_name'].unique())
    
    # Add subtitle with methodology
    fig.text(0.5, 0.02, f'VEI 6 eruptions only - {n_eruptions} eruptions analyzed', 
             ha='center', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def plot_vei7_comparison(before_after_df, volcano_df, save_path='results/vei7_before_after_comparison.png'):
    """
    Create swarm plot for VEI 7 eruptions only.
    """
    print("Creating VEI 7 comparison plot...")
    
    # Filter for VEI 7 eruptions only
    vei7_data = before_after_df[before_after_df['vei'] == 7]
    
    if len(vei7_data) == 0:
        print("  No VEI 7 data found, skipping plot")
        return
    
    # Prepare data for plotting with proper categorical x-positions
    plot_data = []
    windows = sorted(vei7_data['window'].unique())
    
    for _, row in vei7_data.iterrows():
        window = row['window']
        
        # Create categorical labels for clear grouping
        before_label = f"{window}yr\nBefore"
        after_label = f"{window}yr\nAfter"
        
        # Add "before" data point
        plot_data.append({
            'group': before_label,
            'period': 'Before',
            'crisis_count': row['before_count'],
            'window': window,
            'eruption': row['eruption_name']
        })
        
        # Add "after" data point  
        plot_data.append({
            'group': after_label,
            'period': 'After', 
            'crisis_count': row['after_count'],
            'window': window,
            'eruption': row['eruption_name']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create ordered list of groups for proper x-axis ordering
    group_order = []
    for window in windows:
        group_order.extend([f"{window}yr\nBefore", f"{window}yr\nAfter"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Create swarm plots with distinct colors
    sns.swarmplot(data=plot_df, x='group', y='crisis_count', hue='period', 
                  size=5, alpha=0.7, ax=ax, order=group_order,
                  palette={'Before': '#1f77b4', 'After': '#ff7f0e'})
    
    # Calculate and plot means and medians for each window/period combination
    for i, window in enumerate(windows):
        # Before data
        before_data = vei7_data[vei7_data['window'] == window]['before_count']
        before_mean = before_data.mean()
        before_median = before_data.median()
        
        # After data
        after_data = vei7_data[vei7_data['window'] == window]['after_count']
        after_mean = after_data.mean()
        after_median = after_data.median()
        
        # X positions for this window (before and after)
        before_x = i * 2  # Even positions for "before"
        after_x = i * 2 + 1  # Odd positions for "after"
        
        # Plot means (thick horizontal lines)
        ax.hlines(before_mean, before_x - 0.3, before_x + 0.3, colors='darkblue', 
                 linewidth=4, alpha=0.9, label='Before Mean' if i == 0 else "")
        ax.hlines(after_mean, after_x - 0.3, after_x + 0.3, colors='darkred', 
                 linewidth=4, alpha=0.9, label='After Mean' if i == 0 else "")
        
        # Plot medians (thin dashed lines)
        ax.hlines(before_median, before_x - 0.3, before_x + 0.3, colors='darkblue', 
                 linewidth=2, linestyles='dashed', alpha=0.8, label='Before Median' if i == 0 else "")
        ax.hlines(after_median, after_x - 0.3, after_x + 0.3, colors='darkred', 
                 linewidth=2, linestyles='dashed', alpha=0.8, label='After Median' if i == 0 else "")
    
    # Customize plot
    ax.set_xlabel('Time Window', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Active Crises', fontsize=14, fontweight='bold') 
    ax.set_title('Crisis Frequency Before vs After VEI 7 Volcanic Eruptions', 
                fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels for readability
    ax.tick_params(axis='x', rotation=45)
    
    # Add vertical lines to separate time windows
    for i in range(1, len(windows)):
        ax.axvline(x=i*2 - 0.5, color='gray', alpha=0.3, linestyle=':', linewidth=1)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Count unique eruptions
    n_eruptions = len(vei7_data['eruption_name'].unique())
    
    # Add subtitle with methodology
    fig.text(0.5, 0.02, f'VEI 7 eruptions only - {n_eruptions} eruptions analyzed', 
             ha='center', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def create_all_plots(crisis_matrix, volcano_df, before_after_df):
    """Create all visualizations."""
    print("=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Create all plots
    plot_before_after_comparison(before_after_df)  # Combined VEI 6-7
    plot_vei6_comparison(before_after_df, volcano_df)  # VEI 6 only
    plot_vei7_comparison(before_after_df, volcano_df)  # VEI 7 only
    plot_crisis_timeline(crisis_matrix, volcano_df)  # Timeline overview
    
    print("\nAll plots saved to results/")
    print("=" * 50)
