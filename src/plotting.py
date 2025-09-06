"""Plotting functions for volcano-crisis analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def create_all_plots(crisis_matrix, volcano_df, before_after_df, output_subfolder='World'):
    """Create all visualizations in the specified subfolder."""
    # Create output directory structure
    output_dir = Path('results') / output_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Adjust title to include region
    region_label = output_subfolder if output_subfolder != 'World' else 'Global'
    
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