"""
Volcano Crisis Onset Analysis
Analyzes whether major volcanic eruptions (VEI 6-7) trigger new crisis onsets.
Focuses specifically on when crises BEGIN, not when they're active.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style for cleaner plots
plt.style.use('default')
sns.set_palette("husl")


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_crisis_data(filepath='data/CrisisConsequencesData_NavigatingPolycrisis_2023.03.csv', 
                     continent_filter=None):
    """
    Load crisis data with optional continent filtering.
    
    Args:
        filepath: Path to crisis CSV file
        continent_filter: Optional continent name to filter by
    
    Returns:
        DataFrame with crisis data
    """
    df = pd.read_csv(filepath, encoding='latin1')
    
    # Convert dates to numeric, handling errors
    df['Polity.Date.From'] = pd.to_numeric(df['Polity.Date.From'], errors='coerce')
    df['Polity.Date.To'] = pd.to_numeric(df['Polity.Date.To'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Polity.Date.From', 'Polity.Date.To']).astype({
        'Polity.Date.From': int, 
        'Polity.Date.To': int
    })
    
    # Apply continent filter if specified
    if continent_filter and 'Continent' in df.columns:
        df = df[df['Continent'] == continent_filter].copy()
    
    return df


def load_volcano_data(filepath='data/volcano_list.csv', 
                      max_year=1920, 
                      min_year=None, 
                      continent_filter=None):
    """
    Load volcano data for VEI 6-7 eruptions.
    
    Args:
        filepath: Path to volcano CSV file
        max_year: Maximum year to include (default 1920)
        min_year: Optional minimum year filter
        continent_filter: Optional continent name to filter by
    
    Returns:
        DataFrame with filtered volcano data
    """
    df = pd.read_csv(filepath)
    
    # Filter by year range
    df = df[df['Year'] <= max_year].copy()
    if min_year:
        df = df[df['Year'] >= min_year].copy()
    
    # Keep only major eruptions (VEI 6-7)
    df = df[df['VEI'].isin([6, 7])].copy()
    
    # Apply continent filter if specified
    if continent_filter:
        df = df[df['Continent'] == continent_filter].copy()
    
    return df


def get_data_boundaries(crisis_df):
    """
    Determine the actual boundaries of available crisis data.
    
    Args:
        crisis_df: DataFrame with crisis data
    
    Returns:
        Tuple of (earliest_year, latest_year) with data
    """
    # Get the earliest and latest years that have any crisis data
    earliest = crisis_df['Polity.Date.From'].min()
    latest = crisis_df['Polity.Date.To'].max()
    
    return earliest, latest


def create_crisis_matrix(crisis_df, start_year=-2500, end_year=1920):
    """
    Create a binary matrix showing which crises were active in which years.
    Only used for timeline visualization.
    
    Args:
        crisis_df: DataFrame with crisis data
        start_year: First year of matrix
        end_year: Last year of matrix
    
    Returns:
        DataFrame with years as index, crisis names as columns, 1/0 values
    """
    years = range(start_year, end_year + 1)
    crises = crisis_df['Crisis.Case'].unique()
    matrix = pd.DataFrame(0, index=years, columns=crises)
    
    for _, row in crisis_df.iterrows():
        from_year = max(row['Polity.Date.From'], start_year)
        to_year = min(row['Polity.Date.To'], end_year)
        if from_year <= to_year:
            matrix.loc[from_year:to_year, row['Crisis.Case']] = 1
    
    return matrix


def count_crisis_onsets_in_period(crisis_df, start_year, end_year, data_boundaries):
    """
    Count the number of crises that BEGIN within a specified period.
    Returns NaN if the period extends beyond available data.
    
    Args:
        crisis_df: DataFrame with crisis data
        start_year: Beginning of period (inclusive)
        end_year: End of period (inclusive)
        data_boundaries: Tuple of (earliest, latest) years with data
    
    Returns:
        Number of unique crises that started in this period, or NaN if incomplete data
    """
    earliest_data, latest_data = data_boundaries
    
    # Check if the requested period is fully within available data
    # We need to ensure we have data for the ENTIRE period
    if start_year < earliest_data or end_year > latest_data:
        return np.nan
    
    # Filter for crises that start within the window
    onsets = crisis_df[
        (crisis_df['Polity.Date.From'] >= start_year) & 
        (crisis_df['Polity.Date.From'] <= end_year)
    ]
    
    # Count unique crisis cases
    return onsets['Crisis.Case'].nunique()


def compute_onset_before_after_data(crisis_df, volcano_df, 
                                   time_windows=None, 
                                   min_year=None):
    """
    Compute crisis onset counts before and after each volcanic eruption.
    Returns NaN for windows that extend beyond available data.
    
    Args:
        crisis_df: DataFrame with crisis data
        volcano_df: DataFrame with volcano eruption data
        time_windows: List of window sizes in years (default: 10-150 by 10s)
        min_year: Minimum year to include in analysis
    
    Returns:
        DataFrame with onset counts and differences for each eruption/window
    """
    if time_windows is None:
        time_windows = list(range(10, 160, 10))  # 10, 20, ..., 150
    
    # Get data boundaries
    data_boundaries = get_data_boundaries(crisis_df)
    earliest_data, latest_data = data_boundaries
    
    results = []
    
    for _, eruption in volcano_df.iterrows():
        year = eruption['Year']
        name = eruption['Name']
        vei = eruption['VEI']
        
        for window in time_windows:
            before_start = year - window
            before_end = year - 1
            after_start = year + 1
            after_end = year + window
            
            # Skip if before period extends beyond minimum year filter
            if min_year and before_start < min_year:
                continue
            
            # Count crisis onsets in each period (will return NaN if outside data boundaries)
            before_onsets = count_crisis_onsets_in_period(crisis_df, before_start, before_end, data_boundaries)
            after_onsets = count_crisis_onsets_in_period(crisis_df, after_start, after_end, data_boundaries)
            
            # Calculate difference only if both periods have valid data
            if pd.notna(before_onsets) and pd.notna(after_onsets):
                onset_diff = after_onsets - before_onsets
                onset_ratio = after_onsets / before_onsets if before_onsets > 0 else np.nan
            else:
                onset_diff = np.nan
                onset_ratio = np.nan
            
            results.append({
                'eruption_year': year,
                'eruption_name': name,
                'vei': vei,
                'window': window,
                'before_onset_count': before_onsets,
                'after_onset_count': after_onsets,
                'onset_difference': onset_diff,
                'onset_ratio': onset_ratio,
                'data_complete': pd.notna(before_onsets) and pd.notna(after_onsets)
            })
    
    df = pd.DataFrame(results)
    
    # Report data availability
    for window in time_windows:
        window_data = df[df['window'] == window]
        complete = window_data['data_complete'].sum()
        total = len(window_data)
        if total > 0:
            print(f"  Window {window:3d}yr: {complete}/{total} eruptions have complete data")
    
    return df


def test_onset_significance(onset_df):
    """
    Perform statistical tests on crisis onset differences.
    Only uses eruptions with complete data for both before and after periods.
    
    Args:
        onset_df: DataFrame from compute_onset_before_after_data
    
    Returns:
        DataFrame with statistical results for each time window
    """
    results = []
    
    for window in onset_df['window'].unique():
        # Only use eruptions with complete data for this window
        window_data = onset_df[(onset_df['window'] == window) & 
                               (onset_df['data_complete'] == True)].copy()
        
        # Skip if insufficient data
        if len(window_data) < 3:
            continue
        
        before = window_data['before_onset_count'].values
        after = window_data['after_onset_count'].values
        
        # Paired t-test (each eruption is its own control)
        t_stat, p_value = stats.ttest_rel(after, before)
        
        # Calculate effect size (Cohen's d for paired samples)
        differences = after - before
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        results.append({
            'window': window,
            'n_eruptions': len(window_data),
            'mean_before': np.mean(before),
            'mean_after': np.mean(after),
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant_05': p_value < 0.05,
            'significant_01': p_value < 0.01
        })
    
    return pd.DataFrame(results)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_onset_swarm_plot_data(data_df):
    """Convert onset before/after data to format suitable for swarm plot."""
    plot_data = []
    for _, row in data_df.iterrows():
        # Only include if we have complete data
        if not row['data_complete']:
            continue
            
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


def plot_onset_swarm_comparison(data_df, title, save_path, stats_df=None):
    """
    Create swarm plot comparing crisis onsets before vs after eruptions.
    Only plots eruptions with complete data.
    """
    # Filter for complete data only
    complete_data = data_df[data_df['data_complete'] == True].copy()
    
    if len(complete_data) == 0:
        print(f"  No complete data for {title}")
        return
    
    plot_df = create_onset_swarm_plot_data(complete_data)
    windows = sorted(complete_data['window'].unique())
    
    # Create group order for x-axis
    group_order = []
    for window in windows:
        group_order.extend([f"{window}yr\nBefore", f"{window}yr\nAfter"])
    
    # Create figure with better size
    fig, ax = plt.subplots(figsize=(min(20, 2 + len(windows) * 1.5), 8))
    
    # Create swarm plot with distinct colors
    sns.swarmplot(data=plot_df, x='group', y='onset_count', hue='period',
                  size=6, alpha=0.7, ax=ax, order=group_order,
                  palette={'Before': '#2ca02c', 'After': '#d62728'})
    
    # Add mean and median lines with statistical annotations
    for i, window in enumerate(windows):
        window_complete = complete_data[complete_data['window'] == window]
        before_data = window_complete['before_onset_count']
        after_data = window_complete['after_onset_count']
        
        before_x, after_x = i * 2, i * 2 + 1
        
        # Calculate statistics
        mean_before = before_data.mean()
        mean_after = after_data.mean()
        n_samples = len(window_complete)
        
        # Means (thick lines)
        ax.hlines(mean_before, before_x - 0.35, before_x + 0.35, 
                 colors='darkgreen', linewidth=3, alpha=0.9,
                 label='Mean' if i == 0 else "")
        ax.hlines(mean_after, after_x - 0.35, after_x + 0.35,
                 colors='darkred', linewidth=3, alpha=0.9)
        
        # Medians (thin lines)
        ax.hlines(before_data.median(), before_x - 0.25, before_x + 0.25,
                 colors='darkgreen', linewidth=1.5, linestyles='--', alpha=0.7,
                 label='Median' if i == 0 else "")
        ax.hlines(after_data.median(), after_x - 0.25, after_x + 0.25,
                 colors='darkred', linewidth=1.5, linestyles='--', alpha=0.7)
        
        # Add sample size annotation
        ax.text((before_x + after_x) / 2, ax.get_ylim()[0] + 0.5,
               f'n={n_samples}', ha='center', fontsize=8, color='gray')
        
        # Add percentage change annotation
        if mean_before > 0:
            pct_change = ((mean_after - mean_before) / mean_before) * 100
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text((before_x + after_x) / 2, y_pos,
                   f"{pct_change:+.0f}%", ha='center', fontsize=9,
                   color='darkred' if pct_change > 0 else 'darkgreen',
                   fontweight='bold')
    
    # Add year range to title
    year_range = f"{complete_data['eruption_year'].min()}-{complete_data['eruption_year'].max()} CE"
    full_title = f"{title}\n({year_range})"
    
    ax.set_xlabel('Time Window', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crisis Onsets', fontsize=12, fontweight='bold')
    ax.set_title(full_title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Simplified legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], ['Before', 'After', 'Mean'], 
             loc='upper left', fontsize=10)
    
    # Add separators between time windows
    for i in range(1, len(windows)):
        ax.axvline(x=i*2 - 0.5, color='gray', alpha=0.2, linestyle=':', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_onset_difference_heatmap(onset_df, stats_df, save_path):
    """
    Create heatmap showing the difference in crisis onsets (after - before) for each eruption.
    Gray cells indicate missing data (period extends beyond available data).
    """
    if len(onset_df) == 0:
        return
    
    # Pivot data to create matrix for heatmap
    # Use onset_difference which is NaN for incomplete data
    pivot_data = onset_df.pivot_table(
        index='eruption_name',
        columns='window',
        values='onset_difference',
        aggfunc='first'  # Use first since there should be only one value per cell
    )
    
    # Sort by eruption year
    eruption_years = onset_df.groupby('eruption_name')['eruption_year'].first()
    pivot_data = pivot_data.loc[eruption_years.sort_values().index]
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_data) * 0.3)))
    
    # Create custom colormap with gray for missing data
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # Create heatmap with NaN shown as gray
    sns.heatmap(pivot_data, annot=True, fmt='.0f', center=0,
                cmap=cmap, vmin=-8, vmax=8,
                cbar_kws={'label': 'Onset Difference (After - Before)'},
                ax=ax, linewidths=0.5, annot_kws={'size': 8},
                mask=pivot_data.isna())  # This will show NaN as background color
    
    # Add gray patches for missing data
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if pd.isna(pivot_data.iloc[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, 
                                         color='lightgray', alpha=0.5))
    
    # Add significance markers
    if len(stats_df) > 0:
        for j, window in enumerate(pivot_data.columns):
            window_stats = stats_df[stats_df['window'] == window]
            if len(window_stats) > 0:
                n_complete = window_stats.iloc[0]['n_eruptions']
                # Add sample size annotation
                ax.text(j + 0.5, -0.8, f'n={n_complete}', 
                       ha='center', va='top', fontsize=8, color='gray')
                
                # Add significance marker if applicable
                if window_stats.iloc[0]['significant_05']:
                    marker = '**' if window_stats.iloc[0]['significant_01'] else '*'
                    ax.text(j + 0.5, -0.3, marker, 
                           ha='center', va='top', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Time Window (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Volcanic Eruption', fontsize=11, fontweight='bold')
    ax.set_title('Crisis Onset Differences: After minus Before Eruption\n' +
                '(Red = More after, Blue = More before, Gray = Incomplete data)',
                fontsize=12, fontweight='bold')
    
    # Add year labels to eruption names
    y_labels = []
    for name in pivot_data.index:
        year = eruption_years[name]
        vei = onset_df[onset_df['eruption_name'] == name]['vei'].iloc[0]
        y_labels.append(f"{name} ({year}, VEI {vei})")
    ax.set_yticklabels(y_labels, rotation=0, fontsize=8)
    
    # Add legend for significance
    ax.text(1.02, 0.98, '* p < 0.05\n** p < 0.01\nn = sample size', 
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_statistical_summary(stats_df, save_path):
    """
    Create plot showing statistical significance across time windows.
    Shows p-values and effect sizes with sample sizes annotated.
    """
    if len(stats_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: P-values with significance thresholds
    ax1 = axes[0]
    ax1.plot(stats_df['window'], stats_df['p_value'], 
            marker='o', linewidth=2, markersize=8, color='darkblue')
    ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='p = 0.05')
    ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='p = 0.01')
    
    # Add sample size annotations
    for _, row in stats_df.iterrows():
        ax1.annotate(f"n={row['n_eruptions']}", 
                    (row['window'], row['p_value']),
                    textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=7, color='gray')
    
    ax1.set_xlabel('Time Window (years)', fontsize=11)
    ax1.set_ylabel('P-value (paired t-test)', fontsize=11)
    ax1.set_title('Statistical Significance by Time Window', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Color significant points
    sig_05 = stats_df[stats_df['significant_05']]
    sig_01 = stats_df[stats_df['significant_01']]
    ax1.scatter(sig_05['window'], sig_05['p_value'], 
               c='orange', s=100, zorder=5)
    ax1.scatter(sig_01['window'], sig_01['p_value'], 
               c='red', s=100, zorder=6)
    
    # Plot 2: Effect size (Cohen's d) with interpretation lines
    ax2 = axes[1]
    ax2.plot(stats_df['window'], stats_df['cohens_d'], 
            marker='s', linewidth=2, markersize=8, color='purple')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Small')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium')
    ax2.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large')
    
    # Add sample size as secondary y-axis
    ax2_twin = ax2.twinx()
    ax2_twin.bar(stats_df['window'], stats_df['n_eruptions'], 
                alpha=0.2, color='gray', width=8)
    ax2_twin.set_ylabel('Sample Size (n)', fontsize=10, color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')
    
    ax2.set_xlabel('Time Window (years)', fontsize=11)
    ax2.set_ylabel("Cohen's d (Effect Size)", fontsize=11)
    ax2.set_title('Effect Size by Time Window', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Effect Size', loc='best', fontsize=9)
    
    # Add mean difference annotation
    for _, row in stats_df.iterrows():
        if row['significant_05']:
            ax2.annotate(f"{row['mean_difference']:+.1f}", 
                        (row['window'], row['cohens_d']),
                        textcoords="offset points", xytext=(0,5),
                        ha='center', fontsize=8, color='darkred')
    
    plt.suptitle('Statistical Analysis of Crisis Onsets Around Volcanic Eruptions\n' + 
                '(Sample size decreases with window size due to data availability)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_crisis_timeline(crisis_matrix, volcano_df, save_path):
    """
    Create timeline showing crises and eruptions.
    Visualizes the temporal relationship between volcanic eruptions and crisis periods.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot crisis periods as horizontal lines
    crisis_periods = []
    for crisis in crisis_matrix.columns[:50]:  # Limit to first 50 crises for readability
        active_years = crisis_matrix[crisis_matrix[crisis] == 1].index.tolist()
        if not active_years:
            continue
        
        # Find continuous periods
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
    for period in crisis_periods[:100]:  # Limit number of crisis lines for clarity
        crisis = period['crisis']
        if crisis not in crisis_positions:
            crisis_positions[crisis] = y_pos
            y_pos += 1
        ax.plot([period['start'], period['end']], 
               [crisis_positions[crisis], crisis_positions[crisis]], 
               linewidth=1.5, alpha=0.5, color='gray')
    
    # Plot volcanic eruptions as vertical lines
    for _, eruption in volcano_df.iterrows():
        color = 'red' if eruption['VEI'] == 7 else 'orange'
        linewidth = 2.5 if eruption['VEI'] == 7 else 1.5
        alpha = 0.8 if eruption['VEI'] == 7 else 0.6
        ax.axvline(x=eruption['Year'], color=color, alpha=alpha, 
                  linewidth=linewidth)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Crisis Events', fontsize=12, fontweight='bold')
    ax.set_title('Timeline of Global Crises and Major Volcanic Eruptions', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2.5, alpha=0.8, label='VEI 7 Eruption'),
        Line2D([0], [0], color='orange', linewidth=1.5, alpha=0.6, label='VEI 6 Eruption'),
        Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.5, label='Crisis Period')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_analysis(start_year=None, volcano_list=None):
    """
    Run the complete volcano-crisis onset analysis.
    
    Args:
        start_year: Optional minimum year for analysis (e.g., 1000 for medieval onwards)
        volcano_list: Optional list of specific volcano names to analyze
    """
    print("=" * 60)
    print("VOLCANO CRISIS ONSET ANALYSIS")
    print("=" * 60)
    
    if start_year:
        print(f"Analysis period: {start_year} CE onwards")
    if volcano_list:
        print(f"Filtered for volcanoes: {', '.join(volcano_list)}")
    
    # Load global data
    print("\nLoading data...")
    crisis_df = load_crisis_data()
    
    # Report data boundaries
    earliest, latest = get_data_boundaries(crisis_df)
    print(f"Crisis data available from {earliest} to {latest} CE")
    
    crisis_matrix = create_crisis_matrix(crisis_df)
    volcano_df = load_volcano_data(min_year=start_year)
    
    if volcano_list:
        volcano_df = volcano_df[volcano_df['Name'].isin(volcano_list)]
    
    # Get unique continents
    continents = volcano_df['Continent'].dropna().unique().tolist()
    
    # Process global data
    print("\n" + "=" * 60)
    print("GLOBAL ANALYSIS")
    print("=" * 60)
    
    print("\nData availability by time window:")
    onset_df = compute_onset_before_after_data(crisis_df, volcano_df, min_year=start_year)
    stats_df = test_onset_significance(onset_df)
    
    # Create output directory
    output_dir = Path('results/World')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for global analysis
    print("\nCreating global plots...")
    
    # Main comparison plot
    plot_onset_swarm_comparison(
        onset_df,
        'Crisis ONSET Frequency Before vs After Major Volcanic Eruptions (VEI 6-7) - Global',
        output_dir / 'onset_comparison.png',
        stats_df
    )
    
    # VEI-specific plots
    vei6_data = onset_df[onset_df['vei'] == 6]
    if len(vei6_data[vei6_data['data_complete'] == True]) > 0:
        plot_onset_swarm_comparison(
            vei6_data,
            'Crisis ONSET Frequency Before vs After VEI 6 Eruptions - Global',
            output_dir / 'vei6_onset_comparison.png'
        )
    
    vei7_data = onset_df[onset_df['vei'] == 7]
    if len(vei7_data[vei7_data['data_complete'] == True]) > 0:
        plot_onset_swarm_comparison(
            vei7_data,
            'Crisis ONSET Frequency Before vs After VEI 7 Eruptions - Global',
            output_dir / 'vei7_onset_comparison.png'
        )
    
    # Statistical plots
    plot_onset_difference_heatmap(onset_df, stats_df, output_dir / 'onset_heatmap.png')
    plot_statistical_summary(stats_df, output_dir / 'statistical_summary.png')
    plot_crisis_timeline(crisis_matrix, volcano_df, output_dir / 'crisis_timeline.png')
    
    # Print global results
    if len(stats_df) > 0:
        print("\nGlobal Results Summary:")
        complete_eruptions = onset_df.groupby('eruption_name')['data_complete'].any().sum()
        print(f"Analyzed {complete_eruptions} eruptions with at least some complete windows")
        print(f"Date range: {onset_df['eruption_year'].min()}-{onset_df['eruption_year'].max()} CE")
        
        sig_windows = stats_df[stats_df['significant_05']]
        if len(sig_windows) > 0:
            print(f"\nSignificant differences found in {len(sig_windows)}/{len(stats_df)} time windows:")
            for _, row in sig_windows.iterrows():
                print(f"  {row['window']:3d}yr: Δ={row['mean_difference']:+.2f} crises, "
                      f"p={row['p_value']:.4f}, d={row['cohens_d']:+.3f}, n={row['n_eruptions']}")
        else:
            print("No statistically significant differences found")
    
    # Process each continent
    for continent in continents:
        print("\n" + "=" * 60)
        print(f"{continent.upper()} ANALYSIS")
        print("=" * 60)
        
        # Load continent-specific data
        crisis_df_cont = load_crisis_data(continent_filter=continent)
        
        if len(crisis_df_cont) == 0:
            print(f"No crisis data for {continent}")
            continue
            
        crisis_matrix_cont = create_crisis_matrix(crisis_df_cont)
        volcano_df_cont = load_volcano_data(min_year=start_year, continent_filter=continent)
        
        if volcano_list:
            volcano_df_cont = volcano_df_cont[volcano_df_cont['Name'].isin(volcano_list)]
        
        if len(volcano_df_cont) == 0:
            print(f"No eruptions found for {continent}")
            continue
        
        print(f"\nData availability for {continent}:")
        onset_df_cont = compute_onset_before_after_data(crisis_df_cont, volcano_df_cont, min_year=start_year)
        
        if len(onset_df_cont[onset_df_cont['data_complete'] == True]) == 0:
            print(f"No complete data windows for {continent}")
            continue
        
        stats_df_cont = test_onset_significance(onset_df_cont)
        
        # Create output directory
        output_dir_cont = Path(f'results/{continent}')
        output_dir_cont.mkdir(parents=True, exist_ok=True)
        
        # Create plots
        print(f"Creating {continent} plots...")
        
        plot_onset_swarm_comparison(
            onset_df_cont,
            f'Crisis ONSET Frequency Before vs After Major Volcanic Eruptions - {continent}',
            output_dir_cont / 'onset_comparison.png',
            stats_df_cont
        )
        
        # VEI-specific plots
        vei6_data_cont = onset_df_cont[onset_df_cont['vei'] == 6]
        if len(vei6_data_cont[vei6_data_cont['data_complete'] == True]) > 0:
            plot_onset_swarm_comparison(
                vei6_data_cont,
                f'Crisis ONSET Frequency Before vs After VEI 6 Eruptions - {continent}',
                output_dir_cont / 'vei6_onset_comparison.png'
            )
        
        vei7_data_cont = onset_df_cont[onset_df_cont['vei'] == 7]
        if len(vei7_data_cont[vei7_data_cont['data_complete'] == True]) > 0:
            plot_onset_swarm_comparison(
                vei7_data_cont,
                f'Crisis ONSET Frequency Before vs After VEI 7 Eruptions - {continent}',
                output_dir_cont / 'vei7_onset_comparison.png'
            )
        
        plot_onset_difference_heatmap(onset_df_cont, stats_df_cont, 
                                     output_dir_cont / 'onset_heatmap.png')
        plot_statistical_summary(stats_df_cont, output_dir_cont / 'statistical_summary.png')
        plot_crisis_timeline(crisis_matrix_cont, volcano_df_cont, 
                           output_dir_cont / 'crisis_timeline.png')
        
        # Print continent results
        if len(stats_df_cont) > 0:
            print(f"\n{continent} Results:")
            complete_eruptions = onset_df_cont.groupby('eruption_name')['data_complete'].any().sum()
            print(f"Analyzed {complete_eruptions} eruptions with complete data")
            
            sig_windows_cont = stats_df_cont[stats_df_cont['significant_05']]
            if len(sig_windows_cont) > 0:
                print(f"Significant windows: {list(sig_windows_cont['window'].values)}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nResults saved to 'results/' folder")
    print("\nKey files:")
    print("  - onset_comparison.png: Main before/after comparison")
    print("  - onset_heatmap.png: Eruption-by-eruption differences")
    print("  - statistical_summary.png: Significance and effect sizes")
    print("  - crisis_timeline.png: Temporal overview")
    print("\nInterpretation:")
    print("  - Gray cells in heatmap = insufficient data for that time window")
    print("  - Sample sizes decrease with larger windows (data availability)")
    print("  - Statistical tests only use eruptions with complete data")
    print("  - Positive values = MORE crises after eruptions")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configuration
    START_YEAR = None  # Set to None for all data, or e.g. 1000 for medieval onwards
    VOLCANO_LIST = None  # Set to None for all volcanoes, or provide a list of names
    
    # Run the analysis
    run_analysis(start_year=START_YEAR, volcano_list=VOLCANO_LIST)