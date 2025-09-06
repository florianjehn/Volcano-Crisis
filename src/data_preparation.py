"""Data processing for volcano-crisis analysis with crisis onset tracking."""

import pandas as pd
import numpy as np
from scipy import stats


def load_crisis_data(filepath='data/CrisisConsequencesData_NavigatingPolycrisis_2023.03.csv', continent_filter=None):
    """Load crisis data, optionally filtering by continent."""
    df = pd.read_csv(filepath, encoding='latin1')
    df['Polity.Date.From'] = pd.to_numeric(df['Polity.Date.From'], errors='coerce')
    df['Polity.Date.To'] = pd.to_numeric(df['Polity.Date.To'], errors='coerce')
    df = df.dropna(subset=['Polity.Date.From', 'Polity.Date.To']).astype({
        'Polity.Date.From': int, 'Polity.Date.To': int
    })
    
    # Filter by continent if specified and continent column exists
    if continent_filter and 'Continent' in df.columns:
        df = df[df['Continent'] == continent_filter].copy()
    
    return df


def create_crisis_matrix(crisis_df, start_year=-2500, end_year=1920):
    years = range(start_year, end_year + 1)
    crises = crisis_df['Crisis.Case'].unique()
    matrix = pd.DataFrame(0, index=years, columns=crises)
    
    for _, row in crisis_df.iterrows():
        from_year = max(row['Polity.Date.From'], start_year)
        to_year = min(row['Polity.Date.To'], end_year)
        if from_year <= to_year:
            matrix.loc[from_year:to_year, row['Crisis.Case']] = 1
    
    return matrix


def load_volcano_data(filepath='data/volcano_list.csv', max_year=1920, min_year=None, continent_filter=None):
    """Load volcano data, optionally filtering by continent."""
    df = pd.read_csv(filepath)
    df = df[df['Year'] <= max_year].copy()
    if min_year:
        df = df[df['Year'] >= min_year].copy()
    df = df[df['VEI'].isin([6, 7])].copy()
    
    # Filter by continent if specified
    if continent_filter:
        df = df[df['Continent'] == continent_filter].copy()
    
    return df


def count_crises_in_period(crisis_matrix, start_year, end_year):
    if start_year < crisis_matrix.index.min() or end_year > crisis_matrix.index.max():
        return np.nan
    period = crisis_matrix.loc[start_year:end_year]
    return (period.sum(axis=0) > 0).sum()


def count_crisis_onsets_in_period(crisis_df, start_year, end_year):
    """
    Count the number of crises that BEGIN (onset) within the specified period.
    
    This avoids double-counting long-duration crises that span across the eruption date.
    Only counts each crisis once, at its start date.
    
    Args:
        crisis_df: DataFrame with crisis data including 'Polity.Date.From' column
        start_year: Beginning of period (inclusive)
        end_year: End of period (inclusive)
    
    Returns:
        Number of unique crises that started in this period
    """
    # Filter for crises that start within the window
    onsets = crisis_df[
        (crisis_df['Polity.Date.From'] >= start_year) & 
        (crisis_df['Polity.Date.From'] <= end_year)
    ]
    
    # Count unique crisis cases (in case same crisis appears multiple times)
    return onsets['Crisis.Case'].nunique()


def compute_before_after_data(crisis_matrix, volcano_df, time_windows=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], min_year=None):
    results = []
    
    for _, eruption in volcano_df.iterrows():
        year, name, vei = eruption['Year'], eruption['Name'], eruption['VEI']
        
        for window in time_windows:
            before_start = year - window
            if min_year and before_start < min_year:
                continue
                
            before_count = count_crises_in_period(crisis_matrix, before_start, year - 1)
            after_count = count_crises_in_period(crisis_matrix, year + 1, year + window)
            
            results.append({
                'eruption_year': year,
                'eruption_name': name,
                'vei': vei,
                'window': window,
                'before_count': before_count,
                'after_count': after_count
            })
    
    df = pd.DataFrame(results).dropna(subset=['before_count', 'after_count'])
    return df


def compute_onset_before_after_data(crisis_df, volcano_df, time_windows=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], min_year=None):
    """
    Compute crisis onset counts before and after volcanic eruptions.
    
    This function specifically tracks when crises BEGIN, not just when they're active.
    This provides a cleaner signal of whether eruptions trigger new crises.
    
    Args:
        crisis_df: DataFrame with crisis data
        volcano_df: DataFrame with volcano eruption data
        time_windows: List of time window sizes to analyze
        min_year: Minimum year to include in analysis
    
    Returns:
        DataFrame with onset counts before and after each eruption for each window
    """
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
            
            # Skip if before period extends beyond minimum year
            if min_year and before_start < min_year:
                continue
            
            # Count crisis onsets in each period
            before_onsets = count_crisis_onsets_in_period(crisis_df, before_start, before_end)
            after_onsets = count_crisis_onsets_in_period(crisis_df, after_start, after_end)
            
            results.append({
                'eruption_year': year,
                'eruption_name': name,
                'vei': vei,
                'window': window,
                'before_onset_count': before_onsets,
                'after_onset_count': after_onsets,
                'onset_difference': after_onsets - before_onsets,  # Positive means more after
                'onset_ratio': after_onsets / before_onsets if before_onsets > 0 else np.nan
            })
    
    df = pd.DataFrame(results)
    # Remove rows where we couldn't calculate both periods
    df = df.dropna(subset=['before_onset_count', 'after_onset_count'])
    
    return df


def compute_temporal_onset_distribution(crisis_df, volcano_df, years_before=50, years_after=50):
    """
    Compute the temporal distribution of crisis onsets relative to eruptions.
    
    Creates a year-by-year count of crisis onsets relative to each eruption,
    where year 0 is the eruption year.
    
    Args:
        crisis_df: DataFrame with crisis data
        volcano_df: DataFrame with volcano eruption data
        years_before: Number of years before eruption to analyze
        years_after: Number of years after eruption to analyze
    
    Returns:
        DataFrame with columns for relative_year and onset_count
    """
    all_distributions = []
    
    for _, eruption in volcano_df.iterrows():
        eruption_year = eruption['Year']
        
        # Create array for this eruption's distribution
        distribution = []
        
        # Count onsets for each year relative to eruption
        for relative_year in range(-years_before, years_after + 1):
            actual_year = eruption_year + relative_year
            
            # Count onsets in this specific year
            year_onsets = crisis_df[
                crisis_df['Polity.Date.From'] == actual_year
            ]['Crisis.Case'].nunique()
            
            distribution.append({
                'eruption_name': eruption['Name'],
                'eruption_year': eruption_year,
                'vei': eruption['VEI'],
                'relative_year': relative_year,
                'actual_year': actual_year,
                'onset_count': year_onsets
            })
        
        all_distributions.extend(distribution)
    
    return pd.DataFrame(all_distributions)


def test_onset_significance(onset_df):
    """
    Perform statistical tests on crisis onset differences.
    
    Tests whether there's a significant difference between before and after periods
    using paired t-tests and calculates effect sizes.
    
    Args:
        onset_df: DataFrame from compute_onset_before_after_data
    
    Returns:
        DataFrame with statistical test results for each time window
    """
    results = []
    
    for window in onset_df['window'].unique():
        window_data = onset_df[onset_df['window'] == window].copy()
        
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
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        if len(differences) > 0:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(differences, zero_method='wilcox')
        else:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        
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
            'wilcoxon_stat': wilcoxon_stat,
            'wilcoxon_p': wilcoxon_p,
            'significant_05': p_value < 0.05,
            'significant_01': p_value < 0.01
        })
    
    return pd.DataFrame(results)


def analyze_temporal_trends(crisis_matrix):
    """
    Analyze how crisis frequency changes over time to identify recording bias.
    
    This helps determine if apparent increases after eruptions are due to
    better historical documentation in later periods.
    
    Args:
        crisis_matrix: Crisis presence matrix from create_crisis_matrix
    
    Returns:
        Tuple of (yearly_counts Series, century_means Series)
    """
    # Count active crises per year
    yearly_counts = (crisis_matrix > 0).sum(axis=1)
    
    # Create century bins
    century_starts = range(-2500, 2001, 100)
    century_labels = [f"{i:+d} to {i+99:+d}" for i in century_starts[:-1]]
    
    centuries = pd.cut(
        crisis_matrix.index, 
        bins=century_starts,
        labels=century_labels,
        right=False
    )
    
    # Calculate mean crises per century
    century_means = yearly_counts.groupby(centuries).mean()
    
    # Calculate linear trend
    years = crisis_matrix.index.values
    counts = yearly_counts.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)
    
    trend_info = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'crises_per_century_increase': slope * 100
    }
    
    return yearly_counts, century_means, trend_info


def load_all_data(start_year=None, volcano_list=None, continent_filter=None):
    """Load all data with optional filtering by continent, including onset analysis."""
    # Load crisis data with continent filter
    crisis_df = load_crisis_data(continent_filter=continent_filter)
    crisis_matrix = create_crisis_matrix(crisis_df)
    
    # Load volcano data with continent filter
    volcano_df = load_volcano_data(min_year=start_year, continent_filter=continent_filter)
    
    # Filter volcanoes by specific list if provided
    if volcano_list:
        volcano_df = volcano_df[volcano_df['Name'].isin(volcano_list)]
    
    # Compute original before/after data (active crises)
    before_after_df = compute_before_after_data(crisis_matrix, volcano_df, min_year=start_year)
    
    # NEW: Compute onset-based before/after data
    onset_before_after_df = compute_onset_before_after_data(crisis_df, volcano_df, min_year=start_year)
    
    # NEW: Compute temporal distribution of onsets
    onset_distribution_df = compute_temporal_onset_distribution(crisis_df, volcano_df)
    
    # NEW: Statistical testing of onset differences
    onset_stats_df = test_onset_significance(onset_before_after_df)
    
    # Print analysis info
    if len(volcano_df) > 0:
        n = len(before_after_df['eruption_name'].unique()) if len(before_after_df) > 0 else 0
        if n > 0:
            years = f"{before_after_df['eruption_year'].min()}-{before_after_df['eruption_year'].max()}"
            region = f" in {continent_filter}" if continent_filter else " globally"
            print(f"  Analyzing {n} eruptions ({years} CE){region}")
            
            # Print onset analysis summary
            if len(onset_stats_df) > 0:
                sig_windows = onset_stats_df[onset_stats_df['significant_05']]
                if len(sig_windows) > 0:
                    print(f"  Significant onset differences found in {len(sig_windows)} time windows")
        else:
            region = f" for {continent_filter}" if continent_filter else ""
            print(f"  No eruptions with sufficient data{region}")
    
    return crisis_matrix, volcano_df, before_after_df, onset_before_after_df, onset_distribution_df, onset_stats_df