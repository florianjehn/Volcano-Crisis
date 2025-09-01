"""
Data preparation module for Volcano Crisis analysis.
This module reads crisis and volcano data, transforms it into analyzable format,
and performs statistical analysis on the relationship between volcanic eruptions and crises.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def read_crisis_data(filepath: str = '../data/CrisisConsequencesData_NavigatingPolycrisis_2023.03.csv') -> pd.DataFrame:
    """
    Read the crisis data from CSV file.
    
    Args:
        filepath: Path to the crisis data CSV file
        
    Returns:
        DataFrame with crisis data
    """
    print(f"Reading crisis data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Ensure date columns are numeric
    df['Polity.Date.From'] = pd.to_numeric(df['Polity.Date.From'], errors='coerce')
    df['Polity.Date.To'] = pd.to_numeric(df['Polity.Date.To'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['Polity.Date.From', 'Polity.Date.To'])
    
    # Convert to integers
    df['Polity.Date.From'] = df['Polity.Date.From'].astype(int)
    df['Polity.Date.To'] = df['Polity.Date.To'].astype(int)
    
    print(f"Loaded {len(df)} crisis records")
    print(f"Date range: {df['Polity.Date.From'].min()} to {df['Polity.Date.To'].max()}")
    
    return df


def transform_to_yearly_matrix(crisis_df: pd.DataFrame, 
                              start_year: int = -2500, 
                              end_year: int = 1920) -> pd.DataFrame:
    """
    Transform crisis data into a matrix where rows are years and columns are crisis cases.
    Values are binary (1 if crisis was active that year, 0 otherwise).
    
    Args:
        crisis_df: DataFrame with crisis data
        start_year: Starting year for the matrix
        end_year: Ending year for the matrix
        
    Returns:
        DataFrame with years as index and crisis cases as columns
    """
    print(f"Transforming data to yearly matrix from {start_year} to {end_year}...")
    
    # Create year range
    years = range(start_year, end_year + 1)
    
    # Get unique crisis cases
    crisis_cases = crisis_df['crisis.case'].unique()
    
    # Initialize matrix with zeros
    matrix = pd.DataFrame(0, index=years, columns=crisis_cases)
    matrix.index.name = 'year'
    
    # Fill in the matrix
    for _, row in crisis_df.iterrows():
        crisis_name = row['crisis.case']
        from_year = row['Polity.Date.From']
        to_year = row['Polity.Date.To']
        
        # Ensure years are within our range
        from_year = max(from_year, start_year)
        to_year = min(to_year, end_year)
        
        # Mark years when this crisis was active
        if from_year <= to_year:
            matrix.loc[from_year:to_year, crisis_name] = 1
    
    print(f"Created matrix with {len(matrix)} years and {len(crisis_cases)} unique crises")
    print(f"Total crisis-years: {matrix.sum().sum()}")
    
    return matrix


def read_volcano_data(filepath: str = '../data/volcano_list.csv', 
                     max_year: int = 1920) -> pd.DataFrame:
    """
    Read volcano eruption data and filter by year.
    
    Args:
        filepath: Path to the volcano data CSV file
        max_year: Maximum year to include (exclude modern eruptions)
        
    Returns:
        DataFrame with filtered volcano data
    """
    print(f"Reading volcano data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Filter by year
    df = df[df['year'] <= max_year].copy()
    
    print(f"Loaded {len(df)} volcanic eruptions before {max_year}")
    print(f"VEI 6 eruptions: {len(df[df['VEI'] == 6])}")
    print(f"VEI 7 eruptions: {len(df[df['VEI'] == 7])}")
    
    return df


def analyze_crisis_after_eruption(crisis_matrix: pd.DataFrame,
                                 volcano_df: pd.DataFrame,
                                 years_after: int = 10,
                                 vei_level: int = 6,
                                 include_pre_eruption: bool = False) -> dict:
    """
    Analyze the number of crises occurring after volcanic eruptions of a specific VEI level.
    
    Args:
        crisis_matrix: Binary matrix of crisis occurrences by year
        volcano_df: DataFrame with volcano eruption data
        years_after: Number of years after eruption to analyze
        vei_level: VEI level to analyze (6 or 7)
        include_pre_eruption: Whether to also analyze years before eruption
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\nAnalyzing crises after VEI {vei_level} eruptions (window: {years_after} years)...")
    
    # Filter volcanoes by VEI level
    eruptions = volcano_df[volcano_df['VEI'] == vei_level]['year'].values
    
    results = {
        'vei_level': vei_level,
        'years_after': years_after,
        'eruption_years': eruptions,
        'post_eruption_crises': [],
        'post_eruption_crisis_counts': []
    }
    
    if include_pre_eruption:
        results['pre_eruption_crises'] = []
        results['pre_eruption_crisis_counts'] = []
    
    # Analyze each eruption
    for eruption_year in eruptions:
        # Post-eruption analysis
        post_start = eruption_year + 1
        post_end = min(eruption_year + years_after, crisis_matrix.index.max())
        
        if post_start <= post_end and post_start >= crisis_matrix.index.min():
            # Count active crises in the post-eruption period
            post_period = crisis_matrix.loc[post_start:post_end]
            # Count unique crises that were active at any point during this period
            active_crises = post_period.sum(axis=0) > 0
            crisis_count = active_crises.sum()
            
            results['post_eruption_crises'].append(active_crises)
            results['post_eruption_crisis_counts'].append(crisis_count)
        
        # Pre-eruption analysis (if requested)
        if include_pre_eruption:
            pre_start = max(eruption_year - years_after, crisis_matrix.index.min())
            pre_end = eruption_year - 1
            
            if pre_start <= pre_end:
                pre_period = crisis_matrix.loc[pre_start:pre_end]
                active_crises = pre_period.sum(axis=0) > 0
                crisis_count = active_crises.sum()
                
                results['pre_eruption_crises'].append(active_crises)
                results['pre_eruption_crisis_counts'].append(crisis_count)
    
    # Calculate statistics
    if results['post_eruption_crisis_counts']:
        results['post_mean'] = np.mean(results['post_eruption_crisis_counts'])
        results['post_std'] = np.std(results['post_eruption_crisis_counts'])
        results['post_median'] = np.median(results['post_eruption_crisis_counts'])
        print(f"  Post-eruption: mean={results['post_mean']:.2f}, "
              f"std={results['post_std']:.2f}, median={results['post_median']:.1f}")
    
    if include_pre_eruption and results.get('pre_eruption_crisis_counts'):
        results['pre_mean'] = np.mean(results['pre_eruption_crisis_counts'])
        results['pre_std'] = np.std(results['pre_eruption_crisis_counts'])
        results['pre_median'] = np.median(results['pre_eruption_crisis_counts'])
        print(f"  Pre-eruption: mean={results['pre_mean']:.2f}, "
              f"std={results['pre_std']:.2f}, median={results['pre_median']:.1f}")
    
    return results


def random_sampling_analysis(crisis_matrix: pd.DataFrame,
                            years_window: int = 10,
                            n_samples: int = 10000,
                            year_range: Optional[Tuple[int, int]] = None,
                            seed: int = 42) -> dict:
    """
    Perform random sampling to establish baseline crisis frequency.
    
    Args:
        crisis_matrix: Binary matrix of crisis occurrences by year
        years_window: Number of years to count crises after random point
        n_samples: Number of random samples to take
        year_range: Optional tuple of (start_year, end_year) to limit sampling range
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with sampling results
    """
    print(f"\nPerforming random sampling analysis ({n_samples} samples, {years_window}-year window)...")
    
    np.random.seed(seed)
    
    # Determine sampling range
    if year_range:
        sample_min = max(year_range[0], crisis_matrix.index.min())
        sample_max = min(year_range[1], crisis_matrix.index.max() - years_window)
    else:
        sample_min = crisis_matrix.index.min()
        sample_max = crisis_matrix.index.max() - years_window
    
    print(f"  Sampling from year {sample_min} to {sample_max}")
    
    crisis_counts = []
    
    for _ in range(n_samples):
        # Random starting year
        start_year = np.random.randint(sample_min, sample_max + 1)
        end_year = min(start_year + years_window, crisis_matrix.index.max())
        
        # Count active crises in this period
        period = crisis_matrix.loc[start_year:end_year]
        active_crises = period.sum(axis=0) > 0
        crisis_count = active_crises.sum()
        
        crisis_counts.append(crisis_count)
    
    results = {
        'years_window': years_window,
        'n_samples': n_samples,
        'crisis_counts': crisis_counts,
        'mean': np.mean(crisis_counts),
        'std': np.std(crisis_counts),
        'median': np.median(crisis_counts),
        'percentile_5': np.percentile(crisis_counts, 5),
        'percentile_95': np.percentile(crisis_counts, 95)
    }
    
    print(f"  Random sampling: mean={results['mean']:.2f}, std={results['std']:.2f}, "
          f"median={results['median']:.1f}")
    print(f"  90% CI: [{results['percentile_5']:.1f}, {results['percentile_95']:.1f}]")
    
    return results


def perform_statistical_tests(volcano_results: dict, random_results: dict) -> dict:
    """
    Perform statistical tests comparing volcanic eruption periods to random sampling.
    
    Args:
        volcano_results: Results from volcanic eruption analysis
        random_results: Results from random sampling
        
    Returns:
        Dictionary with statistical test results
    """
    from scipy import stats
    
    print(f"\nPerforming statistical tests for VEI {volcano_results['vei_level']}...")
    
    results = {}
    
    # Prepare data
    volcano_counts = volcano_results['post_eruption_crisis_counts']
    random_counts = random_results['crisis_counts']
    
    if len(volcano_counts) > 0:
        # T-test
        t_stat, t_pval = stats.ttest_ind(volcano_counts, random_counts)
        results['t_test'] = {'statistic': t_stat, 'p_value': t_pval}
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(volcano_counts, random_counts, alternative='two-sided')
        results['mann_whitney'] = {'statistic': u_stat, 'p_value': u_pval}
        
        # Permutation test
        def statistic(x, y):
            return np.mean(x) - np.mean(y)
        
        perm_result = stats.permutation_test(
            (volcano_counts, random_counts),
            statistic,
            permutation_type='independent',
            n_resamples=10000,
            random_state=42
        )
        results['permutation'] = {
            'statistic': perm_result.statistic,
            'p_value': perm_result.pvalue
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(volcano_counts) - 1) * np.var(volcano_counts) + 
                              (len(random_counts) - 1) * np.var(random_counts)) / 
                             (len(volcano_counts) + len(random_counts) - 2))
        cohen_d = (np.mean(volcano_counts) - np.mean(random_counts)) / pooled_std
        results['cohen_d'] = cohen_d
        
        print(f"  T-test: p={t_pval:.4f}")
        print(f"  Mann-Whitney U: p={u_pval:.4f}")
        print(f"  Permutation test: p={perm_result.pvalue:.4f}")
        print(f"  Cohen's d: {cohen_d:.3f}")
    
    return results


def main():
    """Main execution function."""
    
    # Create results directory if it doesn't exist
    Path('../results').mkdir(parents=True, exist_ok=True)
    
    # Step 1: Read crisis data
    crisis_df = read_crisis_data()
    
    # Step 2: Transform to yearly matrix
    crisis_matrix = transform_to_yearly_matrix(crisis_df)
    
    # Step 3: Save crisis overview
    print("\nSaving crisis overview matrix...")
    crisis_matrix.to_csv('../results/crisis_overview.csv')
    print(f"Saved to ../results/crisis_overview.csv")
    
    # Step 5: Read volcano data
    volcano_df = read_volcano_data()
    
    # Step 7: Analyze crises after volcanic eruptions
    # Test different time windows: 5, 10, 15, 20 years
    time_windows = [5, 10, 15, 20]
    all_results = {}
    
    for window in time_windows:
        print(f"\n{'='*60}")
        print(f"Analyzing {window}-year window")
        print('='*60)
        
        # VEI 6 eruptions
        vei6_results = analyze_crisis_after_eruption(
            crisis_matrix, volcano_df, 
            years_after=window, vei_level=6, 
            include_pre_eruption=True
        )
        
        # VEI 7 eruptions
        vei7_results = analyze_crisis_after_eruption(
            crisis_matrix, volcano_df,
            years_after=window, vei_level=7,
            include_pre_eruption=True
        )
        
        # Random sampling
        random_results = random_sampling_analysis(
            crisis_matrix,
            years_window=window,
            n_samples=10000
        )
        
        # Statistical tests
        vei6_stats = perform_statistical_tests(vei6_results, random_results)
        vei7_stats = perform_statistical_tests(vei7_results, random_results)
        
        # Store results
        all_results[window] = {
            'vei6': vei6_results,
            'vei7': vei7_results,
            'random': random_results,
            'vei6_stats': vei6_stats,
            'vei7_stats': vei7_stats
        }
    
    # Save comprehensive results
    print("\n" + "="*60)
    print("Saving analysis results...")
    
    # Save as pickle for complete data preservation
    import pickle
    with open('../results/volcano_crisis_analysis.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save summary statistics as CSV for easy access
    summary_data = []
    for window in time_windows:
        for analysis_type in ['vei6', 'vei7', 'random']:
            if analysis_type == 'random':
                mean = all_results[window][analysis_type]['mean']
                std = all_results[window][analysis_type]['std']
                median = all_results[window][analysis_type]['median']
                n_samples = all_results[window][analysis_type]['n_samples']
                
                summary_data.append({
                    'window_years': window,
                    'analysis_type': 'random',
                    'mean_crises': mean,
                    'std_crises': std,
                    'median_crises': median,
                    'n_samples': n_samples
                })
            else:
                results = all_results[window][analysis_type]
                if 'post_mean' in results:
                    summary_data.append({
                        'window_years': window,
                        'analysis_type': f"{analysis_type}_post",
                        'mean_crises': results['post_mean'],
                        'std_crises': results['post_std'],
                        'median_crises': results['post_median'],
                        'n_samples': len(results['post_eruption_crisis_counts'])
                    })
                
                if 'pre_mean' in results:
                    summary_data.append({
                        'window_years': window,
                        'analysis_type': f"{analysis_type}_pre",
                        'mean_crises': results['pre_mean'],
                        'std_crises': results['pre_std'],
                        'median_crises': results['pre_median'],
                        'n_samples': len(results['pre_eruption_crisis_counts'])
                    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('../results/analysis_summary.csv', index=False)
    
    print("Analysis complete! Results saved to:")
    print("  - ../results/crisis_overview.csv")
    print("  - ../results/volcano_crisis_analysis.pkl")
    print("  - ../results/analysis_summary.csv")
    print("\nRun plotting.py to visualize the results.")


if __name__ == "__main__":
    main()