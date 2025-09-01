"""
Data preparation module for Volcano Crisis analysis.
Reads crisis and volcano data, transforms it, and performs statistical analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def read_crisis_data(filepath='data/CrisisConsequencesData_NavigatingPolycrisis_2023.03.csv'):
    """Read and process crisis data."""
    print(f"Reading crisis data...")
    df = pd.read_csv(filepath, encoding='latin1')
    
    # Convert date columns to numeric
    df['Polity.Date.From'] = pd.to_numeric(df['Polity.Date.From'], errors='coerce')
    df['Polity.Date.To'] = pd.to_numeric(df['Polity.Date.To'], errors='coerce')
    df = df.dropna(subset=['Polity.Date.From', 'Polity.Date.To'])
    df['Polity.Date.From'] = df['Polity.Date.From'].astype(int)
    df['Polity.Date.To'] = df['Polity.Date.To'].astype(int)
    
    print(f"Loaded {len(df)} crisis records")
    print(f"Date range: {df['Polity.Date.From'].min()} to {df['Polity.Date.To'].max()}")
    
    return df


def transform_to_yearly_matrix(crisis_df, start_year=-2500, end_year=1920):
    """Transform crisis data into binary matrix (years x crises)."""
    print(f"Transforming to yearly matrix...")
    
    years = range(start_year, end_year + 1)
    crisis_cases = crisis_df['Crisis.Case'].unique()
    
    # Initialize matrix
    matrix = pd.DataFrame(0, index=years, columns=crisis_cases)
    matrix.index.name = 'Year'
    
    # Fill matrix
    for _, row in crisis_df.iterrows():
        crisis_name = row['Crisis.Case']
        from_year = max(row['Polity.Date.From'], start_year)
        to_year = min(row['Polity.Date.To'], end_year)
        
        if from_year <= to_year:
            matrix.loc[from_year:to_year, crisis_name] = 1
    
    print(f"Created matrix: {len(matrix)} years x {len(crisis_cases)} crises")
    return matrix


def read_volcano_data(filepath='data/volcano_list.csv', max_year=1920):
    """Read volcano data and filter by year."""
    print(f"Reading volcano data...")
    df = pd.read_csv(filepath)
    df = df[df['Year'] <= max_year].copy()
    
    print(f"Loaded {len(df)} eruptions before {max_year}")
    print(f"  VEI 6: {len(df[df['VEI'] == 6])} eruptions")
    print(f"  VEI 7: {len(df[df['VEI'] == 7])} eruptions")
    
    return df


def analyze_crisis_after_eruptions(crisis_matrix, volcano_df, years_after=10):
    """Count crises after all volcanic eruptions (VEI 6 and 7 combined)."""
    
    # Get all VEI 6 and 7 eruption years
    eruption_years = volcano_df[volcano_df['VEI'].isin([6, 7])]['Year'].values
    
    crisis_counts = []
    valid_eruptions = []
    
    for eruption_year in eruption_years:
        post_start = eruption_year
        post_end = min(eruption_year + years_after, crisis_matrix.index.max())
        
        if post_start >= crisis_matrix.index.min() and post_start <= crisis_matrix.index.max():
            # Count unique crises active during this period
            period = crisis_matrix.loc[post_start:post_end]
            active_crises = (period.sum(axis=0) > 0).sum()
            crisis_counts.append(active_crises)
            valid_eruptions.append(eruption_year)
    
    return {
        'years_after': years_after,
        'eruption_years': valid_eruptions,
        'crisis_counts': crisis_counts,
        'mean': np.mean(crisis_counts) if crisis_counts else 0,
        'std': np.std(crisis_counts) if crisis_counts else 0,
        'median': np.median(crisis_counts) if crisis_counts else 0
    }


def random_sampling_analysis(crisis_matrix, years_window=10, n_samples=10000, seed=42):
    """Random sampling to establish baseline crisis frequency."""
    
    np.random.seed(seed)
    
    # Sampling range
    sample_min = crisis_matrix.index.min()
    sample_max = crisis_matrix.index.max() - years_window
    
    crisis_counts = []
    
    for _ in range(n_samples):
        start_year = np.random.randint(sample_min, sample_max + 1)
        end_year = min(start_year + years_window, crisis_matrix.index.max())
        
        # Count active crises
        period = crisis_matrix.loc[start_year:end_year]
        active_crises = (period.sum(axis=0) > 0).sum()
        crisis_counts.append(active_crises)
    
    return {
        'years_window': years_window,
        'crisis_counts': crisis_counts,
        'mean': np.mean(crisis_counts),
        'std': np.std(crisis_counts),
        'median': np.median(crisis_counts),
        'percentile_5': np.percentile(crisis_counts, 5),
        'percentile_95': np.percentile(crisis_counts, 95)
    }


def perform_statistical_tests(volcano_results, random_results):
    """Statistical tests comparing volcanic periods to random sampling."""
    from scipy import stats
    
    volcano_counts = volcano_results['crisis_counts']
    random_counts = random_results['crisis_counts']
    
    results = {}
    
    if len(volcano_counts) > 0:
        # T-test
        t_stat, t_pval = stats.ttest_ind(volcano_counts, random_counts)
        results['t_test_pvalue'] = t_pval
        
        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(volcano_counts, random_counts, alternative='two-sided')
        results['mann_whitney_pvalue'] = u_pval
        
        # Permutation test
        perm_result = stats.permutation_test(
            (volcano_counts, random_counts),
            lambda x, y: np.mean(x) - np.mean(y),
            permutation_type='independent',
            n_resamples=10000,
            random_state=42
        )
        results['permutation_pvalue'] = perm_result.pvalue
        
        # Cohen's d
        pooled_std = np.sqrt(((len(volcano_counts) - 1) * np.var(volcano_counts) + 
                              (len(random_counts) - 1) * np.var(random_counts)) / 
                             (len(volcano_counts) + len(random_counts) - 2))
        if pooled_std > 0:
            results['cohen_d'] = (np.mean(volcano_counts) - np.mean(random_counts)) / pooled_std
        else:
            results['cohen_d'] = 0
    
    return results


def main():
    """Main execution function."""
    
    # Create results directory
    Path('results').mkdir(parents=True, exist_ok=True)
    
    # Read and process data
    crisis_df = read_crisis_data()
    crisis_matrix = transform_to_yearly_matrix(crisis_df)
    
    # Save crisis overview
    print("\nSaving crisis overview matrix...")
    crisis_matrix.to_csv('results/crisis_overview.csv')
    
    # Read volcano data
    volcano_df = read_volcano_data()
    
    # Analyze for time windows from 0 to 100 years in 10-year increments
    time_windows = range(0, 110, 10)
    results_data = []
    
    print("\nAnalyzing crisis patterns after volcanic eruptions...")
    print("="*60)
    
    for window in time_windows:
        print(f"\nAnalyzing {window}-year window...")
        
        # Analyze volcanic eruptions
        volcano_results = analyze_crisis_after_eruptions(crisis_matrix, volcano_df, window)
        
        # Random sampling
        random_results = random_sampling_analysis(crisis_matrix, window, n_samples=10000)
        
        # Statistical tests
        stats_results = perform_statistical_tests(volcano_results, random_results)
        
        # Compile results
        results_data.append({
            'window_years': window,
            'volcano_mean': volcano_results['mean'],
            'volcano_std': volcano_results['std'],
            'volcano_median': volcano_results['median'],
            'volcano_n': len(volcano_results['crisis_counts']),
            'random_mean': random_results['mean'],
            'random_std': random_results['std'],
            'random_median': random_results['median'],
            'random_percentile_5': random_results['percentile_5'],
            'random_percentile_95': random_results['percentile_95'],
            'cohen_d': stats_results.get('cohen_d', None),
            't_test_pvalue': stats_results.get('t_test_pvalue', None),
            'mann_whitney_pvalue': stats_results.get('mann_whitney_pvalue', None),
            'permutation_pvalue': stats_results.get('permutation_pvalue', None)
        })
        
        print(f"  Volcanic eruptions: mean={volcano_results['mean']:.2f}, std={volcano_results['std']:.2f}")
        print(f"  Random baseline: mean={random_results['mean']:.2f}, std={random_results['std']:.2f}")
        print(f"  Cohen's d: {stats_results.get('cohen_d', 0):.3f}")
        print(f"  P-value (permutation): {stats_results.get('permutation_pvalue', 1):.4f}")
    
    # Save all results
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('results/volcano_crisis_analysis.csv', index=False)
    
    # Save detailed counts for visualization
    detailed_results = []
    for window in time_windows:
        volcano_results = analyze_crisis_after_eruptions(crisis_matrix, volcano_df, window)
        random_results = random_sampling_analysis(crisis_matrix, window, n_samples=1000)  # Fewer samples for storage
        
        # Save volcano counts
        for count in volcano_results['crisis_counts']:
            detailed_results.append({
                'window': window,
                'type': 'volcano',
                'crisis_count': count
            })
        
        # Save subset of random counts
        for count in random_results['crisis_counts']:
            detailed_results.append({
                'window': window,
                'type': 'random',
                'crisis_count': count
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('results/detailed_crisis_counts.csv', index=False)
    
    print("\n" + "="*60)
    print("Analysis complete! Results saved to:")
    print("  - results/crisis_overview.csv")
    print("  - results/volcano_crisis_analysis.csv")
    print("  - results/detailed_crisis_counts.csv")
    print("\nRun plotting.py to generate visualizations.")


if __name__ == "__main__":
    main()