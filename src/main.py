"""Main script for volcano-crisis analysis with onset tracking."""

from data_preparation import load_all_data, analyze_temporal_trends
from plotting import create_all_plots


def main():
    # Set analysis start year (None = all data, or specific year like 1000, 1500, etc.)
    START_YEAR = None
    # Filter for specific volcanoes
    VOLCANO_LIST = None  # Set back to None for full analysis
    
    print("=" * 60)
    print("VOLCANO CRISIS ANALYSIS WITH ONSET TRACKING")
    print("=" * 60)
    
    if START_YEAR:
        print(f"Analysis period: {START_YEAR} CE onwards")
    if VOLCANO_LIST:
        print(f"Filtered for volcanoes: {', '.join(VOLCANO_LIST)}")
    
    # First load data to get available continents and check for temporal trends
    print("\n=== Checking for Temporal Recording Bias ===")
    crisis_matrix, volcano_df, before_after_df, onset_before_after_df, onset_distribution_df, onset_stats_df = load_all_data(
        START_YEAR, VOLCANO_LIST, continent_filter=None
    )
    
    # Analyze temporal trends to understand recording bias
    yearly_counts, century_means, trend_info = analyze_temporal_trends(crisis_matrix)
    
    print(f"\nTemporal Trend Analysis:")
    print(f"  Crisis recording increases by {trend_info['crises_per_century_increase']:.2f} per century")
    print(f"  Linear trend R² = {trend_info['r_squared']:.3f}")
    if trend_info['p_value'] < 0.001:
        print(f"  Trend is highly significant (p < 0.001)")
        print("  WARNING: Strong temporal bias detected - later periods have more recorded crises")
        print("  This may explain why 'after' periods show more crises than 'before' periods")
    
    # Get unique continents from volcano data
    continents = volcano_df['Continent'].dropna().unique().tolist()
    
    # Process global data (World)
    print("\n" + "=" * 60)
    print("=== Processing World (Global) ===")
    print("=" * 60)
    
    # Global analysis already loaded above
    create_all_plots(crisis_matrix, volcano_df, before_after_df, 
                    onset_before_after_df, onset_distribution_df, onset_stats_df,
                    output_subfolder='World')
    
    # Print global onset analysis results
    if len(onset_stats_df) > 0:
        print("\nGlobal Onset Analysis Results:")
        for _, row in onset_stats_df.iterrows():
            window = row['window']
            mean_diff = row['mean_difference']
            p_val = row['p_value']
            cohens_d = row['cohens_d']
            
            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            print(f"  {window:3d} year window: Δ = {mean_diff:+.2f} onsets, "
                  f"p = {p_val:.4f} {sig_marker}, Cohen's d = {cohens_d:+.3f}")
    
    # Process each continent
    for continent in continents:
        print("\n" + "=" * 60)
        print(f"=== Processing {continent} ===")
        print("=" * 60)
        
        # Load data for this continent
        (crisis_matrix_cont, volcano_df_cont, before_after_df_cont, 
         onset_before_after_df_cont, onset_distribution_df_cont, onset_stats_df_cont) = load_all_data(
            START_YEAR, VOLCANO_LIST, continent_filter=continent
        )
        
        # Only create plots if there are eruptions for this continent
        if len(volcano_df_cont) > 0 and len(before_after_df_cont) > 0:
            create_all_plots(crisis_matrix_cont, volcano_df_cont, before_after_df_cont,
                           onset_before_after_df_cont, onset_distribution_df_cont, onset_stats_df_cont,
                           output_subfolder=continent)
            
            # Print continent-specific onset results
            if len(onset_stats_df_cont) > 0:
                sig_results = onset_stats_df_cont[onset_stats_df_cont['significant_05']]
                if len(sig_results) > 0:
                    print(f"\n  {continent} Onset Analysis - Significant Results:")
                    for _, row in sig_results.iterrows():
                        print(f"    {row['window']:3d} year window: "
                              f"Δ = {row['mean_difference']:+.2f}, "
                              f"p = {row['p_value']:.4f}, "
                              f"Cohen's d = {row['cohens_d']:+.3f}")
        else:
            print(f"  No eruptions found for {continent} in the specified period")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nPlots saved to results/ in respective subfolders:")
    print("  - Original analysis: before_after_comparison.png (active crises)")
    print("  - NEW onset analysis: onset_before_after_comparison.png (crisis starts)")
    print("  - NEW temporal distribution: onset_temporal_distribution.png")
    print("  - NEW statistical summary: onset_statistical_summary.png")
    print("  - NEW difference heatmap: onset_difference_heatmap.png")
    print("\nInterpretation Guide:")
    print("  - Positive values = MORE crises after eruptions")
    print("  - Negative values = FEWER crises after eruptions")
    print("  - Red in heatmaps = increased crisis risk post-eruption")
    print("  - Blue in heatmaps = decreased crisis risk post-eruption")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("Crisis ONSET analysis is more reliable than active crisis counts because:")
    print("  1. It avoids double-counting long-duration crises")
    print("  2. It provides a cleaner signal of crisis initiation")
    print("  3. It's less affected by crisis duration variations")
    print("\nCheck the onset plots to see if volcanic eruptions truly trigger new crises.")


if __name__ == "__main__":
    main()