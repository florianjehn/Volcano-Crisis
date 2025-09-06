"""Main script for volcano-crisis analysis."""

from data_preparation import load_all_data
from plotting import create_all_plots


def main():
    # Set analysis start year (None = all data, or specific year like 1000, 1500, etc.)
    START_YEAR = 1000
    # Filter for specific volcanoes
    VOLCANO_LIST = None  # Set back to None for full analysis
    
    print("VOLCANO CRISIS ANALYSIS")
    if START_YEAR:
        print(f"Analysis period: {START_YEAR} CE onwards")
    if VOLCANO_LIST:
        print(f"Filtered for volcanoes: {', '.join(VOLCANO_LIST)}")
    
    # First load data to get available continents
    crisis_matrix, volcano_df, _ = load_all_data(START_YEAR, VOLCANO_LIST, continent_filter=None)
    
    # Get unique continents from volcano data
    continents = volcano_df['Continent'].dropna().unique().tolist()
    
    # Process global data (World)
    print("\n=== Processing World (Global) ===")
    crisis_matrix, volcano_df, before_after_df = load_all_data(START_YEAR, VOLCANO_LIST, continent_filter=None)
    create_all_plots(crisis_matrix, volcano_df, before_after_df, output_subfolder='World')
    
    # Process each continent
    for continent in continents:
        print(f"\n=== Processing {continent} ===")
        crisis_matrix_cont, volcano_df_cont, before_after_df_cont = load_all_data(
            START_YEAR, VOLCANO_LIST, continent_filter=continent
        )
        
        # Only create plots if there are eruptions for this continent
        if len(volcano_df_cont) > 0 and len(before_after_df_cont) > 0:
            create_all_plots(crisis_matrix_cont, volcano_df_cont, before_after_df_cont, 
                           output_subfolder=continent)
        else:
            print(f"  No eruptions found for {continent} in the specified period")
    
    print("\nPlots saved to results/ in respective subfolders")


if __name__ == "__main__":
    main()