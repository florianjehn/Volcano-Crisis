"""Main script for volcano-crisis analysis."""

from data_preparation import load_all_data
from plotting import create_all_plots


def main():
    # Set analysis start year (None = all data, or specific year like 1000, 1500, etc.)
    START_YEAR = 1000
    
    print("VOLCANO CRISIS ANALYSIS")
    if START_YEAR:
        print(f"Analysis period: {START_YEAR} CE onwards")
    
    crisis_matrix, volcano_df, before_after_df = load_all_data(START_YEAR)
    create_all_plots(crisis_matrix, volcano_df, before_after_df)
    
    print("\nPlots saved to results/")


if __name__ == "__main__":
    main()