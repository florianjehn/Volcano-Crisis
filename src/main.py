"""
Main script for volcano-crisis analysis.
Simple orchestration of data loading and visualization.
"""

from data_preparation import load_all_data
from plotting import create_all_plots


def main():
    """Run the complete volcano-crisis analysis."""
    print("VOLCANO CRISIS ANALYSIS")
    print("Analyzing whether major volcanic eruptions affect global crisis frequency")
    print()
    
    # Load and process all data
    crisis_matrix, volcano_df, before_after_df = load_all_data()
    
    # Create visualizations  
    create_all_plots(crisis_matrix, volcano_df, before_after_df)
    
    print("\nAnalysis complete!")
    print("Check the results/ directory for:")
    print("  - before_after_comparison.png (main analysis - all eruptions)")
    print("  - vei6_before_after_comparison.png (VEI 6 eruptions only)")
    print("  - vei7_before_after_comparison.png (VEI 7 eruptions only)")  
    print("  - crisis_timeline.png (overview timeline)")


if __name__ == "__main__":
    main()
