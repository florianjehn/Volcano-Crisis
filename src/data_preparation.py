"""
Clean data processing for volcano-crisis analysis.
Handles loading crisis data, volcano data, and computing before/after crisis counts.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_crisis_data(filepath='data/CrisisConsequencesData_NavigatingPolycrisis_2023.03.csv'):
    """Load and clean crisis data."""
    print("Loading crisis data...")
    
    df = pd.read_csv(filepath, encoding='latin1')
    
    # Convert dates to integers, drop invalid rows
    df['Polity.Date.From'] = pd.to_numeric(df['Polity.Date.From'], errors='coerce')
    df['Polity.Date.To'] = pd.to_numeric(df['Polity.Date.To'], errors='coerce')
    df = df.dropna(subset=['Polity.Date.From', 'Polity.Date.To'])
    df['Polity.Date.From'] = df['Polity.Date.From'].astype(int)
    df['Polity.Date.To'] = df['Polity.Date.To'].astype(int)
    
    print(f"  Loaded {len(df)} crisis records")
    return df


def create_crisis_matrix(crisis_df, start_year=-2500, end_year=1920):
    """Convert crisis data into binary matrix (years × crises)."""
    print("Creating crisis matrix...")
    
    years = range(start_year, end_year + 1)
    crises = crisis_df['Crisis.Case'].unique()
    
    # Initialize binary matrix
    matrix = pd.DataFrame(0, index=years, columns=crises)
    
    # Fill matrix: 1 where crisis was active
    for _, row in crisis_df.iterrows():
        from_year = max(row['Polity.Date.From'], start_year)
        to_year = min(row['Polity.Date.To'], end_year)
        
        if from_year <= to_year:
            matrix.loc[from_year:to_year, row['Crisis.Case']] = 1
    
    print(f"  Matrix: {len(matrix)} years × {len(crises)} crises")
    return matrix


def load_volcano_data(filepath='data/volcano_list.csv', max_year=1920):
    """Load volcano eruption data."""
    print("Loading volcano data...")
    
    df = pd.read_csv(filepath)
    df = df[df['Year'] <= max_year].copy()
    
    # Only keep VEI 6 and 7 eruptions
    df = df[df['VEI'].isin([6, 7])].copy()
    
    print(f"  Loaded {len(df)} major eruptions (VEI 6-7)")
    return df


def count_crises_in_period(crisis_matrix, start_year, end_year):
    """Count number of unique crises active during a time period."""
    if start_year < crisis_matrix.index.min() or end_year > crisis_matrix.index.max():
        return np.nan
    
    # Get period data and count crises that were active at any point
    period = crisis_matrix.loc[start_year:end_year]
    active_crises = (period.sum(axis=0) > 0).sum()
    
    return active_crises


def compute_before_after_data(crisis_matrix, volcano_df, time_windows=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """
    Compute crisis counts before and after each eruption for all time windows.
    
    Returns DataFrame with columns: eruption_year, eruption_name, vei, window, before_count, after_count
    """
    print("Computing before/after crisis counts...")
    
    results = []
    
    for _, eruption in volcano_df.iterrows():
        eruption_year = eruption['Year']
        eruption_name = eruption['Name']
        eruption_vei = eruption['VEI']
        
        for window in time_windows:
            # Before period: [eruption_year - window, eruption_year - 1]
            before_start = eruption_year - window
            before_end = eruption_year - 1
            before_count = count_crises_in_period(crisis_matrix, before_start, before_end)
            
            # After period: [eruption_year + 1, eruption_year + window]  
            after_start = eruption_year + 1
            after_end = eruption_year + window
            after_count = count_crises_in_period(crisis_matrix, after_start, after_end)
            
            results.append({
                'eruption_year': eruption_year,
                'eruption_name': eruption_name,
                'vei': eruption_vei,
                'window': window,
                'before_count': before_count,
                'after_count': after_count
            })
    
    df = pd.DataFrame(results)
    
    # Remove rows with missing data
    df = df.dropna(subset=['before_count', 'after_count'])
    
    print(f"  Computed data for {len(df)} eruption-window combinations")
    return df


def load_all_data():
    """Load and process all data. Returns crisis_matrix, volcano_df, before_after_df."""
    print("=" * 50)
    print("LOADING AND PROCESSING DATA")
    print("=" * 50)
    
    # Load raw data
    crisis_df = load_crisis_data()
    crisis_matrix = create_crisis_matrix(crisis_df)
    volcano_df = load_volcano_data()
    
    # Compute before/after analysis
    before_after_df = compute_before_after_data(crisis_matrix, volcano_df)
    
    print("\nData loading complete!")
    print("=" * 50)
    
    return crisis_matrix, volcano_df, before_after_df
