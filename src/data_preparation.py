"""Data processing for volcano-crisis analysis."""

import pandas as pd
import numpy as np


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


def load_all_data(start_year=None, volcano_list=None, continent_filter=None):
    """Load all data with optional filtering by continent."""
    # Load crisis data with continent filter
    crisis_df = load_crisis_data(continent_filter=continent_filter)
    crisis_matrix = create_crisis_matrix(crisis_df)
    
    # Load volcano data with continent filter
    volcano_df = load_volcano_data(min_year=start_year, continent_filter=continent_filter)
    
    # Filter volcanoes by specific list if provided
    if volcano_list:
        volcano_df = volcano_df[volcano_df['Name'].isin(volcano_list)]
    
    # Compute before/after data
    before_after_df = compute_before_after_data(crisis_matrix, volcano_df, min_year=start_year)
    
    # Print analysis info
    if len(volcano_df) > 0:
        n = len(before_after_df['eruption_name'].unique()) if len(before_after_df) > 0 else 0
        if n > 0:
            years = f"{before_after_df['eruption_year'].min()}-{before_after_df['eruption_year'].max()}"
            region = f" in {continent_filter}" if continent_filter else " globally"
            print(f"  Analyzing {n} eruptions ({years} CE){region}")
        else:
            region = f" for {continent_filter}" if continent_filter else ""
            print(f"  No eruptions with sufficient data{region}")
    
    return crisis_matrix, volcano_df, before_after_df