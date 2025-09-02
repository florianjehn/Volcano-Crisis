"""Data processing for volcano-crisis analysis."""

import pandas as pd
import numpy as np


def load_crisis_data(filepath='data/CrisisConsequencesData_NavigatingPolycrisis_2023.03.csv'):
    df = pd.read_csv(filepath, encoding='latin1')
    df['Polity.Date.From'] = pd.to_numeric(df['Polity.Date.From'], errors='coerce')
    df['Polity.Date.To'] = pd.to_numeric(df['Polity.Date.To'], errors='coerce')
    df = df.dropna(subset=['Polity.Date.From', 'Polity.Date.To']).astype({
        'Polity.Date.From': int, 'Polity.Date.To': int
    })
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


def load_volcano_data(filepath='data/volcano_list.csv', max_year=1920, min_year=None):
    df = pd.read_csv(filepath)
    df = df[df['Year'] <= max_year].copy()
    if min_year:
        df = df[df['Year'] >= min_year].copy()
    df = df[df['VEI'].isin([6, 7])].copy()
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


def load_all_data(start_year=None):
    crisis_df = load_crisis_data()
    crisis_matrix = create_crisis_matrix(crisis_df)
    volcano_df = load_volcano_data(min_year=start_year)
    before_after_df = compute_before_after_data(crisis_matrix, volcano_df, min_year=start_year)
    
    if start_year:
        n = len(before_after_df['eruption_name'].unique())
        years = f"{before_after_df['eruption_year'].min()}-{before_after_df['eruption_year'].max()}"
        print(f"Analyzing {n} eruptions ({years} CE)")
    
    return crisis_matrix, volcano_df, before_after_df
