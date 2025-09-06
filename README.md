# Volcano Crisis Onset Analysis

A streamlined analysis tool that examines whether major volcanic eruptions (VEI 6-7) trigger new global crises by tracking crisis ONSETS rather than ongoing crises.

## Why Onset Analysis?

Traditional crisis counting has methodological issues:
- **Double-counting**: Long crises spanning eruption dates get counted in both "before" and "after" periods
- **Duration bias**: Mixes crisis initiation with persistence
- **Unclear causality**: Can't distinguish if eruptions trigger NEW crises

Onset analysis solves these by counting only when crises BEGIN, providing a cleaner causal signal.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scipy

# Run analysis
python volcano_crisis_onset.py
```

## Configuration

Edit these variables at the bottom of `volcano_crisis_onset.py`:

```python
START_YEAR = None    # None for all data, or e.g., 1000 for medieval onwards
VOLCANO_LIST = None  # None for all volcanoes, or ['Mount Tambora', 'Krakatoa'] etc.
```

## Output Structure

```
results/
├── World/               # Global analysis
│   ├── onset_comparison.png       # Main before/after comparison
│   ├── onset_heatmap.png         # Eruption-by-eruption differences  
│   ├── statistical_summary.png    # P-values and effect sizes
│   ├── crisis_timeline.png       # Timeline visualization
│   ├── vei6_onset_comparison.png # VEI 6 only
│   └── vei7_onset_comparison.png # VEI 7 only
├── Asia/               # Continental analyses...
├── Europe/
├── North America/
├── South America/
├── Oceania/
└── Antarctica/
```

## Reading the Results

### Swarm Plots (`onset_comparison.png`)
- **Each dot**: One eruption's crisis onset count
- **Green**: Before eruption period
- **Red**: After eruption period  
- **Lines**: Mean (thick) and median (thin)
- **Percentage**: Change from before to after

### Heatmap (`onset_heatmap.png`)
- **Red cells**: MORE crises after eruption
- **Blue cells**: FEWER crises after eruption
- **Numbers**: Difference (after - before)
- **Asterisks**: Statistical significance (* p<0.05, ** p<0.01)

### Statistical Summary (`statistical_summary.png`)
- **Top plot**: P-values (below orange line = significant)
- **Bottom plot**: Effect sizes (0.2=small, 0.5=medium, 0.8=large)

## Method

For each volcanic eruption:
1. Define time windows (10-150 years)
2. Count crisis onsets BEFORE eruption (e.g., year -50 to -1)
3. Count crisis onsets AFTER eruption (e.g., year +1 to +50)
4. Compare using paired t-test (each eruption is its own control)
5. Calculate effect size (Cohen's d)

## Data Requirements

```
data/
├── CrisisConsequencesData_NavigatingPolycrisis_2023.03.csv
└── volcano_list.csv
```

### Volcano data format (CSV):
- `VEI`: Volcanic Explosivity Index (6 or 7)
- `Name`: Volcano name
- `Location`: Geographic location
- `Year`: Eruption year
- `Continent`: Continent name

### Crisis data format (CSV):
- `Crisis.Case`: Unique crisis identifier
- `Polity.Date.From`: Crisis start year
- `Polity.Date.To`: Crisis end year
- `Continent`: (optional) Continental location

## Key Parameters

- **Time windows**: 10 to 150 years in 10-year increments
- **VEI levels**: 6 (large) and 7 (super-colossal)  
- **Statistical test**: Paired t-test
- **Effect size**: Cohen's d for paired samples

## Interpretation Guide

**Positive onset difference**: More crises begin after eruptions
**Negative onset difference**: Fewer crises begin after eruptions

**P-value thresholds**:
- p < 0.05: Statistically significant
- p < 0.01: Highly significant

**Effect size interpretation**:
- |d| < 0.2: Negligible
- |d| = 0.2-0.5: Small
- |d| = 0.5-0.8: Medium
- |d| > 0.8: Large

## Known Limitations

1. **Historical recording bias**: Better documentation in recent centuries
2. **Geographic bias**: Better records for Europe/Asia
3. **Crisis definition**: Varies across time and culture
4. **Sample size**: Limited number of VEI 6-7 eruptions

## Recommendations

- Use `START_YEAR = 1000` or later for better data quality
- Focus on statistical significance across multiple time windows
- Consider regional differences in data quality
- Remember: correlation ≠ causation

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations  
- matplotlib: Plotting
- seaborn: Statistical visualizations
- scipy: Statistical tests