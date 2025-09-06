# Volcano Crisis Analysis

Analyzes whether major volcanic eruptions (VEI 6-7) affect global crisis frequency, using both active crisis counts and crisis onset tracking.

## Key Features

### Two Analysis Methods

1. **Active Crisis Analysis** (Original): Counts all crises active during time windows
2. **Crisis Onset Analysis** (New): Counts only crises that BEGIN during time windows - provides cleaner signal

### Why Crisis Onset Analysis?

The onset analysis addresses key methodological issues:
- **Avoids double-counting**: Long-duration crises spanning eruption dates aren't counted twice
- **Cleaner causality signal**: Tracks if eruptions trigger NEW crises
- **Less affected by crisis duration**: Focuses on initiation rather than persistence
- **Better for statistical testing**: Each crisis counted only once at its start

## Usage

```bash
pip install -r requirements.txt
python src/main.py
```

## Configuration

Edit `START_YEAR` in `src/main.py`:
- `None` - All available data (-2300 to 1920 CE)
- `1000` - Medieval period onwards (better data quality)
- `1500` - Early modern period onwards (best data quality)

## Output Structure

### Folder Organization
```
results/
├── World/                    # Global analysis
├── Asia/                     # Asia-specific analysis
├── Europe/                   # Europe-specific analysis
├── North America/            # North America-specific analysis
├── South America/            # South America-specific analysis
├── Oceania/                  # Oceania-specific analysis
└── Antarctica/               # Antarctica-specific analysis (if applicable)
```

### Plot Types (per folder)

#### Active Crisis Plots (Original)
- `before_after_comparison.png` - All major eruptions (VEI 6-7)
- `vei6_before_after_comparison.png` - VEI 6 eruptions only
- `vei7_before_after_comparison.png` - VEI 7 eruptions only
- `crisis_timeline.png` - Timeline overview

#### Crisis Onset Plots (New)
- `onset_before_after_comparison.png` - Onset frequency comparison
- `vei6_onset_comparison.png` - VEI 6 onset analysis
- `vei7_onset_comparison.png` - VEI 7 onset analysis
- `onset_temporal_distribution.png` - When crises start relative to eruptions
- `onset_difference_heatmap.png` - Differences for each eruption/window
- `onset_statistical_summary.png` - Statistical significance and effect sizes

## Interpretation Guide

### Reading the Plots

- **Swarm plots**: Each dot = one eruption's count for that time window
- **Red/Orange colors**: "After eruption" periods
- **Blue/Green colors**: "Before eruption" periods
- **Thick lines**: Mean values
- **Dashed lines**: Median values
- **Percentages**: Change from before to after

### Statistical Indicators

- `*` = p < 0.05 (statistically significant)
- `**` = p < 0.01 (highly significant)
- `***` = p < 0.001 (very highly significant)
- **Cohen's d**: Effect size (0.2=small, 0.5=medium, 0.8=large)

### Heatmap Colors

- **Red cells**: MORE crises after eruption
- **Blue cells**: FEWER crises after eruption
- **White cells**: No change
- **Numbers**: Difference in crisis count (after - before)

## Method

For each eruption:
1. Defines time windows (10-100 years)
2. Counts crises/onsets BEFORE eruption (e.g., years -50 to -1)
3. Counts crises/onsets AFTER eruption (e.g., years +1 to +50)
4. Compares using paired statistical tests
5. Analyzes patterns globally and by continent

## Known Issues & Limitations

### Temporal Recording Bias
- Historical documentation improves over time
- Later periods have more recorded crises
- "After" periods are always later than "before" periods
- This bias may create false positive results

### Data Quality Issues
- Pre-1000 CE data is sparse
- Geographic bias (better European records)
- Major crises more likely recorded than minor ones
- Crisis definitions vary across time and region

## Recommendations

1. **Focus on recent data**: Use `START_YEAR = 1000` or later
2. **Compare onset plots to active crisis plots**: Onset analysis is more reliable
3. **Check statistical summary**: Look for consistent significance across windows
4. **Consider regional differences**: Some regions have better data
5. **Be cautious with interpretation**: Correlation ≠ causation

## Statistical Methods

- **Paired t-test**: Tests if mean difference is significant
- **Wilcoxon signed-rank test**: Non-parametric alternative
- **Cohen's d**: Standardized effect size
- **Temporal detrending**: Available in code but not default

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- matplotlib: Plotting
- seaborn: Statistical visualizations
- scipy: Statistical tests

## Data Sources

- Crisis data: CrisisConsequencesData_NavigatingPolycrisis_2023.03.csv
- Volcano data: volcano_list.csv (VEI 6-7 eruptions)

## Future Improvements

- [ ] Implement temporal detrending
- [ ] Add bootstrap confidence intervals
- [ ] Include geographic distance weighting
- [ ] Add permutation testing
- [ ] Analyze crisis types separately
- [ ] Add volcanic winter severity estimates