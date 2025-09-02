# Volcano Crisis Analysis

Clean analysis of whether major volcanic eruptions (VEI 6-7) affect global crisis frequency.

## Approach

**Core Question**: Do we see more crises after major volcanic eruptions than before?

**Method**: 
- For each eruption and time window (10-100 years), compare:
  - Crisis count in years BEFORE eruption: [year-X to year-1] 
  - Crisis count in years AFTER eruption: [year+1 to year+X]
- Visualize all comparisons in a single comprehensive swarm plot

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python main.py
```

This creates four plots in `results/`:
- `before_after_comparison.png` - Main analysis comparing crisis frequency before vs after all major eruptions (VEI 6-7)
- `vei6_before_after_comparison.png` - Same analysis but for VEI 6 eruptions only
- `vei7_before_after_comparison.png` - Same analysis but for VEI 7 eruptions only
- `crisis_timeline.png` - Overview showing all crises and eruption dates

## Files

- `main.py` - Orchestrates the analysis
- `data.py` - Loads crisis data, volcano data, computes before/after counts  
- `plotting.py` - Creates the four essential visualizations
- `data/` - Input data files
- `results/` - Generated plots

## Data

- **Crisis data**: CrisisDB dataset with historical global crises
- **Volcano data**: Major eruptions (VEI 6-7) with dates
- **Analysis period**: Up to 1920 CE (where data overlap)