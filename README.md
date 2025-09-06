# Volcano Crisis Analysis

Analyzes whether major volcanic eruptions (VEI 6-7) affect global crisis frequency.

## Usage

```bash
pip install -r requirements.txt
python src/main.py
```

## Configuration

Edit `START_YEAR` in `src/main.py`:
- `None` - All available data 
- `1000` - Medieval period onwards
- `1500` - Early modern period onwards

## Output

Creates plots in `results/` organized by region:

### Folder Structure
- `results/World/` - Global analysis
- `results/Asia/` - Asia-specific analysis  
- `results/Europe/` - Europe-specific analysis
- `results/North America/` - North America-specific analysis
- `results/South America/` - South America-specific analysis
- `results/Oceania/` - Oceania-specific analysis
- `results/Antarctica/` - Antarctica-specific analysis (if applicable)

### Plot Types (per folder)
- `before_after_comparison.png` - All major eruptions
- `vei6_before_after_comparison.png` - VEI 6 only  
- `vei7_before_after_comparison.png` - VEI 7 only
- `crisis_timeline.png` - Timeline overview

## Method

For each eruption, compares crisis counts in X years before vs X years after (windows 10-100 years), both globally and by continent.
