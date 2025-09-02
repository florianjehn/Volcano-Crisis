# Volcano Crisis Analysis

Analyzes whether major volcanic eruptions (VEI 6-7) affect global crisis frequency.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Configuration

Edit `START_YEAR` in `main.py`:
- `None` - All available data 
- `1000` - Medieval period onwards
- `1500` - Early modern period onwards

## Output

Creates 4 plots in `results/`:
- `before_after_comparison.png` - All major eruptions
- `vei6_before_after_comparison.png` - VEI 6 only  
- `vei7_before_after_comparison.png` - VEI 7 only
- `crisis_timeline.png` - Timeline overview

## Method

For each eruption, compares crisis counts in X years before vs X years after (windows 10-100 years).