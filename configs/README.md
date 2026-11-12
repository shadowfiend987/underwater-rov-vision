# Configuration history

`mission5.json` retains all values from the saved original configuration; text formatting follows repository line endings. `before_lab.json` preserves the earlier comparison. These values are not calibrated for underwater use.

| Parameter | Earlier saved config | Current saved config | Original code defaults (D key) |
|---|---:|---:|---:|
| Yellow S low | 100 | 150 | 100 |
| Blue H low | 100 | 85 | 100 |
| Blue S low | 100 | 40 | 100 |
| Minimum area | 500 | 500 | 500 |
| Aspect x10 | 19 | 19 | 20 |
| Fill percent | 45 | 25 | 45 |

The dated original configuration and current configuration have identical content. A saved file cannot prove which unsaved trackbar values were active during a historical test. Raising yellow S low rejects some low-saturation backgrounds but can also reject washed-out targets. Wider blue thresholds can introduce false positives. No controlled before/after performance result is claimed.

The main program validates integer bounds, band counts and lower/upper ordering. H uses 0–179; S and V use 0–255. Missing or invalid files now stop with an actionable error instead of silently using different thresholds. S changes only the explicitly selected configuration in this independent project.
