# pXRF-calibrate
Calculate session-by-session calibration and error propagation factors for pXRF data using the methods outlined in `pXRF_calibration.pdf`. The methodology is designed assuming that you have a number of known reference materials (contained here in `USGS_pXRF_standards.csv`) and that you measure them at the beginning, middle, and end of a day's session (session-by-session pXRF measurements of those standards are contained here in `all_reference_tests.csv`). Included python routines and their purpose:
- `screen_standards.py` applies the various automatic and manual rules to exclude certain standards.
- `calibrate.py` computes the session-by-session calibration factors.
- `eiv_crossplots.py` plots of individual element-session outcomes.
- `plot_per_session_slopes.py` plots the session-by-session calibration factors.

Approach developed by Mark Hoggard.
