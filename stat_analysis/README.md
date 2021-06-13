Class and function definitions for the analysis
===============================================

- `GaisserFlux.py` - Atmopheric neutrino flux definition for weighting outside IceTray (copied from IceTray).
- `event_selection.py` - Functions for seleceting the data for plots and analysis.
- `dtypes.py` - data types definitions for data storage.
- Statistical Analysis classes and functions:
    - `likelihood.py` - Max's binned Poisson likelihood definition.
    - `scan_neyman_plane.py` - Max's fintions for creating test statisitic distribtion from expected signal and background distributions.
    - `signal_injector.py` - Max's class for creating test signal and background distributions for the expected distributions. 
    - `limit_calculator.py` - Max's class for calculation confidence interval upper and lower limits given signal, background, and test statistic distributions.
    - `helper_functions.py` - useful functions for statisitcal analysis (Mostly Max Meier's).
    - `stats.py` - useful statisitcs functions (Max Meier's).
    
