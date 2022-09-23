# Description
Code used for optimizing the EEG-preprocessing and phase forecasting for TMS-EEG studies using a grid-search approach.
Given different parameters of EEG processing such as the filter used and its parameters, see if one approach is better suited for various phase forecasting methods.

The code uses a grid-search approach in which all the possible combinations of all the parameters are tested. 
Ideally the test is firstly made on a synthetic dataset and then tested on real EEG data.

Parameters of interest (that may be changed) are:
- filter type (IIR, FIR and their subtypes)
- filter order
- filter bandwidth
- filter stopband/passband ripple (for specific filters only)
- window length of input signal

Additionally, we look at some phase-forecasting methods that either predict the full signal (e.g., with an autoregressive model) or only predict the next point in time where the signal is expected to have the phase of interest. 
