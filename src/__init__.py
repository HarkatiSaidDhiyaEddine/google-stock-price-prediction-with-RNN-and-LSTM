"""GOOGL price forecasting with a stacked LSTM.

Package layout
--------------
config          all constants, paths and seeds
data_loader     read and validate the raw CSV
preprocessing   chronological split, scaling, windowing
model           the LSTM architecture
train           fitting loop with early stopping
evaluate        metrics and the naive baseline
plots           figures
run_pipeline    end-to-end entry point
"""
