"""Compatibility utilities for different dependency versions."""

import pandas as pd


def hourly_resample(df_or_series):
    """Resample to hourly frequency with backward compatibility.

    pandas 2.2+ requires lowercase frequency aliases ('h'),
    while older versions use uppercase ('H').
    """
    try:
        return df_or_series.resample('h')
    except ValueError:
        return df_or_series.resample('H')


def hourly_freq():
    """Return the hourly frequency string compatible with installed pandas.

    pandas 2.2+ requires lowercase frequency aliases ('h'),
    while older versions use uppercase ('H').
    """
    try:
        # Test if lowercase works
        pd.date_range(start='2020-01-01', periods=1, freq='h')
        return 'h'
    except ValueError:
        return 'H'
