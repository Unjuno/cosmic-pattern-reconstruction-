#!/usr/bin/env python3
"""Install the accepted endian-safe DR11 Tractor reader, then run PR18."""
import tracer_split_native_patch as native
native.install()
import cross_tracer_selection_residual_bispectrum as experiment

if __name__ == '__main__':
    experiment.main()
