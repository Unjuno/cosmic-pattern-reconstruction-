#!/usr/bin/env python3
"""Install the accepted endian-safe DR11 Tractor reader, then run PR21."""
import tracer_split_native_patch as native
native.install()
import fixed_count_cross_bispectrum_validate as experiment

if __name__ == '__main__':
    experiment.main()
