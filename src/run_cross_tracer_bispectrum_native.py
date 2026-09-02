#!/usr/bin/env python3
"""Run cross-tracer bispectrum with the accepted endian-safe Tractor reader."""
import tracer_split_native_patch as native
native.install()
import cross_tracer_bispectrum_validate as experiment
if __name__=='__main__':
    experiment.main()
