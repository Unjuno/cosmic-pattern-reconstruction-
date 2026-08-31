#!/usr/bin/env python3
import tracer_split_native_patch as patch
patch.install()
import tracer_split_validate_36 as experiment
if __name__=='__main__':
    experiment.main()
