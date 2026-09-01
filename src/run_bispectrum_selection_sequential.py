#!/usr/bin/env python3
"""Run selection-aware bispectrum with sequential coadd acquisition.

This replaces only the executor used inside acquire_candidate, matching the
accepted multiband36 acquisition pattern. Science/statistical logic is unchanged.
"""
class _Done:
    def __init__(self,value): self._value=value
    def result(self): return self._value
class _SequentialExecutor:
    def __init__(self,*args,**kwargs): pass
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def submit(self,fn,*args,**kwargs): return _Done(fn(*args,**kwargs))

import bispectrum_selection_residual_validate as experiment
experiment.ThreadPoolExecutor=_SequentialExecutor
if __name__=='__main__': experiment.main()
