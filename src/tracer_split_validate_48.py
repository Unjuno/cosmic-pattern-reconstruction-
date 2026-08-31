#!/usr/bin/env python3
"""48-brick entry point for endian-safe REAL_DR11 tracer-split validation."""
import tracer_split_validate_fixed as fixed

fixed.base.N_FIELDS = 48

if __name__ == '__main__':
    fixed.base.main()
