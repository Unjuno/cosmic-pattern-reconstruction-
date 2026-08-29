#!/usr/bin/env python3
"""48-field replication of the REAL_DR11 multiband pixel selection-null test."""
import selection_null_multiband as mb
mb.TARGET_FIELDS=48
mb.MAX_CANDIDATES=48
# Use non-overlapping-ish stride to keep 48-field LOFO compute bounded; the
# 12-field sensitivity run used stride=4 and remains the fine-grid check.
mb.STRIDE_FINE=8
if __name__=='__main__':
    mb.main()
