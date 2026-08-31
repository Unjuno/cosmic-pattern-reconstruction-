# REAL DR11 x DESI DR1 redshift-view status

## Question

Does the local 2D density continuity seen in Legacy Surveys DR11 correspond to redshift-localized structure in an independent DESI DR1 spectroscopic view?

Science inputs are observed DR11 RA/Dec catalogs and observed DESI DR1 `zpix` galaxy redshifts only. No simulated cosmological field is used.

The primary null permutes observed DESI redshifts within each field while keeping every DESI angular position fixed. This preserves the field's angular fiber/target-selection pattern while destroying redshift-localized associations.

## Initial fixed-expanded48 gate

Only 17 of the original fixed 48 DR11 fields had sufficient DESI DR1 coverage and at least two usable redshift bins.

- mean local actual correlation: 0.1180
- mean local z-shuffle null: 0.1051
- actual-minus-null mean: +0.0130
- median difference: +0.00780
- positive fields: 11/17
- one-sided sign p: 0.166
- one-sided Wilcoxon p: 0.0224
- global 200-permutation p: 0.00498
- broad-map difference: not significant (Wilcoxon p=0.339)

This was treated as suggestive only because the field count was small and the sign test was not significant.

## Independent-sky replication

A second set of 24 candidate fields was selected from a fixed shuffled sky grid using only coverage rules. Every candidate had to be at least 6 degrees from all original expanded48 centers and from other replication centers. Cross-view correlation values were not used in field selection. Seventeen fields passed the final statistical-density requirements.

- mean local actual correlation: 0.07545
- mean local z-shuffle null: 0.08142
- actual-minus-null mean: -0.00597
- median difference: -0.00818
- positive fields: 7/17
- one-sided sign p: 0.834
- one-sided Wilcoxon p: 0.905
- global 200-permutation p: 0.935
- max-slice and broad-map differences: not significant

The independent replication therefore FAILS to reproduce the initial positive local redshift-view gate.

## Combined interpretation

Combining the 17 initial usable fields and 17 independent replication fields gives 34 field-level differences:

- mean actual-minus-null difference: +0.00350
- median difference: +0.00126
- positive fields: 18/34
- one-sided sign p: 0.432
- one-sided Wilcoxon p: 0.324

Therefore the current experiment does **not** establish that the DR11 2D locality signal is tied to DESI redshift-localized 3D structure.

The initial positive is retained as an exploratory result, not deleted, but the accepted scientific status is **UNCERTAIN / NOT REPLICATED**.

## Possible reasons and next controls

1. DESI DR1 spectroscopy is sparse and highly nonuniform over these 0.5-degree fields.
2. Redshift permutation preserves angular sampling but breaks the relation between redshift, target class, galaxy properties, and selection; that can make the null imperfect.
3. The DR11 imaging map contains all source classes, whereas the spectroscopic map is restricted to high-quality DESI galaxies.
4. Fixed broad redshift bins can dilute narrow structures.
5. The initial usable fields may represent a non-representative part of the DESI footprint despite being selected before this test.

A stronger next test should compare matched tracer populations and/or use spectroscopic randoms/target-class-stratified redshift permutation, with redshift bins chosen independently of the observed correlation outcome.

## Implementation history

The first workflow attempt failed before science execution because Data Lab stores `zcat_primary` as a boolean and the query used `zcat_primary=1`. The rerun used `zcat_primary=TRUE` and completed successfully. This is an implementation/schema correction, not a scientific exclusion.
