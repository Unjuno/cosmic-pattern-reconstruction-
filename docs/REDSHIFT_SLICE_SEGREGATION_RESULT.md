# Independent redshift-slice segregation result

Status: **FAIL_REDSHIFT_SLICE_SPATIAL_SEGREGATION**.

The unexpected negative sign in the matched-tracer projection-correlation gate motivated a new hypothesis: because redshift shuffling makes every z bin a random thinning of the same angular catalog, real fixed-z slices might instead be *less mutually similar* than shuffled-label slices. This directional hypothesis was tested only on a new third independent sky set selected without outcome information.

## Confirmation result

- coverage-qualified/evaluable fields: **23**
- median matched DESI fraction: **0.870**
- observed pairwise z-slice local-map correlation median: **-0.04688**
- stratified-null pairwise correlation median: **-0.03338**
- median segregation effect (`null - observed`): **+0.01598**
- positive fields: **14/23**
- one-sided sign p: **0.2024**
- one-sided Wilcoxon p: **0.1056**
- global lower-tail permutation p: **0.0365**

The preregistered PASS required median effect >=0.02, both paired p<0.01, and global p<0.01. None of those joint conditions is met.

A secondary version using full-projection correlation has median effect +0.01024, 15/23 positive, sign p=0.1050, Wilcoxon p=0.0429. This is suggestive but secondary and also fails the confirmatory standard.

## Interpretation

There is a weak tendency for real fixed-z slices to be more spatially distinct than survey/program-stratified random label partitions, but it does not replicate at the preregistered strength on the new independent sky set. The correct status is therefore **not established**.

Together with the matched-tracer projection-correlation FAIL, this closes the current DESI DR1 redshift roadmap without a positive 3D claim. The robust DR11 angular locality remains a two-dimensional observational result under current evidence.

Source workflow run: `34586009672`; artifact id: `10193687149`; artifact digest: `sha256:2ab3743aaf8748b5c9bfb52cc32759f2e05b76d9be66e83caed818e9036a832b`.
