# Results

The exact payload, convolution, ReLU, flattening, scalar embedding, and
concatenated features pass.  The first failing operation is the concatenated
LSTM input dot at time zero.  All four gate slices fail; the stage maximum
absolute error is `0.0001825392246246338`.

The earliest tensor in deterministic record order is the forget-gate input
affine, index `[0,0,2]`, with 195 failing elements and maximum error
`0.00017858296632766724`.  Its downstream cell-state error is
`0.00012597441673278809`, exactly reproducing the parent component probe.

A read-only float64 counterfactual bounds the effect of upstream feature drift
at `4.7408848495404665e-08`; observed gate-affine errors are 3,617--4,437
times larger.  Default-precision backend GEMM arithmetic therefore dominates
the feature perturbation.  This rules out convolution/features as the earliest
failure but does not identify a specific GPU kernel or precision mode.

Safety counts are all zero for training steps, experiment/agent updates,
gradients, optimizer proposals/applications, and parameter mutations.  One GPU
forward-only capture ran.  PID 2786996 was present before and after.
