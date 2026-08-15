# Results

The default branch reproduces the prior input-affine and cell-carry failures.
The highest-precision branch passes every required recurrent stage.  Maximum
CPU/GPU final-carry error falls from `0.00012597441673278809` to
`5.960464477539063e-08`.

On CPU, default and highest recurrent tensors are byte-exact.  On the RTX
5090, the precision intervention changes the input affine by as much as
`0.00018256902694702148` and final carry by
`0.00012592971324920654`, explaining the captured default discrepancy.

The closure is causal for this frozen forward payload and supports designing a
targeted compatibility patch.  It does not validate a source patch, optimizer
update, training, performance, or paper claim.

All forbidden execution counters are zero.  One GPU forward capture ran, and
PID 2786996 was preserved.
