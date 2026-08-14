# Minimax Laplace quadrature tables

`init_para.txt`, `init_error.txt` are the best (minimax) exponential-sum
approximations of 1/x on [1, R],

    1/x = sum_k omega_k exp(-alpha_k x),

taken verbatim from the `laplace-minimax` library (LGPL-3.0),
https://github.com/bhelmichparis/laplace-minimax, which in turn took them from
W. Hackbusch's tables at https://gitlab.mis.mpg.de/scicomp/EXP_SUM (originally
www.mis.mpg.de/scicomp/EXP_SUM/1_x).

Coverage: k = 1..53 points, R = 1.1 .. 4e12 (106 ranges, 1004 grids).

`init_error.txt` holds the max ABSOLUTE error max|1/x - sum_k ...| on [1, R].
The relative error is R times larger, since the absolute error equioscillates.

Cite:
  A. Takatsuka, S. Ten-no, W. Hackbusch, J. Chem. Phys. 129, 044112 (2008)
  B. Helmich-Paris, L. Visscher, J. Comput. Phys. 321, 927 (2016)
