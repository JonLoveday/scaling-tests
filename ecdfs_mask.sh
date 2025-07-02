#!/bin/bash

# Create polygon mask for Rubin DP1 ECDFS field - a 0.3 deg radius circle centred on (53.13, -28.10)
# See https://space.mit.edu/~molly/mangle/manual/weight.html
cat > boundary.txt <<eod
circle 0 1
unit d
53.13 -28.10 0.3
eod
cat > zero <<eod
0
eod
snap boundary.txt boundary.ply
snap holes.txt holes.ply
weight -zzero holes.ply holes.ply
snap boundary.ply holes.ply mask.ply
balkanize mask.ply mask.ply
unify mask.ply mask.ply