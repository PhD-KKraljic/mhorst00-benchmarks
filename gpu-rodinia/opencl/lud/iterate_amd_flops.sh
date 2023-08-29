#!/bin/bash
for i in $(seq 0 15);
do
	/var/tmp/likwid-lua/bin/likwid-perfctr -I 0 -R FLOPS_SP -O -o results-flops-marker/rodinia-lud-flops-`printf %02d $i`.csv -m "./lud -s $((2**i))" >> results-flops-marker/rodinia-lud-flops-`printf %02d $i`.txt
done
