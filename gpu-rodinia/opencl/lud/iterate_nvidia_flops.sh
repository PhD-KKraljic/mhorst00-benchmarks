#!/bin/bash
for i in $(seq 0 15);
do
	/var/tmp/likwid-nvidia/bin/likwid-perfctr -G 0 -W FLOPS_SP -O -o results/nvidia/flops-marker/rodinia-lud-flops-`printf %02d $i`.csv -m "./lud_cuda -s $((2**i))" >> results/nvidia/flops-marker/rodinia-lud-flops-`printf %02d $i`.txt
done
