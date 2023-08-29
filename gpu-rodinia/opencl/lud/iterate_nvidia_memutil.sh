#!/bin/bash
for i in $(seq 0 15);
do
	/var/tmp/likwid-nvidia/bin/likwid-perfctr -G 0 -W MEM -O -o results/nvidia/memutil-timeline/rodinia-lud-memutil-`printf %02d $i`.csv -t 100ms "./lud_cuda -s $((2**i))" >> results/nvidia/memutil-timeline/rodinia-lud-memutil-`printf %02d $i`.txt
done
