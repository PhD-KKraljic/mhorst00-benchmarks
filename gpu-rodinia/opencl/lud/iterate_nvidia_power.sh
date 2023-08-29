#!/bin/bash
for i in $(seq 0 15);
do
	/var/tmp/likwid-nvidia/bin/likwid-perfctr -G 0 -W POWER -O -o results/nvidia/power-timeline/rodinia-lud-power-`printf %02d $i`.csv -t 100ms "./lud -s $((2**i))" >> results/nvidia/power-timeline/rodinia-lud-power-`printf %02d $i`.txt;
done
