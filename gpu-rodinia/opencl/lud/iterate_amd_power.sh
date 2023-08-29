#!/bin/bash
for i in $(seq 0 15);
do
	/var/tmp/likwid-lua/bin/likwid-perfctr -I 0 -R POWER -O -o results-power-timeline/rodinia-lud-power-`printf %02d $i`.csv -t 100ms "./lud -s $((2**i))" >> results-power-timeline/rodinia-lud-power-`printf %02d $i`.txt
done
