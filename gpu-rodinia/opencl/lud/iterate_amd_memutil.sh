#!/bin/bash
for i in $(seq 0 15);
do
	/var/tmp/likwid-lua/bin/likwid-perfctr -I 0 -R MEM_UTIL_GILGAMESH -O -o results-memutil-marker/rodinia-lud-memutil-`printf %02d $i`.csv -m "./lud -s $((2**i))" >> results-memutil-marker/rodinia-lud-memutil-`printf %02d $i`.txt
done
