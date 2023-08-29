#!/bin/csh

mkdir -p bin
mkdir -p result

foreach benchmark ( ft mg cg lu bt is ep sp )
    foreach class ( S W A B C D )
        echo "compiling $benchmark.$class. (OMP-C)"
        make $benchmark CLASS=$class
        foreach num_thread ( 1 2 4 8 16 32 64 )
            setenv OMP_NUM_THREADS $num_thread
            echo "running $benchmark.$class. (OMP-C, $num_thread threads)"
            bin/$benchmark.$class.x > result/$benchmark.$class.$num_thread.out
            echo "done.\n"
        end
    end
end
