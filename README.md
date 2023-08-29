# mhorst00-benchmarks
Benchmarks used for my bachelor thesis on energy efficient memory accesses on GPUs

Every benchmark in this repository contains some code changes to instrument it with LIKWID's Marker API. If that was not possible, use the LIKWID timeline mode.

Except for rocHPL, all benchmarks can be compiled with `make`. rocHPL uses CMake.

Follow the build instructions of each benchmark included inside of the directories. LIKWID instrumentation usage is omitted.

The version of LIKWID used for the thesis is included.

| Benchmark   | Source                                                            | Version/Commit |
|-------------|-------------------------------------------------------------------|----------------|
| STREAM      | <https://github.com/jeffhammond/STREAM> (customized form AMD HIP) | 39d7b16        |
| Rodinia LUD | <https://github.com/yuhc/gpu-rodinia>                             | 6ec8416        |
| NPB MG      | <https://thunder.snu.ac.kr/?page_id=64&page=7>                    | April 01, 2020 |
| rocHPL      | <https://github.com/ROCmSoftwarePlatform/rocHPL>                  | 1919b52        |
| gpumembench | <https://github.com/ekondis/gpumembench>                          | 838a4fc        |


# Thesis

Energy-Efficient Memory Accesses for GPUs

Author: Malte Horst
Date: 2023/09/04
University: DHBW Stuttgart
