#!/bin/sh

# CUDA to MUSA
musify-text --inplace `rg --files -g '*.cu' -g '*.cuh' -g '*.cpp' -g '*.h' -g '*.hpp' -g '*.hxx' -g '*.c' -g '*.inc' -g '*.cxx' ./src`

# MUSA to CUDA
# musify-text --inplace -d m2c `rg --files -g '*.cu' -g '*.cuh' -g '*.cpp' -g '*.h' -g '*.hpp' -g '*.hxx' -g '*.c' -g '*.inc' -g '*.cxx' ./src`