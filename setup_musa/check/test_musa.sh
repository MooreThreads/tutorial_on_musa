#!/bin/bash

mcc test_musa.mu -o test_musa -mtgpu -O2 -lmusart -L${MUSA_INSTALL_PATH}/lib

./test_musa
