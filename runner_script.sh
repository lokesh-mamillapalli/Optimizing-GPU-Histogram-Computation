#!/bin/bash
if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
cd build
cmake .. && make -j
cd ..
./build/bin/tester 536870912 256 111
