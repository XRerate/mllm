#!/bin/bash
cmake -B ../build .. -DCMAKE_BUILD_TYPE=Release -GNinja
cmake --build ../build
