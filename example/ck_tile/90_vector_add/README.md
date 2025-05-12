# Vector Add

This folder contains example for Vector Add using ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
make tile_example_vector_add -j
```
This will result in an executable `build/bin/tile_example_vector_add`
