cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCUDA_ROOT=/usr/local/cuda -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=gcc -DCMAKE_CXX_COMPILER:FILEPATH=g++ -S. -B./build -G "Unix Makefiles" ;
cmake --build ./build --config Debug --target clean -j 18 -- ;
cmake --build ./build --config Debug --target all -j 18 -- ;
./build/src/fc.exe;
