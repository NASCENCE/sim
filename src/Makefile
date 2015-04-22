all:
	nvcc -o main -O3 -L. -I/usr/local/cuda-6.5/include -L/usr/local/cuda-6.5/lib -Iinc -lcurand -lcublas *.cc *.cu
debug:
	nvcc -o main -g -L. -I/usr/local/cuda-6.5/include -L/usr/local/cuda-6.5/lib -Iinc -lcurand -lcublas *.cc *.cu
