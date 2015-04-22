#include <iostream>
#include <fstream>
#include <vector>

#include <cuda.h>
#include <curand_kernel.h>

#include "cudavec.h"

int const N_JUMPS(72);
int const N_SIMULATIONS(512);
int const SEED(2342);

using std::cout;
using std::cin;
using std::endl;

__device__ curandState rand_state[N_SIMULATIONS];

inline void gpuassert(cudaError_t code) {
  if (code != cudaSuccess) {
    std::cout << cudaGetErrorString(code) << std::endl;
    exit(1);
  }
}

__global__ void init_random() {
	int const idx = blockIdx.x;
	curand_init(SEED, idx, 0, &rand_state[idx]);
}

__global__ void simulate(float *g_init, float *c_init, float *A, int n_steps) {
	__shared__ float gamma[N_JUMPS];
	__shared__ float g[N_JUMPS];
	__shared__ float c[N_JUMPS];
	//__shared__ float c[N_JUMPS];
	__shared__ float sample;
	__shared__ int choice;

	int const idx = threadIdx.x;
	int const bidx = blockIdx.x;
	if (idx >= N_JUMPS)
		return;
	
	int const a_offset = bidx * N_JUMPS * N_JUMPS;

	g[idx] = g_init[idx + bidx * N_JUMPS];
	c[idx] = c_init[idx + bidx * N_JUMPS];

	__syncthreads();

	for (int i(0); i < n_steps; ++i) {
		//calculate gamma
	  gamma[idx] = exp(c[idx] + g[idx]);

		//accumulate
		int idx_tmp = idx;
		int step = 1;
		for (int n(N_JUMPS); n > 0; n >>= 1, idx_tmp >>= 1, step <<= 1) {
			if (idx_tmp % 2 == 0) {
			  //int next = idx + step;
			  int next = idx_tmp * step + step;
			  if (next < N_JUMPS)
			    gamma[idx] += gamma[next];
			}
			__syncthreads();
		}

		//sample, init shared
		if (idx == 0) {
			sample = curand_uniform(&rand_state[bidx]) * gamma[0];
			choice = 0;
		}
		__syncthreads();

		//select
		if (sample <= gamma[idx] && (idx == (N_JUMPS - 1) || sample > gamma[idx + 1]))
		  choice = idx;
				
		__syncthreads();

		//add
		g[idx] += A[a_offset + choice * N_JUMPS + idx];
		__syncthreads();
	}
	g_init[idx + bidx * N_JUMPS] = gamma[idx];
}

int main(int argc, char **argv) {
    gpuassert(cudaSetDevice(1));
	CudaVec g_init(N_JUMPS * N_SIMULATIONS);
	CudaVec c_init(N_JUMPS * N_SIMULATIONS);
	CudaVec A(N_JUMPS * N_JUMPS * N_SIMULATIONS);

	cout << "init normal" << endl;
	g_init.init_normal(0, .1);
	c_init.init_normal(0, .1);
	A.init_normal(0, .0001);
	int n_steps(1000000);
	
	init_random<<<N_SIMULATIONS, 1>>>();
	gpuassert( cudaPeekAtLastError() );
	gpuassert( cudaDeviceSynchronize() );
	cout << "simulate" << endl;
	Timer clock;
	simulate<<<N_SIMULATIONS, N_JUMPS>>>(g_init.data, c_init.data, A.data, n_steps);
	gpuassert( cudaPeekAtLastError() );
	gpuassert( cudaDeviceSynchronize() );
	//cout << g_init.to_vector() << endl;
	cout << "took: " << clock.since() << "s" << endl;
	cout << "done" << endl;
}

