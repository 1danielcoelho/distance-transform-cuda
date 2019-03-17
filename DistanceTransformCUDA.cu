#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_profiler_api.h>

using namespace std;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

#define eee(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template<typename T>
__global__ void edt_cols(T* d_input, T* d_output, uint width, uint height)
{
	// x in range [0, width-1]	
	uint x = blockIdx.x * blockDim.x + threadIdx.x; 

	if(x >= width)
		return;
	
	extern __shared__ T g[];

	// Initialize val to either 0 or 'infinity'
	T val = (1-d_input[x]) * (width+height);
	g[0] = val;

	// Scan 1
	for (uint y = 1; y < height; y++)
	{
		val = (1 - d_input[y * width + x]) * (1 + val);
		g[y] = val;
	}

	// Scan 2	
	// y < height is the same as y >= 0, as this uint underflows
	for (uint y = height - 2; y < height; y--)
	{
		if (g[y] > val)
		{
			g[y] = 1 + val;
		}

		val = g[y];
	}
	
	for(uint y = 0; y < height; y++)
		d_output[y * width + x] = g[y];
}

template<typename T>
__global__ void edt_rows(T* d_output, uint width, uint height)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;  // range [0, width-1]
	uint y = blockIdx.y * blockDim.y + threadIdx.y;  // range [0, height-1]

	if(x >= width)
		return;

	extern __shared__ T d_localG[];

	for (uint i = threadIdx.x; i < width; i += blockDim.x)
		d_localG[i] = d_output[y * width + i]; 

	__syncthreads();	

	T minDist = FLT_MAX;
	for (uint i = 0; i < width; i++)
	{
		minDist = fminf(minDist, (x-i)*(x-i) + d_localG[i] * d_localG[i]);
	}

	d_output[y * width + x] = sqrtf(minDist); 
}

template <typename T>
void runCUDA(T* h_inData, T* h_outData, uint width, uint height)
{
	size_t numBytes = height * width * sizeof(T);
	assert(numBytes > 0);

	T* d_inData;
	eee(cudaMalloc((void **) &d_inData, numBytes));
	eee(cudaMemcpy(d_inData, h_inData, numBytes, cudaMemcpyHostToDevice));	

	T* d_outData;
	eee(cudaMalloc((void **)&d_outData, numBytes));
	
	// TODO: Assert width/height are not too large to have a shared memory copy (due to SMEM size limit)

	dim3 colsGrid(width, 1, 1);
	dim3 colsThreads(1, 1, 1);

	uint threadsPerBlock = 1024;		
	dim3 rowsGrid(ceil((1.0f*width) / threadsPerBlock), height, 1);
	dim3 rowsThreads(min(width, 1024), 1, 1);

	uint numtrials = 10000;

	// Warmup
	for (int i = 0; i < (numtrials / 10); i++)
	{
		edt_cols<<<colsGrid, colsThreads, height * sizeof(T)>>>(d_inData, d_outData, width, height);
		edt_rows<<<rowsGrid, rowsThreads, width * sizeof(T)>>>(d_outData, width, height);
		eee(cudaDeviceSynchronize());
	}

	auto start = std::chrono::high_resolution_clock::now();
	{
		for (int i = 0; i < numtrials; i++)
		{
			edt_cols<<<colsGrid, colsThreads, height * sizeof(T)>>>(d_inData, d_outData, width, height);
			edt_rows<<<rowsGrid, rowsThreads, width * sizeof(T)>>>(d_outData, width, height);
			eee(cudaDeviceSynchronize());
		}
	}
	auto duration = std::chrono::high_resolution_clock::now() - start;
	long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	printf("runCUDA executed in %lld microseconds\n", ms / numtrials);    
    
	eee(cudaGetLastError());
	eee(cudaMemcpy(h_outData, d_outData, numBytes, cudaMemcpyDeviceToHost)); 
	
	eee(cudaFree(d_inData));
	eee(cudaFree(d_outData));	

	eee(cudaProfilerStop());
	eee(cudaDeviceReset());
}

int main(int argc, char **argv)
{
	printf("Starting\n");
	
	uint width = 256;
	uint height = 256;

	vector<float> inputData(width * height);
	vector<float> outputData(width * height);
	for (uint x = 0; x < width; x++)
	{
		for (uint y = 0; y < height; y++)
		{
			inputData[y * width + x] = (float)(x > 100 && x < 150 && y > 100 && y < 150 ? 1.0f : 0.0f); 
			inputData[y * width + x] = (float)(inputData[y * width + x] || abs((float)(x - y)) < 3 ? 1.0f : 0.0f);
		}
	}

	runCUDA(inputData.data(), outputData.data(), width, height);

	ofstream fout("input.dat", ios::out | ios::binary);
	fout.write((char*)inputData.data(), inputData.size() * sizeof(inputData[0]));
	fout.close();

	fout = ofstream("output.dat", ios::out | ios::binary);
	fout.write((char*)outputData.data(), outputData.size() * sizeof(outputData[0]));
	fout.close();
}