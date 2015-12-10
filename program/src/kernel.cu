#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#define GLM_FORCE_CUDA
#include "kernel.h"

#define M_PI 3.1415926
#define variance 20.f

__device__ unsigned int utilhash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

//struct RandGen
//{
//	RandGen() {}
//
//	__device__ float operator () (int idx)
//	{
//		thrust::default_random_engine randEng;
//		thrust::uniform_real_distribution<float> uniDist(-1, 1);
//		randEng.discard(idx);
//		return uniDist(randEng);
//	}
//};
//
//const int num = 256*256;
//thrust::device_vector<float> rVec(num);


//__global__ void kernCopyPlanetsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
//    int index = threadIdx.x + (blockIdx.x * blockDim.x);
//
//    float c_scale = -1.0f / s_scale;
//
//    if (index < N) {
//        vbo[4 * index + 0] = pos[index].x * c_scale;
//        vbo[4 * index + 1] = pos[index].y * c_scale;
//        vbo[4 * index + 2] = pos[index].z * c_scale;
//        vbo[4 * index + 3] = 1;
//    }
//}

//void Terrain::copyPlanetsToVBO(float *vbodptr) {
//	dim3 fullBlocksPerGrid((int)ceil(float(numMap) / float(blockSize)));
//
//    kernCopyPlanetsToVBO<<<fullBlocksPerGrid, blockSize>>>(numMap, dev_pos, vbodptr, scene_scale);
//    checkCUDAErrorWithLine("copyPlanetsToVBO failed!");
//
//    cudaThreadSynchronize();
//}

__device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index = 0, int depth = 0) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__global__ void HeightMapping(float *dev_height, unsigned int numPixels)
{
	int i = blockIdx.x, j = threadIdx.x;
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int octaves_ = octaves, seed_ = seeed;
	float amp_ = amplitude, freq_ = frequency;

	thrust::default_random_engine rng = makeSeededRandomEngine(index);
	thrust::uniform_real_distribution<float> u01(0, 1);

	if (index < numPixels){
		float carmen_gaussian_random = sqrtf(-2.0 * logf(1.0f - u01(rng)) * cosf(2.0 * M_PI * u01(rng)));
		dev_height[index] = variance * carmen_gaussian_random;

		for (int iter = 0; iter < order; iter++){
			Perlin perl(octaves_, amp_, freq_, seed_);		
			dev_height[index] += amplitude*perl.Get(i, j);
			octaves_++; amp_ /= 2; freq_ *= 2; seed_++;
		}
	}
}

void MapGen(float *hst_height, unsigned int size) {
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	float *dev_height;
	unsigned int numPixels = size;

	dim3 threadsPerBlock(blockSize);
	dim3 fullBlocksPerGrid((numPixels - blockSize + 1) / blockSize);

	//thrust::transform(
	//	thrust::make_counting_iterator(0),
	//	thrust::make_counting_iterator(num),
	//	rVec.begin(),
	//	RandGen()); 

	cudaMalloc((void**)&dev_height, size*sizeof(float));	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	HeightMapping << <fullBlocksPerGrid, threadsPerBlock >> >(dev_height, numPixels);
	cudaMemcpy(hst_height, dev_height, size*sizeof(float), cudaMemcpyDeviceToHost);
	

	std::cout << time << std::endl;
}
