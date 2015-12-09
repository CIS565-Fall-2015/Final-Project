#define GLM_FORCE_CUDA
#include "kernel.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/* Check for CUDA errors; print and exit if there was a problem.*/
void checkCUDAError(const char *msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

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

///**
// * Wrapper for call to the kernCopyPlanetsToVBO CUDA kernel.
// */
//void Terrain::copyPlanetsToVBO(float *vbodptr) {
//	dim3 fullBlocksPerGrid((int)ceil(float(numMap) / float(blockSize)));
//
//    kernCopyPlanetsToVBO<<<fullBlocksPerGrid, blockSize>>>(numMap, dev_pos, vbodptr, scene_scale);
//    checkCUDAErrorWithLine("copyPlanetsToVBO failed!");
//
//    cudaThreadSynchronize();
//}

dim3 threadsPerBlock(blockSize);

__global__ void HeightMap(float *height, int pixels)
{
	int octaves_ = octaves, seed_ = seed;
	float amp_ = amplitude, freq_ = frequency;
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < pixels)
		for (int i = 0; i < order; i++){
		octaves_++; amp_ /= 2; freq_ *= 2; seed_++;
		Perlin perl(octaves_, amp_, freq_, seed_);
		height[index] += perl.Get(blockIdx.x, threadIdx.x);
		}
}

void Terrain::MapGen(float* hst_height, unsigned int size, float *time) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	float *dev_height;
	unsigned int numPixels = size;
	dim3 fullBlocksPerGrid((int)ceil(float(numPixels) / blockSize));
		
	cudaMalloc((void**)&dev_height, sizeof(hst_height));
	checkCUDAErrorWithLine("cudaMalloc dev_height failed!");

	cudaMemcpy(hst_height, dev_height, sizeof(hst_height), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy hst_height failed!");

	HeightMap << <threadsPerBlock, fullBlocksPerGrid >> >(dev_height, numPixels);

	cudaMemcpy(hst_height, dev_height, sizeof(hst_height), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_height failed!");

	cudaFree(dev_height);
		
	cudaEventRecord(stop);	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(time, start, stop);
	std::cout << *time << std::endl;
}
