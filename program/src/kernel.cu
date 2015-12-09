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

//void Terrain::copyPlanetsToVBO(float *vbodptr) {
//	dim3 fullBlocksPerGrid((int)ceil(float(numMap) / float(blockSize)));
//
//    kernCopyPlanetsToVBO<<<fullBlocksPerGrid, blockSize>>>(numMap, dev_pos, vbodptr, scene_scale);
//    checkCUDAErrorWithLine("copyPlanetsToVBO failed!");
//
//    cudaThreadSynchronize();
//}
__global__ void HeightMapping(thrust::device_vector<glm::vec3> &height, unsigned int numPixels)
{
	unsigned int j = blockIdx.x, i = threadIdx.x;
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	int octaves_ = octaves, seed_ = seed;
	float amp_ = amplitude, freq_ = frequency;

	if (index < numPixels)
		for (int iter = 0; iter < order && octaves_ > 0; iter++){
			octaves_--; amp_ /= 2; freq_ *= 2; seed_++;
			Perlin perl(octaves_, amp_, freq_, seed_);
			height[index] += glm::vec3(i, j, perl.Get(i, j));
		}
}

void MapGen(vector<glm::vec3> &hst_height, unsigned int numPixels) {
	float time(0);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	thrust::device_vector<glm::vec3> dev_height(numPixels, glm::vec3(0));
	dim3 threadsPerBlock(blockSize);
	dim3 fullBlocksPerGrid((int)ceil(float(numPixels) / blockSize));

	HeightMapping << <threadsPerBlock, fullBlocksPerGrid >> >(dev_height, numPixels);
	thrust::copy(hst_height.begin(), hst_height.end(), dev_height.begin());
	
	cudaEventRecord(stop);	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cout << time << endl;
}
