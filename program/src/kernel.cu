#define GLM_FORCE_CUDA
#include "kernel.h"

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

__global__ void HeightMapping(float *height, unsigned int numPixels)
{
	unsigned int i = blockIdx.x, j = threadIdx.x;
	unsigned int index = i * blockDim.x + j;

	int octaves_ = octaves, seed_ = seeed;
	float amp_ = amplitude, freq_ = frequency;

	if (index < numPixels){
		height[index] = 0;
		for (int iter = 0; iter < order && octaves_ > 0; iter++){
			octaves_--; amp_ /= 2; freq_ *= 2; seed_++;
			Perlin perl(octaves_, amp_, freq_, seed_);
			height[index] += perl.Get(i, j);
		}
	}
}

void MapGen(float *hst_height, unsigned int numPixels) {
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	float *dev_height;
	dim3 threadsPerBlock(blockSize);
	dim3 fullBlocksPerGrid((int)ceil(float(numPixels) / blockSize));

	cudaMalloc((void**)&dev_height, numPixels*sizeof(float));
	HeightMapping << <threadsPerBlock, fullBlocksPerGrid >> >(dev_height, numPixels);
	cudaMemcpy(hst_height, dev_height, numPixels*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop);	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cout << time << endl;
}
