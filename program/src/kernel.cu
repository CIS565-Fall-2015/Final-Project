#define GLM_FORCE_CUDA
#include "kernel.h"
#define M_PI 3.1415926
#define variance 10.f

__device__ unsigned int numPixels;
__device__ float *dev_height;

//__device__ int utilhash(int a){
//	a = (a + 0x7ed55d16) + (a << 12);
//	a = (a ^ 0xc761c23c) ^ (a >> 19);
//	a = (a + 0x165667b1) + (a << 5);
//	a = (a + 0xd3a2646c) ^ (a << 9);
//	a = (a + 0xfd7046c5) + (a << 3);
//	a = (a ^ 0xb55a4f09) ^ (a >> 16);
//	return a;
//}
//
//__device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
//	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
//	return thrust::default_random_engine(h);
//}

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

__global__ void HeightMapping()
{
	int i = blockIdx.x, j = threadIdx.x;
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int octaves_ = octaves, seed_ = seeed;
	float amp_ = amplitude, freq_ = frequency;// , variance_ = variance;

	//thrust::default_random_engine rng = makeSeededRandomEngine(numPixels, index, 0);
	//thrust::uniform_real_distribution<float> u01(0, 1);

	if (index < numPixels){
		/*dev_height[index] = 0;
		for (int iter = 0; iter < order; iter++){*/
			float carmen_gaussian_random = sqrtf(-2.0 * logf(1.f - float(index) / numPixels) * cosf(2.0 * M_PI * index / numPixels));
			Perlin perl(octaves_, amp_, freq_, seed_);		
			dev_height[index] = amplitude*perl.Get(i, j) + variance * carmen_gaussian_random + float(index) / numPixels * 10;
		//	variance_ /= 2;
		//	octaves_++; amp_ /= 2; freq_ *= 2; seed_++;
		//}
	}
}

void MapGen(float *hst_height, unsigned int size) {
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	numPixels = size;
	dim3 threadsPerBlock(blockSize);
	dim3 fullBlocksPerGrid((numPixels - blockSize + 1) / blockSize);

	cudaMalloc((void**)&dev_height, numPixels*sizeof(float));
	HeightMapping << <fullBlocksPerGrid, threadsPerBlock>> >();
	cudaMemcpy(hst_height, dev_height, numPixels*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop);	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	std::cout << time << std::endl;
}
