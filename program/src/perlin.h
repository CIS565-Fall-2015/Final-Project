#pragma once

#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <stdlib.h>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define SAMPLE_SIZE 1024
#define B SAMPLE_SIZE
#define BM (SAMPLE_SIZE-1)

#define N 0x1000
#define NP 12
#define NM 0xfff

#define s_curve(t) (t * t * (3.0f - 2.0f * t))
#define lerrp(t, a, b) (a + t * (b - a))

class Perlin {
public:

  __device__ Perlin(int octaves,float freq,float amp,int seed);

  __device__ float Get(float x, float y)
  {
    float vec[2];
    vec[0] = x;
    vec[1] = y;
    return perlin_noise_2D(vec);
  }

  __device__  float Get(float x, float y, float z)
  {
	  float vec[3];
	  vec[0] = x;
	  vec[1] = y;
	  vec[2] = z;
	  return perlin_noise_3D(vec);
  }

  __device__ void init_perlin(int n, float p);
  __device__ float perlin_noise_2D(float vec[2]);
  __device__ float perlin_noise_3D(float vec[3]);

  __device__ float noise1(float arg);
  __device__ float noise2(float vec[2]);
  __device__ float noise3(float vec[3]);
  __device__ void normalize2(float v[2]);
  __device__ void normalize3(float v[3]);
  __device__ void init_rand(int seed);
  __device__ void init(void);

  int mOctaves;
  float mFrequency;
  float mAmplitude;
  int mSeed;

  int p[SAMPLE_SIZE + SAMPLE_SIZE + 2];
  float g3[SAMPLE_SIZE + SAMPLE_SIZE + 2][3];
  float g2[SAMPLE_SIZE + SAMPLE_SIZE + 2][2];
  float g1[SAMPLE_SIZE + SAMPLE_SIZE + 2];
  bool mStart;

  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> unitDistrib;
};