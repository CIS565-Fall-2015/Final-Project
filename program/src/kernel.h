#pragma once

#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <glm/glm.hpp>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "utilityCore.hpp"
#include "perlin.h"

#define blockSize 128
#define order 5

#define octaves 1
#define amplitude 10
#define frequency 0.125
#define seeed 1

void MapGen(float *height, unsigned int numPixels);