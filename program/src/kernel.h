#pragma once

#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <glm/glm.hpp>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "utilityCore.hpp"
#include "perlin.h"

#define blockSize 128
#define order 5

#define octaves 6
#define amplitude 2
#define frequency 0.125
#define seed 1

using namespace std;

void MapGen(float *height, unsigned int numPixels);