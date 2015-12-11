#pragma once

#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "utilityCore.hpp"
#include "perlin.h"

#define blockSize 256
#define order 5

#define octaves 2
#define amplitude 10
#define frequency 0.125
#define seeed 1

void MapGen(float *height, unsigned int numPixels);