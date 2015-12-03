#pragma once

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "scene.h"
#include "perlin.h"
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define OCTAVES 6
#define FREQUENCY 0.125
#define AMPLITUDE 10
#define SEED 1

void terrainInit(Scene *scene);
void terrainFree();
void terrain(uchar4 *pbo, int frame, int iteration);