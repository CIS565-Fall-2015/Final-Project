#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtx/transform.hpp>
#include <glm/glm.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <cstring>
#include <ctime>

#include "glslUtility.hpp"
#include "sceneStructs.h"
#include "terrain.h"
#include "utilities.h"
#include "scene.h"
#include "image.h"

using namespace std;

Scene *scene;
RenderState *renderState;
int iteration;
int width, height;

static string startTimeString;
static bool camchanged = false;
static float theta = 0, phi = 0;
static glm::vec3 cammove;

string currentTimeString();
bool init();
void mainLoop();
