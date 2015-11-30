#pragma once

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "kernel.h"
//#include "sceneStructs.h"
//#include "image.h"
//#include "scene.h"

//====================================
// GL Stuff
//====================================

GLuint normalLocation = 1;
GLuint positionLocation = 0;
const char *attributeLocations[] = { "Position", "Normal" };
GLuint planetVAO = 0;
GLuint planetVBO = 0;
GLuint planetNBO = 0;
GLuint planetIBO = 0;
GLuint displayImage;
GLuint program[2];

const unsigned int PROG_PLANET = 0;

const float fovy = (float) (PI / 4);
const float zNear = 0.10f;
const float zFar = 5.0f;

glm::mat4 projection;
glm::vec3 cameraPosition(1.75, 1.75, 1.35);

static std::string startTimeString;
static bool camchanged = false;
static float theta = 0, phi = 0;
static glm::vec3 cammove;

//extern Scene *scene;
//RenderState *renderState;
//extern int iteration;
//
//extern int width;
//extern int height;

//====================================
// Main
//====================================

const char *projectName;

int main(int argc, char* argv[]);

//====================================
// Main loop
//====================================
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void runCUDA();/*
void saveImage();*/

//====================================
// Setup/init Stuff
//====================================
bool init(int argc, char **argv);
void initVAO();
void initShaders(GLuint *program);
