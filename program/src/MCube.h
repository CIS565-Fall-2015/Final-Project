#pragma once

#include <vector>
#include <tchar.h>
#include <sstream>
#include <string>
#include <fstream> 
#include <iostream>
#include <glm/glm.hpp>
#include "makelevelset3.h"

using namespace glm;
using namespace std;

void MCube(std::vector<vec3> particlePosList);
void computeVertexValue(string objName, std::vector<vec3> particlePosList);
int marchingCube(double isolevel, int indexX, int indexY, int indexZ,
	int meshResX, int meshResY, int meshResZ, double meshSize,
	double* verticesValueArray, vec3 originPos,
	std::vector<vec3> *pointList,
	std::vector<std::vector<int>> *triangleList,
	std::vector<std::vector<int>> *temp);
void createObjFile(string name, vector<vec3> pointList, vector<vector<int>> triangleList);
vec3 vertexInterp(double isolevel, vec3 p1, vec3 p2, double valueP1, double valueP2);
void Level_Set_Method(string name, vector<vec3> pointList, vector<vector<int>> triangleList);

int largestRes = 100;//No need to change
int maxRadius = 5;//Recommend value 2~8
double isolevel = 0.5;//No need to change
double dx = 0.5;//Recommend value 0.1~1.0 #The lower value, the more time is required
double levelSetDiatance = 50;//Recommend value 10~200
string objStorePath = "C:/Users/Zhimin/Desktop/";//Output OBJ file Path
static int frameNumber = 1;
bool useLevelSet = true;