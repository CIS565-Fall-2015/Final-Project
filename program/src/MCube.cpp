#include "MCube.h"

void MCube(std::vector<vec3> particlePosList)
{
	string objName;
	std::stringstream strout;

	strout << "terrain_" << frameNumber << ".obj";
	strout >> objName;
	strout.clear();

	computeVertexValue(objName, particlePosList);
	particlePosList.clear();
	frameNumber++;
}


void computeVertexValue(string objName, std::vector<vec3> particlePosList)
{
	int particleCount = particlePosList.size();
	double maxXPos;
	double maxYPos;
	double maxZPos;
	double minXPos;
	double minYPos;
	double minZPos;

	for (int i = 0; i < particleCount; i++)
	{
		vec3 particlePos = particlePosList[i];
		if (i == 0)
		{
			maxXPos = particlePos[0];
			maxYPos = particlePos[1];
			maxZPos = particlePos[2];
			minXPos = particlePos[0];
			minYPos = particlePos[1];
			minZPos = particlePos[2];
		}
		else
		{
			if (particlePos[0] > maxXPos)
				maxXPos = particlePos[0];
			else if (particlePos[0] < minXPos)
				minXPos = particlePos[0];

			if (particlePos[1] > maxYPos)
				maxYPos = particlePos[1];
			else if (particlePos[1] < minYPos)
				minYPos = particlePos[1];

			if (particlePos[2] > maxZPos)
				maxZPos = particlePos[2];
			else if (particlePos[2] < minZPos)
				minZPos = particlePos[2];
		}
	}


	double diffX = maxXPos - minXPos;
	double diffY = maxYPos - minYPos;
	double diffZ = maxZPos - minZPos;
	double diff = diffX;
	int largestIndex = 0;
	if (diffY > diff)
	{
		diff = diffY;
		largestIndex = 1;
	}
	if (diffZ > diff)
	{
		diff = diffZ;
		largestIndex = 2;
	}

	vec3 centerPos((maxXPos + minXPos) / 2.0f, (maxYPos + minYPos) / 2.0f, (maxZPos + minZPos) / 2.0f);
	double meshSize = diff / largestRes;
	int meshResX;
	int meshResY;
	int meshResZ;
	switch (largestIndex)
	{
	case 0:
		meshResX = largestRes + 2 * maxRadius;
		meshResY = (int)(diffY / meshSize) + 1 + 2 * maxRadius;
		meshResZ = (int)(diffZ / meshSize) + 1 + 2 * maxRadius;
		break;
	case 1:
		meshResX = (int)(diffX / meshSize) + 1 + 2 * maxRadius;
		meshResY = largestRes + 2 * maxRadius;
		meshResZ = (int)(diffZ / meshSize) + 1 + 2 * maxRadius;
		break;
	case 2:
		meshResX = (int)(diffX / meshSize) + 1 + 2 * maxRadius;
		meshResY = (int)(diffY / meshSize) + 1 + 2 * maxRadius;
		meshResZ = largestRes + 2 * maxRadius;
		break;
	}

	int verticesCount = (meshResX + 1) * (meshResY + 1) * (meshResZ + 1);
	vec3 originPos(minXPos - maxRadius * meshSize, minYPos - maxRadius * meshSize, minZPos - maxRadius * meshSize);


	double* verticesValueArray = new double[verticesCount];

	for (int i = 0; i < verticesCount; i++)
	{
		verticesValueArray[i] = 0;
	}


	for (int i = 0; i < particleCount; i++)
	{
		vec3 particlePos = particlePosList[i];;
		int vertexIndexX = (int)round((particlePos[0] - minXPos) / meshSize) + maxRadius;
		int vertexIndexY = (int)round((particlePos[1] - minYPos) / meshSize) + maxRadius;
		int vertexIndexZ = (int)round((particlePos[2] - minZPos) / meshSize) + maxRadius;

		int minVertexIndexX = vertexIndexX - maxRadius;
		int minVertexIndexY = vertexIndexY - maxRadius;
		int minVertexIndexZ = vertexIndexZ - maxRadius;
		int maxVertexIndexX = vertexIndexX + maxRadius;
		int maxVertexIndexY = vertexIndexY + maxRadius;
		int maxVertexIndexZ = vertexIndexZ + maxRadius;


		for (int z = minVertexIndexZ; z <= maxVertexIndexZ; z++)
			for (int y = minVertexIndexY; y <= maxVertexIndexY; y++)
				for (int x = minVertexIndexX; x <= maxVertexIndexX; x++)
				{
			int vertex = z * (meshResX + 1) * (meshResY + 1) + y * (meshResX + 1) + x;

			double distance = sqrt(pow((double)(x - vertexIndexX), 2) + pow((double)(y - vertexIndexY), 2) + pow((double)(z - vertexIndexZ), 2));
			if (distance == 0)
				bool test = true;
			double value;
			if (maxRadius == 0)
				value = 1;
			else value = 2 * (1 - distance / maxRadius);

			if (value < 0)
				value = 0;

			double count = verticesValueArray[vertex];
			count += value;
			verticesValueArray[vertex] = count;
				}
	}

	std::vector<vec3> pointList;
	std::vector<vector<int>> triangleList;
	std::vector<vector<int>> temp;

	for (int k = 0; k < meshResZ; k++)
		for (int j = 0; j < meshResY; j++)
			for (int i = 0; i < meshResX; i++)
			{
		marchingCube(isolevel, i, j, k,
			meshResX, meshResY, meshResZ, meshSize,
			verticesValueArray, originPos,
			&pointList, &triangleList, &temp);
			}

	if (useLevelSet)
		Level_Set_Method(objName, pointList, triangleList);
	else
		createObjFile(objName, pointList, triangleList);

}

void Level_Set_Method(string name, vector<vec3> pointList, vector<vector<int>> triangleList)
{
	//Start with a massive inside out bound box////
	Vec3f min_box(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	Vec3f max_box(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	std::vector<Vec3f> vertList;
	std::vector<Vec3ui> faceList;
	////////////////////
	//Create Face List//
	////////////////////
	for (unsigned int i = 0; i < triangleList.size(); i++)
	{
		faceList.push_back(Vec3ui(triangleList[i][0] - 1, triangleList[i][1] - 1, triangleList[i][2] - 1));
	}
	////////////////////////
	//Create Vertices List//
	////////////////////////
	for (unsigned int i = 0; i < pointList.size(); i++)
	{
		Vec3f point;
		point[0] = pointList[i][0];
		point[1] = pointList[i][1];
		point[2] = pointList[i][2];
		vertList.push_back(point);
		update_minmax(point, min_box, max_box);
	}
	
	double sizeX = max_box[0] - min_box[0];
	double sizeY = max_box[1] - min_box[1];
	double sizeZ = max_box[2] - min_box[2];
	double maxSideLength = sizeX;
	if (sizeY > maxSideLength)
		maxSideLength = sizeY;
	if (sizeZ > maxSideLength)
		maxSideLength = sizeZ;
	double totalSize = sizeX * sizeY * sizeZ;
	//double cubicSize = totalSize/ cubicNum;
	//double dx = pow(cubicSize, (double)(1.0f/ 3.0f));

	//cout << "sizeX = " << sizeX << "  sizeY = " << sizeY << "  sizeZ = " << sizeZ << endl;
	//cout << "dx = " << dx << endl;
	////Add padding around the box.////
	Vec3f unit(dx, dx, dx);
	min_box -= (float)0.001 * unit;
	max_box += (float)0.001 * unit;
	Vec3ui sizes = Vec3ui((max_box - min_box) / dx) + Vec3ui(2, 2, 2);
	//cout << "sizes[0] = " << sizes[0] << "  sizes[1] = " << sizes[1] << "  sizes[2] = " << sizes[2] << endl;
	Array3f phi_grid;



	vector<double> phiValue;
	///////////////////////////////////
	//Use level set method to compute//
	///////////////////////////////////
	make_level_set3(faceList, vertList, min_box, dx, sizes[0], sizes[1], sizes[2], phi_grid);

	double* verticesValueArray = new double[phi_grid.a.size()];
	vector<vec3> marchingCubePointList;
	vector<vector<int>> temp;
	vector<vector<int>> newTriangleList;
	vec3 originPos(min_box[0], min_box[1], min_box[2]);
	double meltingThickness = maxSideLength / levelSetDiatance;

	for (unsigned int i = 0; i < phi_grid.a.size(); i++)
	{
		verticesValueArray[i] = -phi_grid.a[i];
	}
	/////////////////////////////////////////////////////////////////////////////////
	//From returning distance function with marching cube method to create new mesh//
	/////////////////////////////////////////////////////////////////////////////////
	for (unsigned int z = 0; z < sizes[2] - 1; z++)
	{
		//cout << "z = " << z <<endl;
		for (unsigned int y = 0; y < sizes[1] - 1; y++)
		{
			for (unsigned int x = 0; x < sizes[0] - 1; x++)
			{
				marchingCube(meltingThickness, x, y, z,//meltingThickness
					sizes[0] - 1, sizes[1] - 1, sizes[2] - 1, dx,
					verticesValueArray, originPos,
					&marchingCubePointList,
					&newTriangleList,
					&temp);
			}
		}
	}

	createObjFile(name, marchingCubePointList, newTriangleList);

}

int marchingCube(double isolevel, int indexX, int indexY, int indexZ,
	int meshResX, int meshResY, int meshResZ, double meshSize,
	double* verticesValueArray, vec3 originPos,
	vector<vec3>        *pointList,
	vector<vector<int>> *triangleList,
	vector<vector<int>> *temp)
{

	int gridValueIndex0 = indexZ * (meshResX + 1) * (meshResY + 1) + indexY * (meshResX + 1) + indexX;
	int gridValueIndex1 = indexZ * (meshResX + 1) * (meshResY + 1) + indexY * (meshResX + 1) + indexX + 1;
	int gridValueIndex2 = indexZ * (meshResX + 1) * (meshResY + 1) + (indexY + 1) * (meshResX + 1) + indexX + 1;
	int gridValueIndex3 = indexZ * (meshResX + 1) * (meshResY + 1) + (indexY + 1) * (meshResX + 1) + indexX;
	int gridValueIndex4 = (indexZ + 1) * (meshResX + 1) * (meshResY + 1) + indexY * (meshResX + 1) + indexX;
	int gridValueIndex5 = (indexZ + 1) * (meshResX + 1) * (meshResY + 1) + indexY * (meshResX + 1) + indexX + 1;
	int gridValueIndex6 = (indexZ + 1) * (meshResX + 1) * (meshResY + 1) + (indexY + 1) * (meshResX + 1) + indexX + 1;
	int gridValueIndex7 = (indexZ + 1) * (meshResX + 1) * (meshResY + 1) + (indexY + 1) * (meshResX + 1) + indexX;


	double gridValue0 = verticesValueArray[gridValueIndex0];
	double gridValue1 = verticesValueArray[gridValueIndex1];
	double gridValue2 = verticesValueArray[gridValueIndex2];
	double gridValue3 = verticesValueArray[gridValueIndex3];
	double gridValue4 = verticesValueArray[gridValueIndex4];
	double gridValue5 = verticesValueArray[gridValueIndex5];
	double gridValue6 = verticesValueArray[gridValueIndex6];
	double gridValue7 = verticesValueArray[gridValueIndex7];

	vec3 gridPos0(indexX       * meshSize + originPos[0], indexY       * meshSize + originPos[1], indexZ       * meshSize + originPos[2]);
	vec3 gridPos1((indexX + 1) * meshSize + originPos[0], indexY       * meshSize + originPos[1], indexZ       * meshSize + originPos[2]);
	vec3 gridPos2((indexX + 1) * meshSize + originPos[0], (indexY + 1) * meshSize + originPos[1], indexZ       * meshSize + originPos[2]);
	vec3 gridPos3(indexX       * meshSize + originPos[0], (indexY + 1) * meshSize + originPos[1], indexZ       * meshSize + originPos[2]);
	vec3 gridPos4(indexX       * meshSize + originPos[0], indexY       * meshSize + originPos[1], (indexZ + 1) * meshSize + originPos[2]);
	vec3 gridPos5((indexX + 1) * meshSize + originPos[0], indexY       * meshSize + originPos[1], (indexZ + 1) * meshSize + originPos[2]);
	vec3 gridPos6((indexX + 1) * meshSize + originPos[0], (indexY + 1) * meshSize + originPos[1], (indexZ + 1) * meshSize + originPos[2]);
	vec3 gridPos7(indexX       * meshSize + originPos[0], (indexY + 1) * meshSize + originPos[1], (indexZ + 1) * meshSize + originPos[2]);



#pragma region list

	int edgeTable[256] = {
		0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
		0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
		0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
		0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
		0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
		0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
		0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
		0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
		0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
		0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
		0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
		0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
		0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
		0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
		0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
		0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
		0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };


	int triTable[256][16] =
	{ { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
	{ 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
	{ 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
	{ 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
	{ 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
	{ 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
	{ 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
	{ 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
	{ 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
	{ 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
	{ 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
	{ 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
	{ 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
	{ 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
	{ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
	{ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
	{ 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
	{ 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
	{ 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
	{ 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
	{ 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
	{ 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
	{ 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
	{ 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
	{ 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
	{ 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
	{ 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
	{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
	{ 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
	{ 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
	{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
	{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
	{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
	{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
	{ 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
	{ 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
	{ 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
	{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
	{ 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
	{ 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
	{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
	{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
	{ 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
	{ 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
	{ 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
	{ 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
	{ 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
	{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
	{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
	{ 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
	{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
	{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
	{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
	{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
	{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
	{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
	{ 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
	{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
	{ 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
	{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
	{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
	{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
	{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
	{ 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
	{ 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
	{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
	{ 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
	{ 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
	{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
	{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
	{ 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
	{ 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
	{ 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
	{ 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
	{ 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
	{ 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
	{ 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
	{ 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
	{ 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
	{ 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
	{ 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
	{ 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
	{ 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
	{ 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
	{ 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
	{ 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
	{ 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
	{ 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
	{ 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
	{ 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
	{ 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
	{ 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
	{ 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 } };
#pragma endregion
	/*
	Determine the index into the edge table which
	tells us which vertices are inside of the surface
	*/
	int cubeindex = 0;
	if (gridValue0 > isolevel) cubeindex |= 1;
	if (gridValue1 > isolevel) cubeindex |= 2;
	if (gridValue2 > isolevel) cubeindex |= 4;
	if (gridValue3 > isolevel) cubeindex |= 8;
	if (gridValue4 > isolevel) cubeindex |= 16;
	if (gridValue5 > isolevel) cubeindex |= 32;
	if (gridValue6 > isolevel) cubeindex |= 64;
	if (gridValue7 > isolevel) cubeindex |= 128;

	/* Cube is entirely in/out of the surface */
	std::vector<int> pointList_eachgrid;
	if (edgeTable[cubeindex] == 0)
	{
		temp->push_back(pointList_eachgrid);
		return(0);
	}
	/* Find the vertices where the surface intersects the cube */

#pragma region deal edge 
	int edge[12];

	if (edgeTable[cubeindex] & 1)//EDGE 0
	{
		if (indexY == 0 && indexZ == 0)
		{
			vec3 vertex = vertexInterp(isolevel, gridPos0, gridPos1, gridValue0, gridValue1);
			pointList->push_back(vertex);
			edge[0] = pointList->size();
			pointList_eachgrid.push_back(edge[0]);

		}
		else if (indexY == 0)
		{
			int vertexIndex = (indexZ - 1) * meshResX  * meshResY + indexX;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[0] = temp2[4];
			//edge[0] = temp[vertexIndex][4];
			pointList_eachgrid.push_back(edge[0]);
		}
		else if (indexZ == 0)
		{
			int vertexIndex = (indexY - 1) * meshResX + indexX;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[0] = temp2[2];
			//edge[0] = temp[vertexIndex][2];
			pointList_eachgrid.push_back(edge[0]);
		}
		else
		{
			int vertexIndex = (indexZ - 1) * meshResX  * meshResY + (indexY - 1) * meshResX + indexX;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[0] = temp2[6];
			//edge[0] = temp[vertexIndex][6];
			pointList_eachgrid.push_back(edge[0]);
		}
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 2)//EDGE 1
	{
		if (indexZ == 0)
		{
			vec3 vertex = vertexInterp(isolevel, gridPos1, gridPos2, gridValue1, gridValue2);
			//vec3 vertex = (gridPos1 + gridPos2) / 2.0f;
			pointList->push_back(vertex);
			edge[1] = pointList->size();
			pointList_eachgrid.push_back(edge[1]);
		}
		else
		{
			int vertexIndex = (indexZ - 1) * meshResX  * meshResY + indexY * meshResX + indexX;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[1] = temp2[5];
			//edge[1] = temp[vertexIndex][5];
			pointList_eachgrid.push_back(edge[1]);
		}
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 4)//EDGE 2
	{
		if (indexZ == 0)
		{
			vec3 vertex = vertexInterp(isolevel, gridPos2, gridPos3, gridValue2, gridValue3);
			//vec3 vertex = (gridPos2 + gridPos3) / 2.0f;
			pointList->push_back(vertex);
			edge[2] = pointList->size();
			pointList_eachgrid.push_back(edge[2]);
		}
		else
		{
			int vertexIndex = (indexZ - 1) * meshResX  * meshResY + indexY * meshResX + indexX;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[2] = temp2[6];
			//edge[2] = temp[vertexIndex][6];
			pointList_eachgrid.push_back(edge[2]);
		}
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 8)//EDGE 3
	{
		if (indexX == 0 && indexZ == 0)
		{
			vec3 vertex = vertexInterp(isolevel, gridPos3, gridPos0, gridValue3, gridValue0);
			//vec3 vertex = (gridPos3 + gridPos0) / 2.0f;
			pointList->push_back(vertex);
			edge[3] = pointList->size();
			pointList_eachgrid.push_back(edge[3]);
		}
		else if (indexX == 0)
		{
			int vertexIndex = (indexZ - 1) * meshResX  * meshResY + indexY * meshResX;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[3] = temp2[7];
			//edge[3] = temp[vertexIndex][7];
			pointList_eachgrid.push_back(edge[3]);
		}
		else if (indexZ == 0)
		{
			int vertexIndex = indexY * meshResX + indexX - 1;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[3] = temp2[1];
			//edge[3] = temp[vertexIndex][1];
			pointList_eachgrid.push_back(edge[3]);
		}
		else
		{
			int vertexIndex = (indexZ - 1) * meshResX  * meshResY + indexY * meshResX + indexX - 1;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[3] = temp2[5];
			//edge[3] = temp[vertexIndex][5];
			pointList_eachgrid.push_back(edge[3]);
		}
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 16)//EDGE 4
	{
		if (indexY == 0)
		{
			vec3 vertex = vertexInterp(isolevel, gridPos4, gridPos5, gridValue4, gridValue5);
			//vec3 vertex = (gridPos4 + gridPos5) / 2.0f;
			pointList->push_back(vertex);
			edge[4] = pointList->size();
			pointList_eachgrid.push_back(edge[4]);
		}
		else
		{
			int vertexIndex = indexZ * meshResX  * meshResY + (indexY - 1) * meshResX + indexX;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[4] = temp2[6];
			//edge[4] = temp[vertexIndex][6];
			pointList_eachgrid.push_back(edge[4]);
		}
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 32)//EDGE 5
	{
		vec3 vertex = vertexInterp(isolevel, gridPos5, gridPos6, gridValue5, gridValue6);
		//vec3 vertex = (gridPos5 + gridPos6) / 2.0f;
		pointList->push_back(vertex);
		edge[5] = pointList->size();
		pointList_eachgrid.push_back(edge[5]);
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 64)//EDGE 6
	{
		vec3 vertex = vertexInterp(isolevel, gridPos6, gridPos7, gridValue6, gridValue7);
		//vec3 vertex = (gridPos6 + gridPos7) / 2.0f;
		pointList->push_back(vertex);
		edge[6] = pointList->size();
		pointList_eachgrid.push_back(edge[6]);
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 128)//EDGE 7
	{
		if (indexX == 0)
		{
			vec3 vertex = vertexInterp(isolevel, gridPos7, gridPos4, gridValue7, gridValue4);
			//vec3 vertex = (gridPos7 + gridPos4) / 2.0f;
			pointList->push_back(vertex);
			edge[7] = pointList->size();
			pointList_eachgrid.push_back(edge[7]);
		}
		else
		{
			int vertexIndex = indexZ * meshResX  * meshResY + indexY * meshResX + indexX - 1;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[7] = temp2[5];
			//edge[7] = temp[vertexIndex][5];
			pointList_eachgrid.push_back(edge[7]);
		}
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 256)//EDGE 8
	{
		if (indexX == 0 && indexY == 0)
		{
			vec3 vertex = vertexInterp(isolevel, gridPos0, gridPos4, gridValue0, gridValue4);
			//vec3 vertex = (gridPos0 + gridPos4) / 2.0f;
			pointList->push_back(vertex);
			edge[8] = pointList->size();
			pointList_eachgrid.push_back(edge[8]);
		}
		else if (indexX == 0)
		{
			int vertexIndex = indexZ * meshResX  * meshResY + (indexY - 1) * meshResX;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[8] = temp2[11];
			//edge[8] = temp[vertexIndex][11];
			pointList_eachgrid.push_back(edge[8]);
		}
		else if (indexY == 0)
		{
			int vertexIndex = indexZ * meshResX  * meshResY + indexX - 1;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[8] = temp2[9];
			//edge[8] = temp[vertexIndex][9];
			pointList_eachgrid.push_back(edge[8]);
		}
		else
		{
			int vertexIndex = indexZ * meshResX  * meshResY + (indexY - 1) * meshResX + indexX - 1;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[8] = temp2[10];
			//edge[8] = temp[vertexIndex][10];
			pointList_eachgrid.push_back(edge[8]);
		}
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 512)//EDGE 9
	{
		if (indexY == 0)
		{
			vec3 vertex = vertexInterp(isolevel, gridPos1, gridPos5, gridValue1, gridValue5);
			//vec3 vertex = (gridPos1 + gridPos5) / 2.0f;
			pointList->push_back(vertex);
			edge[9] = pointList->size();
			pointList_eachgrid.push_back(edge[9]);
		}
		else
		{
			int vertexIndex = indexZ * meshResX  * meshResY + (indexY - 1) * meshResX + indexX;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[9] = temp2[10];
			//edge[9] = temp[vertexIndex][10];
			pointList_eachgrid.push_back(edge[9]);
		}
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 1024)//EDGE 10
	{
		vec3 vertex = vertexInterp(isolevel, gridPos2, gridPos6, gridValue2, gridValue6);
		//vec3 vertex = (gridPos2 + gridPos6) / 2.0f;
		pointList->push_back(vertex);
		edge[10] = pointList->size();
		pointList_eachgrid.push_back(edge[10]);
	}
	else
		pointList_eachgrid.push_back(-1);

	if (edgeTable[cubeindex] & 2048)//EDGE 11
	{
		if (indexX == 0)
		{
			vec3 vertex = vertexInterp(isolevel, gridPos3, gridPos7, gridValue3, gridValue7);
			//vec3 vertex = (gridPos3 + gridPos7) / 2.0f;
			pointList->push_back(vertex);
			edge[11] = pointList->size();
			pointList_eachgrid.push_back(edge[11]);
		}
		else
		{
			int vertexIndex = indexZ * meshResX  * meshResY + indexY * meshResX + indexX - 1;
			vector<int> temp2 = temp->at(vertexIndex);
			edge[11] = temp2[10];
			//edge[11] = temp[vertexIndex][10];
			pointList_eachgrid.push_back(edge[11]);
		}
	}
	else
		pointList_eachgrid.push_back(-1);

	temp->push_back(pointList_eachgrid);
#pragma endregion

	/* Create the triangle */
	int ntriang = 0;
	for (int i = 0; triTable[cubeindex][i] != -1; i += 3)
	{
		vector<int> triangle;
		triangle.push_back(edge[triTable[cubeindex][i]]);
		triangle.push_back(edge[triTable[cubeindex][i + 1]]);
		triangle.push_back(edge[triTable[cubeindex][i + 2]]);

		triangleList->push_back(triangle);
	}

	return(ntriang);

}


vec3 vertexInterp(double isolevel, vec3 p1, vec3 p2, double valueP1, double valueP2)
{
	vec3 interPt;

	double mu;

	if (abs(isolevel - valueP1) < 0.00001)
		return(p1);
	if (abs(isolevel - valueP2) < 0.00001)
		return(p2);
	if (abs(valueP1 - valueP2) < 0.00001)
		return(p1);
	mu = (isolevel - valueP1) / (valueP2 - valueP1);
	interPt[0] = p1[0] + mu * (p2[0] - p1[0]);
	interPt[1] = p1[1] + mu * (p2[1] - p1[1]);
	interPt[2] = p1[2] + mu * (p2[2] - p1[2]);

	return interPt;

}

void createObjFile(string name, vector<vec3> pointList, vector<vector<int>> triangleList)
{
	string filepath = objStorePath;
	filepath += name;

	ofstream outfile(filepath.c_str());

	for (unsigned int i = 0; i < pointList.size(); ++i)
		outfile << "v " << pointList[i][0] << " " << pointList[i][1] << " " << pointList[i][2] << std::endl;

	for (unsigned int i = 0; i < triangleList.size(); ++i)
		outfile << "f " << triangleList[i][0] << " " << triangleList[i][1] << " " << triangleList[i][2] << std::endl;

	outfile.close();
}

