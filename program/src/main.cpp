/**
 * @file      main.cpp
 * @brief     Example N-body simulation for CIS 565
 * @authors   Liam Boone, Kai Ninomiya
 * @date      2013-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"

// ================
// Configuration
// ================

#define VISUALIZE 1

unsigned int width = 128;
unsigned int height = 72;
unsigned int N_FOR_VIS;

/**
 * C main function.
 */
int main(int argc, char* argv[]) {
    projectName = "565 Final Project : Complex Procedural Terrain";

    if (init(argc, argv)) {
        mainLoop();
        return 0;
    } else {
        return 1;
    }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
 * Initialization of CUDA and GLFW.
 */
bool init(int argc, char **argv) {
    // Set window title to "Student Name: [SM 2.0] GPU Name"
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout
                << "Error: GPU device number is greater than the number of devices!"
                << " Perhaps a CUDA-capable GPU is not installed?"
                << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
    deviceName = ss.str();

    // Window setup stuff
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        std::cout
            << "Error: Could not initialize GLFW!"
            << " Perhaps OpenGL 3.3 isn't available?"
            << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize drawing state
    initVAO();

    // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);

    cudaGLRegisterBufferObject(planetVBO);

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, glm::vec3(0), glm::vec3(0, 0, 1));

    projection = projection * view;

    initShaders(program);

    glEnable(GL_DEPTH_TEST);

    return true;
}

void initVAO() {
	//paramter = runCUDA(time=glfwTime());
	//{bodies,bindexes} = Mube(parameter);

	GLfloat *nbodies = new GLfloat[4 * (N_FOR_VIS + 1)];
    GLfloat *bodies = new GLfloat[4 * (N_FOR_VIS + 1)];
    GLuint *bindices = new GLuint[N_FOR_VIS + 1];

	for (int i = 0; i < N_FOR_VIS + 1; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;

		nbodies[4 * i + 0] = 0.0f;
		nbodies[4 * i + 1] = 1.0f;
		nbodies[4 * i + 2] = 0.0f;
		nbodies[4 * i + 3] = 0.0f;
		
		bindices[i] = i;
	}

    glGenVertexArrays(1, &planetVAO);
    glGenBuffers(1, &planetVBO);
	glGenBuffers(1, &planetIBO);
	glGenBuffers(1, &planetNBO);

    glBindVertexArray(planetVAO);

    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
	glBufferData(GL_ARRAY_BUFFER, 4 * N_FOR_VIS * sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);
	//glBufferData(GL_ARRAY_BUFFER, tribodies.size() * sizeof(glm::vec4), tribodies.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, planetNBO);
	glBufferData(GL_ARRAY_BUFFER, 4 * N_FOR_VIS * sizeof(GLfloat), nbodies, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, N_FOR_VIS * sizeof(GLuint), bindices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(positionLocation);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glEnableVertexAttribArray(normalLocation);
	glVertexAttribPointer((GLuint)normalLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindVertexArray(0);

    delete[] bodies;
	delete[] nbodies;
    delete[] bindices;
}

void initShaders(GLuint * program) {
    GLint location;	

    program[PROG_PLANET] = glslUtility::createProgram(
                     "shaders/planet.vert.glsl",
                     NULL,
                     "shaders/planet.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_PLANET]);

    if ((location = glGetUniformLocation(program[PROG_PLANET], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_PLANET], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
}

//====================================
// Main loop
//====================================
void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer
	if (camchanged) {
		Camera &cam = renderState->camera;
		glm::vec3 v = cam.view;
		glm::vec3 u = cam.up;
		glm::vec3 r = glm::cross(v, u);
		glm::mat4 rotmat = glm::rotate(theta, r) * glm::rotate(phi, u);
		cam.view = glm::vec3(rotmat * glm::vec4(v, 0.f));
		cam.up = glm::vec3(rotmat * glm::vec4(u, 0.f));
		cam.position += cammove.x * r + cammove.y * u + cammove.z * v;
		theta = phi = 0;
		cammove = glm::vec3();
		camchanged = false;
	}

    float4 *dptr = NULL;
    float *dptrvert = NULL;
    cudaGLMapBufferObject((void**)&dptrvert, planetVBO);

	// Initialize N-body simulation
	N_FOR_VIS = width*height;
	float *hst_height = new float[N_FOR_VIS];

    // execute the kernel
	MapGen(hst_height, N_FOR_VIS);
	//ofstream outfile(filepath.c_str());
	//outfile << 1 << endl;
	//for (int i = 0; i < width; i++)
	//	for (int j = 0; j < height; j++){
	//	int index = i*height + j;
	//	//hst_height[index] += height*perl.Get(i, j);
	//	outfile << i << " " << hst_height[index] << " " << j << endl;
	//	}
	//outfile << "CIS565" << endl;
	//outfile.close();
	MCube(hst_height, width, height);
	cout << "OVER" << endl;
#if VISUALIZE
    //Terrain::copyPlanetsToVBO(dptrvert);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(planetVBO);
}

void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        frame++;
        double time = glfwGetTime();

        if (time - timebase > 1.0) {
            fps = frame / (time - timebase);
            timebase = time;
            frame = 0;
        }

		//runCUDA();

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << deviceName;
        glfwSetWindowTitle(window, ss.str().c_str());

		glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#if VISUALIZE
        glUseProgram(program[PROG_PLANET]);
        glBindVertexArray(planetVAO);
        glPointSize(5.0f);
        glDrawElements(GL_POINTS, 3, GL_UNSIGNED_INT, 0);
        glPointSize(1.0f);

        glUseProgram(0);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
#endif
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}


void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:glfwSetWindowShouldClose(window, GL_TRUE); break;
		case GLFW_KEY_SPACE: break;
		case GLFW_KEY_DOWN:  camchanged = true; theta = -0.1f; break;
		case GLFW_KEY_UP:    camchanged = true; theta = +0.1f; break;
		case GLFW_KEY_RIGHT: camchanged = true; phi = -0.1f; break;
		case GLFW_KEY_LEFT:  camchanged = true; phi = +0.1f; break;
		case GLFW_KEY_A:     camchanged = true; cammove -= glm::vec3(.1f, 0, 0); break;
		case GLFW_KEY_D:     camchanged = true; cammove += glm::vec3(.1f, 0, 0); break;
		case GLFW_KEY_W:     camchanged = true; cammove += glm::vec3(0, 0, .1f); break;
		case GLFW_KEY_S:     camchanged = true; cammove -= glm::vec3(0, 0, .1f); break;
		case GLFW_KEY_R:     camchanged = true; cammove += glm::vec3(0, .1f, 0); break;
		case GLFW_KEY_F:     camchanged = true; cammove -= glm::vec3(0, .1f, 0); break;
		case GLFW_KEY_L:     runCUDA();	/*initVAO();*/ break;
		}
	}
}