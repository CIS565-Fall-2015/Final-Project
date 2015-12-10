set(SOURCE_FILES
    "glslUtility.hpp"
    "glslUtility.cpp"
    "utilityCore.hpp"
    "utilityCore.cpp"
    "perlin.h"
    "perlin.cu"
    "kernel.h"
    "kernel.cu"
	"scene.h"
	"scene.cpp"
	"MCube.h"
	"MCube.cpp"
	"makelevelset3.h"
	"makelevelset3.cpp"
	"util.h"
	"vecL.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
