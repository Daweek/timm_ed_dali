cmake_minimum_required(VERSION 3.10)
project(PyFractal2DRenderer LANGUAGES CUDA CXX)

set(CMAKE_THREAD_PREFER_PTHREAD ON)                                                                                                                                                                                                           
set(THREADS_PREFER_PTHREAD_FLAG ON)                                                                                                                                                                                                           
find_package(Threads REQUIRED)

set (CMAKE_CXX_STANDARD 14)

set(USE_SYSTEM_ARGAGG OFF CACHE BOOL "Using argagg on the system")
mark_as_advanced(USE_SYSTEM_ARGAGG)

#find_package(Python3 REQUIRED COMPONENTS Development)
#find_package(Torch REQUIRED)
find_package(OpenCV)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


if(USE_SYSTEM_ARGAGG)
    find_path(argagg_INCLUDE_DIRS "argagg/argagg.hpp" REQUIRED)
else()
    find_path(argagg_INCLUDE_DIRS "argagg/argagg.hpp" REQUIRED PATHS "${CMAKE_SOURCE_DIR}/third_party/argagg/include")
endif()
find_package(flann)

# set(CMAKE_CXX_FLAGS "-pthread -Wno-unused-result -Wsign-compare -fwrapv -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")

if(NOT disable_openmp)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        add_compile_options("${OpenMP_C_FLAGS}")
        add_link_options("${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

if(NOT disable_openmp)
    add_compile_options("-fopenmp")
    add_link_options("-fopenmp")
endif()

add_compile_options("-ffast-math")
add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
add_compile_options("-std=c++14")
# add_compile_options("-lopencv_imgcodecs")
# add_compile_options("-lopencv_core")
# add_compile_options("-lopencv_imgproc")
add_compile_options("-lopencv_highgui")



# # PyFractal2DRenderer
# # depends: Python3, Torch, OpenCV
# add_library(PyFractal2DRenderer MODULE
#     PyFractal2DRenderer.cpp
#     Fractal2DRenderer_cpu.cpp
# )
# target_link_libraries(PyFractal2DRenderer
#     Python3::Python
#     ${TORCH_LIBRARIES}
#     ${OpenCV_LIBS}
# )
# target_include_directories(PyFractal2DRenderer
#     PRIVATE ${TORCH_INCLUDE_DIRS}
#     PRIVATE ${OpenCV_INCLUDE_DIRS}
# )
# target_compile_options(PyFractal2DRenderer
#     PRIVATE ${TORCH_CXX_FLAGS}
# )

# # Fractal2DRenderer_cpu_test
# # depends: OpenCV, FLANN
# add_executable(Fractal2DRenderer_cpu_test
#     Fractal2DRenderer_cpu_test.cpp
#     Fractal2DRenderer_cpu.cpp
# )
# if(${OpenCV_FOUND})
#     target_link_libraries(Fractal2DCategorySearcher ${OpenCV_LIBS})
#     target_include_directories(Fractal2DCategorySearcher PRIVATE ${OpenCV_INCLUDE_DIRS})
#     target_compile_definitions(Fractal2DCategorySearcher PRIVATE OpenCV_FOUND)
# endif()
# if(${flann_FOUND})
#     target_link_libraries(Fractal2DCategorySearcher flann::flann)
#     target_compile_definitions(Fractal2DCategorySearcher PRIVATE flann_FOUND)
# endif()
# 

# # Fractal2DRenderer_gpu_test
# if(NOT DEFINED CMAKE_CUDA_STANDARD)
#     set(CMAKE_CUDA_STANDARD 14)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif()
# add_executable(Fractal2DRenderer_gpu_test
#     Fractal2DRenderer_gpu_test.cu
#     Fractal2DRenderer_gpu.cu
#     Fractal2DRenderer_cpu.cpp
# )
# set_target_properties(Fractal2DRenderer_gpu_test
#     PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_include_directories(Fractal2DRenderer_gpu_test PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(Fractal2DRenderer_gpu_test ${OpenCV_LIBS})
# #target_compile_options(Fractal2DRenderer_gpu_test PRIVATE "--expt-relaxed-constexpr")
# target_compile_options(Fractal2DRenderer_gpu_test PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
# )


# Fractal2DCategorySearcher
add_executable(Fractal2DCategorySearcher
    Fractal2DCategorySearcher.cpp
    Fractal2DRenderer_cpu.cpp
)
target_link_libraries(Fractal2DCategorySearcher
    ${OpenCV_LIBS}
)
target_include_directories(Fractal2DCategorySearcher
    PRIVATE ${argagg_INCLUDE_DIRS}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)


# Fractal2DCategorySearcherGrid
add_executable(Fractal2DCategorySearcherGrid
    Fractal2DCategorySearcherGrid.cpp
    Fractal2DRenderer_cpu.cpp
)
target_link_libraries(Fractal2DCategorySearcherGrid
    ${OpenCV_LIBS}
)
target_include_directories(Fractal2DCategorySearcherGrid
    PRIVATE ${argagg_INCLUDE_DIRS}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)

