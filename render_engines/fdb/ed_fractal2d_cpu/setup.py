from setuptools import setup, Extension
from torch.utils import cpp_extension
import os,sys

extra_compiler_args_libs=[]
library_dirs_libs=[]
libraries_libs=[]
include_dirs_libs=[]

# OpenCV_ROOT=""
# if "OpenCV_ROOT" in os.environ:
#   if os.path.exists(os.path.join(os.environ["OpenCV_ROOT"],"include","opencv4")):
#     OpenCV_ROOT=os.environ["OpenCV_ROOT"]
#     extra_compiler_args_libs.append('-DOpenCV_FOUND')
#     include_dirs_libs.append(OpenCV_ROOT+"/include/opencv4")
#     library_dirs_libs.append(OpenCV_ROOT+"/lib64")
#     libraries_libs+=[
#       "opencv_core",
#       "opencv_features2d",
#       #"opencv_flann",
#       "opencv_highgui",
#       "opencv_imgcodecs",
#       "opencv_imgproc",
#     ]
#   else:
#     print("OpenCV_ROOT is not valid.")
#     sys.exit(0)
# elif len(OpenCV_ROOT)==0:
#   print("Please set OpenCV_ROOT environment variable")
#   sys.exit(0)

# flann_ROOT=""
# if "flann_ROOT" in os.environ:
#   if os.path.exists(os.path.join(os.environ["flann_ROOT"],"include","flann","flann.hpp")):
#     flann_ROOT=os.environ["flann_ROOT"]
#     extra_compiler_args_libs.append('-Dflann_FOUND')
#     include_dirs_libs.append(flann_ROOT+"/include")
#     library_dirs_libs.append(flann_ROOT+"/lib")
#     libraries_libs+=[
#       "flann",
#       "flann_cpp",
#       "lz4"
#     ]
#   else:
#     print("flann_ROOT is not valid.")
#     sys.exit(0)
# elif len(flann_ROOT)==0:
#   print("Please set flann_ROOT environment variable")
#   sys.exit(0)

setup(name='PyFractal2DRenderer',
    ext_modules=[cpp_extension.CppExtension(
        name='PyFractal2DRenderer', 
        sources=['PyFractal2DRenderer.cpp', 'Fractal2DRenderer_cpu.cpp'],
        include_dirs=[
          ]+include_dirs_libs,
        libraries=[
          ]+libraries_libs,
        library_dirs=[]+library_dirs_libs,
        runtime_library_dirs=[
          '/home/aac12092hz/reannotation/training/env1/lib64/python3.8/site-packages/torch/lib'
          ]+library_dirs_libs,
        #extra_compile_args=['-Wall -O0 -g -DDEBUG'],
        # extra_compile_args=['-fopenmp'],
        extra_compile_args=[
          '-ffast-math','-fopenmp',
          ]+extra_compiler_args_libs,
      )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
  )


