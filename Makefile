.PHONY: clean clean-output

#Main compiler
CXX = g++

#Modules
MODULES_SRC = $(wildcard modules/impls/*.cpp)
MODULES := $(patsubst %.cpp,%.o,$(MODULES_SRC))

#LodePNG implementation
LODE_SRC := include/lodepng/lodepng.cpp modules/impls/image_codecs/image_codec_lodepng.cpp
LODE :=  $(patsubst %.cpp,%.o,$(LODE_SRC))

#CUDA implementation
CUDA_MODULES_SRC := $(wildcard modules/impls/**/*.cu)
CUDA_MODULES := $(patsubst %.cu,%.o,$(CUDA_MODULES_SRC))
LDFLAGS_CUDA := -L/opt/cuda/lib
LDLIBS_CUDA := -lcuda -lcudart -lnvjpeg_static -lculibos

#General arguments
LDFLAGS := -I modules/ -I include/lodepng/
CXXFLAGS := $(LDFLAGS) $(MODULES) Program.o -g

#Compile with LodePNG implementation (link object files)
graphics-lode.out: $(MODULES) $(LODE) Program.o
	$(CXX) $(CXXFLAGS) $(LODE) -Wall -Wextra -pedantic -O0 -o graphics-lode.out   

#Compile with CUDA implementation
graphics-cuda.out: $(MODULES) $(CUDA_MODULES) Program.o
	$(CXX) $(CXXFLAGS) $(CUDA_MODULES) $(LDFLAGS_CUDA) $(LDLIBS_CUDA) -Wall -Wextra -pedantic -O0 -o graphics-cuda.out 

#Compile CUDA implementation (target that invokes if *.o with *.cu source is required by other targets)
%.o: %.cu
	nvcc $(LDFLAGS) --debug --device-debug -o $@ -c $^

#Target that invokes if *.o file with *.cpp source is required by other targets
%.o: %.cpp
	$(CXX) $(LDFLAGS) -Wall -Wextra -pedantic -O0 -g -o $@ -c $^

#Clean build files
clean:
	rm -f $(MODULES) $(LODE) $(CUDA_MODULES) Program.o graphics-lode.out graphics-cuda.out

#Clean program output files
clean-output:
	rm -f $(wildcard output/*)
