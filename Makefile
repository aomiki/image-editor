.PHONY: clean clean-output

#Main compiler
CXX = g++

INC=include
SRC=src
MODULES_DIR=$(SRC)/modules

#Modules
MODULES_SRC = $(wildcard $(MODULES_DIR)/impls/*.cpp)
MODULES := $(patsubst %.cpp,%.o,$(MODULES_SRC))

#Command parser
CMD_PARSER_SRC = $(MODULES_DIR)/cmd_parser.cpp
CMD_PARSER_OBJ = $(MODULES_DIR)/cmd_parser.o

#Modules that are used both by LODE implementation and directly in CUDA kernels
MODULES_SHARED_SRC = $(wildcard $(MODULES_DIR)/impls_shared/*.cpp)
MODULES_SHARED_CPP := $(patsubst %.cpp,%.o,$(MODULES_SHARED_SRC))
MODULES_SHARED_CUDA := $(patsubst %.cpp,%.cu.o,$(MODULES_SHARED_SRC))
MODULES_SHARED_CUDA_LINKED := $(patsubst %,%.linked.o,$(MODULES_SHARED_SRC))

#LodePNG implementation
LODE_SRC := $(wildcard $(INC)/lodepng/lodepng.cpp $(MODULES_DIR)/impls_cpu/*.cpp)
LODE :=  $(patsubst %.cpp,%.o,$(LODE_SRC))

#CUDA implementation
CUDA_MODULES_SRC := $(wildcard $(MODULES_DIR)/impls_hw_accel/*.cu)
CUDA_MODULES := $(patsubst %.cu,%.o,$(CUDA_MODULES_SRC))
CUDA_MODULES_LINKED := $(patsubst %,%.linked.o,$(CUDA_MODULES_SRC))

LDFLAGS_CUDA := -I/opt/cuda/include/ -L/opt/cuda/lib
LDLIBS_CUDA := -lcuda -lcudart -lnvjpeg_static -lculibos -lcudart -lcudadevrt

#General arguments
LDFLAGS := -I $(MODULES_DIR)/ -I $(INC)/lodepng/
CXXFLAGS := $(LDFLAGS) $(MODULES) $(SRC)/Program.o -g

#Compile with LodePNG implementation (link object files)
graphics-lode.out: HW_ACCEL = LODE_IMPL
graphics-lode.out: $(MODULES) $(MODULES_SHARED_CPP) $(LODE) $(SRC)/Program.o $(CMD_PARSER_OBJ)
	$(CXX) $(CXXFLAGS) $(MODULES_SHARED_CPP) $(LODE) $(CMD_PARSER_OBJ) -D$(HW_ACCEL) -Wall -Wextra -pedantic -O0 -o graphics-lode.out -lboost_program_options

#Compile with CUDA implementation
graphics-cuda.out: HW_ACCEL = CUDA_IMPL
graphics-cuda.out: $(MODULES) $(MODULES_SHARED_CUDA) $(CUDA_MODULES) $(SRC)/Program.o
	nvcc $(LDFLAGS) -dlink -o cuda_modules_linked.o $(MODULES_SHARED_CUDA) $(CUDA_MODULES) $(LDLIBS_CUDA)
	$(CXX) $(CXXFLAGS) $(MODULES_SHARED_CUDA) cuda_modules_linked.o $(CUDA_MODULES) $(LDFLAGS_CUDA) $(LDLIBS_CUDA) -D$(HW_ACCEL) -Wall -Wextra -pedantic -O0 -o graphics-cuda.out -lboost_program_options

$(MODULES_DIR)/impls_shared/%.cu.o: $(MODULES_DIR)/impls_shared/%.cpp
	nvcc $(LDFLAGS) -x cu -rdc=true --debug --device-debug --cudart shared -o $@ -c $^

#Compile CUDA implementation (target that invokes if *.o with *.cu source is required by other targets)
%.o: %.cu
	nvcc $(LDFLAGS) -rdc=true --debug --device-debug --cudart shared -o $@ -c $^

#Target that invokes if *.o file with *.cpp source is required by other targets
%.o: %.cpp
	$(CXX) $(LDFLAGS) $(LDFLAGS_CUDA) $(LDLIBS_CUDA) -D$(HW_ACCEL) -Wall -Wextra -pedantic -O0 -g -o $@ -c $^

#Clean build files
clean:
	rm -f $(MODULES) $(MODULES_SHARED_CUDA) $(MODULES_SHARED_CUDA_LINKED) $(LODE) $(CUDA_MODULES) $(CUDA_MODULES_LINKED) $(SRC)/Program.o graphics-lode.out graphics-cuda.out

#Clean program output files
clean-output:
	rm -f $(wildcard output/*)
