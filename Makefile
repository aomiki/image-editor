.PHONY: clean clean-output

#Main compiler
CXX = g++

OPTIMIZATION_FLAGS_RELEASE= -march=native -Ofast
OPTIMIZATION_FLAGS_DEBUG= -O0 -g
OPTIMIZATION_FLAGS=$(OPTIMIZATION_FLAGS_RELEASE)

NV_OPTIMIZATION_FLAGS_RELEASE= -use_fast_math -v
NV_OPTIMIZATION_FLAGS_DEBUG= --debug --device-debug --cudart shared
NV_OPTIMIZATION_FLAGS= $(NV_OPTIMIZATION_FLAGS_RELEASE)

INC=include
SRC=src
MODULES_DIR=$(SRC)/modules

#Modules
MODULES_SRC = $(wildcard $(MODULES_DIR)/impls/*.cpp)
MODULES := $(patsubst %.cpp,%.o,$(MODULES_SRC))

#Command parser
#CMD_PARSER_SRC = $(MODULES_DIR)/cmd_parser.cpp
#CMD_PARSER_OBJ = $(MODULES_DIR)/cmd_parser.o

#Modules that are used both by LODE implementation and directly in CUDA kernels
MODULES_SHARED_SRC = $(wildcard $(MODULES_DIR)/impls_shared/*.cpp)
MODULES_SHARED_CPP := $(patsubst %.cpp,%.o,$(MODULES_SHARED_SRC))
MODULES_SHARED_CUDA := $(patsubst %.cpp,%.cu.o,$(MODULES_SHARED_SRC))
MODULES_SHARED_CUDA_LINKED := $(patsubst %,%.linked.o,$(MODULES_SHARED_SRC))

QT_DIR=$(shell qmake6 -query QT_HOST_LIBEXECS)
QT_HEADERS_DIR=$(shell qmake6 -query QT_INSTALL_HEADERS)

GUI_DIR=$(SRC)/gui
#GUI
GUI_SRC=$(GUI_DIR)/mainwindow.cpp $(GUI_DIR)/moc_mainwindow.cpp
GUI=$(patsubst %.cpp,%.o,$(GUI_SRC))

LDFLAGS_GUI=-I/$(QT_HEADERS_DIR) -I$(QT_HEADERS_DIR)/QtGui -I$(QT_HEADERS_DIR)/QtCore -I$(QT_HEADERS_DIR)/QtWidgets -I/usr/lib/qt6/mkspecs/linux-g++ -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB -fPIC
LD_LIBS_GUI=-lQt6Core -lQt6Gui -lQt6Widgets

#LodePNG implementation
LODE_SRC := $(wildcard $(INC)/lodepng/lodepng.cpp $(MODULES_DIR)/impls_cpu/*.cpp)
LODE :=  $(patsubst %.cpp,%.o,$(LODE_SRC))

#CUDA implementation
CUDA_MODULES_SRC := $(wildcard $(MODULES_DIR)/impls_hw_accel/cuda/*.cu)
CUDA_MODULES := $(patsubst %.cu,%.o,$(CUDA_MODULES_SRC))
CUDA_MODULES_LINKED := $(patsubst %,%.linked.o,$(CUDA_MODULES_SRC))

LDFLAGS_CUDA := -I/opt/cuda/include/ -L/opt/cuda/lib
LDLIBS_CUDA := -lcuda -lcudart -lnvjpeg_static -lculibos -lcudart -lcudadevrt

#OpenCL implementation
OPENCL_MODULES_SRC := $(wildcard $(MODULES_DIR)/impls_hw_accel/opencl/*.cpp)
OPENCL_MODULES := $(patsubst %.cpp,%.o,$(OPENCL_MODULES_SRC))
OPENCL_INCLUDE := $(INC)/CL
OPENCL_LIBS := -L/usr/lib/x86_64-linux-gnu -lOpenCL

#General arguments
LDFLAGS := -I$(MODULES_DIR)/ -I$(INC)/lodepng/ -I$(INC)/ -I$(OPENCL_INCLUDE)
CXXFLAGS := $(LDFLAGS) $(LDFLAGS_GUI) $(MODULES) $(GUI) $(SRC)/Program.o -g \-ILRs/ -I/usr/include/openblas -fopenmp -lopenblas


#Compile with LodePNG implementation (link object files)
graphics-lode.out: HW_ACCEL = LODE_IMPL
graphics-lode.out: $(MODULES) $(MODULES_SHARED_CPP) $(LODE) $(GUI) $(SRC)/Program.o $(CMD_PARSER_OBJ)
	$(CXX) $(CXXFLAGS) $(MODULES_SHARED_CPP) $(LODE) $(CMD_PARSER_OBJ) $(LD_LIBS_GUI) -D$(HW_ACCEL) -Wall -Wextra -pedantic $(OPTIMIZATION_FLAGS) -o graphics-lode.out -lboost_program_options

#Compile with CUDA implementation
graphics-cuda.out: HW_ACCEL = CUDA_IMPL
graphics-cuda.out: $(MODULES) $(MODULES_SHARED_CUDA) $(CUDA_MODULES) $(GUI) $(SRC)/Program.o $(CMD_PARSER_OBJ)
	nvcc $(LDFLAGS) -arch=native -dlink -o cuda_modules_linked.o $(MODULES_SHARED_CUDA) $(CUDA_MODULES) $(LDLIBS_CUDA)
	$(CXX) $(CXXFLAGS) $(MODULES_SHARED_CUDA) cuda_modules_linked.o $(CUDA_MODULES) $(CMD_PARSER_OBJ) $(LDFLAGS_CUDA) $(LD_LIBS_GUI) $(LDLIBS_CUDA) -D$(HW_ACCEL) -Wall -Wextra -pedantic $(OPTIMIZATION_FLAGS) -o graphics-cuda.out -lboost_program_options

#Compile with OpenCL implementation
graphics-opencl.out: HW_ACCEL = OPENCL_IMPL
graphics-opencl.out: $(MODULES) $(MODULES_SHARED_CPP) $(LODE) $(OPENCL_MODULES) $(GUI) $(SRC)/Program.o
	$(CXX) $^ -D$(HW_ACCEL) -Wall -Wextra -pedantic $(OPTIMIZATION_FLAGS) -o graphics-opencl.out -lboost_program_options $(LD_LIBS_GUI) $(OPENCL_LIBS)

$(MODULES_DIR)/impls_shared/%.cu.o: $(MODULES_DIR)/impls_shared/%.cpp
	nvcc $(LDFLAGS) -arch=native -x cu -rdc=true $(NV_OPTIMIZATION_FLAGS) -o $@ -c $^

#Compile CUDA implementation (target that invokes if *.o with *.cu source is required by other targets)
%.o: %.cu
	nvcc $(LDFLAGS) -arch=native -rdc=true $(NV_OPTIMIZATION_FLAGS) -o $@ -c $^

$(GUI_DIR)/moc_mainwindow.cpp: $(GUI_DIR)/mainwindow.h $(GUI_DIR)/ui_mainwindow.h
	$(QT_DIR)/moc -I modules/ -I include/lodepng/ $< -o $@

$(GUI_DIR)/ui_mainwindow.h: $(GUI_DIR)/mainwindow.ui
	$(QT_DIR)/uic $(GUI_DIR)/mainwindow.ui -o $(GUI_DIR)/ui_mainwindow.h 

$(GUI_DIR)/mainwindow.o: $(GUI_DIR)/ui_mainwindow.h

#Target that invokes if *.o file with *.cpp source is required by other targets
%.o: %.cpp
	$(CXX) $(LDFLAGS) $(LDFLAGS_CUDA) $(LDFLAGS_GUI) $(LDLIBS_CUDA) $(LD_LIBS_GUI) -D$(HW_ACCEL) -Wall -Wextra -pedantic $(OPTIMIZATION_FLAGS) -o $@ -c $<

#Clean build files
clean:
	rm -f $(MODULES) $(MODULES_SHARED_CUDA) $(MODULES_SHARED_CUDA_LINKED) $(LODE) $(CUDA_MODULES) $(CUDA_MODULES_LINKED) $(SRC)/Program.o $(GUI_DIR)/mainwindow.o $(GUI_DIR)/moc_mainwindow.cpp $(GUI_DIR)/ui_mainwindow.h graphics-lode.out graphics-cuda.out graphics-opencl.out

#Clean program output files
clean-output:
	rm -f $(wildcard output/*)