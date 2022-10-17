CXX?=g++
NVCC?=nvcc
CXXFLAGS?=-std=c++11 -Wall -pedantic -g
NVFLAGS?= --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -g

OPTIMISATION_LEVEL=-O3
TARGET_BINARY=parallel

# all targets
# The first rule in the Makefile is the default target that is made if 'make' is invoked with
# no parameters.  'all' is a dummy target that will make everything
default : all

## Dependencies

# all targets depend on the helper programs
$(TARGET_BINARY) : lattice.hpp util.hpp parallel.hpp util.o lattice.o

util.o: util.hpp util.cpp
lattice.o: lattice.hpp lattice.cpp

%.o : %.cu
	$(NVCC) $(NVFLAGS) $(OPTIMISATION_LEVEL) $(NVFLAGS_$(basename $<)) -c $< -o $@

% : %.cu
	$(NVCC) $(NVFLAGS) $(OPTIMISATION_LEVEL) $(NVFLAGS_$@) $(filter %.o %.cu, $^) $(LDFLAGS) $(LIB) -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(OPTIMISATION_LEVEL) $(CFLAGS_$(basename $<)) -c $< -o $@

% : %.cpp
	$(CXX) $(CXXFLAGS) $(OPTIMISATION_LEVEL) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIB) -o $@

all : $(TARGET_BINARY)

clean:
	rm -f $(TARGET_BINARY) *.o

.PHONY: clean default all
