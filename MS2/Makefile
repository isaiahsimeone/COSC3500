CXX=g++
CXXFLAGS=-Wall -pedantic -g -O3 -std=c++11
OBJS=lattice.o parallel.o util.o
BIN=parallel
DEPS=parallel.hpp lattice.hpp util.hpp

all: $(BIN)

serial: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(BIN)

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o
