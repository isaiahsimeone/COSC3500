CXX=g++
CXXFLAGS=-Wall -pedantic -g -O3 
OBJS=grid.o serial.o
BIN=serial
DEPS=serial.hpp grid.hpp

all: $(BIN)

serial: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(BIN)

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o
