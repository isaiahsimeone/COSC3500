CXX=g++
CXXFLAGS=-Wall -pedantic -g -Ofast
OBJS=lattice.o serial.o util.o
BIN=serial
DEPS=serial.hpp lattice.hpp util.hpp

all: $(BIN)

serial: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(BIN)

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o
