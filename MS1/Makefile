CPP=g++
CPPFLAGS=-Wall -pedantic -g
OBJS=grid.o serial.o
BIN=serial
DEPS=serial.hpp grid.hpp

all: $(BIN)

serial: $(OBJS)
	$(CPP) $(CPPFLAGS) $(OBJS) -o $(BIN)

%.o: %.cpp $(DEPS)
	$(CPP) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
