CPP=g++
CPPFLAGS=-Wall -pedantic -g
OBJS=grid.o serial.o
BIN=serial
DEPS=serial.hpp grid.hpp

all: $(BIN)

serial: $(OBJS)
	$(CPP) $(CPPFLAGS) $(OBJS) -o $(BIN)
	if ! [ -d "./output_img" ]; then mkdir output_img; fi

%.o: %.cpp $(DEPS)
	$(CPP) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -rf ./output_img
