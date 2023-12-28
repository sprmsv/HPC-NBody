CXX = mpiicpc
CXXFLAGS += -Wall -Wextra -pedantic -std=c++11 -O3 -no-multibyte-chars
LD = ${CXX}
LDFLAGS = ${CXXFLAGS}
LDFLAGS += -lm

OBJS = nbody.o reader.o bruteforce.o barneshut.o

all: clean build remove_objs

build: $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o nbody

remove_objs:
	rm *.o

clean:
	rm -Rf nbody *.o *~ gmon.out
