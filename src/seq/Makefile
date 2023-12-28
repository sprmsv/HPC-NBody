CXX = icpc
CXXFLAGS += -Wall -Wextra -pedantic -std=c++11 -O3
LD = ${CXX}
LDFLAGS = ${CXXFLAGS}
LDFLAGS += -lm

OBJS = nbody.o reader.o bruteforce.o barneshut.o

all: clean build remove_objs

build: $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o nbody

profiling: $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -pg -g -o nbody

remove_objs:
	rm *.o

clean:
	rm -Rf nbody *.o *~ gmon.out
