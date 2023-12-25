CXX=g++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -pedantic -O3
LDFLAGS+=-lm -lm

OBJS_NBODY=nbody.o reader.o nbody_barneshut.o

all: nbody

nbody: $(OBJS_NBODY)
	$(LD) $(OBJS_NBODY) $(LDFLAGS) -o $@

clean:
	rm -Rf nbody *.o *~  gmon.out
