CXX = g++
CXXFLAGS += -Wall -Wextra -pedantic -std=c++11 -O3
NVCC = nvcc
NVCCFLAGS += -O3
LD = ${CXX}
LDFLAGS += -lm
# LDFLAGS += $(NVCCFLAGS)

OBJS = nbody.o reader.o bruteforce.o barneshut.o

all: build remove_objs

build: $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o nbody

# nbody-code: nbody.o nbody_bruteforce.o reader.o
# 	$(NVCC) $^ $(LDFLAGS) -o $@

# %.o:%.cu
# 	$(NVCC) $(NVCCFLAGS) -c $< -o $@

remove_objs:
	rm *.o

clean:
	rm -Rf nbody *.o *~ gmon.out
