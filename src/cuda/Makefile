CXX=nvcc
CXXFLAGS+=-Xcompiler="-Wall -Wextra -Werror" -std c++11 -O3
LD = ${CXX}
LDFLAGS+=-lm $(CXXFLAGS)

OBJS = nbody.o reader.o bruteforce.o

all: clean build remove_objs

build: $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o nbody

bruteforce.o: bruteforce.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

remove_objs:
	rm *.o

clean:
	rm -Rf nbody *.o *~ gmon.out
