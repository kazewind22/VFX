CXX = g++
LDFLAGS = `pkg-config --libs opencv` -L/usr/local/opt/lapack/lib
CXXFLAGS = `pkg-config --cflags opencv` -I/usr/local/opt/lapack/include -std=c++0x

all : mtb tonemap

mtb : mtb.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

tonemap : tonemap.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

run_mtb : mtb
	./mtb library

run_tonemap : tonemap
	./tonemap library/library.hdr

clean :
	rm -f mtb tonemap
