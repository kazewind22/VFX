CXX = g++
LDFLAGS = `pkg-config --libs opencv` -L/usr/local/opt/lapack/lib
CXXFLAGS = `pkg-config --cflags opencv` -I/usr/local/opt/lapack/include -std=c++0x

all : main

main : main.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

opencv_hdr : open_hdr.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

run : main
	./main ladder

clean :
	rm -f main
