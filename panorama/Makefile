CXX = g++
LDFLAGS = `pkg-config --libs opencv` -L/usr/local/opt/lapack/lib
CXXFLAGS = `pkg-config --cflags opencv` -I/usr/local/opt/lapack/include -std=c++0x

all : feature

feature : feature.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

run : feature
	./feature parrington/prtn01.jpg

clean :
	rm -f feature
