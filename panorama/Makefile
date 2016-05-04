CXX = g++
LDFLAGS = `pkg-config --libs opencv`
CXXFLAGS = `pkg-config --cflags opencv` -std=c++0x

all : feature

feature : feature.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

run : feature
	./feature parrington

clean :
	rm -f feature
