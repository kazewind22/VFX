CXX=g++
LDFLAGS=`pkg-config --libs opencv`
CXXFLAGS=`pkg-config --cflags opencv`

all : main

main : main.cpp
	    $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean :
	    rm -f main
