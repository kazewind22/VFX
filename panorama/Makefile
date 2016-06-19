export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
CXX = g++
LDFLAGS = `pkg-config --libs opencv` -L/usr/local/lib
CXXFLAGS = `pkg-config --cflags opencv` -I/usr/local/include -std=c++0x

all: feature

feature: feature.cpp
	$(CXX) -g $(CXXFLAGS) $(LDFLAGS) -o $@ $^

run: feature
	./feature square 85 16

clean:
	rm -f feature
