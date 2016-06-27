export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
CXX = g++
LDFLAGS = `pkg-config --libs opencv` -L/usr/local/lib
CXXFLAGS = `pkg-config --cflags opencv` -I/usr/local/include -std=c++0x

all: panorama

panorama: panorama.cpp
	$(CXX) -g $(CXXFLAGS) $(LDFLAGS) -o $@ $^

run: panorama
# ./panorama -m 0.85 -p 0.8 square
	./panorama --do_rectangling --no_stitching square

runp: panorama
# ./panorama -m 0.85 -p 0.8 square
	./panorama --do_rectangling --no_stitching parrington

clean:
	rm -f panorama
