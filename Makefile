CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -pthread

all: coordinator worker client

coordinator: coordinator.cpp model.cpp model.hpp
	$(CXX) $(CXXFLAGS) -o coordinator coordinator.cpp model.cpp

worker: worker.cpp model.cpp model.hpp
	$(CXX) $(CXXFLAGS) -o worker worker.cpp model.cpp

client: client.cpp
	$(CXX) $(CXXFLAGS) -o client client.cpp

clean:
	rm -f coordinator worker client

.PHONY: all clean
