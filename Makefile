CXX      = g++
CXXFLAGS = -std=c++17 -Wall -O2 -pthread

all: coordinator worker client fed_coordinator

coordinator: coordinator.cpp model.cpp model.hpp
	$(CXX) $(CXXFLAGS) -o coordinator coordinator.cpp model.cpp

worker: worker.cpp model.cpp model.hpp
	$(CXX) $(CXXFLAGS) -o worker worker.cpp model.cpp

client: client.cpp
	$(CXX) $(CXXFLAGS) -o client client.cpp

fed_coordinator: fed_coordinator.cpp
	$(CXX) $(CXXFLAGS) -o fed_coordinator fed_coordinator.cpp

clean:
	rm -f coordinator worker client fed_coordinator

.PHONY: all clean