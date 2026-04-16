CXX      = g++
CXXFLAGS = -std=c++17 -Wall -O2 -pthread
SRC      = src

all: coordinator worker client fed_coordinator

coordinator: $(SRC)/coordinator.cpp $(SRC)/model.cpp $(SRC)/model.hpp
	$(CXX) $(CXXFLAGS) -I$(SRC) -o coordinator $(SRC)/coordinator.cpp $(SRC)/model.cpp

worker: $(SRC)/worker.cpp $(SRC)/model.cpp $(SRC)/model.hpp
	$(CXX) $(CXXFLAGS) -I$(SRC) -o worker $(SRC)/worker.cpp $(SRC)/model.cpp

client: $(SRC)/client.cpp
	$(CXX) $(CXXFLAGS) -o client $(SRC)/client.cpp

fed_coordinator: $(SRC)/fed_coordinator.cpp
	$(CXX) $(CXXFLAGS) -o fed_coordinator $(SRC)/fed_coordinator.cpp

clean:
	rm -f coordinator worker client fed_coordinator

.PHONY: all clean