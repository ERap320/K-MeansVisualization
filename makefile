all: kmeans.cpp
	g++ kmeans.cpp -o kmeans-gui -std=gnu++17 -fopenmp -lsfml-graphics -lsfml-window -lsfml-system
	chmod +x kmeans-gui
