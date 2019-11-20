#include <iostream>
#include <cstdlib> //For rand
#include <cmath> //For pow and sqrt
#include <vector>
#include <string>
#include <omp.h>
#include <chrono> //For time keeping
using namespace std;
using std::cout;

//SFML graphics library
#include <SFML/Graphics.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/Font.hpp>

//K-Means default values
#define DIMENSION 600
#define POINTS_NUM 70000
#define CLUSTERS_NUM 10
#define SEED 42

#define COLORS_NUM 10
#define MEAN_SIZE 10 //Size of the red dot used to show the mean's position
#define POINT_SIZE 2 //Size of the dot used to show points

#define MAX_ITERATIONS 500 //Default value for maximum iterations

//Array of colors used for different clustersX
const sf::Color clusterColors[] = {sf::Color::Blue, sf::Color::Cyan, sf::Color::Green, sf::Color::Magenta, sf::Color::Yellow, sf::Color(255,127,0,255), sf::Color(192,192,192,255), sf::Color(150,60,0,255), sf::Color::White, sf::Color(0,120,40,255) };

//Point definition
struct point
{
	long x;
	long y;
	sf::RectangleShape shape; //Component used to diplay the point
};

inline double distance(point a, point b)
{
	return sqrt( pow(a.x - b.x, 2) + pow(a.y - b.y, 2) );
}

//K-Means logic
void initialize(point*& points, point*& means, unsigned long*& clustersX, unsigned long*& clustersY, unsigned long*& clustersCardinality, int windowDimension, long pointsNumber, long clustersNumber, int seed)
{
	srand(seed);
	
	//Allocate points and means memory
	points = new point[pointsNumber];
	means = new point[clustersNumber];
	clustersX = new unsigned long[clustersNumber];
	clustersY = new unsigned long[clustersNumber];
	clustersCardinality = new unsigned long[clustersNumber];

	//Initialize points
	for (long i = 0; i < pointsNumber; i++)
	{
		points[i] = { rand() % windowDimension, rand() % windowDimension };
		points[i].shape.setSize(sf::Vector2f(POINT_SIZE, POINT_SIZE));
		points[i].shape.setPosition(sf::Vector2f(points[i].x, points[i].y));
	}

	//Initialize means
	for (long i = 0; i < clustersNumber; i++)
	{
		means[i] = { rand() % windowDimension, rand() % windowDimension };
		means[i].shape.setSize(sf::Vector2f(MEAN_SIZE, MEAN_SIZE));
		means[i].shape.setFillColor(sf::Color::Red);
		means[i].shape.setPosition(sf::Vector2f(means[i].x, means[i].y));

		cout << "Mean " << i << ": " << means[i].x << "," << means[i].y << endl;
	}

	cout << endl;
}

//Executes one iteration of the algorithm and returns true if the position of the means
//changed (the execution has to continue), false if it stayed the same (the algorithm converged)
bool executeIteration(point* points, point* means, unsigned long* clustersX, unsigned long* clustersY, unsigned long* clustersCardinality, long pointsNumber, long clustersNumber, sf::RenderWindow& window)
{
	//Nearest mean variables
	long nearestMean;
	double minDistance;
	double currDistance;

	//Mean calculation variables
	long xRes, yRes;

	//Clearing cluster sums
	#pragma omp parallel for
	for (long j = 0; j < clustersNumber; j++)
	{
		clustersX[j] = 0;
		clustersY[j] = 0;
		clustersCardinality[j] = 0;
	}

	//Point to cluster assignment
	#pragma omp parallel for private(nearestMean, minDistance, currDistance) reduction(+: clustersX[:clustersNumber], clustersY[:clustersNumber], clustersCardinality[:clustersNumber])
	for (long i = 0; i < pointsNumber; i++)
	{
		nearestMean = 0;
		minDistance = distance(points[i], means[0]);

		//Finding the nearest mean
		for (long j = 1; j < clustersNumber; j++)
		{
			currDistance = distance(points[i], means[j]);
			if (currDistance < minDistance)
			{
				nearestMean = j;
				minDistance = currDistance;
			}
		}

		//Putting the point in the nearest cluster
		clustersX[nearestMean] += points[i].x;
		clustersY[nearestMean] += points[i].y;
		clustersCardinality[nearestMean]++;

		//Color the point
		points[i].shape.setFillColor(clusterColors[nearestMean % COLORS_NUM]);

		//Draw the point
		#pragma omp critical
		window.draw(points[i].shape);
	}

	bool changed = false;
	//Recalculating the means
	#pragma omp parallel for private(xRes, yRes)
	for (long j = 0; j < clustersNumber; j++)
	{
		//Setting the mean
		xRes = clustersX[j] / clustersCardinality[j];
		yRes = clustersY[j] / clustersCardinality[j];

		//Check if it's time to stop
		if (xRes != means[j].x || yRes != means[j].y)
		{
			#pragma omp critical
			changed = true;
		}

		//Assign the values
		means[j].x = xRes;
		means[j].y = yRes;
		means[j].shape.setPosition(sf::Vector2f(means[j].x, means[j].y));

		#pragma omp critical
		window.draw(means[j].shape);
	}

	return changed;
}

//Main
int main(int argc, char* argv[])
{
	int windowDimension = DIMENSION;
	long pointsNumber = POINTS_NUM;
	long clustersNumber = CLUSTERS_NUM;
	int maximumIterations = MAX_ITERATIONS;
	int seed = SEED;

	//Parse command line arguments
	for(int i=1; i<argc; i++)
	{
		if(string(argv[i]) == "-h") //Help
		{
			cout << "K-Means GUI\tElia Battiston" << endl << endl;
			cout << "Flags:" << endl;
			cout << "-h\tShow the help message" << endl;
			cout << "-w\tWindow dimensions\tDefault: 600" << endl;
			cout << "-p\tPoints number\t\tDefault: 70000" << endl;
			cout << "-c\tClusters number\t\tDefault: 10" << endl;
			cout << "-i\tMaximum iterations\tDefault: 500" << endl;
			cout << "-s\tSeed\t\t\tDefault: 42" << endl;
			cout << endl;
			return 0;
		}
		else if(string(argv[i]) == "-w") //Window dimension
		{
			i++;
			windowDimension = stoi(argv[i]);
		}
		else if(string(argv[i]) == "-p") //Points number
		{
			i++;
			pointsNumber = stol(argv[i]);
		}
		else if(string(argv[i]) == "-c") //Clusters number
		{
			i++;
			clustersNumber = stol(argv[i]);
		}
		else if(string(argv[i]) == "-i") //Maximum iterations
		{
			i++;
			maximumIterations = stoi(argv[i]);
		}
		else if(string(argv[i]) == "-s") //Seed
		{
			i++;
			seed = stoi(argv[i]);
		}
	}

	//Tell the complete configuration
	cout << "Window dimension:\t" << windowDimension << endl;
	cout << "Points number:\t\t" << pointsNumber << endl;
	cout << "Clusters number:\t" << clustersNumber << endl;
	cout << "Maximum iterations:\t" << maximumIterations << endl;
	cout << "Seed:\t\t\t" << seed << endl;
	cout << endl;
	
	//Font for writing the number of iterations
	sf::Font arial;
	if (!arial.loadFromFile("arial.ttf"))
	{
		cout << "Could not load the font. Make sure 'arial.ttf' is placed in the executable's directory." << endl;
		return 1;
	}
	sf::Text text;
	text.setFont(arial);
	text.setCharacterSize(24);
	text.setStyle(sf::Text::Bold);
	text.setFillColor(sf::Color::Red);
	text.setOutlineColor(sf::Color::White);
	text.setOutlineThickness(5);

	//Create the program's window
	sf::RenderWindow window(sf::VideoMode(windowDimension, windowDimension), "K-Means");

	//Problem's initialization
	point* points;
	point* means;
	unsigned long* clustersX;
	unsigned long* clustersY;
	unsigned long* clustersCardinality;

	initialize(points, means, clustersX, clustersY, clustersCardinality, windowDimension, pointsNumber, clustersNumber, seed);

	//Elaboration
	unsigned long iterations = 0;

	//Clock used to track elapsed time
	auto startTime = std::chrono::high_resolution_clock::now();

	bool changed = true;
	while (window.isOpen())
	{
		//Window's events loop
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		if (changed)
		{
			//Clear the window's content from the last iteration
			window.clear(sf::Color::Black);

			//Execute one iteration of the algorithm's logic
			changed = executeIteration(points, means, clustersX, clustersY, clustersCardinality, pointsNumber, clustersNumber, window);

			iterations++;

			if (!changed || iterations >= maximumIterations)
			{
				//Stop the timer
				auto endTime = std::chrono::high_resolution_clock::now();
				
				cout << endl;
				for (long j = 0; j < clustersNumber; j++)
					cout << "Cluster " << j << ": size " << clustersCardinality[j] << " with coordinates " << means[j].x << "," << means[j].y << endl;


				text.setString(" Time: " + to_string(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0) + "   Iterations: " + to_string(iterations));
				window.draw(text);

				window.display();

				cout << endl << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0 << endl;
			}
			else
			{
				text.setString(" Iterations: " + to_string(iterations));
				window.draw(text);

				window.display();
			}
		}
	}

	//Free allocated memory
	delete[] points;
	delete[] means;
	delete[] clustersX;
	delete[] clustersY;
	delete[] clustersCardinality;
	
	return 0;
}