#include "tipsyIO.h"
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>

//#define Nmax  8000000000
#define Nmax  80000000
#define DWARFID  4000000000000000000


void readDumpedAscii(real4* bodyPositions, 
		real4* bodyVelocities, 
		ullong* bodiesIDs, 
		std::string fileName,
		long long  &n,
		double reduceWeight = 1.0
		) 
{

	char fullFileName[256];
	sprintf(fullFileName, "%s", fileName.c_str());

	std::cerr << "Trying to read file: " << fullFileName << std::endl;

	std::ifstream inputFile(fullFileName, std::ios::in);

	if(!inputFile.is_open())
	{
		std::cerr << "Can't open input file \n";
		exit(0);
	}

	//Skip the  header lines
	std::string tempLine;
	std::getline(inputFile, tempLine);


	ullong pid  = 0;
	int type;
	real4 positions;
	real4 velocity;
	int cntr = 0;
	//     float r2 = 0;
	while(std::getline(inputFile, tempLine))
	{
		if(tempLine.empty()) continue; //Skip empty lines

		std::stringstream ss(tempLine);
		ss >> pid >> type >> positions.w >>
			positions.x >> positions.y >> positions.z >>
			velocity.x >> velocity.y >> velocity.z;

		if (type == 0) pid += DARKMATTERID;
		else if (type == 1) pid += BULGEID;

		positions.w *= reduceWeight;

		bodyPositions[n] = positions;
		bodyVelocities[n] = velocity;
		bodiesIDs[n] = pid;

		n++;
		cntr++;
	}

	inputFile.close();

	fprintf(stderr, "read %d bodies from ascii file \n", cntr);
}

void centerOfMassShift(real4* bodyPositions, long long n) {
	double mx = 0.0, my = 0.0, mz = 0.0;
	for (long long i = 0; i < n; i++) {
		real4 pos = bodyPositions[i];
		mx += pos.x;
		my += pos.y;
		mz += pos.z;
	}
	mx /= n;
	my /= n;
	mz /= n;
	for (int i = 0; i < n; i++) {
		bodyPositions[i].x -= mx;
		bodyPositions[i].y -= my;
		bodyPositions[i].z -= mz;
	}
}

void readAMUSEFile(real4* bodyPositions, 
		real4* bodyVelocities, 
		ullong* bodiesIDs, 
		std::string fileName,
		long long  &n
		) 
{

	char fullFileName[256];
	sprintf(fullFileName, "%s", fileName.c_str());

	std::cerr << "Trying to read file: " << fullFileName << std::endl;

	std::ifstream inputFile(fullFileName, std::ios::in);

	if(!inputFile.is_open())
	{
		std::cerr << "Can't open input file \n";
		exit(0);
	}

	//Skip the  header lines
	std::string tempLine;
	std::getline(inputFile, tempLine);
	std::getline(inputFile, tempLine);


	ullong pid  = 0;
	real4       positions;
	real4       velocity;
	int cntr = 0;
	//     float r2 = 0;
	while(std::getline(inputFile, tempLine))
	{
		if(tempLine.empty()) continue; //Skip empty lines

		std::stringstream ss(tempLine);
		ss >> positions.w >>
			velocity.x >> velocity.y >> velocity.z  >>
			positions.x >> positions.y >> positions.z;
		pid = DWARFID + cntr;
		bodyPositions[n] = positions;
		bodyVelocities[n] = velocity;
		bodiesIDs[n] = pid;

		n++;
		cntr++;
	}

	inputFile.close();

	fprintf(stderr, "read %d bodies from amuse file \n", cntr);
}


int main(int argc, char * argv[]) {
	MPI_Comm comm = MPI_COMM_WORLD;

	MPI_Init(&argc, &argv);

	int nranks, rank;
	MPI_Comm_size(comm, &nranks);
	MPI_Comm_rank(comm, &rank);
//int comm = 0, rank = 0;

	if (argc != 4 && argc != 5)
	{
		//if (rank == 0)
		{
			fprintf(stderr, " ------------------------------------------------------------------------\n");
			fprintf(stderr, " Usage: \n");
			fprintf(stderr, " %s  bonsaiASCII amuseASCII tipsyOut reduceWeight\n", argv[0]);
			fprintf(stderr, " ------------------------------------------------------------------------\n");
		}
		exit(-1);
	}

	const std::string basciiName(argv[1]);
	const std::string amuseName(argv[2]);
	const std::string outName(argv[3]);

	double reduceWeight = 1.0;
	if (argc == 5) reduceWeight = atof(argv[4]);

	real4* bodyPositions = nullptr;
	real4* bodyVelocities = nullptr;
	ullong* bodiesIDs = nullptr;
	bodyPositions	 = new real4[Nmax];
	bodyVelocities	 = new real4[Nmax];
	bodiesIDs = new ullong[Nmax];

	long long n = 0;


	readDumpedAscii(bodyPositions, bodyVelocities, bodiesIDs, basciiName, n, reduceWeight);
	centerOfMassShift(bodyPositions, n);
	readAMUSEFile(bodyPositions, bodyVelocities, bodiesIDs, amuseName, n);

	std::cout << "NUMBER: " << n << std::endl;

	tipsyIO tio;
	tio.writeFile(bodyPositions, bodyVelocities, bodiesIDs, n, outName, 0, rank, 1, comm, true);


	delete[] bodyPositions; 
	delete[] bodyVelocities; 
	delete[] bodiesIDs;
	


	//MPI_Finalize();
	return 0;
}
