#include "BonsaiIO.h"
#include "IDType.h"
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>

typedef struct real4
{
  float x,y,z,w;
} real4;

typedef float vec3[3];

//#define DARKMATTERID  3000000000000000000
//#define DISKID        0
//#define BULGEID       2000000000000000000


static IDType lGetIDType (const long long id, int type=4)
{
  IDType ID;
  ID.setID(id);
  ID.setType(type);    //Type of satellite
  
  return ID;
};


static double readAMUSE(
		std::vector<real4> &bodyPositions,
		std::vector<real4> &bodyVelocities,
		std::vector<IDType> &bodyIDs,
		const int reduce, 
		std::string fileName)
{
	double dtRead = 0;
	double t0 = MPI_Wtime();

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

	int pid  = 0;
	int cntr = 0;
	real4 positions;
	real4 velocity;

	while(std::getline(inputFile, tempLine))
	{
		if(tempLine.empty()) continue; //Skip empty lines

		std::stringstream ss(tempLine);
		//Amuse format
		//       inputFile >> positions.w >> r2 >> r2 >> 
		//                velocity.x  >> velocity.y  >> velocity.z  >>
		//                positions.x >> positions.y >> positions.z;
		ss >> positions.w >>
			velocity.x >> velocity.y >> velocity.z  >>
			positions.x >> positions.y >> positions.z;
		if(reduce > 0)
		{
			if(cntr % reduce == 0)
			{
				positions.w *= reduce;

				bodyIDs.push_back(lGetIDType(pid++));  //Convert the ID to a star (disk for now) particle
				bodyPositions.push_back(positions);
				bodyVelocities.push_back(velocity);
			}
		}      
		cntr++;   
	}
	inputFile.close();

	
	dtRead += MPI_Wtime() - t0;
	return dtRead;
}


static double readBonsai(
		const int rank, const MPI_Comm &comm,
		const std::vector<BonsaiIO::DataTypeBase*> &data, 
		BonsaiIO::Core &in, 
		const int reduce, 
		const bool restartFlag = true)
{
	double dtRead = 0;
	for (auto &type : data)
	{
		double t0 = MPI_Wtime();
		if (rank == 0)
			fprintf(stderr, " Reading %s ...\n", type->getName().c_str());
		if (in.read(*type, restartFlag, reduce))
		{
			long long int nLoc = type->getNumElements();
			long long int nGlb;
			MPI_Allreduce(&nLoc, &nGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
			if (rank == 0)
			{
				fprintf(stderr, " Read %lld of type %s\n",
						nGlb, type->getName().c_str());
				fprintf(stderr, " ---- \n");
			}
		} 
		else if (rank == 0)
		{
			fprintf(stderr, " %s  is not found, skipping\n", type->getName().c_str());
			fprintf(stderr, " ---- \n");
		}

		dtRead += MPI_Wtime() - t0;
	}

	return dtRead;
}

static double write(
		const int rank, const MPI_Comm &comm,
		const std::vector<BonsaiIO::DataTypeBase*> &data,
		BonsaiIO::Core &out)
{
	double dtWrite = 0;
	for (const auto &type : data)
	{
		double t0 = MPI_Wtime();
		if (rank == 0)
			fprintf(stderr, " Writing %s ... \n", type->getName().c_str());
		long long int nLoc = type->getNumElements();
		long long int nGlb;
		MPI_Allreduce(&nLoc, &nGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
		if (nGlb > 0)
		{
			if (rank == 0)
				fprintf(stderr, " Writing %lld of type %s\n",
						nGlb, type->getName().c_str());
			assert(out.write(*type));
			if (rank == 0)
				fprintf(stderr, " ---- \n");
		}
		else if (rank == 0)
		{
			fprintf(stderr, " %s is empty... not writing \n", type->getName().c_str());
			fprintf(stderr, " ---- \n");
		}
		dtWrite += MPI_Wtime() - t0;
	}

	return dtWrite;
}


#if 1
	template<typename IO, size_t N>
static double writeStars(std::vector<real4>    &bodyPositions, 
		std::vector<real4>    &bodyVelocities, 
		std::vector<IDType>   &bodiesIDs, 
		IO &out,
		std::array<size_t,N> &count)
{
	double dtWrite = 0;

	const int pCount  = bodyPositions.size();

	/* write IDs */
	{
		BonsaiIO::DataType<IDType> ID("Stars:IDType", pCount);
		for (int i = 0; i< pCount; i++)
		{
			//ID[i] = lGetIDType(bodiesIDs[i]);
			ID[i] = bodiesIDs[i];
			assert(ID[i].getType() > 0);
			if (ID[i].getType() < count.size())
				count[ID[i].getType()]++;
		}
		double t0 = MPI_Wtime();
		out.write(ID);
		dtWrite += MPI_Wtime() - t0;
	}

	/* write pos */
	{
		BonsaiIO::DataType<real4> pos("Stars:POS:real4",pCount);
		for (int i = 0; i< pCount; i++)
			pos[i] = bodyPositions[i];
		double t0 = MPI_Wtime();
		out.write(pos);
		dtWrite += MPI_Wtime() - t0;
	}

	/* write vel */
	{
		typedef float vec3[3];
		BonsaiIO::DataType<vec3> vel("Stars:VEL:float[3]",pCount);
		for (int i = 0; i< pCount; i++)
		{
			vel[i][0] = bodyVelocities[i].x;
			vel[i][1] = bodyVelocities[i].y;
			vel[i][2] = bodyVelocities[i].z;
		}
		double t0 = MPI_Wtime();
		out.write(vel);
		dtWrite += MPI_Wtime() - t0;
	}

	return dtWrite;
}
#endif

int main(int argc, char * argv[])
{
	MPI_Comm comm = MPI_COMM_WORLD;

	MPI_Init(&argc, &argv);

	int nranks, rank;
	MPI_Comm_size(comm, &nranks);
	MPI_Comm_rank(comm, &rank);


	if (argc < 3)
	{
		if (rank == 0)
		{
			fprintf(stderr, " ------------------------------------------------------------------------\n");
			fprintf(stderr, " Usage: \n");
			fprintf(stderr, " %s  bonsaiName amuseName outputName [reduceStar]\n", argv[0]);
			fprintf(stderr, " ------------------------------------------------------------------------\n");
		}
		exit(-1);
	}

	const std::string bonsaiName(argv[1]);
	const std::string amuseName(argv[2]);
	const std::string outputName(argv[3]);

	int reduceDWF  = 1;

	if( rank == 0) fprintf(stderr, "bonsai name: %s amuseASCII name: %s  outputname: %s Reducing Stars by factor: %d \n",bonsaiName.c_str(),  amuseName.c_str(), outputName.c_str(), reduceDWF);

	double dtRead = 0.;

	// read bonsai
	const double tOpen = MPI_Wtime(); 
	BonsaiIO::Core in(rank, nranks, comm, BonsaiIO::READ,  bonsaiName);
	double dtOpen = MPI_Wtime() - tOpen;
	std::vector<BonsaiIO::DataTypeBase*> data;
	int reduceDM = 0, reduceS = 0;
	if (reduceDM > 0)
	{
		std::vector<BonsaiIO::DataTypeBase*> dataDM;
		typedef float float4[4];
		typedef float float3[3];
		typedef float float2[2];
		dataDM.push_back(new BonsaiIO::DataType<IDType>("DM:IDType"));
		dataDM.push_back(new BonsaiIO::DataType<float4>("DM:POS:real4"));
		dataDM.push_back(new BonsaiIO::DataType<float3>("DM:VEL:float[3]"));
		dataDM.push_back(new BonsaiIO::DataType<float2>("DM:RHOH:float[2]"));

		dtRead += readBonsai(rank, comm, dataDM, in, reduceDM);

		data.insert(data.end(), dataDM.begin(), dataDM.end());
	}
	if (reduceS > 0)
	{
		std::vector<BonsaiIO::DataTypeBase*> dataStars;
		typedef float float4[4];
		typedef float float3[3];
		typedef float float2[2];
		dataStars.push_back(new BonsaiIO::DataType<IDType>("Stars:IDType"));
		dataStars.push_back(new BonsaiIO::DataType<float4>("Stars:POS:real4"));
		dataStars.push_back(new BonsaiIO::DataType<float3>("Stars:VEL:float[3]"));
		dataStars.push_back(new BonsaiIO::DataType<float2>("Stars:RHOH:float[2]"));

		dtRead += readBonsai(rank, comm, dataStars, in, reduceS);

		data.insert(data.end(), dataStars.begin(), dataStars.end());
	}

	// read amuseASCII
	if (reduceDWF > 0) {
		std::vector<BonsaiIO::DataTypeBase*> dataDWF;
		std::vector<real4> bodyPositions;
		std::vector<real4> bodyVelocities;
		std::vector<IDType> bodyIDs;
		dtRead += readAMUSE(bodyPositions, bodyVelocities, bodyIDs, reduceDWF, amuseName);
		const int pCount  = bodyPositions.size();
		typedef float float4[4];
		typedef float float3[3];
		BonsaiIO::DataType<IDType> ID("Stars:IDType", pCount);
		BonsaiIO::DataType<float4> pos("Stars:POS:real4", pCount);
		BonsaiIO::DataType<float3> vel("Stars:VEL:float[3]", pCount);
		for (int i = 0; i< pCount; i++)
		{
			ID[i] = bodyIDs[i];
			pos[i][0] = bodyPositions[i].x;
			pos[i][1] = bodyPositions[i].y;
			pos[i][2] = bodyPositions[i].z;
			pos[i][3] = bodyPositions[i].w;
			vel[i][0] = bodyVelocities[i].x;
			vel[i][1] = bodyVelocities[i].y;
			vel[i][2] = bodyVelocities[i].z;
		}
		dataDWF.push_back(&ID);
		dataDWF.push_back(&pos);
		dataDWF.push_back(&vel);

		data.insert(data.end(), dataDWF.begin(), dataDWF.end());
	}

	if (rank == 0) fprintf(stderr, "dtRead = %g sec \n", dtRead);

	const double tAll = MPI_Wtime();
	{
		const double tOpen = MPI_Wtime(); 
		BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::WRITE,  outputName);
		double dtOpen = MPI_Wtime() - tOpen;

		double dtWrite = 0;
		dtWrite += write(rank, comm, data, out);

		double writeBW = out.computeBandwidth();
		const double tClose = MPI_Wtime(); 
		out.close();
		double dtClose = MPI_Wtime() - tClose;

		double dtOpenGlb = 0;
		double dtWriteGlb = 0;
		double dtCloseGlb = 0;
		MPI_Allreduce(&dtOpen, &dtOpenGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(&dtWrite, &dtWriteGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(&dtClose, &dtCloseGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
		if (rank == 0)
		{
			fprintf(stderr, "dtOpen = %g sec \n", dtOpenGlb);
			fprintf(stderr, "dtWrite = %g sec \n", dtWriteGlb);
			fprintf(stderr, "dtClose= %g sec \n", dtCloseGlb);
			fprintf(stderr, "Write BW= %g MB/s \n", writeBW/1e6);
		}
	}
	double dtAllLoc = MPI_Wtime() - tAll;
	double dtAllGlb;
	MPI_Allreduce(&dtAllLoc, &dtAllGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
	if (rank == 0)
		fprintf(stderr, "All operations done in   %g sec \n", dtAllGlb);


	MPI_Finalize();


	return 0;
}


