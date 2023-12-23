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

static double read(
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


static IDType lGetIDType (const long long id, int type=4)
{
	IDType ID;
	ID.setID(id);
	ID.setType(type);
	return ID;
};


void readAMUSEFile(std::vector<real4>    &bodyPositions, 
		std::vector<real4>    &bodyVelocities, 
		std::vector<IDType>   &bodiesIDs, 
		std::string fileName,
		const int reduceFactor,
		int type=4
		) {

	//bodyPositions.clear();

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
	real4       positions;
	real4       velocity;
	int cntr = 0;
	//     float r2 = 0;
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

		//               positions.x /= 1000;
		//               positions.y /= 1000;
		//               positions.z /= 1000;

		//       idummy = pid; //particleIDtemp;

		//       cout << idummy << "\t"<< positions.w << "\t"<<  positions.x << "\t"<<  positions.y << "\t"<< positions.z << "\t"
		//       << velocity.x << "\t"<< velocity.y << "\t"<< velocity.z << "\t" << velocity.w << "\n";

		if(reduceFactor > 0)
		{
			if(cntr % reduceFactor == 0)
			{
				positions.w *= reduceFactor;
				bodyPositions.push_back(positions);
				bodyVelocities.push_back(velocity);

				//Convert the ID to a star (disk for now) particle
				bodiesIDs.push_back(lGetIDType(pid++, type));
			}
		}      
		cntr++;   
	}
	inputFile.close();

	fprintf(stderr, "read %d bodies from dump file \n", cntr);
};


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


int main(int argc, char * argv[]) {
	MPI_Comm comm = MPI_COMM_WORLD;

	MPI_Init(&argc, &argv);

	int nranks, rank;
	MPI_Comm_size(comm, &nranks);
	MPI_Comm_rank(comm, &rank);

	int reduceDM = 1, reduceS = 1, reduceDWF = 1;

	if (argc == 7)
	{
		reduceDM = std::stoi(argv[4]);
		reduceS = std::stoi(argv[5]);
		reduceDWF = std::stoi(argv[6]);
	}
	else if (argc != 4)
	{
		if (rank == 0)
		{
			fprintf(stderr, " ------------------------------------------------------------------------\n");
			fprintf(stderr, " Usage: \n");
			fprintf(stderr, " %s  bonsaiName amuseName outName [reduceDM reduceStar reduceDWF]\n", argv[0]);
			fprintf(stderr, " ------------------------------------------------------------------------\n");
		}
		exit(-1);
	}

	const std::string bonsaiName(argv[1]);
	const std::string amuseName(argv[2]);
	const std::string outName(argv[3]);

	if( rank == 0) fprintf(stderr,"bonsaiName: %s  amuseName: %s outName: %s \n", bonsaiName.c_str(), amuseName.c_str(), outName.c_str());

	std::vector<real4>    bodyPositions;
	std::vector<real4>    bodyVelocities;
	std::vector<IDType>   bodyIDs;

	const double tOpen = MPI_Wtime(); 
	BonsaiIO::Core in(rank, nranks, comm, BonsaiIO::READ,  bonsaiName);
	double dtOpen = MPI_Wtime() - tOpen;
	double dtRead = 0.;

	std::vector<BonsaiIO::DataTypeBase*> data;

	if (reduceDM > 0)
	{
		std::vector<BonsaiIO::DataTypeBase*> dataDM;
		typedef float float4[4];
		typedef float float3[3];
		typedef float float2[2];
		dataDM.push_back(new BonsaiIO::DataType<IDType>("DM:IDType"));
		dataDM.push_back(new BonsaiIO::DataType<float4>("DM:POS:real4"));
		dataDM.push_back(new BonsaiIO::DataType<float3>("DM:VEL:float[3]"));
		//dataDM.push_back(new BonsaiIO::DataType<float2>("DM:RHOH:float[2]"));

		dtRead += read(rank, comm, dataDM, in, reduceDM);

		data.insert(data.end(), dataDM.begin(), dataDM.end());
	}
	if (reduceS > 0)
	{
		std::vector<BonsaiIO::DataTypeBase*> dataStars;
		typedef float float4[4];
		typedef float float3[3];
		typedef float float2[2];
		auto idt = BonsaiIO::DataType<IDType>("Stars:IDType");
		auto pos = BonsaiIO::DataType<float4>("Stars:POS:real4");
		auto vel = BonsaiIO::DataType<float3>("Stars:VEL:float[3]"); 		
		dataStars.push_back(&idt);
		dataStars.push_back(&pos);
		dataStars.push_back(&vel);
		//dataStars.push_back(new BonsaiIO::DataType<IDType>("Stars:IDType"));
		//dataStars.push_back(new BonsaiIO::DataType<float4>("Stars:POS:real4"));
		//dataStars.push_back(new BonsaiIO::DataType<float3>("Stars:VEL:float[3]"));
		//dataStars.push_back(new BonsaiIO::DataType<float2>("Stars:RHOH:float[2]"));

		dtRead += read(rank, comm, dataStars, in, reduceS);

		const auto n = idt.size();
		real4 positions;
		real4 velocity;
		for (size_t i = 0; i < n; i++) 
		{
			positions.x = pos[i][0];
			positions.y = pos[i][1];
			positions.z = pos[i][2];
			positions.w = pos[i][3];
			velocity.x = vel[i][0];
			velocity.y = vel[i][1];
			velocity.z = vel[i][2];
			bodyIDs.push_back(idt[i]);
			bodyPositions.push_back(positions);
			bodyVelocities.push_back(velocity);
		}
	}

	if (reduceDWF > 0) {
		readAMUSEFile(bodyPositions, bodyVelocities, bodyIDs, amuseName, reduceDWF);
	}


	// write
	{
		const double tOpen = MPI_Wtime(); 
		BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::WRITE,  outName);
		double dtOpen = MPI_Wtime() - tOpen;

		double dtWrite = 0;
		dtWrite += write(rank, comm, data, out);

		std::array<size_t,10> ntypeloc, ntypeglb;
		std::fill(ntypeloc.begin(), ntypeloc.end(), 0);
		MPI_Barrier(comm);
		dtWrite += writeStars(bodyPositions, bodyVelocities, bodyIDs, out, ntypeloc);

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

	MPI_Finalize();
	return 0;
}
