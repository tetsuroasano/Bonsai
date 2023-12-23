#include "BonsaiIO.h"
#include "IDType.h"
#include "tipsyIO.h"
#include <memory>
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>

//#define Nmax  8000000000
#define Nmax  80000000
#define DWARFID  4000000000000000000

static double read(
		const int rank, const MPI_Comm &comm,
		const std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> &data, 
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



int main(int argc, char * argv[]) {
	MPI_Comm comm = MPI_COMM_WORLD;

	MPI_Init(&argc, &argv);

	int nranks, rank;
	MPI_Comm_size(comm, &nranks);
	MPI_Comm_rank(comm, &rank);
	//int comm = 0, rank = 0;

	if (argc != 5 && argc != 6)
	{
		if (rank == 0)
		{
			fprintf(stderr, " ------------------------------------------------------------------------\n");
			fprintf(stderr, " Usage: \n");
			fprintf(stderr, " %s  fileIn tipsyOut reduceDM reduceStar time \n", argv[0]);
			fprintf(stderr, " ------------------------------------------------------------------------\n");
		}
		exit(-1);
	}

	const std::string fileIn(argv[1]);
	const std::string outName(argv[2]);
	const int reduceDM = atof(argv[3]);
	const int reduceS = atof(argv[4]);
	float time = 0;
	if (argc == 6) time = atof(argv[5]); 

	real4* bodyPositions = nullptr;
	real4* bodyVelocities = nullptr;
	ullong* bodiesIDs = nullptr;
	bodyPositions	 = new real4[Nmax];
	bodyVelocities	 = new real4[Nmax];
	bodiesIDs = new ullong[Nmax];

	long long n = 0;

	// read /////////////////////////////////////////////////////////////
	{
		const double tOpen = MPI_Wtime(); 
		BonsaiIO::Core in(rank, nranks, comm, BonsaiIO::READ,  fileIn);
		double dtOpen = MPI_Wtime() - tOpen;


		if (rank == 0)
			in.getHeader().printFields();

		double dtRead;
		if (reduceDM > 0)
		{
		  typedef float float4[4];
		  typedef float float3[3];
		  typedef float float2[2];
		  auto idt  = std::make_shared<BonsaiIO::DataType<IDType>>("DM:IDType");
		  auto pos  = std::make_shared<BonsaiIO::DataType<float4>>("DM:POS:real4");
		  auto vel  = std::make_shared<BonsaiIO::DataType<float3>>("DM:VEL:float[3]");
		  auto rhoh = std::make_shared<BonsaiIO::DataType<float2>>("DM:RHOH:float[2]");
		
		  std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> dataDM;
		  dataDM.push_back(idt);
		  dataDM.push_back(pos);
		  dataDM.push_back(vel);
		  dataDM.push_back(rhoh);
		  dtRead += read(rank, comm, dataDM, in, reduceDM);
		
		  const auto nDM = idt->size();

		  for (size_t i = 0; i < nDM; i++)
		  {
				bodiesIDs[n] = idt->operator[](i).getID() + DARKMATTERID;
				bodyPositions[n].x = pos->operator[](i)[0];
				bodyPositions[n].y = pos->operator[](i)[1];
				bodyPositions[n].z = pos->operator[](i)[2];
				bodyPositions[n].w = pos->operator[](i)[3];
				bodyVelocities[n].x = vel->operator[](i)[0];
				bodyVelocities[n].y = vel->operator[](i)[1];
				bodyVelocities[n].z = vel->operator[](i)[2];
				n++;
			}

		}

		if (reduceS > 0)
		{
			typedef float float4[4];
			typedef float float3[3];
			typedef float float2[2];
			auto idt  = std::make_shared<BonsaiIO::DataType<IDType>>("Stars:IDType");
			auto pos  = std::make_shared<BonsaiIO::DataType<float4>>("Stars:POS:real4");
			auto vel  = std::make_shared<BonsaiIO::DataType<float3>>("Stars:VEL:float[3]");
			auto rhoh = std::make_shared<BonsaiIO::DataType<float2>>("Stars:RHOH:float[2]");

			std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> dataS;
			dataS.push_back(idt);
			dataS.push_back(pos);
			dataS.push_back(vel);
			dataS.push_back(rhoh);
			dtRead += read(rank, comm, dataS, in, reduceS);

		  const auto nS = idt->size();

			for (size_t i = 0; i < nS; i++)
		  {
				bodiesIDs[n] = idt->operator[](i).getID();
				if (idt->operator[](i).getType() == 1) bodiesIDs[n] += BULGEID;
				bodyPositions[n].x = pos->operator[](i)[0];
				bodyPositions[n].y = pos->operator[](i)[1];
				bodyPositions[n].z = pos->operator[](i)[2];
				bodyPositions[n].w = pos->operator[](i)[3];
				bodyVelocities[n].x = vel->operator[](i)[0];
				bodyVelocities[n].y = vel->operator[](i)[1];
				bodyVelocities[n].z = vel->operator[](i)[2];
				n++;
			}
			
		}

		double readBW = in.computeBandwidth();

		const double tClose = MPI_Wtime(); 
		in.close();
		double dtClose = MPI_Wtime() - tClose;

		double dtOpenGlb = 0;
		double dtReadGlb = 0;
		double dtCloseGlb = 0;
		MPI_Allreduce(&dtOpen, &dtOpenGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(&dtRead, &dtReadGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(&dtClose, &dtCloseGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
		if (rank == 0)
		{
			fprintf(stderr, "dtOpen = %g sec \n", dtOpenGlb);
			fprintf(stderr, "dtRead = %g sec \n", dtReadGlb);
			fprintf(stderr, "dtClose= %g sec \n", dtCloseGlb);
			fprintf(stderr, "Read BW= %g MB/s \n", readBW/1e6);
		}
	}
	// end read

	std::cout << "NUMBER: " << n << std::endl;

	tipsyIO tio;
	tio.writeFile(bodyPositions, bodyVelocities, bodiesIDs, n, outName, time, rank, 1, comm, true);


	delete[] bodyPositions; 
	delete[] bodyVelocities; 
	delete[] bodiesIDs;

	//MPI_Finalize();
	return 0;
}
