 // CC -Wall -O3 -ffast-math -funroll-loops -I/users/jbedorf/codes/BonsaiOkt13_14/tools/IO -std=c++11  analizeDisk.cpp -o  analizeDisk

//#include "config.h"
#include  <cmath>
#include  <fstream>
#include  <iostream>
#include <algorithm>
#include <vector>


#include <memory>
#include "BonsaiIO.h"
#include "IDType.h"


#include "Astro.h"
#include "Constants.h"

// Simulation Units
double UnitLength;
double UnitTime;
double UnitVelocity;
double UnitMass;
double GravConst;
double SDUnit;    

// cosmological constants
double hubble;
double OmegaM;
double OmegaL;
double Hubble;

// Galaxy Model Parameters
double Mblg;
double Rblg;
double Mdisk;
double Rdisk;
double Zdisk;
double Qref;
double Mhalo;
double Rhalo;
double Cnfw;

struct StructPbody{
    uint64_t ID;
    double Mass;
    double Pos[3];
    double Vel[3];
};
struct StructPbody *Pbody;

//static void InitializeMPI(int argc, char **argv);
static void Analysis(int Nstars, std::ofstream &out);
int read_file(std::ifstream & in, std::ofstream &out);
double HubbleOverH0( double redshift ){
    double z = redshift;
    return sqrt((1-OmegaM) + OmegaM*pow(1+z,3));
}

typedef float float4[4];
typedef float float3[3];
typedef float float2[2];

int read(
    const int rank, const MPI_Comm &comm,
    const std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> &data,
    BonsaiIO::Core &in,
    const int reduce,
    std::shared_ptr<BonsaiIO::DataType<IDType>> &idt,
    std::shared_ptr<BonsaiIO::DataType<float4>> &pos,
    std::shared_ptr<BonsaiIO::DataType<float3>> &vel,
    const bool restartFlag = true
    )
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
      if(rank == 0) fprintf(stderr,"Read %lld  of %s \n", nGlb, type->getName().c_str());
    }
    else if (rank == 0)
    {
      fprintf(stderr, " %s  is not found, skipping\n", type->getName().c_str());
      fprintf(stderr, " ---- \n");
    }

    dtRead += MPI_Wtime() - t0;
  }

  const auto n = idt->size();

  //Count number of disk particles
  size_t countS = 0;
  for(size_t z=0; z < n; z++)
  {
    if(idt->operator[](z).getType() != 1)
	    countS++;
  }

  Pbody = new StructPbody[countS+2];


  int bulgeCount = 0;

  if(rank == 0)  fprintf(stderr, "Rank: %d has read: %ld items: \n", rank, n);
  int i = 0;
  for(size_t z=0; z < n; z++)
  {
    //getType() ->  1, then its a bulge particle, else it is a disk particle
    if(idt->operator[](z).getType() != 1)
    {
	Pbody[i].ID = idt->operator[](z).getID(); 
        Pbody[i].Mass   = pos->operator[](z)[3];
        Pbody[i].Pos[0] = pos->operator[](z)[0];
        Pbody[i].Pos[1] = pos->operator[](z)[1];
        Pbody[i].Pos[2] = pos->operator[](z)[2];

        Pbody[i].Vel[0] = vel->operator[](z)[0];
        Pbody[i].Vel[1] = vel->operator[](z)[1];
        Pbody[i].Vel[2] = vel->operator[](z)[2];
        i++;
    }
    else
    {
    	bulgeCount++;
    }
  } //for

  fprintf(stderr,"Ignored %d bulge particles! \n" , bulgeCount);

  return i;

}//end read function

//////////////////////////////////////
int main(int argc, char *argv[])
{

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    int nRank, myRank;
    MPI_Comm_size(comm, &nRank);
    MPI_Comm_rank(comm, &myRank);

    const std::string fileIn (argv[1]);
    const int doDM = atoi(argv[2]);
    
    GravConst = 1.0;
    UnitLength = 1.0*KPC_CGS;                      //[kpc]->[cm]
    UnitMass = 2.33e+9*MSUN_CGS;
    SDUnit = 2.3e+9/1.e6;
    double tmp = CUBE(UnitLength)/(GRAVITY_CONSTANT_CGS*UnitMass);
#define ONE_YEAR 3.15582e7
    UnitTime = sqrt(tmp);                           //[s]
    UnitVelocity = UnitLength/UnitTime;             //[cm/s]

    // For Cosmology
    hubble = 0.7;
    OmegaM = 0.3;
    OmegaL = 1.0 - OmegaM;
    Hubble = hubble*100*(1.e+5/UnitLength)/(1.0/UnitTime)/1000.0;
    
    // galaxy model
    //int Number = ReadParameterFile("InputParameter.dat");
    

    BonsaiIO::Core in(myRank, nRank, comm, BonsaiIO::READ, fileIn);

	std::string baseName = "Stars:";
	if(doDM)
	  baseName.assign("DM:");

	auto idt  = std::make_shared<BonsaiIO::DataType<IDType>>(baseName + "IDType");
	auto pos  = std::make_shared<BonsaiIO::DataType<float4>>(baseName + "POS:real4");
	auto vel  = std::make_shared<BonsaiIO::DataType<float3>>(baseName + "VEL:float[3]");

	std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> dataDM;
	dataDM.push_back(idt);
	dataDM.push_back(pos);
	dataDM.push_back(vel);
	const int reduceDM = 1;
	int n = read(myRank, comm, dataDM, in, reduceDM, idt, pos, vel);
	in.close();
	std::cerr << "n=" << n << std::endl;

    char outfile[128];
	sprintf(outfile, "%s%s", fileIn.c_str(), "_Jz.dat");
	std::ofstream out;
	if(myRank == 0)
		out.open(outfile, std::ofstream::out);

    //All processes call the analysis function
	Analysis(n, out);
	out.close();

    //Wait until all processes have finished their analysis tasks
    MPI_Barrier(MPI_COMM_WORLD);

    return 0;
}

void Analysis(int Nstars, std::ofstream &out){

    const int nR = 60;
    float RrotEnd = 30.0;
    float RrotMin = 0.0;
    // rotation curve
    const int iMax = nR;
    float dR = (RrotEnd - RrotMin)/iMax;
    fprintf(stderr, "dR=%f", dR);
    int Ns[iMax];
    float Rs[iMax];
    float Mass[iMax];
    float Jz[iMax];

    for(int i=0; i<iMax; i++){
        Rs[i] = RrotMin + (i+0.5)*dR;
        Ns[i] = 0;
    	Mass[i] = 0.0;
        Jz[i]=0.0;
    }

    // stars
    for(int pn=0; pn<Nstars; pn++){
        double R = sqrt(SQ(Pbody[pn].Pos[0]) + SQ(Pbody[pn].Pos[1]));
        //double z = Pbody[pn].Pos[2];
        double jz = Pbody[pn].Vel[1]*Pbody[pn].Pos[0] - Pbody[pn].Vel[0]*Pbody[pn].Pos[1];
        if( R <= RrotMin || R >= RrotEnd )    continue;
        int i = floor((R-RrotMin)/dR);
        Ns[i] += 1;
        Mass[i] += (float)Pbody[pn].Mass;
        Jz[i] += jz;
    }


    //Combine the partial results from all processes used
    MPI_Comm comm = MPI_COMM_WORLD;
    int procID = 0;
    MPI_Comm_rank(comm, &procID);


    if( procID == 0)
    {
        MPI_Reduce( MPI_IN_PLACE, Ns,   iMax, MPI_INT,   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce( MPI_IN_PLACE, Mass, iMax, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce( MPI_IN_PLACE, Jz,   iMax, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Reduce(Ns, Ns,     iMax, MPI_INT,   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(Mass, Mass, iMax, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(Jz, Jz,     iMax, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if(procID != 0) return;

     //Process the combined results on process 0
    for(int i=0; i < 10; i++)
    	fprintf(stderr, "index: %d value: %d \n", i, Ns[i]);

    // average
    for(int i=0; i<iMax; i++){
        Jz[i] *= UnitVelocity*UnitLength;
    	Mass[i] *= 2.33e9;//UnitMass;
    }

    out.setf(std::ios::scientific);
    out.precision(6);

    for(int i=0; i<iMax; i++){
    	out << Rs[i] << "  " << Jz[i] << "  "  // 1,2
	        <<  Mass[i] << "  " << Ns[i]  << std::endl; // 3, 4
    }

}

