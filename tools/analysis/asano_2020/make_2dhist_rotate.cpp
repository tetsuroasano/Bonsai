//#include "config.h"
#include  <cmath>
#include  <fstream>
#include  <iostream>
using namespace std;

#include <vector>

#include "Astro.h"

#include "Constants.h"

#include <memory>
#include "../BonsaiIO.h"
#include "../IDType.h"


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

#define NMAX 30000000
struct StructPbody{
    uint64_t ID;
		int Type;
    double Mass;
    double Pos[3];
    double Vel[3];
};
struct StructPbody *Pbody;
struct StructPbody *Pbody_SN;
struct StructPbody *Pbody_tot;

void Create_MPI_Pbody(MPI_Datatype &MPI_StructPbody){
	int count = 5;
	int array_of_blocklength[] = {1, 1, 1, 3, 3};
	MPI_Aint array_of_displacements[] = {
		offsetof(StructPbody, ID),
		offsetof(StructPbody, Type),
		offsetof(StructPbody, Mass),
		offsetof(StructPbody, Pos),
		offsetof(StructPbody, Vel)
	};
	MPI_Datatype array_of_types[] = {
		MPI_UINT64_T, 
		MPI_INT, 
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE
	};
	//MPI_Datatype tmp_type;
	//MPI_Aint lb, extent;

	MPI_Type_create_struct(count, array_of_blocklength, array_of_displacements, array_of_types, &MPI_StructPbody);
	MPI_Type_commit(&MPI_StructPbody);
}

//static void InitializeMPI(int argc, char **argv);
//static void Analysis(int Nstars, ofstream &out, float vc);
void rotate_xy(float angl, double val[]){
  float rot[3];
  rot[0] = cos(angl)*val[0] - sin(angl)*val[1];
  rot[1] = sin(angl)*val[0] + cos(angl)*val[1];
  rot[2] = val[2];
  
  val[0] = rot[0];
  val[1] = rot[1];
  val[2] = rot[2];
  //return rot;
}
void rotate(float angl, int N)
{
  for(int i=0;i<N;i++){
    rotate_xy(angl, Pbody[i].Pos);
    rotate_xy(angl, Pbody[i].Vel);
  }

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
    const bool restartFlag  // = true,
		//float dSun,
		//float maxDistanceTest
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
    }
    else if (rank == 0)
    {
      fprintf(stderr, " %s  is not found, skipping\n", type->getName().c_str());
      fprintf(stderr, " ---- \n");
    }

    dtRead += MPI_Wtime() - t0;
  }
  

  const auto n = idt->size();
	long int count_SN=0;
	float total_Z=0;
	float total_VZ=0;

// count SN canditates and calculate their mean Z and mean VZ
	for(size_t z=0; z < n; z++)
  {
    	count_SN++;
			total_Z += pos->operator[](z)[2];
			total_VZ += vel->operator[](z)[2];
  } 
	float avgZ = total_Z/count_SN;
	float avgVZ = total_VZ/count_SN;

////////////////////////////////////////////////////////////
  Pbody = new StructPbody[count_SN];
  Pbody_SN = new StructPbody[count_SN];
  if(rank == 0)   fprintf(stderr, "Rank: %d has read: %ld items \n", rank, n);
  int i = 0;
  for(size_t z=0; z < n; z++)
  {
        Pbody[i].ID     = idt->operator[](z).getID();
        Pbody[i].Type   = idt->operator[](z).getType();
        Pbody[i].Mass   = pos->operator[](z)[3];
        Pbody[i].Pos[0] = pos->operator[](z)[0];
        Pbody[i].Pos[1] = pos->operator[](z)[1];
        Pbody[i].Pos[2] = pos->operator[](z)[2] - avgZ;

        Pbody[i].Vel[0] = vel->operator[](z)[0];
        Pbody[i].Vel[1] = vel->operator[](z)[1];
        Pbody[i].Vel[2] = vel->operator[](z)[2] - avgVZ;
        i++;
  } //for

  return i;

}//end read function


////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]){
	
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    int nRank, myRank;
    MPI_Comm_size(comm, &nRank);
    MPI_Comm_rank(comm, &myRank);

    fprintf(stderr,"Usage: prog inFile angl(bar angle in rad)\n");

    if(argc != 3)
	    exit(0);

    const std::string filename (argv[1]);
    if (myRank == 0) {
	    fprintf(stderr,"Number of processes: %d \n", nRank);
	    fprintf(stderr,"Filename: %s \n", filename.c_str());
    }

    cerr << "bar angle(rad)=";
    float angl = atof(argv[2]);
   
    GravConst = 1.0;
    UnitLength = 1.0*KPC_CGS;                      //[kpc]->[cm]
    //UnitMass = 1.e+10*MSUN_CGS;                    //[Msun]->[g]
    UnitMass = 2.3e+9*MSUN_CGS;
    //SDUnit = 1.e10/1.e6;                           //[Msun/pc^2]
    SDUnit = 2.3e+9/1.e6;
    double tmp = CUBE(UnitLength)/(GRAVITY_CONSTANT_CGS*UnitMass);
#define ONE_YEAR 3.15582e7
    UnitTime = sqrt(tmp);                           //[s]
    UnitVelocity = UnitLength/UnitTime;             //[cm/s]
    cerr << "UnitVel=" << UnitVelocity << endl;
 

	BonsaiIO::Core in(myRank, nRank, comm, BonsaiIO::READ, filename);

	auto idt  = std::make_shared<BonsaiIO::DataType<IDType>>("Stars:IDType");
	auto pos  = std::make_shared<BonsaiIO::DataType<float4>>("Stars:POS:real4");
	auto vel  = std::make_shared<BonsaiIO::DataType<float3>>("Stars:VEL:float[3]");

	std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> dataDM;
	dataDM.push_back(idt);
	dataDM.push_back(pos);
	dataDM.push_back(vel);
	const int reduceDM = 1;

//READ////////////////////////////////////////////////////////////////////
	int n = read(myRank, comm, dataDM, in, reduceDM, idt, pos, vel, true );
	in.close();

	rotate(-angl, n); // bar rest frame

	int *Rcounts, *Displs;
  Rcounts	= new int[nRank];
  Displs	= new int[nRank+1];
	MPI_Allgather(&n, 1, MPI_INT, Rcounts, 1, MPI_INT, comm);	
	Displs[0] = 0;
	for (int i = 0; i < nRank; i++) {
		Displs[i+1] = Displs[i] + Rcounts[i];
	}
  Pbody_tot = new StructPbody[Displs[nRank]];

	MPI_Datatype MPI_StructPbody;
	Create_MPI_Pbody(MPI_StructPbody);
	MPI_Allgatherv(&Pbody[0], n, MPI_StructPbody, &Pbody_tot[0], Rcounts, Displs, MPI_StructPbody, comm);

	delete[] Pbody;

		if (myRank == 0){
			//Define parameters for the histgram
			std::vector<std::vector<long long>> hist2d(2400, std::vector<long long>(2400));
			const float xmin = -12.0;
			const float ymin = -12.0;
			const float dx = 0.01;

			for (size_t i = 0; i < Displs[nRank]; i++)
			{
				float x = Pbody_tot[i].Pos[0];
				float y = Pbody_tot[i].Pos[1];
				int id_x = floor((x-xmin)/dx);
				int id_y = floor((y-ymin)/dx);
				if (id_x < 0 || id_x >= (int)hist2d.size()) continue;
				if (id_y < 0 || id_y >= (int)hist2d[0].size()) continue;
				hist2d[id_x][id_y]++;
			}

			// output file
			std::string fileOut = "";
			for (int i = 0; i < (int)filename.size(); i++) {
				fileOut += filename[i];
				if (filename[i]=='/') fileOut = "";
			}
			fileOut += ".hist2d.csv";
			std::ofstream ofs(fileOut);
			for (int i = 0; i < (int)hist2d.size(); i++) {
				for (int j = 0; j < (int)hist2d[i].size()-1; j++) {
					ofs << hist2d[i][j] <<",";
				}
				ofs << hist2d[i].back() <<"\n";
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();


		return 0;
}

