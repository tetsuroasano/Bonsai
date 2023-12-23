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
struct StructPbody *Pbody_SN_tot;

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
static int Analysis(
		const int nRank, 
		const int rank, 
		const MPI_Comm &comm, 
		int Nstars, 
		float dSun, 
		float vc, 
		float maxDistanceTest
		//	const std::string &filename
		); 
int read_file(ifstream & in, ofstream &out);
/*
double HubbleOverH0( double redshift ){
    double z = redshift;
    return sqrt((1-OmegaM) + OmegaM*pow(1+z,3));
}
*/

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
void rotate2(float angl, int N)
{
  for(int i=0;i<N;i++){
    rotate_xy(angl, Pbody_SN_tot[i].Pos);
    rotate_xy(angl, Pbody_SN_tot[i].Vel);
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
    const bool restartFlag,  // = true,
		float dSun,
		float maxDistanceTest,
		float angle
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
		double vec[3] = {pos->operator[](z)[0], 
										pos->operator[](z)[1],
										pos->operator[](z)[2]};
		rotate_xy(angle, vec);
		float d2 = (SQ(vec[0])+SQ(vec[1]-dSun));
		if( d2 < SQ(maxDistanceTest+0.5)){
    	count_SN++;
			total_Z += pos->operator[](z)[2];
			total_VZ += vel->operator[](z)[2];
		}
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
		double vec[3] = {pos->operator[](z)[0], 
										pos->operator[](z)[1],
										pos->operator[](z)[2]};
		rotate_xy(angle, vec);
		float d2 = (SQ(vec[0])+SQ(vec[1]-dSun));
		if( d2 < SQ(maxDistanceTest+0.5))
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
		}
  } //for

  if(rank == 0) fprintf(stderr,"Number of SN_cand items: %d \n", i);

  return i;

}//end read function

std::string divide_dir_and_file(std::string dir_filename){
	std::string filename;
	int sla_position = 0;
	for (int i = 0; i < dir_filename.length(); i++) {
		if(dir_filename[i]=='/') sla_position = i;
	}
	filename = dir_filename.substr(sla_position);
	return filename;
}


void output_SN_snaps(const int myRank, std::string filename, float maxDistanceTest, float dSun, float angleToUse, int count_SN){
	char outfile1[128];
	std::string filename_only = divide_dir_and_file(filename);
	sprintf(outfile1, "./%s_SN_%g_%g_%g_local_cyl2.csv", filename_only.c_str(), maxDistanceTest, dSun, angleToUse);
	if(myRank==0){
    ofstream out1(outfile1);
		for (int i = 0; i < count_SN; i++) {
			out1 << Pbody_SN_tot[i].ID <<  "," 
			     << Pbody_SN_tot[i].Type <<  "," 
			     << Pbody_SN_tot[i].Pos[0] << "," 
				   << Pbody_SN_tot[i].Pos[1] << "," 
					 << Pbody_SN_tot[i].Pos[2] << "," 
			     << Pbody_SN_tot[i].Vel[0] << "," 
				   << Pbody_SN_tot[i].Vel[1] << "," 
					 << Pbody_SN_tot[i].Vel[2] 
					 << endl; 
		}
	out1.close();
	}

}
////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]){
	
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    int nRank, myRank;
    MPI_Comm_size(comm, &nRank);
    MPI_Comm_rank(comm, &myRank);

    fprintf(stderr,"Usage: prog inFile dist angl(bar angle in rad) angleToUse(relative angle) dSun vc \n");

    if(argc != 7)
	    exit(0);

    const std::string filename (argv[1]);
    if (myRank == 0) {
	    fprintf(stderr,"Number of processes: %d \n", nRank);
	    fprintf(stderr,"Filename: %s \n", filename.c_str());
    }

    float maxDistanceTest = -1;
		float temp = atof(argv[2]);
		fprintf(stderr,"Max distance set to: %f \n", temp);
		maxDistanceTest = temp;

    cerr << "bar angle(rad)=";
    float angl = atof(argv[3]);
    cout << "Using angle: " << angl << std::endl;

    float angleToUse = 25;
		temp = atof(argv[4]);
		fprintf(stderr,"Angle set to: %f \n", temp);
		angleToUse = temp;


    float dSun = 10;
    dSun  = atof(argv[5]);
    fprintf(stderr,"Using the following dSun value: %f \n", dSun);
    
	
    cerr << "circular vel=";
    float vc = atof(argv[6]);
    cout << "Using vc: " << vc << std::endl;
    
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
	int n = read(myRank, comm, dataDM, in, reduceDM, idt, pos, vel, true, dSun, maxDistanceTest,0.5*M_PI-angl+angleToUse*M_PI/180.);
	in.close();

	cerr << "n=" << n << endl;

	rotate(0.5*M_PI-angl, n); // bar rest frame
	
//Analysis////////////////////////////////////////////////////////////////
 float angle_list[46] = {175, -170, -165, -160, -155, -150, -145, -140, -135, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180};
 int count_SN;
 //for (int i = 0; i < 46; i++) {
	 //angleToUse = angle_list[i];
	 float angl2 = angleToUse*M_PI/180.;
	 rotate(angl2, n);
	 count_SN = Analysis(nRank, myRank, comm, n, dSun, vc, maxDistanceTest);
	 rotate2(-angl2, count_SN);
	 rotate2(-0.5*M_PI+angl, count_SN);
	 output_SN_snaps(myRank, filename, maxDistanceTest, dSun, angleToUse, count_SN);
	 cout << "CHECK" << endl;
	 delete[] Pbody_SN_tot;
 //}

//Output//////////////////////////////////////////////////////////////////
	

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();


    return 0;
}

int Analysis(
		const int nRank, 
		const int rank, 
		const MPI_Comm &comm, 
		int Nstars, 
		float dSun, 
		float vc, 
		float maxDistanceTest
		//const std::string &filename
)
{ 
 	float RMax = dSun;
  float RMin = 0.;
  int iMax = 80;
  int jMax = 100;
  float VelUnit = UnitVelocity/VELOCITY_KMS_CGS;
  //int counts[4][80][100]={0};
  //int counts2[80][100]={0};
  float dR = (RMax - RMin)/iMax;

  float vamin = -150;
  float vamax = 150;
  float dva = (vamax - vamin)/jMax;

  float vrmin = -150;
  float vrmax = 150;
  float dvr = (vrmax - vrmin)/jMax;

  double vc_OLR = 244.7;
  double R_OLR = 9.1;


	double ave_z = 0.0;

  // stars
  int n_counts = 0;
  for(int pn=0; pn<Nstars; pn++){
      //if(Pbody[pn].Pos[1]<0.0) continue;
      double r = sqrt(SQ(Pbody[pn].Pos[0]) + SQ(Pbody[pn].Pos[1]-dSun)+SQ(Pbody[pn].Pos[2]));
			// if(r>maxDistanceTest) continue;
      //if(r>1.) continue;
      double dist = sqrt(SQ(Pbody[pn].Pos[0]) + SQ(Pbody[pn].Pos[1]-dSun));
			if(dist>maxDistanceTest) continue;
      double b = atan(Pbody[pn].Pos[2]/dist);
      double b_deg = b*180/M_PI;
      //if(fabs(b_deg)>10) continue;
      //if(Pbody[pn].Pos[1]<0.0) continue;   
      //if(Pbody[pn].Pos[1]>dSun) continue; 
      //if(fabs(Pbody[pn].Pos[0])>0.5) continue;
      //if(fabs(Pbody[pn].Pos[2])>0.5) continue;
      double R = sqrt(SQ(Pbody[pn].Pos[0]) + SQ(Pbody[pn].Pos[1]));     
      double vr =  Pbody[pn].Vel[0]*Pbody[pn].Pos[0]/R + Pbody[pn].Vel[1]*Pbody[pn].Pos[1]/R;
      double va = -Pbody[pn].Vel[0]*Pbody[pn].Pos[1]/R + Pbody[pn].Vel[1]*Pbody[pn].Pos[0]/R;
      vr *= -VelUnit;
      va *= VelUnit;
      double l = atan2(Pbody[pn].Pos[1]-dSun, Pbody[pn].Pos[0]); 
      double l_deg = l*180./M_PI;
      l_deg = l_deg - 90;
      if(l_deg<-180) l_deg += 360;
      l_deg = l_deg*-1.;
      
      l_deg = l_deg + 45;
      if(l_deg>180) l_deg -= 360;
			//out1 << Pbody[pn].Pos[0] <<  "\t" << Pbody[pn].Pos[1]  << "\t" << vr << "\t" << va  <<endl;
			Pbody_SN[n_counts].ID = Pbody[pn].ID;
			Pbody_SN[n_counts].Type = Pbody[pn].Type;
			Pbody_SN[n_counts].Pos[0] = Pbody[pn].Pos[0];
			Pbody_SN[n_counts].Pos[1] = Pbody[pn].Pos[1];
			Pbody_SN[n_counts].Pos[2] = Pbody[pn].Pos[2];
			Pbody_SN[n_counts].Vel[0] = Pbody[pn].Vel[0];
			Pbody_SN[n_counts].Vel[1] = Pbody[pn].Vel[1];
			Pbody_SN[n_counts].Vel[2] = Pbody[pn].Vel[2];

			ave_z += Pbody[pn].Pos[2]/Nstars;
			n_counts++;   
  }
    cerr << "n(<" << maxDistanceTest <<"kpc)=" << n_counts << endl;
		cout << "ave_z= " << ave_z << endl;

	delete[] Pbody;
	int *Rcounts, *Displs;
  Rcounts	= new int[nRank];
  Displs	= new int[nRank+1];
	MPI_Allgather(&n_counts, 1, MPI_INT, Rcounts, 1, MPI_INT, comm);	
	Displs[0] = 0;
	for (int i = 0; i < nRank; i++) {
		Displs[i+1] = Displs[i] + Rcounts[i];
	}
  Pbody_SN_tot = new StructPbody[Displs[nRank]];

	MPI_Datatype MPI_StructPbody;
	Create_MPI_Pbody(MPI_StructPbody);
	MPI_Allgatherv(&Pbody_SN[0], n_counts, MPI_StructPbody, &Pbody_SN_tot[0], Rcounts, Displs, MPI_StructPbody, comm);

	delete[] Pbody_SN;

	return Displs[nRank];
}

