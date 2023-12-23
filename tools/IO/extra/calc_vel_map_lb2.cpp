//#include "config.h"
#include  <cmath>
#include  <fstream>
#include  <iostream>
using namespace std;

#include <vector>

#include "Astro.h"

#include "Constants.h"


#include <memory>
#include "BonsaiIO.h"
#include "IDType.h"


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
    double Mass;
    double Pos[3];
    double Vel[3];
};
struct StructPbody *Pbody;

//static void InitializeMPI(int argc, char **argv);
//static void Analysis(int Nstars, ofstream &out, float vc);
static void Analysis(int Nstars, float dSun, float vc, const std::string &filename); //, ofstream &out);
int read_file(ifstream & in, ofstream &out);
double HubbleOverH0( double redshift ){
    double z = redshift;
    return sqrt((1-OmegaM) + OmegaM*pow(1+z,3));
}

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
    const bool restartFlag,  // = true,
    const float avgZ,
    const float avgVZ
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
      if (rank == 0)
      {
 //       fprintf(stderr, " Read %lld of type %s\n",
   //         nGlb, type->getName().c_str());
     //   fprintf(stderr, " ---- \n");
      }
    }
    else if (rank == 0)
    {
      fprintf(stderr, " %s  is not found, skipping\n", type->getName().c_str());
      fprintf(stderr, " ---- \n");
    }

    dtRead += MPI_Wtime() - t0;
  }
  

  const auto n = idt->size();
  Pbody = new StructPbody[n];
  if(rank == 0)   fprintf(stderr, "Rank: %d has read: %ld items \n", rank, n);
  int i = 0;
  for(size_t z=0; z < n; z++)
  {
    //if(idt->operator[](z).getType() == 2)
    if(1)
    {
        Pbody[i].ID     = idt->operator[](z).getID();
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

  if(rank == 0) fprintf(stderr,"Number of bulge items: %d \n", i);

  return i;

}//end read function




int output(int n)
{
  ofstream out("output0000", ios::out|ios::binary);
    int  d=3;
    float time=0.0;
    out.write((char*)&n, sizeof(int));
    out.write((char*)&d, sizeof(int));
    out.write((char*)&time, sizeof(float));


    for(int i=0;i<n;i++){
      float tmp;
      tmp = Pbody[i].Mass;
      out.write((char*)&tmp, sizeof(float));        
        
    }
    for(int i=0;i<n;i++){
      float tmp[3];
      tmp[0] = Pbody[i].Pos[0];
      tmp[1] = Pbody[i].Pos[1];
      tmp[2] = Pbody[i].Pos[2];
      out.write((char*)tmp, 3*sizeof(float));
        
    }
    for(int i=0;i<n;i++){
      float tmp[3];
      tmp[0] = Pbody[i].Vel[0];
      tmp[1] = Pbody[i].Vel[1];
      tmp[2] = Pbody[i].Vel[2];
      out.write((char*)tmp, 3*sizeof(float));
        
	}
   
    return n;
}


//////////////////////////////////////
int main(int argc, char *argv[]){
	
	
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    int nRank, myRank;
    MPI_Comm_size(comm, &nRank);
    MPI_Comm_rank(comm, &myRank);

    fprintf(stderr,"Usage: prog inFile dist angl angleToUse avgZ avgVZ dSun vc \n");

    if(argc < 8)
	    exit(0);

    const std::string filename (argv[1]);
    if (myRank == 0) {
	    fprintf(stderr,"Number of processes: %d \n", nRank);
	    fprintf(stderr,"Filename: %s \n", filename.c_str());
    }


    float maxDistanceTest = -1;
    if(argc > 2) 
    {
	    float temp = atof(argv[2]);
	    fprintf(stderr,"Max distance set to: %f \n", temp);
	    maxDistanceTest = temp;
    }
    cerr << "bar angle(rad)=";
    float angl = atof(argv[3]);
    cout << "Using angle: " << angl << std::endl;

    float angleToUse = 25;
    if(argc > 4)
    {
	    float temp = atof(argv[4]);
	    fprintf(stderr,"Angle set to: %f \n", temp);
	    angleToUse = temp;
    }


    float avgZ = 0, avgVZ = 0, dSun = 10;
    if(argc > 5) avgZ  = atof(argv[5]);
    if(argc > 6) avgVZ = atof(argv[6]);
    if(argc > 7) dSun  = atof(argv[7]);
    fprintf(stderr,"Using the following avg Z values: %f and %f \n", avgZ, avgVZ);
    fprintf(stderr,"Using the following dSun value: %f \n", dSun);
    
	
    cerr << "circular vel=";
    float vc = atof(argv[8]);
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
    // For Cosmology
    hubble = 0.7;
    OmegaM = 0.3;
    OmegaL = 1.0 - OmegaM;
    Hubble = hubble*100*(1.e+5/UnitLength)/(1.0/UnitTime)/1000.0;
 

	BonsaiIO::Core in(myRank, nRank, comm, BonsaiIO::READ, filename);

	auto idt  = std::make_shared<BonsaiIO::DataType<IDType>>("Stars:IDType");
	auto pos  = std::make_shared<BonsaiIO::DataType<float4>>("Stars:POS:real4");
	auto vel  = std::make_shared<BonsaiIO::DataType<float3>>("Stars:VEL:float[3]");

	std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> dataDM;
	dataDM.push_back(idt);
	dataDM.push_back(pos);
	dataDM.push_back(vel);
	const int reduceDM = 1;
	int n = read(myRank, comm, dataDM, in, reduceDM, idt, pos, vel, true,avgZ, avgVZ);
	in.close();

	
	
	cerr << "n=" << n << endl;


	rotate(0.5*M_PI-angl, n);
	
	float angl2 = angleToUse*M_PI/180.;
	rotate(angl2, n);


	Analysis(n, dSun, vc, filename);
	
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();


    return 0;
}


void Analysis(int Nstars, float dSun, float vc, const std::string &filename)
{
//  float dSun = 8.0;
  float RMax = dSun;
  float RMin = 0.;
  int iMax = 80;
  int jMax = 100;
  float VelUnit = UnitVelocity/VELOCITY_KMS_CGS;
  int counts[4][80][100]={0};
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

  // stars
  int n_counts = 0;
  //double v0 = 0.0;
  for(int pn=0; pn<Nstars; pn++){
      if(Pbody[pn].Pos[1]<0.0) continue;
      double r = sqrt(SQ(Pbody[pn].Pos[0]) + SQ(Pbody[pn].Pos[1]-dSun)+SQ(Pbody[pn].Pos[2]));
      if(r>0.5) continue;
      //if(r>1.) continue;
      double dist = sqrt(SQ(Pbody[pn].Pos[0]) + SQ(Pbody[pn].Pos[1]-dSun));
      double b = atan(Pbody[pn].Pos[2]/dist);
      double b_deg = b*180/M_PI;
      if(fabs(b_deg)>10) continue;
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
      //if(Pbody[pn].Pos[1]<0){
	//cerr << l_deg << endl;
      //}
      //v0 += va;
      int k = (int)floor((l_deg+180)/90);
      int i = (int)floor((vr-vrmin)/dvr);
      int j = (int)floor((va-vamin-vc)/dva);
      if(i>=0 && i<iMax){
	if(j>=0 && j<jMax){
	  //if(k==2){
	  //  cerr << l_deg << "  " << Pbody[pn].Pos[0] << "  " << Pbody[pn].Pos[1] << endl;
	  //}
	  counts[k][i][j]++;	
	  n_counts++;
	}
      }
      
  }
    cerr << "n(<0.5kpc)=" << n_counts << endl;
    //cerr << "v0=" << v0/n_counts << " " << v0/n_counts/dSun << endl; 

    //out.setf(ios::scientific);
    //out.precision(6);
    char outfile1[128];
    sprintf(outfile1, "%s.vel_map_v2_dSun%f.dat", filename.c_str(), dSun);
    
    ofstream out1(outfile1);

    for(int k=0;k<4;k++){
      for(int i=0;i<iMax;i++){
	for(int j=0;j<jMax;j++){
	  out1 << vrmin+i*dvr <<  "  " << vamin+j*dva << "  " << log10(counts[k][i][j])<< endl;
	}
	out1<< endl;
      }
      out1<< endl;
    }
    out1.close();
   
}
