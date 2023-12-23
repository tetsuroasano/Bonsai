//#include "config.h"
#include  <cmath>
#include  <fstream>
#include  <iostream>
using namespace std;

#include <vector>

#include "Astro.h"
//#include "MPIParameters.h"
//#include "RandomNumberGenerator.h"
//#include "DiskGalaxyModel.h"
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
static void Analysis(int Nstars, ofstream &out, float maxDistanceTest);
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

    float angleToUse = 25;
    if(argc > 4)
    {
	    float temp = atof(argv[4]);
	    fprintf(stderr,"Angle set to: %f \n", temp);
	    angleToUse = temp;
    }


    float avgZ = 0, avgVZ = 0;
    if(argc > 5) avgZ  = atof(argv[5]);
    if(argc > 6) avgVZ = atof(argv[6]);
    fprintf(stderr,"Using the following avg Z values: %f and %f \n", avgZ, avgVZ);


    
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
    
    // galaxy model
     
    //char filename[128];
    //cerr << "FILE: ";
    //cin >> filename;

/*
	ifstream in(filename, ios::in|ios::binary);
	if (!in) {
	    cerr << "Error: No file!" << endl;
	    //exit(1);
	    //break;
	}*/
	
	
	char outfile[128];
	sprintf(outfile, "%s%s%f%s%f%s", filename.c_str(), "_v_bulge_distance_", maxDistanceTest, "_angle_", angleToUse, "_zcor_lb_v2.dat");
	ofstream out(outfile);
	
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
	
	
	//int n;
	//n = read_file(in, out);
	//in.close();
	
	cerr << "n=" << n << endl;

	cerr << "bar angle(rad)=";
	float angl = atof(argv[3]);
	cout << "Using angle: " << angl << std::endl;
	//cin >> angl;
	//if(y0<0) angl *=-1;
	//rotate(angl, n);

	rotate(0.5*M_PI - angl, n);
	
	//float angl2 = 25.*M_PI/180.;
	float angl2 = angleToUse*M_PI/180.;
	rotate(angl2, n);
	//output(n);// for test
	Analysis(n, out, maxDistanceTest);
	out.close();
	//filenum++;
	//}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();


    return 0;
}




void Analysis(int Nstars, ofstream &out, float maxDistanceTest){

    const int nR = 60;
    float RrotEnd = 30.0;
    float RrotMin = -30.0;
    float Rrot[nR], Vrot[nR], dRrot = (RrotEnd-RrotMin)/nR;
    //CreateCircularVelocity(nR,Rrot,Vrot,dRrot,bRotDir);

    // rotation curve
    const int iMax = nR;
    int Ns[iMax][3];
    float dR = (RrotEnd - RrotMin)/iMax;
    fprintf(stderr, "dR=%f", dR);
    float Rs[iMax];

    float Vlos[iMax][3];
    float Vlos_d[iMax][3];
    //float Vrs[iMax], Vas[iMax], Vzs[iMax];  	// mean speed
    //float Drs[iMax], Das[iMax], Dzs[iMax];  	// dispersion
    //float Vcdm[iMax];				// dark matter halo
    //float Vcdsk[iMax];				// exp disk
    float VelUnit = UnitVelocity/VELOCITY_KMS_CGS;
    
    float m[iMax];
    float Mass[iMax];
    float zrms[iMax][3];
    vector<float> vp[iMax][3];

    for(int i=0; i<iMax; i++){
        Rs[i] = RrotMin + (i+0.5)*dR;
        //Qs[i] = Gam[i] = mX[i] = Sigs[i] = 0.0;
        //Vcdm[i] = Vcdsk[i] = 0.0;
        //Omgs[i] = kapps[i] = 0.0;
        //Omgc[i] = kappc[i] = dOmg[i] = 0.0;
        //OmgILR[i] = OmgOLR[i] = 0.0;
        //OmgILR4[i] = OmgOLR4[i] = 0.0;
        //Vrs[i] = Vas[i] = Vzs[i] = 0.0;
        //Drs[i] = Das[i] = Dzs[i] = 0.0;
	for(int k=0;k<3;k++){
	  Vlos[i][k] = 0.0;
	  Vlos_d[i][k] = 0.0;
	  Ns[i][k] = 0;
	  zrms[i][k] = 0.0;
	}
	Mass[i] = 0.0;
	//Sigs[i] = 0.0;
	
    }
  
    MPI_Comm comm = MPI_COMM_WORLD;
    int procID, nProcs = 0;    
    MPI_Comm_size(comm, &nProcs);
	MPI_Comm_rank(comm, &procID);
  	const int nSendMax = 1000000;

  if(procID != 0)
  {
	fprintf(stderr,"Going to send data from rank: %d to 0, stars: %d \n", procID, Nstars);
	MPI_Send(&Nstars, 1, MPI_INT, 0, 123, MPI_COMM_WORLD);
	
	for(int writeIdx=0; writeIdx < Nstars; writeIdx += nSendMax)
	{
		int nToSend = std::min(nSendMax, Nstars-writeIdx);	
		fprintf(stderr, "Rank %d sending: %d items \n", procID, nToSend);	
		MPI_Send(&Pbody[writeIdx], nToSend*sizeof(StructPbody), MPI_BYTE, 0, 124, MPI_COMM_WORLD);
	}
	return;
  }
  else
  {
	  for(int rank = 0; rank < nProcs; rank++)
	  {
			// stars
			for(int pn=0; pn<Nstars; pn++){
			  double r = sqrt(SQ(Pbody[pn].Pos[0]) + SQ(Pbody[pn].Pos[1]));

			  if(maxDistanceTest >= 0) 
			  {
			  	if(r>=maxDistanceTest) continue; // wihtin 3 kpc
			  }
			  else
			  {
			  	if(fabs(Pbody[pn].Pos[1])>1.) continue;
			  }

			  double dist = sqrt(SQ(Pbody[pn].Pos[0]) + SQ(Pbody[pn].Pos[1]-8.));
		          double b = atan(Pbody[pn].Pos[2]/dist);
	        	  double b_deg = b*180/M_PI;

			  if(b_deg>-3) continue;
			  if(b_deg<-9) continue;

			  int k = 2;
			  if(b_deg>-5){
			    k = 0;
			  }else if(b_deg>-7){ 
			    k = 1;
		          }
			
		          double l = atan(Pbody[pn].Pos[0]/(9.-Pbody[pn].Pos[1])); 
			  double l_deg = l*180./M_PI;

			  double vlos1 = Pbody[pn].Vel[1]*cos(l) + Pbody[pn].Vel[0]*sin(l);
			  double vlos = Pbody[pn].Vel[2]*sin(b) + vlos1*cos(b);

			  if( l_deg <= RrotMin || l_deg >= RrotEnd )    continue;
			  int i = (int)((l_deg-RrotMin)/dR);

			  Ns[i][k] += 1;
			  Vlos[i][k] += vlos;
			  vp[i][k].push_back(vlos);			  
		       }
	
		      if(rank+1 < nProcs)
		      {		
			//Get the data from the other processes
			int nNew = 0;
			MPI_Status status;
			MPI_Recv(&nNew, 1, MPI_INT, rank+1, 123, MPI_COMM_WORLD, &status);
			
			fprintf(stderr,"Going to receive data from rank: %d, stars: %d \n", rank+1, nNew);
			
			//Resize the receive buffer
			delete[] Pbody;
			Pbody = new StructPbody[nNew];
			
			
			for(int writeIdx=0; writeIdx < nNew; writeIdx += nSendMax)
			{
				int nToSend = std::min(nSendMax, nNew-writeIdx);		
				fprintf(stderr, "Rank receiving: %d items from %d \n", nToSend, rank+1);	
				MPI_Recv(&Pbody[writeIdx], nToSend*sizeof(StructPbody), MPI_BYTE, rank+1, 124, MPI_COMM_WORLD, &status);
			}
			Nstars = nNew;
		     }
			
	  }
  }

    // average
    //Mass[0] = Sigs[0];
    for(int k=0;k<3;k++){
      for(int i=0; i<iMax; i++) {
		if(Ns[i][k] != 0)
		{
			Vlos[i][k] /= (float)Ns[i][k];
			float tmp = 0.0;
			for(int j=0;j<(int)vp[i][k].size();j++)
			{
				tmp += (vp[i][k][j]-Vlos[i][k])*(vp[i][k][j]-Vlos[i][k]);
			}
			Vlos_d[i][k] = sqrt(tmp/(float)Ns[i][k]);
		}	
      }
    }

    out.setf(ios::scientific);
    out.precision(6);
    
    for(int i=0; i<iMax; i++){
      out << Rs[i] << "  " ;
      for(int k=0;k<3;k++){
	out<< Vlos[i][k]*VelUnit << "  "  // 1,2
	   << Vlos_d[i][k]*VelUnit << "  " << Ns[i][k] << "  ";
      }
      out << endl;
    }
}


