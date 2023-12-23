//#include "config.h"
#include  <cmath>
#include  <fstream>
#include  <iostream>
#include <sstream>
#include <vector>
#include "Astro.h"
#include "Constants.h"
#include <memory>
#include <algorithm>
#include "../BonsaiIO.h"
#include "../IDType.h"
using namespace std;


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
struct StructPbody *Pbody_all;

#define NMAX_test_par 524288
struct Struct_ID_Type{
	uint64_t ID;
	int Type;
};
struct Struct_ID_Type *test_ID_Type;

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

vector<string> split(string& input, char delimiter){
	istringstream stream(input);
	string field;
	vector<string> result;
	while(getline(stream, field, delimiter)){
		result.push_back(field);
	}
	return result;
}

int read_csv(string filename, vector< vector<string> > &out_vector){
	ifstream ifs(filename);
	if(!ifs){
		cout << filename <<": cannot be read" << endl;
		getchar();
		exit(0);
	}
	string line;
	int row_num = 0;
	while (getline(ifs, line)) {
		vector<string> strvec = split(line, ',');
		out_vector.push_back(strvec);
		row_num++;
	}
	return row_num;
}

double read(
    const int myRank, 
		const int nRank,
		const MPI_Comm &comm,
		string filename,
		const int test_par_num,
		StructPbody Pbody_all[],
		Struct_ID_Type test_ID_Type[],
    const bool restartFlag  // = true,
    )
{	
  double dtRead = 0;
  double t0 = MPI_Wtime();
	BonsaiIO::Core in(myRank, nRank, comm, BonsaiIO::READ, filename);
	auto idt  = std::make_shared<BonsaiIO::DataType<IDType>>("Stars:IDType");
	auto pos  = std::make_shared<BonsaiIO::DataType<float4>>("Stars:POS:real4");
	auto vel  = std::make_shared<BonsaiIO::DataType<float3>>("Stars:VEL:float[3]");
	std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> dataDM;
	dataDM.push_back(idt);
	dataDM.push_back(pos);
	dataDM.push_back(vel);
	const int reduceDM = 1;

  for (auto &type : dataDM)
  {
    if (myRank == 0)
      fprintf(stderr, " Reading %s ...\n", type->getName().c_str());
    if (in.read(*type, restartFlag, reduceDM))
    {
      long long int nLoc = type->getNumElements();
      long long int nGlb;
      MPI_Allreduce(&nLoc, &nGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    }
    else if (myRank == 0)
    {
      fprintf(stderr, " %s  is not found, skipping\n", type->getName().c_str());
      fprintf(stderr, " ---- \n");
    }

  }
  
  const auto n = idt->size();
	long double total_Z = 0; 
	long double total_VZ = 0;
	//If type is not long double but float or double, correct average value cannot be calculated.

  Pbody = new StructPbody[test_par_num];
  int i = 0; //test particle number read in each process
	int test_par_num_read = 0; //test particle number read in all processes
	long int par_num_read = 0; //all particle number read in each process
	//binary search
	vector<pair<long long, int>> list_pair_idtype;
	for (int j = 0; j < test_par_num; j++) {
		long long ID_push = test_ID_Type[j].ID;
		int Type_push = test_ID_Type[j].Type;
		list_pair_idtype.push_back(pair<long long, int>(ID_push, Type_push));
	}
	sort(list_pair_idtype.begin(), list_pair_idtype.end());
	//
	for(size_t z=0; z < n; z++)
  {      
		par_num_read++;
		total_Z += pos->operator[](z)[2];
		total_VZ += vel->operator[](z)[2];

		/* 
		//Linear search
		for (int j = 0; j < test_par_num; j++) {
			if((idt->operator[](z).getID() == test_ID_Type[j].ID)
					&&(idt->operator[](z).getType() == test_ID_Type[j].Type)
					){
				Pbody[i].ID     = idt->operator[](z).getID();
				Pbody[i].Type   = idt->operator[](z).getType();
				Pbody[i].Mass   = pos->operator[](z)[3];
				Pbody[i].Pos[0] = pos->operator[](z)[0];
				Pbody[i].Pos[1] = pos->operator[](z)[1];
				Pbody[i].Pos[2] = pos->operator[](z)[2];

				Pbody[i].Vel[0] = vel->operator[](z)[0];
				Pbody[i].Vel[1] = vel->operator[](z)[1];
				Pbody[i].Vel[2] = vel->operator[](z)[2];
				i++;
				break;
			}
		}
		*/
		//Binary search
		if (binary_search(list_pair_idtype.begin(), list_pair_idtype.end(), 
					pair<long long, int>(idt->operator[](z).getID(),idt->operator[](z).getType())) 
			 ){
			Pbody[i].ID     = idt->operator[](z).getID();
			Pbody[i].Type   = idt->operator[](z).getType();
			Pbody[i].Mass   = pos->operator[](z)[3];
			Pbody[i].Pos[0] = pos->operator[](z)[0];
			Pbody[i].Pos[1] = pos->operator[](z)[1];
			Pbody[i].Pos[2] = pos->operator[](z)[2];

			Pbody[i].Vel[0] = vel->operator[](z)[0];
			Pbody[i].Vel[1] = vel->operator[](z)[1];
			Pbody[i].Vel[2] = vel->operator[](z)[2];
			i++;
		}
		//
		
		MPI_Allreduce(&i, &test_par_num_read, 1, MPI_INT, MPI_SUM, comm);
		if(test_par_num_read == test_par_num) break;
	} 
	long double total_Z_all;
	long int par_num_read_all;
	MPI_Allreduce(&par_num_read, &par_num_read_all, 1, MPI_LONG, MPI_SUM, comm);
	MPI_Allreduce(&total_Z, &total_Z_all, 1, MPI_LONG_DOUBLE, MPI_SUM, comm);
	double avgZ = total_Z_all/par_num_read_all;
	MPI_Allreduce(&total_VZ, &total_Z_all, 1, MPI_LONG_DOUBLE, MPI_SUM, comm);
	double avgVZ = total_Z_all/par_num_read_all;
	//they cannot be correct average values? if test particle selection is biased
	if(i!=0){
		for (int j = 0; j < i; j++) {
			Pbody[j].Pos[2] -= avgZ;
			Pbody[j].Vel[2] -= avgVZ;
		}
	}

	int *Rcounts, *Displs;
	Rcounts = new int[nRank];
	Displs = new int[nRank+1];
	MPI_Allgather(&i, 1, MPI_INT, Rcounts, 1, MPI_INT, comm);
	Displs[0] = 0;
	for (int j = 0; j < nRank; j++) {
		Displs[j+1] = Displs[j] + Rcounts[j];
	}
	MPI_Datatype MPI_StructPbody;
	Create_MPI_Pbody(MPI_StructPbody);
	MPI_Allgatherv(&Pbody[0], i, MPI_StructPbody, &Pbody_all[0], Rcounts, Displs, MPI_StructPbody, comm);
	delete[] Pbody;
	delete[] Rcounts;
	delete[] Displs;

  dtRead += MPI_Wtime() - t0;
	in.close();
  return dtRead;
}//end read function


////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]){
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    int nRank, myRank;
    MPI_Comm_size(comm, &nRank);
    MPI_Comm_rank(comm, &myRank);

    fprintf(stderr,"Usage: prog ID_Type_list snap_list bar_angle_list outfile\n");

		if(argc != 5)
			exit(0);

    const string ID_Type_filename (argv[1]);
    if (myRank == 0) {
	    fprintf(stderr,"Number of processes: %d \n", nRank);
    }
		test_ID_Type = new Struct_ID_Type[NMAX_test_par];
		int test_par_num;
		vector< vector<string> > tmp;
		test_par_num = read_csv(ID_Type_filename, tmp);
		for (int i = 0; i < test_par_num; i++) {
			test_ID_Type[i].ID = stoi(tmp[i][0]);
			test_ID_Type[i].Type = stoi(tmp[i][1]);
		}

		const string snap_list_filename(argv[2]);
		vector< vector<string> > vec_snap_list;
		int snap_num;
		snap_num = read_csv(snap_list_filename, vec_snap_list);

    if (myRank == 0) {
			cout << "trace orbits of " << test_par_num << " particles" << endl;
			cout << "from " << snap_num << " snapshots" << endl;
    }
		Pbody_all = new StructPbody[test_par_num];

		const string bar_angle_list_filename(argv[3]);
		vector< vector<string> > vec_bar_angle;
		read_csv(bar_angle_list_filename, vec_bar_angle);
		MPI_Barrier(MPI_COMM_WORLD);

		const string out_filename(argv[4]);
    ofstream out1(out_filename);
		if(myRank==0){
			out1 << "#ID,,";
			for (int i = 0; i < test_par_num; i++) {
					out1 << test_ID_Type[i].ID << "," 
						<< test_ID_Type[i].ID << "," 
						<< test_ID_Type[i].ID << "," 
						<< test_ID_Type[i].ID << "," 
						<< test_ID_Type[i].ID << "," 
						<< test_ID_Type[i].ID; 
					if(i != test_par_num-1) out1 << ",";
					else out1 << endl;
				}
			out1 << "#Type,,";
			for (int i = 0; i < test_par_num; i++) {
				out1 << test_ID_Type[i].Type << ","
					<< test_ID_Type[i].Type << ","
					<< test_ID_Type[i].Type << ","
					<< test_ID_Type[i].Type << ","
					<< test_ID_Type[i].Type << ","
					<< test_ID_Type[i].Type;
				if(i != test_par_num-1) out1 << ",";
				else out1 << endl;
			}
			out1 << "t,bar_angle,"; 
			for (int i = 0; i < test_par_num; i++) {
				out1 <<"x,y,z,Vx,Vy,Vz";
				if(i != test_par_num-1) out1 << ",";
				else out1 << endl;
			}
		}

		string snap_filename;
		string bar_angle;
		for (auto vec_1d_filename : vec_snap_list) {
			snap_filename = vec_1d_filename[0];	
			if(myRank==0) cout << "reading " << snap_filename << endl; 
			// snap id (= t)
			//if(myRank==0) out1 << split(split(snap_filename, '.')[3], '_')[3] << ",";	
			//path in the NAS
			if(myRank==0) out1 << split(split(snap_filename, '.')[0], '_')[3] << ",";	
			// bar angle
			for(auto v_snap_angle : vec_bar_angle){
				if(v_snap_angle[0] == snap_filename) bar_angle = v_snap_angle[1];
			}
			if(myRank==0) out1 << bar_angle << ",";
			//READ
			cout << "dt=" << read(myRank, nRank, comm, snap_filename, test_par_num, Pbody_all, test_ID_Type, true) << endl;

			// must be carefull for the output order 
			// the order of test_ID_Type[].ID is not equal to that of Pbody_SN_tot[].ID 
			if(myRank==0){
				for (int i = 0; i < test_par_num; i++) {
					for (int j = 0; j < test_par_num; j++) {
						if ((test_ID_Type[i].ID == Pbody_all[j].ID)&&(test_ID_Type[i].Type == Pbody_all[j].Type)){
							out1 << Pbody_all[j].Pos[0] << "," 
								<< Pbody_all[j].Pos[1] << "," 
								<< Pbody_all[j].Pos[2] << "," 
								<< Pbody_all[j].Vel[0] << "," 
								<< Pbody_all[j].Vel[1] << "," 
								<< Pbody_all[j].Vel[2]; 

						}
					}
					if(i != test_par_num-1) out1 << ",";
					else out1 << endl;
				}
			}
		}
		out1.close();

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();


		return 0;
}


