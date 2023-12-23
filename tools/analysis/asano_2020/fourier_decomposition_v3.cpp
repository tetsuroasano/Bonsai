#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <memory>
#include <vector>
#include <fftw3.h>
#include "Astro.h"
#include "Constants.h"
#include "../BonsaiIO.h"
#include "../IDType.h"

using namespace std;

#define NMAX 30000000
struct StructPbody{
    //uint64_t ID;
    //double Mass;
    double Pos[3];
    //double Vel[3];
};
struct StructPbody *Pbody;

//r-phi map parameters
#define N_phi 256
#define N_phi_max 1024
#define N_R 128
#define N_R_max 256
#define unit_length 0.2
////////////////////////

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
    //std::shared_ptr<BonsaiIO::DataType<float3>> &vel,
    const bool restartFlag,  // = true,
    const float avgZ
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
  Pbody = new StructPbody[n];
  fprintf(stderr, "Rank: %d has read: %ld items \n", rank, n);
  int i = 0;
  for(size_t z=0; z < n; z++)
  {
		//Pbody[i].ID     = idt->operator[](z).getID();
		//Pbody[i].Mass   = pos->operator[](z)[3];
		Pbody[i].Pos[0] = pos->operator[](z)[0];
		Pbody[i].Pos[1] = pos->operator[](z)[1];
		Pbody[i].Pos[2] = pos->operator[](z)[2] - avgZ;

		//Pbody[i].Vel[0] = vel->operator[](z)[0];
		//Pbody[i].Vel[1] = vel->operator[](z)[1];
		//Pbody[i].Vel[2] = vel->operator[](z)[2] - avgVZ;
		i++;
	} //for

  return i;

}

void calc_surface_density(const int rank, const MPI_Comm &comm, float Omega[][N_phi_max], float Omega_sum[][N_phi_max], int N_particle){
	float phi;
  float	delta_phi = 2.0*M_PI/N_phi;
	float R;
	int index_R, index_phi;

	for (int i = 0; i < N_R_max; i++) {
		for (int j = 0; j < N_phi_max; j++) {
			Omega[i][j] = 0;
			Omega_sum[i][j] = 0;
		}
	}

	for (int i = 0; i < N_particle; i++) {
		if(Pbody[i].Pos[0]>=0.0){
			R = sqrt(Pbody[i].Pos[0]*Pbody[i].Pos[0] +Pbody[i].Pos[1]*Pbody[i].Pos[1]);
			// add 2pi so that phi will not be negative
			phi = atan(Pbody[i].Pos[1]/Pbody[i].Pos[0]) + 2.0*M_PI; 
			index_R = (int)floor(R/unit_length);
			index_phi = ((int)floor(phi/delta_phi))%N_phi;
			
			Omega[index_R][index_phi] += 1;
		}

		if(Pbody[i].Pos[0]<0.0){
			R = sqrt(Pbody[i].Pos[0]*Pbody[i].Pos[0] +Pbody[i].Pos[1]*Pbody[i].Pos[1]);
			phi = atan(Pbody[i].Pos[1]/Pbody[i].Pos[0]) + M_PI; 
			index_R = (int)floor(R/unit_length);
			index_phi = (int)floor(phi/delta_phi);
			
			Omega[index_R][index_phi] += 1;
		}
	}

	MPI_Allreduce(Omega, Omega_sum, N_R_max*N_phi_max, MPI_FLOAT, MPI_SUM, comm);

	//Normalization
	long int count_tot = 0;
	for (int i = 0; i < N_R; i++) {
		count_tot = 0;
		for (int j = 0; j < N_phi; j++) {
			count_tot += Omega_sum[i][j];
		}
		for (int j = 0; j < N_phi; j++) {
			Omega_sum[i][j] = Omega_sum[i][j]/count_tot;
			// if count_R=0, Omega will be nan
		}
	}
	delete[] Pbody;
}

// Omega is normalized at each radius
void fourier_decomposition(float Omega[][N_phi_max], float phase[][N_R_max], float Amp[][N_R_max]){
	float angle_a;

	for (int i = 0; i < N_R; i++) {
		fftwf_complex *fft_in, *fft_out;
		fftwf_plan plan;
		const int fftsize = N_phi;
		fft_in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftsize);
		fft_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftsize);
		plan = fftwf_plan_dft_1d(fftsize, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
		for (int j = 0; j < N_phi; j++) {
			fft_in[j][0] = Omega[i][j];
			fft_in[j][1] = 0.0;
		}

		fftwf_execute(plan);

		for (int mode = 2; mode <= 10; mode++) {
			Amp[mode][i] = fft_out[mode][0]*fft_out[mode][0]+fft_out[mode][1]*fft_out[mode][1];

			//phase angles are between 0 and pi
			angle_a = atan2(-fft_out[mode][1], fft_out[mode][0])/mode;
			while (angle_a < 0) {
				angle_a += 2*M_PI/mode;
			}
			phase[mode][i] = angle_a;
		}
		
		fftwf_destroy_plan(plan);
		fftwf_free(fft_in);
		fftwf_free(fft_out);
	}
}



int main(int argc, char *argv[]){
	
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);
	int nRank, myRank;
	MPI_Comm_size(comm, &nRank);
	MPI_Comm_rank(comm, &myRank);

	int  N_particle;
	float phase[16][N_R_max];
	float Amp[16][N_R_max];
	float Omega[N_R_max][N_phi_max];
	float Omega_sum[N_R_max][N_phi_max];

	fprintf(stderr,"Usage: prog inFile outFile\n");

	if(argc < 3) exit(0);
	const string filename (argv[1]);

	const string Outfile (argv[2]);
	ofstream ofs(Outfile);
	if(myRank==0) ofs << "R"<<"\t";
	if(myRank==0) {
		for (int i = 2; i <= 10; i++) {
			ofs << "m" << i << "\t";
		}
		for (int i = 2; i < 10; i++) {
			ofs << "Amp" << i << "\t";
		}
			ofs << "Amp" << 10 << endl;
	}
	
	
	if (myRank == 0) {
		fprintf(stderr,"Number of processes: %d \n", nRank);
		fprintf(stderr,"Filename: %s \n", filename.c_str());
	}

	//Read
	BonsaiIO::Core in(myRank, nRank, comm, BonsaiIO::READ, filename);
	auto idt = std::make_shared<BonsaiIO::DataType<IDType>>("Stars:IDType");
	auto pos = std::make_shared<BonsaiIO::DataType<float4>>("Stars:POS:real4");
	std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> dataDM;
	dataDM.push_back(idt);
	dataDM.push_back(pos);
	const int reduceDM = 1;
	N_particle = read(myRank, comm, dataDM, in, reduceDM, idt, pos, true, 0);
	in.close();

	//FFT
	calc_surface_density(myRank, comm, Omega, Omega_sum, N_particle);
	fourier_decomposition(Omega_sum, phase, Amp);

	//Out
	if(myRank==0){
		for (int i = 0; i < N_R; i++) {
			ofs << unit_length*(i+0.5) << "\t";
			for (int j = 2; j <= 10; j++) {
				ofs << phase[j][i] << "\t";
			}
			for (int j = 2; j < 10; j++) {
				ofs << Amp[j][i] << "\t";
			}
			ofs << Amp[10][i] << endl;
		}
	}


	MPI_Finalize();
	return 0;
}
