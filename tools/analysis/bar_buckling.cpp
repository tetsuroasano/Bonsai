#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <memory>
#include <vector>
#include <omp.h>
#include "BonsaiIO.h"
#include "IDType.h"

typedef float float4[4];
typedef float float3[3];
typedef float float2[2];

const float DR = 0.5;
const float R_MAX = 15.;
const int N_R = ceil(R_MAX / DR);
const int FOURIER_M_MAX = 5;

int read(
    const int rank, const MPI_Comm &comm,
    const std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> &data,
    BonsaiIO::Core &in,
    const int reduce,
    std::shared_ptr<BonsaiIO::DataType<IDType>> &idt,
    std::shared_ptr<BonsaiIO::DataType<float4>> &pos,
    const bool restartFlag  // = true,
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
	std::cerr << "Read time: " << dtRead << std::endl;

  const auto n = idt->size();
  fprintf(stderr, "Rank: %d has read: %ld items \n", rank, n);

  return n;

}

void calc(
		int n_tot,
		std::vector<int> &num,
		std::vector<float> &a_bpx,
		std::vector<std::vector<float>> &a_buck,
		std::shared_ptr<BonsaiIO::DataType<float4>> &pos)
{
  double dtCalc = 0;
	double t0 = MPI_Wtime();

	std::vector<std::vector<float>> tmp_cos(N_R, std::vector<float>(FOURIER_M_MAX));
	std::vector<std::vector<float>> tmp_sin(N_R, std::vector<float>(FOURIER_M_MAX));
	std::vector<float> avg_pos(3);

	for (int i = 0; i < n_tot; i++) {
		avg_pos[0] += pos->operator[](i)[0];
		avg_pos[1] += pos->operator[](i)[1];
		avg_pos[2] += pos->operator[](i)[2];
	}
	avg_pos[0] /= (float)n_tot;
	avg_pos[1] /= (float)n_tot;
	avg_pos[2] /= (float)n_tot;

	for (auto i = 0; i < n_tot; i++) {
		float x = pos->operator[](i)[0] - avg_pos[0];
		float y = pos->operator[](i)[1] - avg_pos[1];
		float z = pos->operator[](i)[2] - avg_pos[2];

		int R_idx = (int)floor(sqrt(x*x + y*y)/DR);
		if (R_idx >= N_R) continue;
		float phi = atan2(y, x);

		num[R_idx]++;
		a_bpx[R_idx] += z*z;
		for (int m = 0; m <= FOURIER_M_MAX; m++) {
			tmp_cos[R_idx][m] += z*cos(m*phi);
			tmp_sin[R_idx][m] += z*sin(m*phi);
		}
	}


	for (int r = 0; r < N_R; r++) {
		a_bpx[r] /= (float)num[r];
		a_bpx[r] = sqrt(a_bpx[r]);
		for (int m = 0; m <= FOURIER_M_MAX; m++) {
			tmp_cos[r][m] /= (float)num[r];
			tmp_sin[r][m] /= (float)num[r];
			a_buck[r][m] = sqrt(tmp_cos[r][m]*tmp_cos[r][m]
					+ tmp_sin[r][m]*tmp_sin[r][m]);
		}
	}

	dtCalc += MPI_Wtime() - t0;
	std::cerr << "Calculation time: " << dtCalc << std::endl;
}

int main(int argc, char *argv[]){

	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);
	int nRank, myRank;
	MPI_Comm_size(comm, &nRank);
	MPI_Comm_rank(comm, &myRank);

	fprintf(stderr,"Usage: prog inFile outFile\n");

	if(argc < 3) exit(0);
	const std::string filename (argv[1]);
	const std::string outfile (argv[2]);

	if (myRank == 0) {
		fprintf(stderr,"Number of processes: %d \n", nRank);
		fprintf(stderr,"Filename: %s \n", filename.c_str());
	}

	// Read a bosai file
	BonsaiIO::Core in(myRank, nRank, comm, BonsaiIO::READ, filename);
	auto idt = std::make_shared<BonsaiIO::DataType<IDType>>("Stars:IDType");
	auto pos = std::make_shared<BonsaiIO::DataType<float4>>("Stars:POS:real4");
	std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> dataS;
	dataS.push_back(idt);
	dataS.push_back(pos);
	const int reduceDM = 1;
	const auto n_tot = read(myRank, comm, dataS, in, reduceDM, idt, pos, true);
	in.close();

	// Calc A_BPX, A_buckle
	std::vector<int> num(N_R);
	std::vector<float> a_bpx(N_R);
	std::vector<std::vector<float>> a_buck(N_R, std::vector<float>(FOURIER_M_MAX + 1));
	calc(n_tot, num, a_bpx, a_buck, pos);

	std::ofstream ofs(outfile);
	if (myRank == 0) {
		ofs << "R\tN\tA_BPX\t";
		for (int m = 1; m <= FOURIER_M_MAX; m++) {
			if (m ==  FOURIER_M_MAX) ofs << "A_buckle(" << m <<")\n";
			else ofs << "A_buckle(" << m <<")\t";
		}

		for (int r = 0; r < N_R; r++) {
			ofs << DR*r << "\t" << num[r] << "\t" << a_bpx[r] << "\t";
			for (int m = 1; m <= FOURIER_M_MAX; m++) {
				if (m ==  FOURIER_M_MAX) ofs << a_buck[r][m] <<"\n";
				else ofs << a_buck[r][m] <<"\t";
			}
		}
	}

	MPI_Finalize();
	return 0;
}
