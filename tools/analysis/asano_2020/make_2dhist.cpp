#include "../BonsaiIO.h"
#include "../IDType.h"
#include <memory>
#include <cmath>
#include <fstream>


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




int main(int argc, char * argv[])
{
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
    
  int nRank, myRank;
  MPI_Comm_size(comm, &nRank);
  MPI_Comm_rank(comm, &myRank);

  if (argc < 4)
  {
    if (myRank == 0)
    {
      fprintf(stderr, " ------------------------------------------------------------------------\n");
      fprintf(stderr, " Usage: \n");
      fprintf(stderr, " %s  fileIn reduceDM reduceStars  > output \n", argv[0]);
      fprintf(stderr, " ------------------------------------------------------------------------\n");
    }
    exit(-1);
  }
  
  const std::string fileIn (argv[1]);
  const int reduceDM = atoi(argv[2]);
  const int reduceS  = atoi(argv[3]);

  if (myRank == 0)
  {
    fprintf(stderr, " Input file:  %s\n", fileIn.c_str());
    fprintf(stderr, "    reduceStars= %d \n", reduceS);
    fprintf(stderr, "    reduceDM   = %d \n", reduceDM);
  }
  
  /************* read ***********/

  {
    const double tOpen = MPI_Wtime(); 
    BonsaiIO::Core in(myRank, nRank, comm, BonsaiIO::READ,  fileIn);
    double dtOpen = MPI_Wtime() - tOpen;


    if (myRank == 0)
      in.getHeader().printFields();
    
      

    double dtRead;
    if (reduceDM > 0) return -1;


		//Define parameters for the histgram
		std::vector<std::vector<long long>> hist2d(2400, std::vector<long long>(2400));
		const float xmin = -12.0;
		const float ymin = -12.0;
		const float dx = 0.01;

    if (reduceS > 0)
    {
      typedef float float4[4];
      typedef float float3[3];
      typedef float float2[2];
      auto idt  = std::make_shared<BonsaiIO::DataType<IDType>>("Stars:IDType");
      auto pos  = std::make_shared<BonsaiIO::DataType<float4>>("Stars:POS:real4");
      auto vel  = std::make_shared<BonsaiIO::DataType<float3>>("Stars:VEL:float[3]");
      auto rhoh = std::make_shared<BonsaiIO::DataType<float2>>("Stars:RHOH:float[2]");

      std::vector<std::shared_ptr<BonsaiIO::DataTypeBase>> dataDM;
      dataDM.push_back(idt);
      dataDM.push_back(pos);
      dataDM.push_back(vel);
      dataDM.push_back(rhoh);
      dtRead += read(myRank, comm, dataDM, in, reduceDM);
      
      const auto n = idt->size();
      
      for (size_t i = 0; i < n; i++)
			{
				float x = pos->operator[](i)[0];
				float y = pos->operator[](i)[1];
				int id_x = floor((x-xmin)/dx);
				int id_y = floor((y-ymin)/dx);
				if (id_x < 0 || id_x >= (int)hist2d.size()) continue;
				if (id_y < 0 || id_y >= (int)hist2d[0].size()) continue;
				hist2d[id_x][id_y]++;
			}
		}
		std::string fileOut = "";
		for (int i = 0; i < (int)fileIn.size(); i++) {
			fileOut += fileIn[i];
			if (fileIn[i]=='/') fileOut = "";
		}
		fileOut += ".hist2d.csv";
		std::ofstream ofs(fileOut);
		for (int i = 0; i < (int)hist2d.size(); i++) {
			for (int j = 0; j < (int)hist2d[i].size()-1; j++) {
				ofs << hist2d[i][j] <<",";
			}
			ofs << hist2d[i].back() <<"\n";
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
    if (myRank == 0)
    {
      fprintf(stderr, "dtOpen = %g sec \n", dtOpenGlb);
      fprintf(stderr, "dtRead = %g sec \n", dtReadGlb);
      fprintf(stderr, "dtClose= %g sec \n", dtCloseGlb);
      fprintf(stderr, "Read BW= %g MB/s \n", readBW/1e6);
    }
  }


  MPI_Finalize();




  return 0;
}


