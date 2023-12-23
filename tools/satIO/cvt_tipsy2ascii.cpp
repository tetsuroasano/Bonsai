#include "BonsaiIO.h"
#include "IDType.h"
#include "read_tipsy.h"
#include <array>


static IDType lGetIDType (const long long id)
{
  IDType ID;
  ID.setID(id);
  ID.setType(3);     /* Everything is Dust until told otherwise */
  if(id >= DISKID  && id < BULGEID)       
  {
    ID.setType(2);  /* Disk */
    ID.setID(id - DISKID);
  }
  else if(id >= BULGEID && id < DARKMATTERID)  
  {
    ID.setType(1);  /* Bulge */
    ID.setID(id - BULGEID);
  }
  else if (id >= DARKMATTERID)
  {
    ID.setType(0);  /* DM */
    ID.setID(id - DARKMATTERID);
  }
  return ID;
};

template<typename IO, size_t N>
static double writeDM(ReadTipsy &data, IO &out,
    std::array<size_t,N> &count)
{
  double dtWrite = 0;
  const int pCount  = data.firstID.size();
  /* write IDs */
  {
    BonsaiIO::DataType<IDType> ID("DM:IDType", pCount);
    for (int i = 0; i< pCount; i++)
    {
      ID[i] = lGetIDType(data.firstID[i]);
      assert(ID[i].getType() == 0);
      if (ID[i].getType() < count.size())
        count[ID[i].getType()]++;
    }
    double t0 = MPI_Wtime();
    out.write(ID);
    dtWrite += MPI_Wtime() - t0;
  }
  
  /* write pos */
  {
    BonsaiIO::DataType<ReadTipsy::real4> pos("DM:POS:real4",pCount);
    for (int i = 0; i< pCount; i++)
      pos[i] = data.firstPos[i];
    double t0 = MPI_Wtime();
    out.write(pos);
    dtWrite += MPI_Wtime() - t0;
  }
    
  /* write vel */
  {
    typedef float vec3[3];
    BonsaiIO::DataType<vec3> vel("DM:VEL:float[3]",pCount);
    for (int i = 0; i< pCount; i++)
    {
      vel[i][0] = data.firstVel[i].x;
      vel[i][1] = data.firstVel[i].y;
      vel[i][2] = data.firstVel[i].z;
    }
    double t0 = MPI_Wtime();
    out.write(vel);
    dtWrite += MPI_Wtime() - t0;
  }

  return dtWrite;
}

template<typename IO, size_t N>
static double writeStars(ReadTipsy &data, IO &out,
    std::array<size_t,N> &count)
{
  double dtWrite = 0;

  const int pCount  = data.secondID.size();

  /* write IDs */
  {
    BonsaiIO::DataType<IDType> ID("Stars:IDType", pCount);
    for (int i = 0; i< pCount; i++)
    {
      ID[i] = lGetIDType(data.secondID[i]);
      assert(ID[i].getType() > 0);
      if (ID[i].getType() < count.size())
        count[ID[i].getType()]++;
    }
    double t0 = MPI_Wtime();
    out.write(ID);
    dtWrite += MPI_Wtime() - t0;
  }
    
  /* write pos */
  {
    BonsaiIO::DataType<ReadTipsy::real4> pos("Stars:POS:real4",pCount);
    for (int i = 0; i< pCount; i++)
      pos[i] = data.secondPos[i];
    double t0 = MPI_Wtime();
    out.write(pos);
    dtWrite += MPI_Wtime() - t0;
  }
    
  /* write vel */
  {
    typedef float vec3[3];
    BonsaiIO::DataType<vec3> vel("Stars:VEL:float[3]",pCount);
    for (int i = 0; i< pCount; i++)
    {
      vel[i][0] = data.secondVel[i].x;
      vel[i][1] = data.secondVel[i].y;
      vel[i][2] = data.secondVel[i].z;
    }
    double t0 = MPI_Wtime();
    out.write(vel);
    dtWrite += MPI_Wtime() - t0;
  }

  return dtWrite;
}


int main(int argc, char * argv[])
{
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
    
  int nranks, rank;
  MPI_Comm_size(comm, &nranks);
  MPI_Comm_rank(comm, &rank);


  if (argc < 4)
  {
    if (rank == 0)
    {
      fprintf(stderr, " ------------------------------------------------------------------------\n");
      fprintf(stderr, " Usage: \n");
      fprintf(stderr, " %s  baseName nDomains outputName \n", argv[0]);
      fprintf(stderr, " ------------------------------------------------------------------------\n");
    }
    exit(-1);
  }
  
  const std::string baseName(argv[1]);
  const int nDomains = atoi(argv[2]);
  const std::string outputName(argv[3]);


  int reduceFactorFirst  = 1;
  int reduceFactorSecond = 1;
  
  if(argc > 4)
  {
	  reduceFactorFirst  = atoi(argv[4]);
	  reduceFactorSecond = atoi(argv[5]);
  }

  if(rank == 0)
	  fprintf(stderr,"Reducing DM: %d  Stars: %d \n", reduceFactorFirst, reduceFactorSecond);



  ReadTipsy data(
      baseName, 
      rank, nranks,
      nDomains, 
      reduceFactorFirst,
      reduceFactorSecond);

  long long nFirstLocal = data.firstID.size();
  long long nSecondLocal = data.secondID.size();

  long long nFirst, nSecond;
  MPI_Allreduce(&nFirstLocal, &nFirst, 1, MPI_LONG, MPI_SUM, comm);
  MPI_Allreduce(&nSecondLocal, &nSecond, 1, MPI_LONG, MPI_SUM, comm);

  if (rank == 0)
  {
    fprintf(stderr, " nFirst = %lld \n", nFirst);
    fprintf(stderr, " nSecond= %lld \n", nSecond);
    fprintf(stderr, " nTotal= %lld \n", nFirst + nSecond);
  }

  const double tAll = MPI_Wtime();
  
  double dtAllLoc = MPI_Wtime() - tAll;
  double dtAllGlb;
  MPI_Allreduce(&dtAllLoc, &dtAllGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
  if (rank == 0)
    fprintf(stderr, "All operations done in   %g sec \n", dtAllGlb);




  MPI_Finalize();




  return 0;
}


