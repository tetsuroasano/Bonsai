
#ifndef	__ASTRO_H_INCLUDED__
#define	__ASTRO_H_INCLUDED__

#define MaxCharactersInLine (256)

#define	SQ(x)		((x)*(x))
#define	CUBE(x)		((x)*(x)*(x))
#define	SWAP(type,x,y)	{type t = x; x = y; y = t; }
//#define	swap(x,y)	(tmp=(x),(x)=(y),(y)=tmp)
#define	MAX(x,y)	(((x)>(y))?(x):(y))
#define	MIN(x,y)	(((x)<(y))?(x):(y))
#define	ABS(x)		(((x)>0)?(x):(-(x)))
#define	SGN(x)		(((x)>0)?(+1):(((x)<0)?(-1):(0)))
#define BitSign(x)      (((x)==1)?(+1):(-1))
#define Sign(x)         (((x)>=0)?(+1):(-1))
#define Parity(x)   (((x)%2==0)?(+1):(-1))

#define	NORM(x)		( sqrt( SQ((x)[0]) + SQ((x)[1]) + SQ((x)[2]) ) )
#define	NORM2(x)	( SQ((x)[0]) + SQ((x)[1]) + SQ((x)[2]) )
#define	DISTANCE(x,y)	( sqrt(SQ((x)[0]-(y)[0])+SQ((x)[1]-(y)[1])+SQ((x)[2]-(y)[2])) )
#define	DISTANCE2(x,y)	(SQ((x)[0]-(y)[0])+SQ((x)[1]-(y)[1])+SQ((x)[2]-(y)[2]))
#define	DOT_PRODUCT(x,y)	(((x)[0]*(y)[0])+((x)[1]*(y)[1])+((x)[2]*(y)[2]))

//typedef char bool;
//typedef _Bool bool;
#define TRUE  (true)
#define FALSE (false)

#define ON      (1)
#define OFF     (0)
#define NONE    (-1)

#define PI (3.1415926535897932384626433832795)
#define IPI (0.31830988618379067153776752674503)

#define bprint(f)       printf("*** " #f " = %d\n", (int)f)
#define dprint(f)       printf("*** " #f " = %d\n", f)
#define zdprint(f)      printf("*** " #f " = %zd\n", f)
#define dlprint(f)      printf("*** " #f " = %ld\n", f)
#define fprint(f)       printf("*** " #f " = %f\n", f)
#define eprint(f)       printf("*** " #f " = %e\n", f)
#define gprint(f)       printf("*** " #f " = %g\n", f)
#define sprint(f)       printf("*** " #f " = %s\n", f)
#define pprint(f)       printf("*** " #f " = %p\n", f)
#define dbgprt(f)       printf("DBG print %s:line%d:%s()\n",__FILE__,__LINE__,__FUNCTION__)

#define bprintl(f)       printf("*** " #f " = %d:%s:line%d:%s()\n", (int)f,__FILE__,__LINE__,__FUNCTION__)
#define dprintl(f)       printf("*** " #f " = %d:%s:line%d:%s()\n", f,__FILE__,__LINE__,__FUNCTION__)
#define zdprintl(f)      printf("*** " #f " = %zd:%s:line%d:%s()\n", f,__FILE__,__LINE__,__FUNCTION__)
#define dlprintl(f)      printf("*** " #f " = %ld:%s:line%d:%s()\n", f,__FILE__,__LINE__,__FUNCTION__)
#define fprintl(f)       printf("*** " #f " = %f:%s:line%d:%s()\n", f,__FILE__,__LINE__,__FUNCTION__)
#define eprintl(f)       printf("*** " #f " = %e:%s:line%d:%s()\n", f,__FILE__,__LINE__,__FUNCTION__)
#define gprintl(f)       printf("*** " #f " = %g:%s:line%d:%s()\n", f,__FILE__,__LINE__,__FUNCTION__)
#define sprintl(f)       printf("*** " #f " %s:line%d:%s()\n", __FILE__,__LINE__,__FUNCTION__)
//#define sprintl(f)       printf("*** " #f " = %s:%s:line%d:%s()\n", f,__FILE__,__LINE__,__FUNCTION__)
#define pprintl(f)       printf("*** " #f " = %p:%s:line%d:%s()\n", f,__FILE__,__LINE__,__FUNCTION__)

#define FileOpen(fp, fname, mode)				\
	(fp) = fopen((fname),(mode));				\
	if ((fp) == NULL){					\
		fprintf(stderr,"Can't open file %s\n",(fname));	\
		exit(1);					\
	}

#define Snprintf(buf,expression, ...)                       \
    if( snprintf((buf),MaxCharactersInLine,expression, __VA_ARGS__ ) >= (MaxCharactersInLine) ){ \
        fprintf(stderr,"buffer overflow: %s:line%d:%s\n",       \
                __FILE__,__LINE__,__FUNCTION__);              \
        exit(1);                                            \
    }


#endif /* ASTRO_H_INCLUDED */


