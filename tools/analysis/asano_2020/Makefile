CXX = mpicxx
CC  = mpicc
LD  = mpicxx

OMPFLAGS  = -fopenmp 
#OMPFLAGS += -D_GLIBCXX_PARALLEL

OFLAGS = -O3 -g -Wall -ffast-math -funroll-loops  
# OMPFLAGS=

CXXFLAGS =  -fPIC $(OFLAGS) $(OMPFLAGS) -I ../ -std=c++11 
CXXFLAGS += -Werror
CXXFLAGS += -Wno-literal-suffix  # ignores warrning in openmpi 1.6.5

LDFLAGS = 

SRCPATH = ./

SRC1 = cvt_bonsai2ASCII_agama.cpp
SRC2 = cvt_bonsai2ASCII_mod1000.cpp
SRC3 = cvt_bonsai2ASCII_rotate_mod1000.cpp
SRC4 = cvt_bonsai2ASCII_rotate_mod10.cpp
SRC5 = cvt_bonsai2ASCII_rotate_mod2.cpp
SRC6 = extract_local_cyl.cpp
SRC7 = extract_local_cyl_non_rotate.cpp
SRC8 = extract_rec.cpp
SRC9 = extract_SN.cpp
SRC10 = extract_SN2.cpp
SRC11 = extract_SN3.cpp
SRC12 = fourier_decomposition.cpp
SRC13 = fourier_decomposition_v2.cpp
SRC14 = fourier_decomposition_v3.cpp
SRC15 = make_2dhist.cpp
SRC16 = make_2dhist_rotate.cpp
SRC17 = ring_extract.cpp
SRC18 = trace_orbit_v2.cpp
SRC19 = cvt_bonsai2ASCII.cpp

PROG1 = cvt_bonsai2ASCII_agama
PROG2 = cvt_bonsai2ASCII_mod1000
PROG3 = cvt_bonsai2ASCII_rotate_mod1000
PROG4 = cvt_bonsai2ASCII_rotate_mod10
PROG5 = cvt_bonsai2ASCII_rotate_mod2
PROG6 = extract_local_cyl
PROG7 = extract_local_cyl_non_rotate
PROG8 = extract_rec
PROG9 = extract_SN
PROG10 = extract_SN2
PROG11 = extract_SN3
PROG12 = fourier_decomposition
PROG13 = fourier_decomposition_v2
PROG14 = fourier_decomposition_v3
PROG15 = make_2dhist
PROG16 = make_2dhist_rotate
PROG17 = ring_extract
PROG18 = trace_orbit_v2
PROG19 = cvt_bonsai2ASCII

RM = /bin/rm

all:	  $(PROG1) $(PROG2) $(PROG3) $(PROG4) $(PROG5) $(PROG6) $(PROG7) $(PROG8) $(PROG9) $(PROG10) $(PROG11) $(PROG12) $(PROG13) $(PROG14) $(PROG15) $(PROG16) $(PROG18) $(PROG19)



$(PROG1): $(SRC1) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG2): $(SRC2) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG3): $(SRC3) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG4): $(SRC4) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG5): $(SRC5) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG6): $(SRC6) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG7): $(SRC7) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG8): $(SRC8) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG9): $(SRC9) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG10): $(SRC10) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG11): $(SRC11) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG12): $(SRC12) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG13): $(SRC13) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS) -lfftw3f -lm
$(PROG14): $(SRC14) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS) -lfftw3f -lm
$(PROG15): $(SRC15) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG16): $(SRC16) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG17): $(SRC17) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG18): $(SRC18) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG19): $(SRC19) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)


clean:	  
	-$(RM) $(PROG1) 
	-$(RM) $(PROG2) 
	-$(RM) $(PROG3) 
	-$(RM) $(PROG4) 
	-$(RM) $(PROG5) 
	-$(RM) $(PROG6) 
	-$(RM) $(PROG7) 
	-$(RM) $(PROG8) 
	-$(RM) $(PROG9) 
	-$(RM) $(PROG10) 
	-$(RM) $(PROG11) 
	-$(RM) $(PROG12) 
	-$(RM) $(PROG13) 
	-$(RM) $(PROG14) 
	-$(RM) $(PROG15) 
	-$(RM) $(PROG16) 
	-$(RM) $(PROG17) 
	-$(RM) $(PROG18) 
	-$(RM) $(PROG19) 

