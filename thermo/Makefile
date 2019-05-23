CXX = g++
CXXFLAGS = -O3 -Wall
INCLUDES = -I. -I../src -I$(CUDA_HDR_PATH)
OBJS = enrichment.o nucleation.o parabola625.o

all: check-nucleation
.PHONY: all

debug: CXXFLAGS += -O0 -g
debug: check-nucleation $(OBJS)

profile: CXXFLAGS += -O1 -pg
profile: $(OBJS)

# === Executables ===

check-nucleation: check-nucleation.cpp ../src/globals.h nucleation.c $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) $< -o $@ -lm

ternary-diagram.png: phase-diagram.py parabola625.c parabola625.so enrichment.so
	python $<

# === Generated Sources ===

enrichment.c: enrichment.py
	python $<

parabola625.c: CALPHAD_energies.py constants.py
	python $<

# === Objects ===

enrichment.o: enrichment.c enrichment.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

enrichment.so: enrichment.c enrichment.c
	$(CXX) $(CXXFLAGS) -shared -fPIC $< -o $@

nucleation.o: nucleation.c nucleation.h parabola625.c
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

parabola625.o: parabola625.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

parabola625.so: parabola625.c
	$(CXX) $(CXXFLAGS) -shared -fPIC $< -o $@

# === Extras ===

ternary-landscape.png: ternary-landscape.py tie-lines.npz
	python $<

tie-lines.npz: phase-diagram.py
	python $<

.PHONY: clean
clean:
	rm -vf check-nucleation enrichment.c enrichment.o nucleation.o parabola625.* sigma.csv sigma.png
