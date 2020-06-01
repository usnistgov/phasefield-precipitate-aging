CFLAGS = -O3 -Wall
OBJS = enrichment.o nucleation.o parabola625.o enrichment.so parabola625.so

all: check-nucleation
.PHONY: all

debug: CFLAGS += -O0 -g
debug: check-nucleation $(OBJS)

profile: CFLAGS += -O1 -pg
profile: $(OBJS)

CXX = g++
CXXFLAGS = ${CFLAGS} -std=c++11
INCLUDES = -I. -I../src -I$(CUDA_HDR_PATH)

# === Executables ===

check-nucleation: check-nucleation.cpp ../src/parameters.h nucleation.c $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) $< -o $@ -lm

ternary-diagram.png: phase-diagram.py parabola625.c parabola625.so enrichment.so
	python $<

# === Generated Sources ===

enrichment.c: enrichment.py
	python $<

parabola625.c: CALPHAD_energies.py constants.py
	python $< && astyle --style=linux --indent-col1-comments --indent=tab --indent-preprocessor --indent-preproc-cond --pad-header --align-pointer=type --align-reference=type --keep-one-line-blocks --suffix=none parabola625.c

# === Objects ===

enrichment.o: enrichment.c enrichment.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

enrichment.so: enrichment.c enrichment.c
	gcc $(CFLAGS) -shared -fPIC $< -o $@

nucleation.o: nucleation.c nucleation.h parabola625.c
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

parabola625.o: parabola625.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

parabola625.so: parabola625.c
	gcc $(CFLAGS) -shared -fPIC $< -o $@

# === Extras ===

ternary-landscape.png: ternary-landscape.py tie-lines.npz
	python $<

tie-lines.npz: phase-diagram.py
	python $<

.PHONY: clean
clean:
	rm -vf check-nucleation enrichment.h enrichment.c enrichment.o enrichment.so nucleation.o parabola625.* sigma.csv sigma.png
