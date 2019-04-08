OBJS = parabola625.o enrichment.o

all: check-nucleation
.PHONY: all

check-nucleation: check-nucleation.cpp $(OBJS)
	g++ -O2 -Wall $(OBJS) $< -o $@ -lm

parabola625.o: parabola625.c
	g++ -O2 -Wall -c $< -o $@

ternary-diagram.png: phase-diagram.py parabola625.c parabola625.so
	python $<

parabola625.c: CALPHAD_energies.py
	python $<

parabola625.so: parabola625.c
	gcc -O2 -Wall -fPIC -shared $< -o $@

enrichment.o: enrichment.c
	gcc -O2 -Wall -fPIC -c $< -o $@

enrichment.c: enrichment.py
	python $<

ternary-landscape.png: ternary-landscape.py tie-lines.npz
	python $<

tie-lines.npz: phase-diagram.py
	python $<

.PHONY: clean
clean:
	rm -f check-enrichment enrichment.c enrichment.o parabola625.*
