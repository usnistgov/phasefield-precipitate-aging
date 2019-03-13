OBJS = parabola625.o nucleation.o

check-nucleation: check-nucleation.cpp $(OBJS)
	g++ -O0 -g -Wall $(OBJS) $< -o $@

parabola625.o: parabola625.c
	g++ -O0 -g -Wall -c $< -o $@

nucleation.o: nucleation.c
	g++ -O0 -g -Wall -c $< -o $@
