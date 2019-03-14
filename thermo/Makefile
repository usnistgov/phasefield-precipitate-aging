OBJS = parabola625.o nucleation.o

check-nucleation: check-nucleation.cpp $(OBJS)
	g++ -O2 -Wall $(OBJS) $< -o $@

parabola625.o: parabola625.c
	g++ -Wall -c $< -o $@

parabola625.so: parabola625.c
	gcc -Wall -fPIC -shared $< -o $@

nucleation.o: nucleation.c
	gcc -Wall -fPIC -c $< -o $@
