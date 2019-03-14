OBJS = parabola625.o nucleation.o
HDRS = /usr/local/tnk10/opt/conda/envs/mmsp/include/python2.7

check-nucleation: check-nucleation.cpp $(OBJS)
	g++ -O0 -g -Wall $(OBJS) $< -o $@

parabola625.py: parabola625.i parabola625.h
	swig -python $< && \
	gcc -fPIC -c parabola625_wrap.c -I$(HDRS) && \
	gcc -fPIC -c parabola625.c -o pyParabola625.o && \
	gcc -shared pyParabola625.o parabola625_wrap.o -o _parabola625.so

parabola625_wrap.o: parabola625_wrap.c parabola625.c
	gcc -fPIC -c $< -I/usr/local/include/python2.7

_parabola625.so: parabola625_wrap.o

parabola625.o: parabola625.c
	g++ -O0 -g -Wall -c $< -o $@

nucleation.o: nucleation.c
	g++ -O0 -g -Wall -c $< -o $@
