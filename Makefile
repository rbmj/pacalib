all: cooling

CC=mpicc
CFLAGS=-std=gnu99 -O2 -Wall

cooling: cooling.o world.o
	${CC} ${CFLAGS} -o cooling cooling.o world.o -lm

world.o: world.c world.h
	${CC} ${CFLAGS} -c -o world.o world.c

cooling.o: cooling.c world.h
	${CC} ${CFLAGS} -c -o cooling.o cooling.c

clean:
	rm -f *.o fire
