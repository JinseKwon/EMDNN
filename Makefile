CC=g++
CFLAGS=-g -W -Wall -O2 -lm
OBJS= emdnn.o skeleton.o blas.o detection_layer.o
TARGET=skeleton
 
$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(CFLAGS) $(OBJS)
 
emdnn.o : emdnn.c
	$(CC) -c -o emdnn.o emdnn.c

skeleton.o : skeleton.c
	$(CC) -c -o skeleton.o skeleton.c

blas.o : blas.c
	$(CC) -c -o blas.o blas.c

detection_layer.o : detection_layer.c
	$(CC) -c -o detection_layer.o detection_layer.c

clean :
	rm *.o skeleton