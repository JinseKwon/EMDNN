OPENCL=0
CLBLAST=0
OPENBLAS=0

CC=g++
CFLAGS= -g -W -Wall -O2
OBJS=emdnn.o  blas.o detection_layer.o image_io.o classification_layer.o
LIBS=-lm `pkg-config --libs opencv`
DEF= 

ifeq ($(OPENCL), 1) 
LIBS+= -lOpenCL
DEF+= -DOPENCL
endif

ifeq ($(OPENBLAS), 1) 
LIBS+= -lOpenBLAS
endif

TARGET=skeleton
TAR_OBJS = skeleton.o
TARGET2=alexnet
TAR_OBJS2 = alexnet.o
TARGET3=vgg-16
TAR_OBJS3=vgg-16.o

$(TARGET): $(OBJS) $(TAR_OBJS)
	$(CC) -o $(TARGET)  $(CFLAGS) $(TAR_OBJS)  $(OBJS) $(LIBS) $(DEF)
	rm *.o

$(TARGET2): $(OBJS) $(TAR_OBJS2)
	$(CC) -o $(TARGET2) $(CFLAGS) $(TAR_OBJS2) $(OBJS) $(LIBS) $(DEF)
	rm *.o
$(TARGET3): $(OBJS) $(TAR_OBJS3)
	$(CC) -o $(TARGET3) $(CFLAGS) $(TAR_OBJS3) $(OBJS) $(LIBS) $(DEF)
	rm *.o

alexnet.o : alexnet.c
	$(CC) -c -o alexnet.o alexnet.c

vgg-16.o : vgg-16.c
	$(CC) -c -o vgg-16.o vgg-16.c


skeleton.o : skeleton.c
	$(CC) -c -o skeleton.o skeleton.c $(DEF)

emdnn.o : emdnn.c
	$(CC) -c -o emdnn.o emdnn.c $(DEF)

blas.o : blas.c
	$(CC) -c -o blas.o blas.c

detection_layer.o : detection_layer.c
	$(CC) -c -o detection_layer.o detection_layer.c

classification_layer.o : classification_layer.c
	$(CC) -c -o classification_layer.o classification_layer.c

image_io.o : image_io.c
	$(CC) -c -o image_io.o image_io.c `pkg-config --cflags --libs opencv`

clean :
	rm *.o skeleton alexnet vgg-16
