OPENCL=1
CLBLAST=0
OPENBLAS=1

CLBLAST_PATH=/home/shunya/CLBlast/
OPENBLAS_PATH=/home/shunya/OpenBLAS/

CC=g++
CFLAGS= -g -W -Wall -O2
OBJS=emdnn.o  blas.o detection_layer.o classification_layer.o image_io.o 
LIBS=-lm `pkg-config --libs opencv`
DEF=
INC=

ifeq ($(OPENCL), 1) 
LIBS+= -lOpenCL
DEF+= -DOPENCL
endif

ifeq ($(CLBLAST), 1) 
DEF+= -DCLBLAST
LIBS+= -fPIC $(CLBLAST_PATH)/build/libclblast.so
INC+= -I $(CLBLAST_PATH)/include
endif

ifeq ($(OPENBLAS), 1) 
DEF+= -DOPENBLAS
LIBS+= -fPIC $(OPENBLAS_PATH)/libopenblas.so
INC+= -I $(OPENBLAS_PATH)
endif

TARGET=skeleton
TAR_OBJS = skeleton.o
TARGET2=alexnet
TAR_OBJS2 = alexnet.o
 
$(TARGET): $(OBJS) $(TAR_OBJS)
	$(CC) -o $(TARGET)  $(CFLAGS) $(TAR_OBJS)  $(OBJS) $(LIBS) $(DEF) $(INC)
	rm *.o

$(TARGET2): $(OBJS) $(TAR_OBJS2)
	$(CC) -o $(TARGET2) $(CFLAGS) $(TAR_OBJS2) $(OBJS) $(LIBS) $(DEF) $(INC)
	rm *.o

alexnet.o : alexnet.c
	$(CC) -c -o alexnet.o alexnet.c $(DEF)

skeleton.o : skeleton.c
	$(CC) -c -o skeleton.o skeleton.c $(DEF)

emdnn.o : emdnn.c
	$(CC) -c -o emdnn.o emdnn.c $(DEF)  $(INC)

blas.o : blas.c
	$(CC) -c -o blas.o blas.c $(DEF) $(INC)

detection_layer.o : detection_layer.c 
	$(CC) -c -o detection_layer.o detection_layer.c $(DEF)

classification_layer.o : classification_layer.c 
	$(CC) -c -o classification_layer.o classification_layer.c $(DEF)

image_io.o : image_io.c 
	$(CC) -c -o image_io.o image_io.c `pkg-config --cflags --libs opencv` $(DEF)

clean :
	rm *.o skeleton alexnet
