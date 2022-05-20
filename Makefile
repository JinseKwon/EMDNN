OPENCL=1
CLBLAST=1
OPENBLAS=1
NNPACK=1

DEBUG_PRINT=0

CLBLAST_PATH=/home/shunya/CLBlast
OPENBLAS_PATH=/home/shunya/OpenBLAS
NNPACK_PATH=/home/shunya/NNPACK-darknet



CC=g++
CFLAGS= -g -W -Wall -O2
OBJS=emdnn.o  blas.o detection_layer.o classification_layer.o image_io.o quantization.o
SRCS=$(patsubst %.o, %.c, $(OBJS))
LIBS=-lm `pkg-config --libs opencv`
DEF=
INC=

ifeq ($(DEBUG_PRINT), 1) 
DEF+= -DDEBUG_PRINT
endif

ifeq ($(OPENCL), 1) 
LIBS+= -lOpenCL
DEF+= -DOPENCL
endif

ifeq ($(CLBLAST), 1) 
DEF+= -DCLBLAST -DOPENCL
LIBS+= -lOpenCL -fPIC $(CLBLAST_PATH)/build/libclblast.so
INC+= -I $(CLBLAST_PATH)/include
endif

ifeq ($(OPENBLAS), 1) 
DEF+= -DOPENBLAS
LIBS+= -fPIC $(OPENBLAS_PATH)/libopenblas.so
INC+= -I $(OPENBLAS_PATH)
endif

ifeq ($(NNPACK), 1) 
DEF+= -DNNPACK
LIBS+= -fPIC $(NNPACK_PATH)/lib/libnnpack.a
LIBS+= -fPIC $(NNPACK_PATH)/lib/libpthreadpool.a -lpthread -lm
INC+= -I $(NNPACK_PATH)/include
endif

TARGET=yolotiny
TAR_OBJS=yolotiny.o
TARGET2=alexnet
TAR_OBJS2=alexnet.o
TARGET3=tinydark
TAR_OBJS3=tinydark.o
TARGET4=vgg16
TAR_OBJS4=vgg16.o
TARGET5=mobilenet
TAR_OBJS5=mobilenet.o
TARGET6=mnist
TAR_OBJS6=mnist.o

all: $(TARGET) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) #$(TARGET6) 

$(TARGET): $(OBJS) $(TAR_OBJS)
	$(CC) -o $(TARGET)  $(CFLAGS) $(TAR_OBJS)  $(OBJS) $(LIBS) $(DEF) $(INC)

$(TARGET2): $(OBJS) $(TAR_OBJS2)
	$(CC) -o $(TARGET2) $(CFLAGS) $(TAR_OBJS2) $(OBJS) $(LIBS) $(DEF) $(INC)

$(TARGET3): $(OBJS) $(TAR_OBJS3)
	$(CC) -o $(TARGET3) $(CFLAGS) $(TAR_OBJS3) $(OBJS) $(LIBS) $(DEF) $(INC)

$(TARGET4): $(OBJS) $(TAR_OBJS4)
	$(CC) -o $(TARGET4) $(CFLAGS) $(TAR_OBJS4) $(OBJS) $(LIBS) $(DEF) $(INC)

$(TARGET5): $(OBJS) $(TAR_OBJS5)
	$(CC) -o $(TARGET5) $(CFLAGS) $(TAR_OBJS5) $(OBJS) $(LIBS) $(DEF) $(INC)

$(TARGET6): $(OBJS) $(TAR_OBJS6)
	$(CC) -o $(TARGET6) $(CFLAGS) $(TAR_OBJS6) $(OBJS) $(LIBS) $(DEF) $(INC)
	rm *.o

alexnet.o : alexnet.c
	$(CC) -c -o $@ $^ $(DEF) $(INC)

yolotiny.o : yolotiny.c
	$(CC) -c -o $@ $^ $(DEF) $(INC)

tinydark.o : tinydark.c
	$(CC) -c -o $@ $^ $(DEF) $(INC)

vgg16.o : vgg16.c
	$(CC) -c -o $@ $^ $(DEF) $(INC)

mobilenet.o : mobilenet.c
	$(CC) -c -o $@ $^ $(DEF) $(INC)

mnist.o : mnist.c
	$(CC) -c -o $@ $^ $(DEF) $(INC)

emdnn.o : emdnn.c
	$(CC) -c -o emdnn.o emdnn.c $(DEF)  $(INC)

blas.o : blas.c
	$(CC) -c -o blas.o blas.c $(DEF) $(INC)

detection_layer.o : detection_layer.c 
	$(CC) -c -o detection_layer.o detection_layer.c $(DEF) $(INC)

classification_layer.o : classification_layer.c 
	$(CC) -c -o classification_layer.o classification_layer.c $(DEF) $(INC)

image_io.o : image_io.c 
	$(CC) -c -o image_io.o image_io.c `pkg-config --cflags --libs opencv` $(DEF) $(INC)

quantization.o : quantization.c
	$(CC) -c -o quantization.o quantization.c $(DEF) $(INC)

clean :
	rm *.o $(TARGET) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6)
