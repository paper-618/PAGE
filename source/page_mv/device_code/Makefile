include ../SLmake.inc
CC=MT-3000-gcc
AR=MT-3000-ar
LD=MT-3000-ld

PAGE_FLAGS ?=

CFLAGS=-c -O2 -fenable-m3000 -ffunction-sections -flax-vector-conversions \
       $(PAGE_FLAGS) \
       -I./ -I$(ENV_ROOT)/include -I$(GCCROOT)/include/
LDFLAGS= -L$(ENV_ROOT)/lib --gc-sections -Tdsp.lds

SRC=kernel.c
OBJ=kernel.o
EXE=kernel.out
DAT=kernel.dat

ALL: $(EXE)
	@if [ -e ../bin ] && [ ! -d ../bin ]; then rm -f ../bin; fi
	@mkdir -p ../bin
	$(GCCROOT)/bin/MT-3000-makedat -J $(EXE)
	cp ${DAT} ../bin/

$(OBJ): $(SRC)
	$(GCCROOT)/bin/$(CC) $(CFLAGS) $(SRC) -o $(OBJ)

$(EXE): $(OBJ)
	$(GCCROOT)/bin/$(LD) $(LDFLAGS) $(OBJ) $(ENV_ROOT)/lib/libhthread_device.a \
		$(GCCROOT)/lib/vlib3000.a $(GCCROOT)/lib/slib3000.a -o $(EXE)

clean:
	rm -f $(EXE) $(OBJ) $(DAT)
