include ../SLmake.inc
PAGE_FLAGS ?=

CC      = gcc
CFLAGS  = -O2 -std=c99 -fopenmp $(PAGE_FLAGS) -I./ -I$(ENV_ROOT)/include
LDLIBS  = $(ENV_ROOT)/lib/libhthread_host.a -lpthread -lm
HEADERS = $(wildcard ../../header/*.h)

EXES = page

ALL: $(EXES)
	@if [ -e ../bin ] && [ ! -d ../bin ]; then rm -f ../bin; fi
	@mkdir -p ../bin
	cp $(EXES) ../bin/

page: page.c $(HEADERS)
	$(CC) $(CFLAGS) page.c -fopenmp -o $@ $(LDLIBS)

clean:
	rm -f $(EXES)
