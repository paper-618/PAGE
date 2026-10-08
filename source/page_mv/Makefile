DIRS = device_code host_code

all:
	@if [ -e bin ] && [ ! -d bin ]; then rm -f bin; fi
	@mkdir -p bin
	@ \
	for i in ${DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then (cd $$i; $(MAKE)) || exit 1; \
	  fi; \
	done

clean:
	@ \
	for i in ${DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then (cd $$i; $(MAKE) clean); \
	  fi; \
	done

veryclean: clean
	@if [ -e bin ] && [ ! -d bin ]; then rm -f bin; fi
	@if [ -d bin ]; then rm -f bin/page bin/kernel.dat bin/*.log; fi
