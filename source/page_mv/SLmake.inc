DSP_ROOT ?= $(DEV_PRO_PATH)
GCCROOT ?= $(DSP_ROOT)/dsp_compiler
ENV_ROOT ?= $(DSP_ROOT)/hthreads
export LD_LIBRARY_PATH := $(DSP_ROOT)/third-party-lib:$(GCCROOT)/lib:$(ENV_ROOT)/lib:$(LD_LIBRARY_PATH)
