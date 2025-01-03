#
# Simple Makefile for compiling programs for initio robot car 
#
SHELL	= bash
GCC	= gcc
CFLAGS	= -Wall -mfloat-abi=hard
LFLAGS	= -linitio -lcurses -lwiringPi -lpthread -lm -ljpeg

CROSSGCC	= arm-linux-gnueabi-gcc
CROSSINCLUDEPATH	= -I/usr/local/arm-linux-gnueabi/include

PROG 	= camcar

.PHONY: all run cross-compile cross-link help

all: $(PROG)

$(PROG): $(PROG).o detect_blob.o quickblob.o

run: $(PROG)
	./$<

schedule: $(PROG)
	rtcs_schedule $<

%.o : %.c
	$(GCC) -c -o $@ $(CFLAGS) $<

% : %.o
	$(GCC) -o $@ $(LFLAGS) $< detect_blob.o quickblob.o

# cross compilation: compiler on host machine:
cross-compile: cross_$(PROG).o

cross_$(PROG).o: $(PROG).c
	$(CROSSGCC) -c -o cross_$(PROG).o $(CFLAGS) $(CROSSINCLUDEPATH) $<

# cross compilation: linker on target machine:
# (need to first copy compiled object file from host to target machine)
cross-link:
	$(GCC) -o cross_$(PROG) $(LFLAGS) cross_$(PROG).o detect_blob.o quickblob.o

clean:
	rm -f detect_blob.o quickblob.o $(PROG).o $(PROG)

help:
	@echo
	@echo "Possible commands:"
	@echo " > make run"
	@echo " > make schedule"
	@echo " > make cross-compile"
	@echo " > make cross-link"
	@echo " > make clean"
	@echo

