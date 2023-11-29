# Change this to wherever you keep the NDK
NDK            = /opt/android-ndk-r26b
SRCDIR         = .
OBJDIR         = .
DBG           ?= 0

# Debug/Release configuration
ifeq ($(DBG),1)
MODE_FLAGS     = -DDEBUG -g -O0
else
MODE_FLAGS     = -fdata-sections -ffunction-sections
endif

# NDK configuration (clang)
NDK_TARGET     = aarch64-linux-android26
NDK_TOOL       = $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang-17
NDK_SYSROOT    = $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/sysroot

# Compiler and Linker Flags
CFLAGS         = $(MODE_FLAGS) --target=aarch64-unknown-linux-gui -mcpu=cortex-a76 -std=c99 -fPIE -Wall --target=$(NDK_TARGET) --sysroot=$(NDK_SYSROOT)
LDFLAGS        = $(MODE_FLAGS) -fPIE -pie --target=$(NDK_TARGET) --sysroot=$(NDK_SYSROOT)


# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS) -o run run.c -lm
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS) -o runq runq.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS)  -g -o run run.c -lm
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS)  -g -o runq runq.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS) -Ofast run.c -o run -lm
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS) -Ofast runq.c -o runq -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS) -Ofast -static-openmp -fopenmp run.c  -lm  -o run
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS) -Ofast -static-openmp -fopenmp runq.c  -lm  -o runq

.PHONY: myrun
myrun: run.c
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS) -Ofast -static-openmp -fopenmp -DMY_OPT -o run run.c microkernels.c -lm -mfpu=neon -mfloat-abi=hard
	$(NDK_TOOL) $(CFLAGS) $(LDFLAGS) -Ofast -static-openmp -fopenmp -DMY_OPT -o runq runq.c -lm -mfpu=neon -mfloat-abi=hard


.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. run.c win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(NDK_TOOL) $(CFLAGS) -Ofast -std=gnu11 -o run run.c -lm
	$(NDK_TOOL) $(CFLAGS) -Ofast -std=gnu11 -o runq runq.c -lm

.PHONY: runompgnu
runompgnu:
	$(NDK_TOOL) $(CFLAGS) -Ofast -fopenmp -std=gnu11 run.c  -lm  -o run
	$(NDK_TOOL) $(CFLAGS) -Ofast -fopenmp -std=gnu11 runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
	rm -f runq

push:
	adb push run /data/local/tmp/run
	adb push runq /data/local/tmp/runq