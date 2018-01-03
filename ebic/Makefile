program_NAME := ebic
program_CXX_SRCS := $(wildcard *.cxx *.cu)
program_C_OBJS := ${program_C_SRCS:.c=.o}
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_OBJS := $(program_C_OBJS) $(program_CXX_OBJS)
program_INCLUDE_DIRS := 
program_LIBRARY_DIRS := /usr/local/cuda/lib
program_LIBRARIES := cuda


CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(program_LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(program_LIBRARIES),-l$(library))

.PHONY: all clean distclean

all: $(program_NAME)

$(program_NAME): $(program_OBJS)
	nvcc -O3 -std=c++11 --expt-extended-lambda -Xcompiler -fopenmp --default-stream per-thread $(program_OBJS) $(LDFLAGS) -o $(program_NAME) 

clean:
	rm $(program_NAME)