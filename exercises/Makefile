NVCC = nvcc
RM = rm -f 

SRCS = $(shell find . -name '*.cu' | sort)
EXE = $(SRCS:.cu=)

all: $(EXE)

clean:
	@$(RM) $(EXE)

%: %.cu
	$(NVCC) -o $@ $<

.PHONY: all clean

