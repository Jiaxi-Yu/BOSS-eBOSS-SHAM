CC = gcc -std=c99
LIBS = -lgsl -lgslcblas -DOMP -fopenmp -L/global/homes/z/zhaoc/lib/gsl-2.4/lib
INCL = -I/global/homes/z/zhaoc/lib/gsl-2.4/include
OPTS = -O3 -Wall $(INCL) $(LIBS) -lm
OBJS = 
SRCS = load_conf.c read_data.c rdz2xyz.c algebra.c count_bin.c save_res.c kdtree.c kdtree_cnt.c 2pcf.c
EXEC = 2pcf

all: 2pcf


2pcf:
	$(CC) $(SRCS) -o $(EXEC) $(OPTS)

clean:
	rm -f $(OBJS) $(EXEC)
