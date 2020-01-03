CPP=g++ -std=c++11 -O3
CFLAGS=-Wall -Wextra -Wno-sign-conversion -Wno-sign-compare -Wno-unused-parameter -pedantic-errors -pipe -fPIC -g

all: sparse_hlda

sparse_hlda.o: sparse_hlda.cpp
	$(CPP) $(CFLAGS) -c $^
model.o: model.cpp
	$(CPP) $(CFLAGS) -c $^
utils.o: utils.cpp
	$(CPP) $(CFLAGS) -c $^

sparse_hlda: sparse_hlda.o utils.o model.o
	$(CPP) -lm -o $@ $^

.PHONY: clean install
clean:
	rm -rf sparse_hlda
	rm -rf *.o
	rm -rf *.gch

install:
	mkdir -p ../bin
	mv sparse_hlda ../bin
