all:
	mpic++ -O3 -fopenmp -DNDEBUG -Iinclude src/main/cpp/*.cpp -o pregel

clean: 
	/bin/rm -f a.out src/main/cpp/*~ include/*~
