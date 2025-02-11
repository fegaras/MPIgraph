all:
	mpic++ -O2 -std=c++11 -DNDEBUG -Iinclude src/main/cpp/*.cpp -o pregel

debug:
	mpic++ -g -std=c++11 -D _GLIBCXX_DEBUG -D _GLIBCXX_DEBUG_PEDANTIC -Iinclude src/main/cpp/*.cpp -o pregel

clean: 
	/bin/rm -f pregel *~ src/main/cpp/*~ include/*~
