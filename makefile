all: fastLDA.cpp
	g++ -o fastLDA -fopenmp fastLDA.cpp

serial: fastLDA.cpp
	g++ -o fastLDA fastLDA.cpp
clean:
	rm fastLDA
