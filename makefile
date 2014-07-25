#put here relative or absolute path to directory where CAPD is compiled
CAPD = /home/wilczak/capd4/bin/bin
INCLUDE = `$(CAPD)/capd-config --cflags`
LIBS = `$(CAPD)/capd-config --libs`

#put here your programs name
PROGRAMS = restrictedFourBody.exe

all: $(PROGRAMS)


restrictedFourBody.exe: restrictedFourBody.cpp restrictedFourBody.hpp
	g++ -Wall -O2 $(INCLUDE) restrictedFourBody.cpp $(LIBS) -fopenmp -o $@

clean:
	rm -f *~ *.o

destroy:
	rm -f $(PROGRAMS).exe
