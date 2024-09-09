
all: point module tv

point: point.c
	gcc -fPIC point.c -c -o point.o

module: winding.c winding.o point.o
	gcc -fPIC -c winding.c -o winding.o
	gcc -fPIC -shared -o winding.so winding.o point.o
	rm -f winding.o

tv: tv.c tv.o point.o
	gcc -fPIC -c tv.c -o tv.o
	gcc -fPIC -shared -o tv.so tv.o point.o
	rm -f tv.o

clean: 
	rm -f point.o
	rm -f winding.so
	rm -f tv.so