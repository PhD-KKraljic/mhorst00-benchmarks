all: lud

lud:
	cd ocl; make
	mv ocl/lud .;

clean:
	cd ocl; make clean

run: lud
	#./lud -i ../../data/lud/256.dat
	#./lud -s 1024 -v
	./lud -s 8192 -p 0 -d 0
