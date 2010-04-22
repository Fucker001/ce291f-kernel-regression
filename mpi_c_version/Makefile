#
# Franklin
#
CC = CC
CFLAGS = -O3 -Mbounds
LIBS = 


TARGETS = traffic 

all:	$(TARGETS)
traffic: traffic.o
	$(CC) -o $@ $(LIBS) traffic.o 
	
traffic.o: traffic.cpp
	$(CC) -c $(CFLAGS) traffic.cpp
	
clean:
	rm -f *.o $(TARGETS)
