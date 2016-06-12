OBJ = main.o neural.o sigmoid.o
FLAGS = -std=c++11 -I./includes/

all: $(OBJ)
	g++ $(FLAGS) $(OBJ) -o neur

%.o: %.cpp
	g++ -c $(FLAGS) $< -o $@

clean:
	rm -f $(OBJ) neur
