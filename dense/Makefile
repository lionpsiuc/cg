CC = gcc
CFLAGS = -fopenmp -O3 -Wall -Wextra

LDFLAGS = -fopenmp -lm

TARGET = cg

SRCS = cg.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)
	python3 plot.py

clean:
	rm -f $(TARGET) $(OBJS) *.dat
