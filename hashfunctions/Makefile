CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LIBS    = -lgmp
TARGET  = poseidon

$(TARGET): poseidon.c
	$(CC) $(CFLAGS) -o $(TARGET) poseidon.c $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
