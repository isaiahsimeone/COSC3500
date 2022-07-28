#include <iostream>
#include <random>
#include <unistd.h>

using namespace std;

#define M 40
#define N 30

void    populate(char**, int, int);
void    display(char**, int, int);

int main(int argc, char** argv) {
    char** screen_buffer = new char*[M];
    for (int i = 0; i < M; i++)
        screen_buffer[i] = new char[N];

    for (;; sleep(1)) {
        // Fill with trash
        populate(screen_buffer, M, N);
        // Display
        display(screen_buffer, M, N);
        // Move cursor back N lines
        cout << "\033[40A";
    }




}

char random_char() {
    return (char)(0x21 + rand() % 0x7E);
}

void populate(char** screen_buffer, int width, int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            screen_buffer[x][y] = random_char();
        }
    }
}

void display(char** screen_buffer, int width, int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            cout << screen_buffer[x][y];
        }
        cout << endl;
    }
}