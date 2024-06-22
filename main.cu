#include <iostream>

__global__ void add_one(int& x) {
    x += 1;
}

int main() {
    int x = 1;
    add_one << <1, 1 >> > (x);
    std::cout << "x = " << x << std::endl;
    return 0;
}