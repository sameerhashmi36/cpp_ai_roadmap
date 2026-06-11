#include <iostream>

int main() {
    // 1. Static Typing: must declare the exact data type
    double learning_rate = 0.001; 
    float activation = -1.5f;

    std::cout << "Learning Rate is set to: " << learning_rate << std::endl;

    // 2. Control Flow: A basic ReLU (Rectified Linear Unit) concept
    if (activation < 0.0f) {
        activation = 0.0f;
    }

    std::cout << "Output after ReLU activation: " << activation << std::endl;

    return 0; // 0 means the program executed successfully
}