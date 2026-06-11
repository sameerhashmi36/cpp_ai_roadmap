/*
 * PROJECT: Deep Learning Model State Tracking
 * DESCRIPTION: Demonstrates the structural mechanics of Pass-by-Value, 
 * Pass-by-Pointer, and Pass-by-Reference on a complex multi-variable model state.
 * INPUT: Initialized weights (0.85), bias (0.1), and epoch counter (1).
 * OUTPUT: Console logs proving which passing methods successfully update the global state.
 */

#include <iostream>

// A structure to hold our simulated deep learning model metrics
struct ModelState {
    float weight;
    float bias;
    int epoch;
};

// 1. Pass-by-Value: Creates a total duplicate copy of the structure in a new memory location
void updateByValue(ModelState model_copy) {
    model_copy.weight = 0.99f;
    model_copy.bias = 0.5f;
    model_copy.epoch = 999;
    std::cout << "[Inside updateByValue] Modified Copy Address: " << &model_copy << std::endl;
}

// 2. Pass-by-Pointer: Receives the physical memory address of the original structure
// must use the '->' arrow operator to open the address and change the values inside
void updateByPointer(ModelState* model_ptr) {
    if (model_ptr != nullptr) { // Professional safety check: ensure the pointer isn't empty
        model_ptr->weight = 0.12f;
        model_ptr->bias = 0.01f;
        model_ptr->epoch = 5;
    }
    std::cout << "[Inside updateByPointer] Stored Target Address:  " << model_ptr << std::endl;
}

// 3. Pass-by-Reference: Receives a direct nickname (alias) for the original structure
// can use the clean '.' syntax, just like Python, but it updates the original hardware memory instantly
void updateByReference(ModelState& model_ref) {
    model_ref.weight = 0.02f;
    model_ref.bias = 0.001f;
    model_ref.epoch = 10;
    std::cout << "[Inside updateByReference] Nickname Target Address: " << &model_ref << std::endl;
}

int main() {
    // Initialize our model state
    ModelState my_model = {0.85f, 0.1f, 1};
    
    std::cout << "=== INITIAL STATE ===" << std::endl;
    std::cout << "Original Model Address in main: " << &my_model << std::endl;
    std::cout << "Weight: " << my_model.weight << " | Bias: " << my_model.bias << " | Epoch: " << my_model.epoch << "\n\n";

    // Test 1: Pass-by-Value
    std::cout << "--- Executing Pass-By-Value ---" << std::endl;
    updateByValue(my_model);
    std::cout << "Result in main -> Weight: " << my_model.weight << " (No change!)\n\n";

    // Test 2: Pass-by-Pointer
    std::cout << "--- Executing Pass-By-Pointer ---" << std::endl;
    // use the '&' symbol to explicitly extract and pass the physical address of my_model
    updateByPointer(&my_model); 
    std::cout << "Result in main -> Weight: " << my_model.weight << " (Successfully updated via address!)\n\n";

    // Test 3: Pass-by-Reference
    std::cout << "--- Executing Pass-By-Reference ---" << std::endl;
    // pass my_model normally. The function signature automatically hooks it up as a nickname.
    updateByReference(my_model);
    std::cout << "Result in main -> Weight: " << my_model.weight << " (Successfully updated via alias!)\n\n";

    std::cout << "=== FINAL MODEL STATE ===" << std::endl;
    std::cout << "Weight: " << my_model.weight << " | Bias: " << my_model.bias << " | Epoch: " << my_model.epoch << std::endl;

    return 0;
}