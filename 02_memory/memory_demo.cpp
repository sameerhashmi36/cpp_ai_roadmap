/*
 * PROJECT: Memory Manipulation in AI Pipelines
 * DESCRIPTION: Demonstrates the practical difference between passing by Value,
 * passing by Pointer, and passing by Reference using a loss metric.
 * INPUT: An initial training loss value (2.5).
 * OUTPUT: Console logs showing how memory addresses dictate whether data is mutated.
 */

#include <iostream>

// 1. Pass by Value (making a duplicate)
// This copies the entire data. The original variable outside stays untouched.
void modifyLossByValue(float loss_copy) {
    loss_copy = 0.5f; // Modifying only the local duplicate
}

// 2. Pass by Pointer (The old C-style method)
// I pass the raw memory address. Must use '*' to "dereference" it to change the value.
void modifyLossByPointer(float* loss_address) {
    if (loss_address != nullptr) { // Safety check: ensure pointer isn't pointing to nothing
        *loss_address = 0.2f;      // Go to the address and change the original value
    }
}

// 3. Pass by Reference (The modern C++ standard for Tensors and Images)
// 'float&' creates an alias (nickname). No copy is made, syntax stays clean.
void modifyLossByReference(float& loss_alias) {
    loss_alias = 0.01f; // Changes the original variable directly!
}

int main() {
    float training_loss = 2.5f;

    std::cout << "=== INITIAL STATE ===" << std::endl;
    std::cout << "Initial Loss Value: " << training_loss << std::endl;
    std::cout << "Memory Address of training_loss: " << &training_loss << "\n\n"; // '&' gets the raw address

    // Test 1: Pass by Value
    modifyLossByValue(training_loss);
    std::cout << "After modifyLossByValue: " << training_loss << " (No change, a copy was modified)" << std::endl;

    // Test 2: Pass by Pointer
    modifyLossByPointer(&training_loss); // explicitly pass the address using '&'
    std::cout << "After modifyLossByPointer: " << training_loss << " (Changed! Accessed via raw address)" << std::endl;

    // Test 3: Pass by Reference
    modifyLossByReference(training_loss); // Syntax looks completely clean, like Python
    std::cout << "After modifyLossByReference: " << training_loss << " (Changed! Accessed via alias)" << std::endl;

    return 0;
}