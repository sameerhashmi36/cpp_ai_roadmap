/*
 * PROJECT: Object-Oriented AI Layer Simulation
 * DESCRIPTION: Implements a custom Dense/Linear neural network layer class, 
 * demonstrating Encapsulation (public/private split) and optimized 
 * Constructor Initializer Lists in C++.
 * INPUT: A simulated input weight modifier and an input feature value (5.0).
 * OUTPUT: The computed linear activation result (Input * Weight + Bias).
 */

#include <iostream>

class DenseLayer {
    // === 1. PRIVATE SECTION (Encapsulation) ===
    // These variables are hidden. The outside world cannot read or modify them directly.
    private:
        float weight;
        float bias;
        int input_features;
        int output_features;

    // === 2. PUBLIC SECTION (The User Interface / API) ===
    // Anything here can be safely accessed by the user of this class.
    public:
        // 3. THE CONSTRUCTOR with an Initializer List
        // The colon ':' starts the initializer list and assigns incoming arguments 
        // directly to the private variables *before* the function body `{}` executes.
        DenseLayer(int in_dim, int out_dim, float init_weight): input_features(in_dim), output_features(out_dim), weight(init_weight), bias(0.01f) // Hardcoding a tiny initial bias default
        {
            // This is the constructor body. The variables are already initialized!
            std::cout << "[Constructor] Layer initialized. Shape: (" << input_features << ", " << output_features << ")" << std::endl;
        }

        // 4. THE FORWARD PASS FUNCTION
        // Mimics PyTorch's forward pass: computes (X * W) + B
        float forward(float input_value) {
            std::cout << "[Forward Pass] Processing input: " << input_value << std::endl;
            return (input_value * weight) + bias;
        }

        // 5. GETTER FUNCTION (Read-Only Access)
        // Since 'bias' is private, provide a safe way to check its value without allowing modifications.
        float getBias() const {
            return bias;
        }

        // 6. GETTER FUNCTION (Read-Only Access)
        // Since 'weight' is private, provide a safe way to check its value without allowing modifications.
        float getWeight() const {
            return weight;
        }
        
        // 7. SETTER FUNCTION (Controlled Modification)
        // Allows safe modification of the private 'weight' variable.
        void setWeight(float new_weight) {
            if (new_weight < -10.0f || new_weight > 10.0f) {
                std::cerr << "Warning: Extreme weight value skipped!" << std::endl;
                return; // Abort update to prevent gradient explosions
            }
            weight = new_weight;
        }
};

int main() {
    std::cout << "=== Creating Neural Network Layer ===" << std::endl;
    
    // Instantiate the object (allocates memory on the Stack and runs the constructor)
    // Corresponds to: layer = DenseLayer(in_dim=10, out_dim=5, init_weight=0.5)
    DenseLayer my_layer(10, 5, 0.5f);

    std::cout << "\n=== Running Inference ===" << std::endl;
    float input_tensor = 5.0f; // Simulated incoming tensor feature
    float output_tensor = my_layer.forward(input_tensor);

    std::cout << "\n=== Layer Properties ===" << std::endl;
    std::cout << "Output Tensor Value: " << output_tensor << std::endl;
    std::cout << "Inspecting Layer Bias safely: " << my_layer.getBias() << std::endl;
    std::cout << "Inspecting Layer Weight safely: " << my_layer.getWeight() << std::endl;

    my_layer.setWeight(0.8f); // Update the weight using the setter function

    // COMPILER ERROR TEST (Uncomment the line below to see C++ Encapsulation protect the code):
    // my_layer.weight = 2.0f; // The compiler will freeze and refuse to build!

    // but this can still safely check the weight using the getter function:
    std::cout << "Inspecting Layer Weight safely: " << my_layer.getWeight() << std::endl;


    return 0;
}