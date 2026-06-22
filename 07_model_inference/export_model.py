import torch
import torchvision.models as models

# 1. Instantiate a pre-trained image classification network (ResNet18)
# Use modern weights initialization parameters
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Crucial: Put the model into evaluation mode
# This locks down dropout and batch-normalization layers for static deployment
model.eval()

# 3. Create a dummy input tensor matching the target shape layout
# Layout: [Batch Size (1), Channels (3), Height (224), Width (224)]
dummy_input = torch.rand(1, 3, 224, 224)

print("[INFO] Tracing the neural execution graph via TorchScript...")
# 4. Trace the execution path of the dummy tensor through the model graph
traced_model = torch.jit.trace(model, dummy_input)

# 5. Serialize and save the compiled model asset directly to disk
model_filename = "build/resnet18.pt"
traced_model.save(model_filename)

print(f"[SUCCESS] Compiled model successfully exported to: {model_filename}")