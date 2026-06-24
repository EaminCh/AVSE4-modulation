import torch
from model import AVSE4BaselineModule

def run_diagnostic():
    print("1. Initializing your Dictionary-Aware Lightning Module...")
    model = AVSE4BaselineModule(num_channels=2)
    
    print("2. Constructing mock dictionary batch mirroring your dataset keys...")
    mock_batch = {
        "noisy": torch.randn(2, 2, 257, 64),      # [Batch, Channel, Freq, Time]
        "video": torch.randn(2, 1, 64, 88, 88),    # [Batch, Channel, Time, Height, Width]
        "clean": torch.randn(2, 2, 257, 64)       # Clean Ground Truth
    }
    
    print("3. Testing Evaluation Forward Pass (via model(batch))...")
    model.eval()
    with torch.no_grad():
        eval_output = model(mock_batch)
    print(f"   ?? Success! Cleaned output tensor shape: {eval_output.shape}")
    
    print("4. Testing Training Step Optimization (Loss & Gradients)...")
    model.train()
    total_loss = model.training_step(mock_batch, batch_idx=0)
    print(f"   ?? Success! Combined Multi-Task Loss: {total_loss.item():.4f}")
    
    print("\n?? ALL SYSTEM CHECKS PASSED: Your codebase is structurally synchronized!")

if __name__ == "__main__":
    run_diagnostic()