import os

class Config:
    # -----------------------
    # PATHS & DIRECTORIES
    # -----------------------
    # Directory containing the dataset (Vimeo or Kinetics)
    DATASET_DIR = "/path/to/vimeo_septuplet" 
    
    # Directory to save checkpoints during training
    CHECKPOINT_DIR = "./checkpoints"
    
    # Paths to the pre-trained weights (The "Pentagon" of models)
    PRETRAINED_WEIGHTS = {
        "motion": "weights/motion/FlowVAE_finetune_ep11.pth",
        "refine": "weights/motion/RefiFlow.pth",
        "residual": "weights/residual/ResidualVAE_HardMode_Ep4.pth",
        "adaptive": "weights/reconstruction/AdaptiveNET.pth",
        "post": "weights/reconstruction/ResRefiNET.pth"
    }
    
    # -----------------------
    # TRAINING HYPERPARAMETERS
    # -----------------------
    BATCH_SIZE = 4           # Use 8 if running on 48GB VRAM
    LEARNING_RATE = 1e-5     # Low LR for fine-tuning
    EPOCHS = 10
    NUM_WORKERS = 4          # For data loading
    
    # Lambda determines the trade-off between Bitrate (R) and Distortion (D)
    # Higher Lambda = Better Quality, Higher Bitrate
    # Lower Lambda = Lower Quality, Lower Bitrate
    LAMBDA_RD = 2048         # Standard value for high quality (can range 256 - 4096)
    
    # -----------------------
    # IMAGE SETTINGS
    # -----------------------
    CROP_SIZE = (256, 256)   # Training crop size for Vimeo-90k
    INFERENCE_SIZE = (1024, 576) # QHD resolution for testing