import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VimeoSeptupletDataset(Dataset):
    """
    Standard Dataset: Loads Frame 1 and Frame 2 (consecutive) from 7-frame sequences.
    Used for the basic training of the Residual VAE.
    """
    def __init__(self, root_dir, split='train', crop_size=(256, 256)):
        self.root_dir = root_dir
        self.crop_size = crop_size
        
        filename = "sep_trainlist.txt" if split == 'train' else "sep_testlist.txt"
        list_path = os.path.join(root_dir, filename)
        
        # Fallback for Kaggle directory structure
        if not os.path.exists(list_path):
             list_path = os.path.join(os.path.dirname(root_dir), filename)

        self.sequences = []
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                self.sequences = [line.strip() for line in f if line.strip()]
            print(f" [Septuplet Standard] Loaded {len(self.sequences)} seqs")
        else:
            print(f" Split file not found: {list_path}")

        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        full_path = os.path.join(self.root_dir, "sequences", seq_path)
        
        img1 = Image.open(os.path.join(full_path, "im1.png")).convert('RGB')
        img2 = Image.open(os.path.join(full_path, "im2.png")).convert('RGB')

        # Synchronized Crop
        # We must apply the exact same random crop to both frames.
        seed = torch.random.seed()
        torch.manual_seed(seed)
        im1 = self.transform(img1)
        torch.manual_seed(seed)
        im2 = self.transform(img2)
        
        return im1, im2


class VimeoTripletSkipDataset(Dataset):
    """
    Specific for Motion VAE on the Triplet dataset.
    Always skips the middle frame (im1 -> im3).
    This creates a larger motion gap for robust flow training.
    """
    def __init__(self, root_dir, split='train', crop_size=(256, 256)):
        self.root_dir = root_dir
        self.crop_size = crop_size
        
        filename = "tri_trainlist.txt" if split == 'train' else "tri_testlist.txt"
        list_path = os.path.join(root_dir, filename)
        
        # Fallback for Kaggle directory structure
        if not os.path.exists(list_path):
             list_path = os.path.join(os.path.dirname(root_dir), filename)
        
        self.sequences = []
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                self.sequences = [line.strip() for line in f if line.strip()]
            print(f" [Triplet Skip] Loaded {len(self.sequences)} seqs")
        else:
            print(f" Split file not found: {list_path}")
            
        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        full_path = os.path.join(self.root_dir, "sequences", seq_path)
        
        img_prev = Image.open(os.path.join(full_path, "im1.png")).convert("RGB")
        img_curr = Image.open(os.path.join(full_path, "im3.png")).convert("RGB")
        
        # Synchronized Crop
        seed = torch.random.seed()
        torch.manual_seed(seed)
        frame_prev = self.transform(img_prev)
        torch.manual_seed(seed)
        frame_curr = self.transform(img_curr)
        
        return frame_prev, frame_curr


class VimeoHardModeDataset(Dataset):
    """
    Advanced Mode: Uses long sequences (Septuplets) but introduces a variable Gap.
    E.g. Gap=2 selects (im1->im3) or (im2->im4).
    Drastically increases difficulty for Motion Estimation training (Data Augmentation).
    """
    def __init__(self, root_dir, gap=1, split="train", crop_size=(256, 256)):
        self.root_dir = root_dir
        self.gap = gap
        self.crop_size = crop_size
        
        filename = "sep_trainlist.txt" if split == "train" else "sep_testlist.txt"
        list_path = os.path.join(root_dir, filename)
        
        # Fallback path
        if not os.path.exists(list_path):
             list_path = os.path.join(os.path.dirname(root_dir), filename)
        
        self.sequences = []
        if os.path.exists(list_path):
            with open(list_path, "r") as f:
                self.sequences = [line.strip() for line in f if line.strip()]
            print(f" [HardMode Gap={gap}] Loaded {len(self.sequences)} seqs")
        else:
             print(f" Split file not found: {list_path}")

        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        full_path = os.path.join(self.root_dir, "sequences", self.sequences[idx])
        
        # HardMode Logic: Choose a valid random starting point for the gap
        # Sequence len = 7 frames. Max start index depends on the gap.
        max_start = 7 - self.gap
        if max_start < 1: max_start = 1
        
        idx_start = random.randint(1, max_start)
        idx_end = idx_start + self.gap
        
        try:
            img1 = Image.open(os.path.join(full_path, f"im{idx_start}.png")).convert("RGB")
            img2 = Image.open(os.path.join(full_path, f"im{idx_end}.png")).convert("RGB")
        except FileNotFoundError:
            # Emergency fallback: load first and last available frames if specific indices fail
            files = sorted(os.listdir(full_path))
            img1 = Image.open(os.path.join(full_path, files[0])).convert("RGB")
            img2 = Image.open(os.path.join(full_path, files[min(len(files)-1, self.gap)])).convert("RGB")

        # CRITICAL FIX: Synchronized Random Crop
        # Without this, motion training fails (impossible to match features).
        seed = torch.random.seed()
        
        torch.manual_seed(seed)
        im1 = self.transform(img1)
        
        torch.manual_seed(seed)
        im2 = self.transform(img2)
        
        return im1, im2