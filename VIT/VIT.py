# -*- coding: utf-8 -*-
import argparse
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from tqdm import tqdm
from PIL import Image

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== Model Components ====================
class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        out = x + self.pos_embedding
        return self.dropout(out) if self.dropout else out

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x) if self.dropout1 else x
        x = self.fc2(x)
        return self.dropout2(x) if self.dropout2 else x

class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        return torch.tensordot(x, self.weight, dims=dims) + self.bias

class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim**0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        b, n, _ = x.shape
        q = self.query(x, dims=([2], [0])).permute(0, 2, 1, 3)
        k = self.key(x, dims=([2], [0])).permute(0, 2, 1, 3)
        v = self.value(x, dims=([2], [0])).permute(0, 2, 1, 3)

        attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1)
        out = torch.matmul(attn_weights, v).permute(0, 2, 1, 3)
        out = self.out(out, dims=([2, 3], [0, 1]))
        return self.dropout(out) if self.dropout else out

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, num_heads, attn_dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = residual + (self.dropout(self.attn(x)) if self.dropout else self.attn(x))
        residual = x
        x = self.norm2(x)
        return residual + self.mlp(x)

class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, 
                 dropout_rate=0.1, attn_dropout_rate=0.0):
        super().__init__()
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(emb_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.pos_embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.norm(x)

class VisionTransformer(nn.Module):
    def __init__(self, image_size=(256, 256), patch_size=(16, 16), emb_dim=768, mlp_dim=3072,
                 num_heads=12, num_layers=12, attn_dropout_rate=0.0, dropout_rate=0.1):
        super().__init__()
        h, w = image_size
        fh, fw = patch_size
        self.gh, self.gw = h // fh, w // fw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.transformer = Encoder(
            self.gh * self.gw, emb_dim, mlp_dim, num_layers, num_heads, dropout_rate, attn_dropout_rate)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 3, 1)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)
        return self.transformer(emb)

class CAFIA_Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vit = VisionTransformer(
            image_size=(args.image_size, args.image_size),
            patch_size=(args.patch_size, args.patch_size),
            emb_dim=args.emb_dim,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            attn_dropout_rate=args.attn_dropout_rate,
            dropout_rate=args.dropout_rate)
        self._init_weights(args)
        self.classifier = nn.Linear(args.emb_dim, args.num_classes)

    def _init_weights(self, args):
        if os.path.exists(args.vit_model):
            state_dict = torch.load(args.vit_model, map_location='cpu').get('state_dict', {})
            for key in ['classifier.weight', 'classifier.bias']:
                state_dict.pop(key, None)
            self.vit.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        feat = self.vit(x)
        return self.classifier(feat[:, 0])

# ==================== Scheduler ====================
def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            e, es = step - warmup_length, steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

# ==================== Training Function ====================
def train_model(args, trainloader, testloader):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = CAFIA_Transformer(args).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                          betas=(args.beta1, args.beta2),
                          eps=args.eps,
                          lr=args.learning_rate,
                          weight_decay=args.weight_decay)
    
    total_steps = len(trainloader) * args.epoches
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)
    
    # Training loop
    logging.info("Starting Training")
    
    # Open log files once at the beginning
    with open("VIT-train-log.txt", "a+") as ftrain, open("VIT-test-log.txt", "a+") as ftest:
        ftrain.write("-----------------------------------------\n")
        ftest.write("-------------------------------------------\n")
        
        iter_count = 1
        for epoch in range(args.epoches):
            model.train()
            total_loss = 0.0
            
            best_acc=0
            # Use memory-efficient training loop
            with tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epoches}", 
                     mininterval=1.0) as progress:
                for step, (images, labels) in enumerate(progress):
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)  # More efficient memory usage
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler(epoch * len(trainloader) + step)
                    
                    total_loss += loss.item()
                    progress.set_postfix(loss=loss.item())
                    
                    # Write training log in batches to reduce I/O
                    if (step + 1) % 1 == 0 or step == len(trainloader) - 1:
                        ftrain.write(f"{iter_count},{loss.item():.5f}\n")
                        ftrain.flush()  # Ensure data is written to disk
                    iter_count += 1
            
            # Evaluation
            model.eval()
            correct, total = 0, 0
            all_preds = []
            all_labels = []
            with torch.no_grad(), tqdm(testloader, desc="Evaluating") as eval_progress:
                for images, labels in eval_progress:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            cm = confusion_matrix(all_labels, all_preds)
            print("Confusion Matrix:")
            print(cm)
            acc = 100 * correct / total

            logging.info(f"Epoch {epoch+1} | Loss: {total_loss/len(trainloader):.4f} | Accuracy: {acc:.5f}%")
            ftest.write(f"{epoch},{acc:.5f}\n")
            ftest.flush()
            
            # Save checkpoint with memory cleanup
            os.makedirs(args.output, exist_ok=True)
            checkpoint_path = os.path.join(args.output, f"checkpoint.pt")
            if acc >best_acc:
                best_acc=acc
                # Save model state without unnecessary data
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)
                np.savetxt(os.path.join(args.output, 'confusion_matrix.txt'), cm, fmt='%d')  # 保存文本
            # Explicit memory cleanup
            del images, labels, outputs
            torch.cuda.empty_cache()
    
    logging.info("Training Completed")

# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(description="Vision Transformer Training")
    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--beta1", type=float, default=0.99)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=500)
    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoches", type=int, default=50)
    # Model
    parser.add_argument("--image_size", type=int, default=224, choices=[224, 384])
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--mlp_dim", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--attn_dropout_rate", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=10)
    # Paths
    parser.add_argument("--output", default="./output", type=str)
    parser.add_argument("--vit_model", default="./Vit_weights/imagenet21k+imagenet2012_ViT-B_16-224.pth", type=str)
    
    args = parser.parse_args()
    
    os.makedirs(args.output,exist_ok=True)
    # Data preparation with optimized settings
    transform = transforms.Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use pinned memory for faster data transfer to GPU
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 
                   'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    test_kwargs = {'batch_size': args.batch_size, 'shuffle': False,
                  'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    testloader = torch.utils.data.DataLoader(testset, **test_kwargs)

    # Start training
    train_model(args, trainloader, testloader)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    main()