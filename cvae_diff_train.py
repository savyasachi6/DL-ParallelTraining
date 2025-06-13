import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from transformers import CLIPProcessor, CLIPModel
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
import argparse
import subprocess
from datetime import timedelta
from functools import partial

# Environment settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
os.environ["NCCL_DEBUG"] = "INFO"

# Configuration
IMG_SIZE = 64
LATENT_DIM = 256
TEXT_EMBED_DIM = 512
EPOCHS = 100
T = 1000
BATCH_SIZE = 32

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

clip_model = None
clip_processor = None

def encode_text(text, device):
    global clip_model, clip_processor
    if clip_model is None:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = clip_processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return clip_model.get_text_features(**inputs)

# Model Definitions
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3 * c, h * w)
        q, k, v = qkv.split(c, dim=1)
        attn = (q @ k.transpose(-1, -2)).softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return x + self.out(out)

class CrossAttention(nn.Module):
    def __init__(self, channels, text_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(text_dim, channels * 2)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, text_embed):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, c, h * w).transpose(1, 2)
        kv = self.kv(text_embed).reshape(b, 2, c)
        k, v = kv[:, 0], kv[:, 1]
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        attn = (q @ k.transpose(-1, -2)).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.out(out)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, text_dim):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ensure valid channels
        assert in_channels > 0 and out_channels > 0, "in_channels and out_channels must be positive"
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.cross_attn = CrossAttention(out_channels, text_dim)

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, text_embed):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Rank {rank}: ResBlock input shape: {x.shape}")
        residual = self.res_conv(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.cross_attn(x, text_embed)
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)

class CVAE_Diffusion(nn.Module):
    def __init__(self, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = LATENT_DIM
        self.text_embed_dim = TEXT_EMBED_DIM
        self.latent_h = img_size // 16
        self.enc_channels = max(128, LATENT_DIM // 2)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, self.enc_channels, 4, 2, 1),
            nn.ReLU()
        )
        enc_out_dim = self.enc_channels * self.latent_h * self.latent_h
        total_in_dim = enc_out_dim + self.text_embed_dim
        self.fc_mu = nn.Linear(total_in_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(total_in_dim, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
        )

        unet_channels = [self.latent_dim, self.latent_dim // 2, self.latent_dim // 4]  # [256, 128, 64]
        self.unet_down = nn.ModuleList([
            ResBlock(unet_channels[0], unet_channels[1], self.text_embed_dim),  # 256 -> 128
            ResBlock(unet_channels[1], unet_channels[2], self.text_embed_dim),  # 128 -> 64
            ResBlock(unet_channels[2], unet_channels[2], self.text_embed_dim),  # 64 -> 64
        ])
        self.downsample = nn.ModuleList([
            nn.Conv2d(unet_channels[1], unet_channels[1], 4, 2, 1),
            nn.Conv2d(unet_channels[2], unet_channels[2], 4, 2, 1),
        ])
        self.bottleneck = ResBlock(unet_channels[2], unet_channels[2], self.text_embed_dim)


        # self.unet_up = nn.ModuleList([
        #     ResBlock(unet_channels[2] * 2, unet_channels[2], self.text_embed_dim),  # 128 -> 64
        #     ResBlock(unet_channels[1], unet_channels[1], self.text_embed_dim),      # 128 -> 128
        #     ResBlock(unet_channels[1] * 2, unet_channels[0], self.text_embed_dim),  # 256 -> 256
        # ])

        self.unet_up = nn.ModuleList([
            ResBlock(unet_channels[2] * 2, unet_channels[2], self.text_embed_dim),  # 128 -> 64
            ResBlock(unet_channels[2] * 2, unet_channels[1], self.text_embed_dim),  # 128 -> 128
            ResBlock(unet_channels[1] * 2, unet_channels[0], self.text_embed_dim),  # 256 -> 256
        ])

        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(unet_channels[2], unet_channels[2], 4, 2, 1),
            nn.ConvTranspose2d(unet_channels[1], unet_channels[1], 4, 2, 1),
        ])

        self.out = nn.Sequential(
            nn.Conv2d(unet_channels[0] * 2, unet_channels[0], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(unet_channels[0], self.latent_dim, 3, 1, 1),
        )


    def encode(self, x, text_embed):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Rank {rank}: Input shape to encoder: {x.shape}")
        print(f"Rank {rank}: Encoder first layer weight shape: {self.encoder[0].weight.shape}")
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, text_embed], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_unet(self, z, t, text_embed):
        x = z
        skips = []
        for i, block in enumerate(self.unet_down):
            print(f"Rank {dist.get_rank()}: unet_down[{i}] input shape: {x.shape}")
            x = block(x, text_embed)
            print(f"Rank {dist.get_rank()}: unet_down[{i}] output shape: {x.shape}")
            skips.append(x)
            if i < len(self.unet_down) - 1:
                x = self.downsample[i](x)
                print(f"Rank {dist.get_rank()}: downsample[{i}] output shape: {x.shape}")
        x = self.bottleneck(x, text_embed)
        print(f"Rank {dist.get_rank()}: bottleneck output shape: {x.shape}")
        for i, block in enumerate(self.unet_up):
            if i > 0:
                x = self.upsample[i - 1](x)
                print(f"Rank {dist.get_rank()}: upsample[{i-1}] output shape: {x.shape}")
            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            print(f"Rank {dist.get_rank()}: unet_up[{i}] concat shape: {x.shape}")
            x = block(x, text_embed)
            print(f"Rank {dist.get_rank()}: unet_up[{i}] output shape: {x.shape}")
        x = torch.cat([x, z], dim=1)
        print(f"Rank {dist.get_rank()}: final concat shape: {x.shape}")
        x = self.out(x)
        print(f"Rank {dist.get_rank()}: output shape: {x.shape}")
        return x

    def forward(self, x, text_embed, t=None):
        mu, logvar = self.encode(x, text_embed)
        z = self.reparameterize(mu, logvar)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        z = F.interpolate(z, size=(self.latent_h, self.latent_h), mode='nearest')
        if t is None:
            return z, mu, logvar
        else:
            noise_pred = self.forward_unet(z, t, text_embed)
            return noise_pred, mu, logvar

def setup_distributed():
    try:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        master_addr = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list]).decode().splitlines()[0]
    except KeyError:
        rank = local_rank = 0
        world_size = 1
        master_addr = 'localhost'

    os.environ.update({
        'RANK': str(rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': master_addr,
        'MASTER_PORT': '29500',
        'LOCAL_RANK': str(local_rank)
    })

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(minutes=30),
            rank=rank,
            world_size=world_size,
            device_id=device  # Use the torch.device object here
        )

    return rank, local_rank, device


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))

def load_checkpoint(model, optimizer, checkpoint_path, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Rank {rank}: Loaded checkpoint from epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch, loss

def test_encoder_conv(device):
    conv = nn.Conv2d(3, 32, 4, stride=2, padding=1).to(device)
    x = torch.randn(32, 3, 64, 64).to(device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"Rank {rank}: Testing Conv2d: input shape={x.shape}, weight shape={conv.weight.shape}")
    try:
        y = conv(x)
        print(f"Rank {rank}: Conv2d succeeded, output shape={y.shape}")
    except Exception as e:
        print(f"Rank {rank}: Conv2d failed: {e}")

def get_mixed_precision_policy():
    return MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

# def get_wrap_policy():
#     return partial(size_based_auto_wrap_policy, min_num_params=1e6)

from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# def get_wrap_policy():
#     return ModuleWrapPolicy({nn.Sequential, ResBlock})  # Wrap encoder, decoder, and ResBlock modules

# def get_wrap_policy():
#     return None  # FSDP wraps the entire model

# def get_wrap_policy():
#     return lambda module, recurse, nonwrapped_numel: size_based_auto_wrap_policy(
#         module, recurse, nonwrapped_numel, min_num_params=1e6
#     )

def get_wrap_policy():
    return None  # FSDP wraps the entire model


def main():
    parser = argparse.ArgumentParser(description="FSDP Training on Frontera")
    parser.add_argument('--resume', type=str, help='Checkpoint path to resume training from')
    parser.add_argument('--no-fsdp', action='store_true', help='Disable FSDP for debugging')
    args = parser.parse_args()

    rank, local_rank, device = setup_distributed()
    print(f"Rank {rank}: PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    print(f"Rank {rank}: CUDA available: {torch.cuda.is_available()}, Device name: {torch.cuda.get_device_name(device)}")
    print(f"Rank {rank}: CUDA device count: {torch.cuda.device_count()}")

    # Test encoder convolution
    test_encoder_conv(device)

    # Create model
    model = CVAE_Diffusion(img_size=IMG_SIZE).cpu()
    model = model.to(device)

    # Load model weights only on rank 0
    if dist.get_rank() == 0 and os.path.exists("model.pth"):
        print(f"Rank {rank}: Loading model.pth")
        state_dict = torch.load("model.pth", map_location=device)
        model.load_state_dict(state_dict)

    # THEN wrap with FSDP
    if dist.is_initialized() and not args.no_fsdp:
        model = FSDP(
            model,
            auto_wrap_policy=get_wrap_policy(),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=get_mixed_precision_policy(),
            device_id=device,
            cpu_offload=CPUOffload(offload_params=False),
            sync_module_states=False,  # Manual broadcast follows
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            use_orig_params=True,
        )

    # Now broadcast weights AFTER wrapping
    if dist.is_initialized():
        dist.barrier()
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    # Debug model
    if hasattr(model, 'module'):
        weight_shape = model.module.encoder[0].weight.shape
        total_params = sum(p.numel() for p in model.module.parameters())
    else:
        weight_shape = model.encoder[0].weight.shape
        total_params = sum(p.numel() for p in model.parameters())
    print(f"Rank {rank}: Encoder first layer weight shape: {weight_shape}")
    print(f"Rank {rank}: Total model params after wrapping: {total_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    start_epoch = 0

    if args.resume:
        print(f"Rank {rank}: Loading checkpoint from {args.resume}")
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank) if dist.is_initialized() else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True
    )
    # Define get_alphas
    def get_alphas():
        alphas = torch.linspace(1.0, 0.0001, T)
        return alphas, torch.cumprod(alphas, dim=0)


    alphas, alphas_cumprod = get_alphas()
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)

    text_embeds = encode_text(CIFAR10_CLASSES, device).to(device)

    for epoch in range(start_epoch, EPOCHS):
        print(f"Rank {rank}: Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        if sampler:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        for step, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            print(f"Rank {rank}: Batch {step} - Inputs shape: {inputs.shape}, dtype: {inputs.dtype}, device: {inputs.device}")
            text_embed = text_embeds[labels]
            print(f"Rank {rank}: Batch {step} - Text embed shape: {text_embed.shape}, dtype: {text_embed.dtype}, device: {text_embed.device}")

            with torch.no_grad():
                mu, logvar = model.encode(inputs, text_embed)
                z = model.reparameterize(mu, logvar)
                z = F.interpolate(z.view(z.size(0), model.latent_dim, 1, 1), size=(model.latent_h, model.latent_h), mode='nearest')

            t = torch.randint(0, T, (z.size(0),), device=device)
            noise = torch.randn_like(z)
            noisy_z = (
                alphas_cumprod[t].sqrt()[:, None, None, None] * z +
                (1 - alphas_cumprod[t]).sqrt()[:, None, None, None] * noise
            )

            optimizer.zero_grad(set_to_none=True)
            noise_pred = model.forward_unet(noisy_z, t, text_embed)
            loss = F.mse_loss(noise_pred, noise)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if rank == 0 and step % 10 == 0:
                print(f"Rank {rank} Step {step} | Loss: {loss.item():.4f}")

        if rank == 0:
            save_dir = os.path.expandvars("$SCRATCH/checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, epoch_loss / len(dataloader), save_dir)
            print(f"Checkpoint saved: {save_dir}/checkpoint_epoch_{epoch}.pt")

    if dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
