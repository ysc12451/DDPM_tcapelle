"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""

import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext
from PIL import Image
import os

import torch
from torch import optim
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
from fastprogress import progress_bar

import wandb
from utils import *
from modules import UNet_conditional, EMA


config = SimpleNamespace(    
    run_name = "DDPM_conditional_1220",
    remove_deep_conv = True,
    basic_dim = 32, # 64
    epochs = 100, # 100
    noise_steps= 800, # 1000
    seed = 42,
    batch_size = 10,
    img_size = 64, # 64
    num_classes = 101,
    # dataset_path = get_cifar(img_size=64),
    # dataset_path = "/mnt/c/Code/cs771_project/data/",
    dataset_path = "/storage08/shuchen/DDPM/",
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    lr = 2e-3 # 5e-3
) 

inverse_transform = torchvision.transforms.Compose([
    # Inverse of Normalize
    T.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),  
    # Inverse of ToTensor
    T.Lambda(lambda x: x * 0.5 + 0.5),  
    # Inverse of RandomResizedCrop
    # T.Lambda(lambda x: F.resize(x, int(1.0 / 0.8)))
    # Inverse of Resize
    # T.Lambda(lambda x: F.resize(x, int(4 / 3 * args.img_size))),
])

# Note: F.resize is a placeholder for the actual function used for resizing in your code (e.g., PIL's Image.resize).
# You may need to replace it with the appropriate function based on the library you are using for image processing.

def combine_images(image_list, output_path, images_per_row=10):
    """
    Combine a list of images into a single image with a specified number of images per row.

    Parameters:
    - image_list: List of image objects (Pillow Image objects or equivalent).
    - output_path: Path to save the combined image.
    - images_per_row: Number of images to display in each row (default is 10).
    """
    # Calculate the dimensions of the combined image
    _, image_width, image_height = image_list[0].size()
    combined_width = images_per_row * image_width
    combined_height = ((len(image_list) - 1) // images_per_row + 1) * image_height

    # Create a new image for the combined result
    combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

    # Paste each image onto the combined image
    for i, image_tensor in enumerate(image_list):
        to_pil = transforms.ToPILImage()
        image = to_pil(image_tensor)
        x_offset = (i % images_per_row) * image_width
        y_offset = (i // images_per_row) * image_height
        box = (x_offset, y_offset, x_offset + image_width, y_offset + image_height)
        combined_image.paste(image, box)

    # Save the combined image
    combined_image.save(output_path)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=101, c_in=3, c_out=3, device="cuda", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, time_dim=256, num_classes=num_classes, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3): # cfg_scale: linear weight w for cond and uncond
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                wandb.log({"train_mse": loss.item(),
                            "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    def log_images(self, epoch):
        "Log images to wandb and save them to disk"
        labels = torch.arange(self.num_classes).long().to(self.device)
        sampled_images = self.sample(use_ema=False, labels=labels)
        combine_images(sampled_images, os.path.join("/storage08/shuchen/DDPM/DDPM_tcapelle/outputs/", f'ckpt_ep{epoch}.png'), images_per_row=10)
        wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})

        # EMA model sampling
        ema_sampled_images = self.sample(use_ema=True, labels=labels)
        combine_images(ema_sampled_images, os.path.join("/storage08/shuchen/DDPM/DDPM_tcapelle/outputs/", f'ema_ckpt_ep{epoch}.png'), images_per_row=10)
        # plot_images(sampled_images)  #to display on jupyter if available
        wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
        at.add_dir(os.path.join("models", run_name))
        wandb.log_artifact(at)

    def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True)
            
            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                wandb.log({"val_mse": avg_loss})
            
            # log predicitons
            if epoch % args.log_every_epoch == 0:
                self.log_images(epoch)
                self.save_model(run_name=f"{args.run_name}_ep{epoch}", epoch=epoch)

        # save model
        self.save_model(run_name=args.run_name, epoch=epoch)


def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--rm_deep', type=str, default=config.remove_deep_conv, help='remove deep conv layers')
    parser.add_argument('--basic_dim', type=str, default=config.basic_dim, help='basic dimension for UNet')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes, remove_deep_conv=config.remove_deep_conv, basic_dim=config.basic_dim)
    with wandb.init(project="train_sd", group="train", config=config):
        diffuser.prepare(config)
        diffuser.fit(config)
