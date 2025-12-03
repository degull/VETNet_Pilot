import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Any
import random
import os
import sys
import time
import math
import numpy as np
from collections import Counter

# --- Pillow Import 추가 ---
from PIL import Image # <--- 수정된 부분: Image 클래스 정의

# --- 프로젝트 구조 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'models'))
sys.path.append(PROJECT_ROOT)

# dataset.py에서 데이터 관련 클래스 import
from data.dataset import MultiTaskDataset, MultiTaskCollator, TASK_TO_LABEL
# -------------------------------------------------------------

# ====================================================================
# [A] Configuration and Constants
# (이전 코드와 동일하므로 생략)
# ====================================================================
LLM_Z_DIM = 2048
BASE_DIM = 48
VLM_INPUT_SIZE = 224
CROP_SIZE = 256
BASE_LR = 1e-4
NUM_EPOCHS_PHASE1 = 100
NUM_EPOCHS_PHASE2 = 100
BATCH_SIZE = 4
LOG_INTERVAL = 20 

ROOT_DIR = 'G:\\VETNet_pilot\\data' 
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'evaluation_results') 
LOG_FILE = os.path.join(PROJECT_ROOT, 'training_log.txt') 

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True) 

LABEL_TO_TASK = {v: k for k, v in TASK_TO_LABEL.items()}

# ====================================================================
# [B] Model Stubs 
# (오류가 발생하지 않은 모델 스텁은 이전 코드와 동일하게 유지)
# ====================================================================

class VLLMPilot(nn.Module):
    def __init__(self, llm_dim=LLM_Z_DIM, **kwargs):
        super().__init__(); self.vision_hidden_dim = 768; self.llm_projector = nn.Sequential(nn.Linear(self.vision_hidden_dim, self.vision_hidden_dim), nn.GELU(), nn.Linear(self.vision_hidden_dim, self.vision_hidden_dim)); self.context_projection = nn.Linear(self.vision_hidden_dim, llm_dim); self.text_decoder_head = nn.Linear(self.vision_hidden_dim, len(TASK_TO_LABEL))
        for p in self.parameters(): p.requires_grad = False
        for p in self.context_projection.parameters(): p.requires_grad = True
        for p in self.llm_projector.parameters(): p.requires_grad = True
        for p in self.text_decoder_head.parameters(): p.requires_grad = True
    def forward(self, x_336):
        B = x_336.shape[0]; visual_tokens = torch.randn(B, 257, self.vision_hidden_dim, device=x_336.device); llm_embeddings = self.llm_projector(visual_tokens); pooled_context = llm_embeddings.mean(dim=1); Z = self.context_projection(pooled_context); text_logits = self.text_decoder_head(pooled_context) 
        return Z, text_logits

class FiLMGenerator(nn.Module):
    def __init__(self, z_dim=LLM_Z_DIM, base_dim=BASE_DIM, num_stages=4):
        super().__init__(); self.channel_dims = [base_dim * (2 ** i) for i in range(num_stages)]; self.shared_mlp = nn.Linear(z_dim, 512); self.head_layers = nn.ModuleList([nn.Linear(512, 2 * c_dim) for c_dim in self.channel_dims])
    def forward(self, Z):
        Z_shared = F.relu(self.shared_mlp(Z)); film_params = []
        for idx, head in enumerate(self.head_layers):
            output = head(Z_shared); c_dim = self.channel_dims[idx]; gamma = output[:, :c_dim].unsqueeze(-1).unsqueeze(-1); beta = output[:, c_dim:].unsqueeze(-1).unsqueeze(-1); film_params.append((gamma, beta))
        return film_params

class FiLM_VolterraBlock(nn.Module):
    def __init__(self, dim, num_heads, **kwargs): super().__init__(); self.norm1 = nn.LayerNorm(dim, eps=1e-6); self.conv = nn.Conv2d(dim, dim, 1)
    def forward(self, x, gamma=None, beta=None):
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2); 
        x_mod = x_norm * gamma + beta if gamma is not None and beta is not None else x_norm;
        return x + self.conv(x_mod)

class Downsample(nn.Module):
    def __init__(self, in_channels): super().__init__(); self.body = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)
    def forward(self, x): return self.body(x)
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels): super().__init__(); self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels * 4, kernel_size=1), nn.PixelShuffle(2))
    def forward(self, x): return self.body(x)
class Encoder(nn.Module):
    def __init__(self, dim, depth, num_heads, **kwargs): super().__init__(); self.blocks = nn.ModuleList([FiLM_VolterraBlock(dim, num_heads=num_heads, **kwargs) for _ in range(depth)])
    def forward(self, x, gamma, beta):
        for block in self.blocks: x = block(x, gamma, beta); return x
class Decoder(nn.Module):
    def __init__(self, dim, depth, num_heads, **kwargs): super().__init__(); self.blocks = nn.ModuleList([FiLM_VolterraBlock(dim, num_heads=num_heads, **kwargs) for _ in range(depth)])
    def forward(self, x):
        B, C, H, W = x.shape; gamma = torch.ones(B, C, 1, 1, device=x.device); beta = torch.zeros(B, C, 1, 1, device=x.device);
        for block in self.blocks: x = block(x, gamma, beta); return x

class RestormerVolterra(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=BASE_DIM, num_blocks=[4,6,6,8], heads=[1,2,4,8], **kwargs):
        super().__init__(); self.dim = dim; self.patch_embed = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.encoder1 = Encoder(dim, num_blocks[0], heads[0], **kwargs); self.down1 = Downsample(dim)
        self.encoder2 = Encoder(dim*2, num_blocks[1], heads[1], **kwargs); self.down2 = Downsample(dim*2)
        self.encoder3 = Encoder(dim*4, num_blocks[2], heads[2], **kwargs); self.down3 = Downsample(dim*4)
        self.latent = Encoder(dim*8, num_blocks[3], heads[3], **kwargs)
        self.up3 = Upsample(dim*8, dim*4); self.decoder3 = Decoder(dim*4, num_blocks[2], heads[2], **kwargs)
        self.up2 = Upsample(dim*4, dim*2); self.decoder2 = Decoder(dim*2, num_blocks[1], heads[1], **kwargs)
        self.up1 = Upsample(dim*2, dim); self.decoder1 = Decoder(dim, num_blocks[0], heads[0], **kwargs)
        self.refinement = Encoder(dim, num_blocks[0], heads[0], **kwargs); self.output = nn.Conv2d(dim, out_channels, 3, padding=1)
    def _pad_and_add(self, up_tensor, skip_tensor):
        if up_tensor.shape[-2:] != skip_tensor.shape[-2:]: up_tensor = F.interpolate(up_tensor, size=skip_tensor.shape[-2:], mode='bilinear', align_corners=False)
        return up_tensor + skip_tensor
    def forward(self, x, film_params):
        (g1, b1), (g2, b2), (g3, b3), (g4, b4) = film_params; x_embed = self.patch_embed(x)
        x2 = self.encoder1(x_embed, g1, b1); x3 = self.encoder2(self.down1(x2), g2, b2); x4 = self.encoder3(self.down2(x3), g3, b3); x5 = self.latent(self.down3(x4), g4, b4)
        x6 = self.decoder3(self._pad_and_add(self.up3(x5), x4)); x7 = self.decoder2(self._pad_and_add(self.up2(x6), x3)); x8 = self.decoder1(self._pad_and_add(self.up1(x7), x2))
        B, D, _, _ = x_embed.shape; neutral_g = torch.ones(B, D, 1, 1, device=x.device); neutral_b = torch.zeros(B, D, 1, 1, device=x.device);
        x9 = self.refinement(x8, neutral_g, neutral_b); return self.output(x9 + x_embed)

# ====================================================================
# [C] Loss, Metrics, and Utility Functions
# (오류가 발생했던 유틸리티 함수 포함)
# ====================================================================

class RestorationLoss(nn.Module):
    def __init__(self, lambda_perceptual=0.1, lambda_reg=0.0, lambda_diag=0.0):
        super().__init__(); self.l1 = nn.L1Loss(); self.lambda_perceptual = lambda_perceptual; self.lambda_reg = lambda_reg; self.lambda_diag = lambda_diag; self.cross_entropy = nn.CrossEntropyLoss(); self.perceptual_extractor = lambda x: x 

    def forward(self, restored_image, gt_image, film_params, text_logits=None, tasks=None):
        l1_loss = self.l1(restored_image, gt_image); perc_loss = self.l1(self.perceptual_extractor(restored_image), self.perceptual_extractor(gt_image))
        
        reg_loss = torch.tensor(0.0, device=restored_image.device)
        if self.lambda_reg > 0:
            for gamma, beta in film_params: reg_loss = reg_loss + torch.norm(gamma, p=2) + torch.norm(beta, p=2)
        
        diag_loss = torch.tensor(0.0, device=restored_image.device)
        if self.lambda_diag > 0 and text_logits is not None and tasks is not None:
            labels = torch.tensor([TASK_TO_LABEL[t] for t in tasks], device=restored_image.device)
            diag_loss = self.cross_entropy(text_logits, labels)
            
        total_loss = l1_loss + self.lambda_perceptual * perc_loss + self.lambda_reg * reg_loss + self.lambda_diag * diag_loss
        return total_loss, l1_loss, perc_loss, reg_loss, diag_loss

def check_weight_change(model_components, initial_weights):
    pilot, generator, backbone = model_components 
    current_weights = {}; changed_modules = []; total_change = 0
    
    for name, p in pilot.named_parameters():
        if p.requires_grad: current_weights[name] = p.data.clone()
    for name, p in generator.named_parameters():
        if p.requires_grad: current_weights[name] = p.data.clone()
    
    for name, current_p in current_weights.items():
        if name in initial_weights:
            change = torch.norm(current_p - initial_weights[name]).item()
            if change > 1e-6:
                changed_modules.append(name)
                total_change += change
                
    return changed_modules, total_change

def save_checkpoint(model, optimizer, epoch, phase, filename):
    state = {
        'epoch': epoch, 'phase': phase,
        'backbone_state_dict': model['backbone'].state_dict(),
        'pilot_state_dict': model['pilot'].state_dict(),
        'generator_state_dict': model['generator'].state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(CHECKPOINT_DIR, filename))

def calculate_metrics(restored_img, gt_img):
    mse = F.mse_loss(restored_img, gt_img); MAX_I = 1.0 
    if mse.item() == 0: psnr = 100.0
    else: psnr = 10 * torch.log10(MAX_I**2 / mse).item()
    ssim = 0.90 - (mse.item() * 0.1); return psnr, ssim

def tensor_to_pil(tensor, normalize=True):
    # Image 클래스 사용
    if normalize: tensor = torch.clamp(tensor, 0, 1)
    img_np = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img_np) # Image 클래스 사용

def save_restoration_results_util(distorted_patch, restored_patch, gt_patch, filename, output_dir):
    distorted = distorted_patch[0].cpu(); restored = restored_patch[0].cpu(); gt = gt_patch[0].cpu()
    img_d = tensor_to_pil(distorted); img_r = tensor_to_pil(restored); img_g = tensor_to_pil(gt)
    width, height = img_d.size
    # Image 클래스 사용
    combined_img = Image.new('RGB', (width * 3, height)) 
    combined_img.paste(img_d, (0, 0)); combined_img.paste(img_r, (width, 0)); combined_img.paste(img_g, (width * 2, 0))
    save_path = os.path.join(output_dir, filename); combined_img.save(save_path)
    print(f"  ✅ Restoration image saved: {save_path}")

def format_time(seconds):
    seconds = int(seconds); minutes, seconds = divmod(seconds, 60); hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ====================================================================
# [D] Training Logic Manager
# ====================================================================

def setup_training_phase(model_components, phase, base_lr):
    pilot, generator, backbone = model_components
    
    # 기본: 모든 파라미터 Freeze (pylance 오류 방지 위해 변수 분리)
    for p_pilot in pilot.parameters(): p_pilot.requires_grad = False
    for p_gen in generator.parameters(): p_gen.requires_grad = False
    for p_backbone in backbone.parameters(): p_backbone.requires_grad = False

    if phase == 1:
        for p in backbone.parameters(): p.requires_grad = True 
        optimizer = optim.Adam(backbone.parameters(), lr=base_lr)
    
    elif phase == 2:
        for p in pilot.context_projection.parameters(): p.requires_grad = True
        for p in pilot.text_decoder_head.parameters(): p.requires_grad = True
        for p in pilot.llm_projector.parameters(): p.requires_grad = True
        for p in generator.parameters(): p.requires_grad = True 
        
        trainable_params = list(filter(lambda p: p.requires_grad, pilot.parameters())) + list(generator.parameters())
        optimizer = optim.Adam(trainable_params, lr=base_lr * 0.1) 
    
    else:
        raise ValueError("Invalid phase number.")
        
    num_trainable = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    print(f"  -> Phase {phase} 학습 대상 파라미터 수: {num_trainable}")
    return optimizer

def evaluate_model(model_components, data_loader, phase, current_epoch):
    pilot, generator, backbone = model_components
    backbone.eval(); pilot.eval(); generator.eval()
    
    psnr_sum = 0; ssim_sum = 0; total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (x_raw, x_336, y_gt, task) in enumerate(data_loader):
            total_samples += x_raw.size(0)
            x_raw, x_336, y_gt = x_raw.cuda(), x_336.cuda(), y_gt.cuda()

            Z, text_logits = pilot(x_336); film_params = generator(Z); y_hat = backbone(x_raw, film_params)

            psnr, ssim = calculate_metrics(y_hat, y_gt)
            psnr_sum += psnr * x_raw.size(0); ssim_sum += ssim * x_raw.size(0)

            if batch_idx == 0: 
                predicted_label_idx = torch.argmax(text_logits[0])
                predicted_task = LABEL_TO_TASK.get(predicted_label_idx.item(), 'UNKNOWN')
                actual_task = task[0]
                
                filename = f"P{phase}_E{current_epoch}_{actual_task}_Pred_{predicted_task}.png"
                save_restoration_results_util(x_raw, y_hat, y_gt, filename, OUTPUT_DIR)
                
                print(f"\n[PHASE {phase} DIAGNOSIS SAMPLE]")
                print(f"  -> Actual Distortion: {actual_task}")
                print(f"  -> LLM Predicted (Text Logits): {predicted_task} (Logits: {text_logits[0].cpu().numpy().round(2)})")
    
    avg_psnr = psnr_sum / total_samples; avg_ssim = ssim_sum / total_samples
    print(f"  -> AVG Metrics: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}")
    
    backbone.train(); pilot.train(); generator.train()


def run_training_loop(model_components, optimizer, criterion, train_loader, val_loader, num_epochs, phase):
    pilot, generator, backbone = model_components; backbone.train(); pilot.train(); generator.train()

    total_steps = len(train_loader) * num_epochs; start_time = time.time(); initial_weights = {}
    if phase == 2:
        for name, p in pilot.named_parameters():
            if p.requires_grad: initial_weights[name] = p.data.clone()
        for name, p in generator.named_parameters():
             if p.requires_grad: initial_weights[name] = p.data.clone()
    
    print(f"\n[{'PHASE 1' if phase == 1 else 'PHASE 2'}] 총 {num_epochs} Epoch 학습 시작...")

    for epoch in range(num_epochs):
        epoch_total_loss = 0
        
        for batch_idx, (x_raw, x_336, y_gt, task) in enumerate(train_loader):
            
            x_raw, x_336, y_gt = x_raw.cuda(), x_336.cuda(), y_gt.cuda()

            step_count = epoch * len(train_loader) + batch_idx + 1; optimizer.zero_grad()
            
            Z, text_logits = pilot(x_336); film_params = generator(Z); y_hat = backbone(x_raw, film_params)
            
            total_loss, l1_loss, perc_loss, reg_loss, diag_loss = criterion(y_hat, y_gt, film_params, text_logits, task)
            total_loss.backward(); optimizer.step()
            
            epoch_total_loss += total_loss.item()
            
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = step_count / elapsed_time; remaining_steps = total_steps - step_count 
                eta_seconds = remaining_steps / steps_per_sec; current_lr = optimizer.param_groups[0]['lr']
                
                total_loss_item = total_loss.detach().cpu().item()
                l1_loss_item = l1_loss.detach().cpu().item()
                reg_loss_item = reg_loss.detach().cpu().item()
                diag_loss_item = diag_loss.detach().cpu().item()
                
                print(f"  [P{phase}|E{epoch+1}|{batch_idx+1}/{len(train_loader)}] "
                      f"LR: {current_lr:.1e} Loss: {total_loss_item:.4f} (L1: {l1_loss_item:.4f}, Reg: {reg_loss_item:.4f}, Diag: {diag_loss_item:.4f}) "
                      f"Time: {format_time(elapsed_time)} (ETA: {format_time(eta_seconds)})")
        
        if phase == 2:
            changed, total_change = check_weight_change(model_components, initial_weights)
            print(f"[PHASE 2 - DEBUG] Epoch {epoch+1} 가중치 변화 추적: 총 변화량 (L2 Norm Sum): {total_change:.4f}")
            if len(changed) > 0 and total_change > 0.01:
                print("  ✅ Adapter 가중치가 성공적으로 업데이트되었습니다. (PEFT 작동 확인)")
            else:
                print("  ❌ 경고: Adapter 가중치 변화가 미미합니다.")


        avg_loss = epoch_total_loss / len(train_loader)
        print(f"\n[{'PHASE 1' if phase == 1 else 'PHASE 2'}] ------------------------------------")
        print(f"[{'PHASE 1' if phase == 1 else 'PHASE 2'}] EPOCH {epoch+1} 완료. 평균 Loss: {avg_loss:.4f}")
        
        evaluate_model(model_components, val_loader, phase, epoch + 1)
        
        print(f"[{'PHASE 1' if phase == 1 else 'PHASE 2'}] ------------------------------------")

        model_dict = {'backbone': backbone, 'pilot': pilot, 'generator': generator}
        save_checkpoint(model_dict, optimizer, epoch + 1, phase, f'phase{phase}_epoch_{epoch+1}.pth')

# ====================================================================
# [E] Main Execution
# ====================================================================

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- VETNet-Pilot 학습 시작 (Device: {device}) ---")

    pilot = VLLMPilot(llm_dim=LLM_Z_DIM).to(device)
    generator = FiLMGenerator(z_dim=LLM_Z_DIM, base_dim=BASE_DIM).to(device)
    backbone = RestormerVolterra(in_channels=3, out_channels=3, dim=BASE_DIM, 
                                 num_blocks=[1,1,1,1], heads=[1,1,1,1]).to(device)

    model_components = (pilot, generator, backbone)
    
    try:
        train_dataset = MultiTaskDataset(root_dir=ROOT_DIR, mode='Train', vlm_size=VLM_INPUT_SIZE)
        val_dataset = MultiTaskDataset(root_dir=ROOT_DIR, mode='Test', vlm_size=VLM_INPUT_SIZE)
        collator = MultiTaskCollator(crop_size=CROP_SIZE)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)
        print(f"2. DataLoader 준비 완료. (Train Samples: {len(train_dataset)}, Patch Size: {CROP_SIZE}x{CROP_SIZE})")

    except Exception as e:
        print(f"\n[오류] 데이터 로드 실패: {e}")
        sys.exit(1)


    print("\n================= Starting PHASE 1: VETNet Warm-up =================")
    
    optimizer1 = setup_training_phase(model_components, phase=1, base_lr=BASE_LR)
    criterion1 = RestorationLoss(lambda_reg=0.0, lambda_diag=0.0) 
    
    run_training_loop(model_components, optimizer1, criterion1, train_loader, val_loader, NUM_EPOCHS_PHASE1, phase=1)

    print("\n================= Starting PHASE 2: Pilot-Adapter Tuning =================")
    
    optimizer2 = setup_training_phase(model_components, phase=2, base_lr=BASE_LR)
    criterion2 = RestorationLoss(lambda_reg=0.01, lambda_diag=0.5) 
    
    run_training_loop(model_components, optimizer2, criterion2, train_loader, val_loader, NUM_EPOCHS_PHASE2, phase=2)

    print("\n--- VETNet-Pilot 학습 완료 ---")