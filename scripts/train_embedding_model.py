# =========================================================
# train_embedding_model.py
# deTACTer 프로젝트용 시퀀스 임베딩 모델 학습 스크립트
# =========================================================
# 주요 기능:
# 1. GRU 및 Transformer Autoencoder 학습
# 2. Optuna TPE를 활용한 하이퍼파라미터 최적화
# 3. Y-axis flip 증강 (학습 시에만)
# 4. 버전별 모델 관리 (results/models/{version}/)
# =========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import optuna
import yaml
import os
import sys
import argparse
from pathlib import Path

# 터미널 한글 출력 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# =========================================================
# 설정 로드
# =========================================================
# 설정 로드
# v5.9: 경로 소프트 코딩 및 BASE_DIR 통일
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

VERSION = config['version']
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "data", "refined", VERSION))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "results", "models", VERSION))
os.makedirs(MODEL_DIR, exist_ok=True)

# Device 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] {DEVICE}")

# =========================================================
# Dataset 클래스
# =========================================================
class SequenceDataset(Dataset):
    """시퀀스 좌표 데이터셋 (Autoencoder용)"""
    def __init__(self, sequences, max_len=20, augment_yflip=False):
        """
        Args:
            sequences: List of numpy arrays, shape (seq_len, 2) for (x, y)
            max_len: 최대 시퀀스 길이 (패딩/자르기)
            augment_yflip: Y축 반전 증강 여부 (학습 시에만 True)
        """
        self.sequences = sequences
        self.max_len = max_len
        self.augment_yflip = augment_yflip
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()  # (seq_len, 2)
        
        # Y-flip 증강 (50% 확률)
        if self.augment_yflip and np.random.rand() < 0.5:
            seq[:, 1] = 1.0 - seq[:, 1]  # Y 좌표 반전 (0~1 정규화 가정)
        
        # Padding or Truncation
        seq_len = len(seq)
        if seq_len < self.max_len:
            # Padding with zeros
            padded = np.zeros((self.max_len, 2), dtype=np.float32)
            padded[:seq_len] = seq
            mask = np.zeros(self.max_len, dtype=np.float32)
            mask[:seq_len] = 1.0
        else:
            # Truncation
            padded = seq[:self.max_len].astype(np.float32)
            mask = np.ones(self.max_len, dtype=np.float32)
        
        return torch.tensor(padded), torch.tensor(mask)

# =========================================================
# 모델 정의
# =========================================================
class GRUAutoencoder(nn.Module):
    """GRU 기반 Sequence Autoencoder"""
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Decoder
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, mask):
        # x: (batch, seq_len, 2)
        # Encode
        _, hidden = self.encoder(x)  # hidden: (num_layers, batch, hidden_dim)
        
        # Decode (teacher forcing 없이 자기 출력 재귀)
        batch_size, seq_len, _ = x.shape
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        decoder_out, _ = self.decoder(decoder_input, hidden)
        
        # Output
        recon = self.output_layer(decoder_out)  # (batch, seq_len, 2)
        return recon, hidden[-1]  # hidden[-1]: (batch, hidden_dim) - 임베딩

class TransformerAutoencoder(nn.Module):
    """Transformer 기반 Sequence Autoencoder"""
    def __init__(self, input_dim=2, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))  # max_len=100
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=embed_dim*4, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, input_dim)
        
    def forward(self, x, mask):
        # x: (batch, seq_len, 2)
        batch_size, seq_len, _ = x.shape
        
        # Embedding
        x = self.input_proj(x)  # (batch, seq_len, embed_dim)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Mask for Transformer (True = ignore)
        src_key_padding_mask = (mask == 0)  # (batch, seq_len)
        
        # Encode
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Decode (reconstruction)
        recon = self.output_proj(encoded)  # (batch, seq_len, 2)
        
        # Embedding: mean pooling over valid positions
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        embedding = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        return recon, embedding  # embedding: (batch, embed_dim)

# =========================================================
# 학습 함수 (개선: 상세 메트릭 및 모니터링)
# =========================================================
def train_epoch(model, dataloader, optimizer, criterion, device, epoch_num=0):
    model.train()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    n_batches = 0
    
    # Real-time progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num+1} [Train]", leave=False)
    
    for x, mask in pbar:
        x, mask = x.to(device), mask.to(device)
        
        optimizer.zero_grad()
        recon, _ = model(x, mask)
        
        # Loss: MSE on valid positions only
        mask_expanded = mask.unsqueeze(-1)
        valid_recon = recon * mask_expanded
        valid_x = x * mask_expanded
        
        loss = criterion(valid_recon, valid_x)
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            mse = torch.mean((valid_recon - valid_x) ** 2)
            mae = torch.mean(torch.abs(valid_recon - valid_x))
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_mae += mae.item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}', 'mse': f'{mse.item():.6f}'})
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'mae': total_mae / n_batches
    }

def validate(model, dataloader, criterion, device, epoch_num=0):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num+1} [Valid]", leave=False)
    
    with torch.no_grad():
        for x, mask in pbar:
            x, mask = x.to(device), mask.to(device)
            recon, _ = model(x, mask)
            
            mask_expanded = mask.unsqueeze(-1)
            valid_recon = recon * mask_expanded
            valid_x = x * mask_expanded
            
            loss = criterion(valid_recon, valid_x)
            mse = torch.mean((valid_recon - valid_x) ** 2)
            mae = torch.mean(torch.abs(valid_recon - valid_x))
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_mae += mae.item()
            n_batches += 1
            
            pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'mae': total_mae / n_batches
    }

# =========================================================
# Optuna Objective (개선: 상세 로깅 및 Validation 기반 최적화)
# =========================================================
def objective(trial, model_type, sequences, max_len, log_file):
    # Hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    import datetime
    
    # Buffer all logs for this trial
    trial_logs = []
    
    # Log trial start
    trial_log = f"\n{'='*60}\n"
    trial_log += f"Trial {trial.number} - {model_type.upper()}\n"
    trial_log += f"  hidden_dim: {hidden_dim}, num_layers: {num_layers}\n"
    trial_log += f"  dropout: {dropout:.3f}, lr: {lr:.6f}, batch_size: {batch_size}\n"
    
    # Model
    if model_type == 'gru':
        model = GRUAutoencoder(input_dim=2, hidden_dim=hidden_dim, 
                              num_layers=num_layers, dropout=dropout).to(DEVICE)
        trial_log += f"  Model: GRU Autoencoder\n"
    else:  # transformer
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        if hidden_dim % num_heads != 0:
            hidden_dim = (hidden_dim // num_heads) * num_heads
        model = TransformerAutoencoder(input_dim=2, embed_dim=hidden_dim,
                                      num_heads=num_heads, num_layers=num_layers,
                                      dropout=dropout).to(DEVICE)
        trial_log += f"  Model: Transformer Autoencoder (heads: {num_heads})\n"
    
    # Data split (80/20 for train/validation)
    split_idx = int(len(sequences) * 0.8)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]
    
    trial_log += f"  Train: {len(train_seqs)} seqs, Valid: {len(val_seqs)} seqs\n"
    
    train_dataset = SequenceDataset(train_seqs, max_len=max_len, augment_yflip=True)
    val_dataset = SequenceDataset(val_seqs, max_len=max_len, augment_yflip=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_val_metrics = None
    patience = 5
    patience_counter = 0
    
    print(trial_log)
    
    for epoch in range(50):  # Max epochs
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        val_metrics = validate(model, val_loader, criterion, DEVICE, epoch)
        
        # Buffer epoch log (don't write to file yet)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"  [{timestamp}] Epoch {epoch+1:2d}: Train Loss={train_metrics['loss']:.6f}, "
        log_msg += f"Valid Loss={val_metrics['loss']:.6f}, MAE={val_metrics['mae']:.6f}\n"
        trial_logs.append(log_msg)
        print(log_msg, end='')
        
        # Early stopping based on VALIDATION loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stop_msg = f"\n{'='*60}\n"
                early_stop_msg += f"⚠️  EARLY STOPPING 발동 (정상 종료)\n"
                early_stop_msg += f"   Epoch {epoch+1}에서 조기 종료\n"
                early_stop_msg += f"   이유: {patience}번 연속 validation loss 개선 없음\n"
                early_stop_msg += f"   최고 성능: Epoch {epoch+1-patience} (Val Loss: {best_val_loss:.6f})\n"
                early_stop_msg += f"{'='*60}\n"
                trial_logs.append(early_stop_msg)
                print(early_stop_msg)
                break
        
        # Pruning
        trial.report(val_metrics['loss'], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Final log
    final_log = f"  BEST Valid Loss: {best_val_loss:.6f}, MAE: {best_val_metrics['mae']:.6f}\n"
    final_log += f"{'='*60}\n"
    print(final_log)
    
    # Write all logs at once with file locking (Windows-compatible)
    import time
    max_retries = 10
    total_log_content = trial_log + "".join(trial_logs) + final_log
    
    for retry in range(max_retries):
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                # Try to get exclusive access (Windows specific)
                try:
                    import msvcrt
                    # Lock the file for writing
                    f.seek(0, 2) # Move to end
                    # Lock some large enough area if possible, or just write
                except ImportError:
                    pass
                
                f.write(total_log_content)
                f.flush()
            break
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(0.1)
            else:
                print(f"Warning: Failed to write log after {max_retries} retries: {e}")
    
    return best_val_loss  # Optuna minimizes this

# =========================================================
# 메인 실행 (개선: 로깅 및 50 trials)
# =========================================================
# =========================================================
# 메인 실행 (개선: 로깅 및 200 trials)
# =========================================================
# =========================================================
# 메인 실행 (개선: 로깅 및 200 trials)
# =========================================================
import sequence_extraction  # 모듈 임포트
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score

def main(model_type='transformer', n_trials=200, version_arg=None):
    # 버전 결정 (인자가 있으면 우선, 없으면 config)
    global VERSION, DATA_DIR, MODEL_DIR
    if version_arg:
        VERSION = version_arg
    else:
        VERSION = config['version']
        
    # BASE_DIR은 상단에서 정의된 값 사용
    DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "data", "refined", VERSION))
    MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "results", "models", VERSION))
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"============================================================")
    print(f"deTACTer 임베딩 모델 TPE Search ({model_type.upper()}) - v5.8")
    print(f"  Target: Minimize Validation Loss (MSE)")
    print(f"  버전: {VERSION}")
    print(f"  모델 저장 경로: {MODEL_DIR}")
    print(f"  TPE Trials: {n_trials}")
    print(f"============================================================")
    
    # Log file
    log_file = f"{MODEL_DIR}{model_type}_training_log.txt"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"deTACTer {model_type.upper()} Final Training Session - {VERSION}\n")
        f.write(f"{'='*60}\n\n")
    
    # [v5.8] 데이터 로드 (Preprocessed Data)
    raw_df = sequence_extraction.load_preprocessed_data()
    outcome_df = sequence_extraction.identify_outcome_events(raw_df)
    
    print(f"[{VERSION}] 성과 이벤트 식별 완료. TPE Loop 진입...")
    
    # Optuna Study (Maximize Silhouette)
    study_name = f"{model_type}_{VERSION}_optics_search"
    storage = f"sqlite:///{MODEL_DIR}{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize', # Minimize MSE
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Check existing trials
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, n_trials - n_completed)
    
    print(f"\n[{VERSION}] Optuna Study 상태:")
    print(f"  기존 완료: {n_completed} trials")
    print(f"  추가 실행: {n_remaining} trials")
    
    if n_remaining > 0:
        print(f"\n[{VERSION}] TPE 탐색 시작... ({n_remaining}개 trial 추가 실행)")
        study.optimize(lambda trial: objective_dynamic(trial, model_type, outcome_df, log_file), 
                      n_trials=n_remaining)
        print(f"\n{'='*60}")
        print(f"✅ [{VERSION}] TPE 탐색 정상 완료!")
    
    # Best params summary
    print(f"\n{'='*60}")
    print(f"[{VERSION}] {model_type.upper()} 최적 파라미터 탐색 완료")
    print(f"  Best Silhouette Score: {study.best_value:.6f}")
    print(f"  Best Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print(f"{'='*60}\n")
    
    # Save best model logic
    best_params = study.best_params
    
    # 최적 파라미터로 최종 데이터 추출
    print(f"[{VERSION}] 최적 파라미터로 최종 시퀀스 데이터 추출 중...")
    final_total_len = best_params['total_seq_len']
    final_min_passes = best_params['min_pass_count']
    
    final_seq_df = sequence_extraction.extract_sequences(
        outcome_df, total_len=final_total_len, min_passes=final_min_passes
    )
    
    # 시퀀스별 좌표 변환
    final_sequences = []
    for seq_id in final_seq_df['sequence_id'].unique():
        seq_data = final_seq_df[final_seq_df['sequence_id'] == seq_id].sort_values('seq_position')
        coords = seq_data[['start_x', 'start_y']].values
        final_sequences.append(coords)
    
    print(f"  -> 최종 시퀀스: {len(final_sequences)}개 (Len: {final_total_len})")
    
    # 모델 재학습 (100 에폭 + Early Stopping)
    if model_type == 'gru':
        best_model = GRUAutoencoder(
            input_dim=2,
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        ).to(DEVICE)
    else:
        best_model = TransformerAutoencoder(
            input_dim=2,
            embed_dim=best_params['hidden_dim'],
            num_heads=best_params['num_heads'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        ).to(DEVICE)
        
    # 최종 학습 데이터 분할 (Early Stopping용)
    split_idx = int(len(final_sequences) * 0.9)
    f_train_seqs = final_sequences[:split_idx]
    f_val_seqs = final_sequences[split_idx:]
    
    f_train_dataset = SequenceDataset(f_train_seqs, max_len=final_total_len, augment_yflip=True)
    f_val_dataset = SequenceDataset(f_val_seqs, max_len=final_total_len, augment_yflip=False)
    
    f_train_loader = DataLoader(f_train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    f_val_loader = DataLoader(f_val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()
    
    print(f"[{VERSION}] 최적 파라미터로 최종 모델 학습 중 (Max 100 epochs, EarlyStopping)...")
    
    best_f_val_loss = float('inf')
    f_patience = 10
    f_counter = 0
    best_state = None
    
    for epoch in range(100):
        train_metrics = train_epoch(best_model, f_train_loader, optimizer, criterion, DEVICE, epoch)
        val_metrics = validate(best_model, f_val_loader, criterion, DEVICE, epoch)
        
        if val_metrics['loss'] < best_f_val_loss:
            best_f_val_loss = val_metrics['loss']
            best_state = best_model.state_dict()
            f_counter = 0
        else:
            f_counter += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"  [{VERSION}] Epoch {epoch+1}/100, TrLoss: {train_metrics['loss']:.6f}, ValLoss: {val_metrics['loss']:.6f}")
            
        if f_counter >= f_patience:
            print(f"  [{VERSION}] Early Stopping at epoch {epoch+1}")
            break
            
    # Best state 복구
    if best_state is not None:
        best_model.load_state_dict(best_state)
            
    # Save Model & Metadata
    model_path = Path(MODEL_DIR) / f"{model_type}_best.pth"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'params': best_params,
        'max_len': final_total_len,
        'best_score': study.best_value,
        'final_val_loss': best_f_val_loss,
        'version': VERSION,
        'model_type': model_type
    }, model_path)
    
    # [v5.8] 최종 시퀀스 데이터도 저장 (분석용)
    seq_save_path = f"{DATA_DIR}/attack_sequences.csv"
    final_seq_df.to_csv(seq_save_path, index=False, encoding='utf-8-sig')
    print(f"[{VERSION}] 최종 시퀀스 데이터 저장 완료: {seq_save_path}")
    
    print(f"\n[{VERSION}] 최적 모델 저장 완료: {model_path}")

# =========================================================
# Dynamic Objective Function (OPTICS + Expanded Space)
# =========================================================
def objective_dynamic(trial, model_type, outcome_df, log_file):
    # Dynamic Data Generation for this Trial
    # [v5.9] User Fixed Parameters Check
    if VERSION == 'v5.9':
        total_seq_len = trial.suggest_int('total_seq_len', 8, 8) 
        min_pass_count = trial.suggest_int('min_pass_count', 3, 3)
        optics_min_samples = trial.suggest_int('optics_min_samples', 3, 10)
    else:
        # Suggest parameters for sequence extraction (Standard)
        total_seq_len = trial.suggest_int('total_seq_len', 3, 10)
        min_pass_count = trial.suggest_int('min_pass_count', 2, 5)
        optics_min_samples = trial.suggest_int('optics_min_samples', 3, 15)
        
    print(f"\n[Trial {trial.number}] Data Params: Len={total_seq_len}, MP={min_pass_count}, MinS={optics_min_samples}")
    
    # 2. Extract Sequences
    try:
        # config의 값을 덮어쓰지 않고 함수 인자로 전달
        seq_df = sequence_extraction.extract_sequences(
            outcome_df, total_len=total_seq_len, min_passes=min_pass_count
        )
    except Exception as e:
        print(f"[Trial Error] Extraction failed: {e}")
        return -1.0
        
    if seq_df.empty or len(seq_df['sequence_id'].unique()) < 50:
        print(f"[Trial Pruned] Not enough sequences ({len(seq_df) if not seq_df.empty else 0})")
        raise optuna.TrialPruned()
        
    sequences = []
    for seq_id in seq_df['sequence_id'].unique():
        seq_data = seq_df[seq_df['sequence_id'] == seq_id].sort_values('seq_position')
        coords = seq_data[['start_x', 'start_y']].values
        sequences.append(coords)
        
    # 3. Model Params (Expanded)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 1, 8)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Logging
    import datetime
    trial_logs = []
    trial_log = f"\n{'='*60}\n"
    trial_log += f"Trial {trial.number} - {model_type.upper()}\n"
    trial_log += f"  [Data] Len:{total_seq_len}, MP:{min_pass_count} -> {len(sequences)} seqs\n"
    trial_log += f"  [Model] H:{hidden_dim}, L:{num_layers}, D:{dropout:.3f}, lr:{lr:.5f}, B:{batch_size}\n"
    
    # Create Model
    try:
        model = None
        if model_type == 'gru':
            model = GRUAutoencoder(input_dim=2, hidden_dim=hidden_dim, 
                                  num_layers=num_layers, dropout=dropout).to(DEVICE)
        else:  # transformer
            # Heads: 2, 4, 8, 16. Must divide hidden_dim
            possible_heads = [h for h in [2, 4, 8, 16] if hidden_dim % h == 0]
            if not possible_heads: 
                # Fallback if no valid head (unlikely with powers of 2)
                possible_heads = [1]
                
            num_heads = trial.suggest_categorical('num_heads', possible_heads)
            model = TransformerAutoencoder(input_dim=2, embed_dim=hidden_dim,
                                          num_heads=num_heads, num_layers=num_layers,
                                          dropout=dropout).to(DEVICE)
            trial_log += f"  [Transf] heads:{num_heads}\n"
            # Save extra params
            trial.set_user_attr("num_heads", num_heads)
                
        # Train (Short Epochs for TPE)
        dataset = SequenceDataset(sequences, max_len=total_seq_len, augment_yflip=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Validation Dataset
        val_dataset = SequenceDataset(sequences, max_len=total_seq_len, augment_yflip=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model.train()
        for epoch in range(15): # Fast TPE
            train_epoch(model, dataloader, optimizer, criterion, DEVICE, epoch)
            
        # Evaluation: Validation Loss
        val_metrics = validate(model, val_loader, criterion, DEVICE, 0)
        val_loss = val_metrics['loss']
            
        trial_log += f"  [Model] H:{hidden_dim}, L:{num_layers}, D:{dropout:.3f}, lr:{lr:.5f}, B:{batch_size}\n"
        if model_type == 'transformer':
            trial_log += f"  [Transf] heads:{num_heads}\n"
        trial_log += f"  [Result] Val Loss: {val_loss:.6f} (Minimize)\n"
        trial_log += f"{'='*60}\n"
        
        # Log to file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(trial_log)
            
        print(f"[Trial {trial.number}] Val Loss: {val_loss:.6f}")
        return val_loss

    except Exception as e:
        print(f"[Trial Error] Model training failed: {e}")
        return float('inf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gru', choices=['gru', 'transformer'])
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--fresh', action='store_true', 
                       help='기존 study DB를 삭제하고 처음부터 새로 시작')
    args = parser.parse_args()
    
    # Handle fresh start
    if args.fresh and args.version:
        version_to_clean = args.version
        # BASE_DIR은 임포트 시점에 정의됨
        model_dir = os.path.join(BASE_DIR, 'results', 'models', version_to_clean)
        db_file = Path(model_dir) / f"{args.model}_{version_to_clean}.db"
        if db_file.exists():
            print(f"[Fresh Start] 기존 study DB 삭제: {db_file}")
            db_file.unlink()
    
    main(model_type=args.model, n_trials=args.trials, version_arg=args.version)
