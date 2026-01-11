# =========================================================
# extract_embeddings.py
# GRU/Transformer 모델로부터 시퀀스 임베딩 추출
# =========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
from pathlib import Path
import sys

# 터미널 한글 출력
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# =========================================================
# 설정
# =========================================================
# 설정
# v5.9: 경로 및 버전 참조 정상화
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================
# Dataset (train_embedding_model.py와 동일)
# =========================================================
class SequenceDataset(Dataset):
    def __init__(self, sequences, max_len=20):
        self.sequences = sequences
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        seq_len = len(seq)
        
        if seq_len < self.max_len:
            padded = np.zeros((self.max_len, 2), dtype=np.float32)
            padded[:seq_len] = seq
            mask = np.zeros(self.max_len, dtype=np.float32)
            mask[:seq_len] = 1.0
        else:
            padded = seq[:self.max_len].astype(np.float32)
            mask = np.ones(self.max_len, dtype=np.float32)
        
        return torch.tensor(padded), torch.tensor(mask)

# =========================================================
# 모델 정의 (train_embedding_model.py와 동일)
# =========================================================
class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, mask):
        _, hidden = self.encoder(x)
        batch_size, seq_len, _ = x.shape
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder(decoder_input, hidden)
        recon = self.output_layer(decoder_out)
        return recon, hidden[-1]

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=embed_dim*4, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, input_dim)
        
    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        src_key_padding_mask = (mask == 0)
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        recon = self.output_proj(encoded)
        
        mask_expanded = mask.unsqueeze(-1)
        embedding = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        return recon, embedding

# =========================================================
# 임베딩 추출
# =========================================================
def extract_embeddings(model_type, version):
    """모델로부터 임베딩 추출"""
    
    print(f"\n{'='*60}")
    print(f"임베딩 추출: {model_type.upper()} - {version}")
    print(f"{'='*60}")
    
    # 데이터 로드
    data_dir = f"{BASE_DIR}/data/refined/{version}/"
    seq_df = pd.read_csv(f"{data_dir}attack_sequences.csv", encoding='utf-8-sig')
    
    # 시퀀스 추출
    sequences = []
    sequence_ids = []
    for seq_id in seq_df['sequence_id'].unique():
        seq_data = seq_df[seq_df['sequence_id'] == seq_id].sort_values('seq_position')
        coords = seq_data[['start_x', 'start_y']].values
        sequences.append(coords)
        sequence_ids.append(seq_id)
    
    print(f"총 {len(sequences)}개 시퀀스 로드")
    
    # 모델 로드
    model_dir = f"{BASE_DIR}/results/models/{version}/"
    model_path = Path(model_dir) / f"{model_type}_best.pth"
    
    if not model_path.exists():
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    params = checkpoint['params']
    max_len = checkpoint['max_len']
    
    print(f"모델 파일 로드: {model_path.name}")
    print(f"최적 파라미터: {params}")
    
    # 모델 생성
    if model_type == 'gru':
        model = GRUAutoencoder(
            input_dim=2,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(DEVICE)
    else:  # transformer
        model = TransformerAutoencoder(
            input_dim=2,
            embed_dim=params['hidden_dim'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 데이터셋 및 로더
    dataset = SequenceDataset(sequences, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 임베딩 추출
    all_embeddings = []
    
    print(f"임베딩 추출 중...")
    with torch.no_grad():
        for x, mask in dataloader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            _, embeddings = model(x, mask)
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"임베딩 shape: {all_embeddings.shape}")
    
    # 결과 저장
    output_dir = Path(f"{BASE_DIR}/results/embeddings/{version}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{model_type}_embeddings.npy"
    np.save(output_path, all_embeddings)
    
    # 메타데이터 저장
    meta_df = pd.DataFrame({
        'sequence_id': sequence_ids,
        'embedding_dim': all_embeddings.shape[1]
    })
    meta_path = output_dir / f"{model_type}_metadata.csv"
    meta_df.to_csv(meta_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 저장 완료:")
    print(f"   임베딩: {output_path}")
    print(f"   메타데이터: {meta_path}")
    print(f"{'='*60}\n")
    
    return all_embeddings

# =========================================================
# 메인
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['gru', 'transformer'])
    parser.add_argument('--version', type=str, required=True)
    args = parser.parse_args()
    
    extract_embeddings(args.model, args.version)
