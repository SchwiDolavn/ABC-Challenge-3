
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset


# ---------------------Delete old file in virtual folder-------------------------
def delete_files_in_directory(directory_path):
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Usage
directory_path = './data/virtual'
delete_files_in_directory(directory_path)
# -------------------------------------------------------------------------------
# ------------------- Helper Function: Random Window Sampling -------------------
def random_window_sampling(data, seq_len):
    """
    Cut data (n_samples, 6) into random windows of length seq_len.
    Returns an array of shape (num_segments, seq_len, 6).
    """
    n = data.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len, data.shape[1]))
    num_segments = n // seq_len
    segments = []
    for _ in range(num_segments):
        start = np.random.randint(0, n - seq_len + 1)
        seg = data[start:start+seq_len, :]
        segments.append(seg)
    return np.array(segments)

# ------------------- Helper Module: Positional Encoding -------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

# ------------------- Model: Transformer Autoencoder-------------------
class TransformerAutoencoder(nn.Module):
    def __init__(self, seq_len=300, input_dim=6, model_dim=256, num_layers=16, num_heads=8, dropout=0.01):
        super(TransformerAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.flatten_dim = seq_len * input_dim

        # Encoder: Linear projection, positional encoding, and TransformerEncoder.
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout=dropout, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent representation
        self.fc_latent = nn.Linear(model_dim, model_dim)

        # Decoder
        self.fc_decode = nn.Linear(model_dim, self.flatten_dim)

    def encode(self, x):
        # x: (batch, seq_len, input_dim)
        x_emb = self.embedding(x)               # (batch, seq_len, model_dim)
        x_emb = self.pos_encoder(x_emb)
        encoded = self.transformer_encoder(x_emb)  # (batch, seq_len, model_dim)
        pooled = encoded.mean(dim=1)            # (batch, model_dim)
        latent = self.fc_latent(pooled)         # (batch, model_dim)
        return latent

    def decode(self, latent):
        out_flat = self.fc_decode(latent)       # (batch, flatten_dim)
        out_seq = out_flat.view(-1, self.seq_len, self.input_dim)  # (batch, seq_len, input_dim)
        return out_seq

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

def transformer_autoencoder_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')


def save_virtual_data(data, filename):
    data.to_csv(os.path.join(virt_directory, filename+'.csv'), index=False)
    return

def custom_virtual_data_generation(train_data_dict):
    seq_len = 300
    synthetic_data_list = []

    # Calculate overall label distribution from train_data_dict
    all_labels = []
    for u, df in train_data_dict.items():
        all_labels.extend(df[selected_columns[-1]].values)
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    max_count = np.max(counts)
    label2count = dict(zip(unique_labels, counts))

    MINORITY_THRESHOLD = 0.1 
    scale_factor_minor = 6  
    scale_factor_major = 3
    num_epochs = 100     
    minority_labels = [lb for lb, c in label2count.items() if c < MINORITY_THRESHOLD * max_count]
    
    dfs=[]
    for u, df in train_data_dict.items():
        print('Gener ating virtual data from user %s.'% u)
        raw_data = df[selected_columns[:6]].copy()
        labels = df[selected_columns[-1]].copy().astype(str)
        df=pd.concat([raw_data,labels],axis=1)
        dfs.append(df)
    dfs=np.asarray(dfs, dtype="object")
    for u in range(len(dfs)):
        try:
            df_user=pd.concat(dfs[[u,u+1]],axis=0)
        except:
            df_user=pd.concat(dfs[[0,u]],axis=0)    
    # for u, df_user in train_data_dict.items():
        # print(f"Generating virtual data from user {u}")
        sensor_data = df_user[selected_columns[:6]]
        labels_series = df_user[selected_columns[-1]]
        df_user_ = pd.concat([sensor_data, labels_series], axis=1)

        synthetic_segments_all = []
        for lbl in df_user_[selected_columns[-1]].unique():
            df_label = df_user_[df_user_[selected_columns[-1]] == lbl]
            data_array = df_label[selected_columns[:6]].values
            if data_array.shape[0] < seq_len:
                continue

            # Cut the data into random windows
            segments = random_window_sampling(data_array, seq_len)
            if segments.shape[0] == 0:
                continue

            segments_tensor = torch.tensor(segments, dtype=torch.float).to(device)
            dataset = TensorDataset(segments_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Initialize the Transformer Autoencoder
            model_tr = TransformerAutoencoder(seq_len=seq_len, input_dim=6, model_dim=256, num_layers=8, num_heads=8, dropout=0.01).to(device)
            optimizer = torch.optim.Adam(model_tr.parameters(), lr=1e-3)

            # Train the autoencoder on windows from this label
            model_tr.train()
            for epoch in range(num_epochs):
                total_loss = 0.0
                for batch in loader:
                    batch_x = batch[0]  # (batch, seq_len, 6)
                    optimizer.zero_grad()
                    recon, latent = model_tr(batch_x)
                    loss = transformer_autoencoder_loss(recon, batch_x)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch_x.size(0)
                # Optionally, print average loss per epoch:
                # avg_loss = total_loss / len(dataset)
                # print(f"User {u}, Label {lbl}, Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            if lbl in minority_labels:
                scale_factor = scale_factor_minor
            else:
                scale_factor = scale_factor_major

            # Generate synthetic data using the trained model
            model_tr.eval()
            with torch.no_grad():
                num_segments = segments_tensor.size(0)
                gen_count = int(num_segments * scale_factor)
                latent_samples = torch.randn(gen_count, 256).to(device)
                generated = model_tr.decode(latent_samples)  # (gen_count, seq_len, 6)
                out_seq = generated.cpu().numpy()

            for seg in out_seq:
                df_seg = pd.DataFrame(seg, columns=selected_columns[:6])
                df_seg[selected_columns[-1]] = lbl
                synthetic_segments_all.append(df_seg)

        if len(synthetic_segments_all) == 0:
            save_virtual_data(df_user_, str(u))
        else:
            df_synthetic = pd.concat(synthetic_segments_all, axis=0)
            df_synthetic = df_synthetic.sample(frac=1).reset_index(drop=True)
            save_virtual_data(df_synthetic, str(u))
            synthetic_data_list.append(df_synthetic)
        print('done')

custom_virtual_data_generation(train_data_dict)