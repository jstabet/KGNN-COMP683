# %%
#Set paths to parquet and edges data
import pandas as pd
import numpy as np


data = pd.read_parquet('../results/kis_viability_filt.parquet')
edges_df = pd.read_csv('../results/kinome.edgelist', names=['source', 'target'])

# %%
#Removing missing inhibition states

import torch
import torch_geometric
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv




def load_and_process_data(edge_df, parquet_df):
    edges_df = edge_df

    data = parquet_df


    measured_kinases = [col[4:] for col in data.columns if col.startswith('inh_')]
    print("\nKinase statistics:")
    print(f"Number of measured kinases: {len(measured_kinases)}")
    print("First 5 measured kinases:", measured_kinases[:5])

    edges_df_filtered = edges_df[
        (edges_df['source'].isin(measured_kinases)) &
        (edges_df['target'].isin(measured_kinases))
    ]
    network_kinases = set(measured_kinases) #set(edges_df_filtered['source']) | set(edges_df_filtered['target'])

    # 2) prune your feature list to match
    surviving_kinases = [k for k in measured_kinases if k in network_kinases]
    print(f"After pruning: {len(measured_kinases)} kinases (should equal unique nodes)")

    # 3) rebuild the kinase → index map
    kinase_to_idx = {k: i for i, k in enumerate(surviving_kinases)}

    # 4) rebuild edge_index using only the surviving kinases
    edge_index = torch.tensor([
        [kinase_to_idx[s] for s in edges_df_filtered['source']],
        [kinase_to_idx[t] for t in edges_df_filtered['target']]
    ], dtype=torch.long)
    print("\nEdge index statistics:")
    print(f"Edge index shape: {edge_index.shape}")
    print("First 5 edges (node indices):", edge_index[:, :5])

    print("\nVerifying data consistency:")
    print(f"Number of kinase features in data: {len(surviving_kinases)}")
    print(f"Number of unique nodes in edge_index: {len(torch.unique(edge_index))}")

    return data, edge_index, kinase_to_idx, surviving_kinases



data, edge_index, kinase_to_idx, measured_kinases = load_and_process_data(edges_df, data)

# %%
# map each drug and each cell line to a unique integer
drug2idx = {d:i for i,d in enumerate(data['drug'].unique())}
cl2idx   = {c:i for i,c in enumerate(data['depmap_id'].unique())}


# %%


import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import SAGEConv, BatchNorm, GlobalAttention
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn.aggr import AttentionalAggregation


# 1) Feature columns for the surviving kinases:
feature_cols = [f'inh_{k}' for k in measured_kinases]

# 2) Dataset class
from torch_geometric.data import Dataset, Data

from torch_geometric.data import Data

class KinomeDataset(Dataset):
    def __init__(self, df, edge_index, feature_cols, drug2idx, cl2idx):
        super().__init__()
        self.df           = df
        self.edge_index   = edge_index
        self.feature_cols = feature_cols
        self.drug2idx     = drug2idx
        self.cl2idx       = cl2idx

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        # dynamic node features
        dyn = row[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        x = torch.from_numpy(dyn).unsqueeze(1)  # [N,1]

        # target
        y = torch.tensor(row['viability'], dtype=torch.float32)

        # build Data object
        data = Data(x=x,
                    edge_index=self.edge_index,
                    y=y)

        # attach the new attributes *after* creation
        data.drug_idx = torch.tensor(self.drug2idx[row['drug']],    dtype=torch.long)
        data.cell_idx = torch.tensor(self.cl2idx[row['depmap_id']], dtype=torch.long)

        return data

    __getitem__ = get




# %%
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, GlobalAttention
from torch_geometric.nn.aggr import AttentionalAggregation

class KinomeGNN(nn.Module):
    def __init__(self, hidden=32, emb_dim=16,
                 num_drugs=len(drug2idx),
                 num_cells=len(cl2idx),
                 p_drop=0.4):
        super().__init__()
        # graph conv layers
        self.conv1    = SAGEConv(1,    hidden)
        self.bn1      = BatchNorm(hidden)
        self.conv2    = SAGEConv(hidden, hidden)
        self.bn2      = BatchNorm(hidden)
        self.drop     = nn.Dropout(p_drop)
        self.pool     = AttentionalAggregation(nn.Linear(hidden, 1))

        # new: drug & cell‐line embeddings
        self.drug_emb = nn.Embedding(num_drugs, emb_dim)
        self.cell_emb = nn.Embedding(num_cells, emb_dim)

        # final head: hidden + 2*emb_dim → hidden → 1
        self.head     = nn.Sequential(
            nn.Linear(hidden + 2*emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch, drug_idx, cell_idx):
        x = self.drop(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.drop(F.relu(self.bn2(self.conv2(x, edge_index))))
        g = self.pool(x, batch)         # [batch, hidden]

        # embed drug & cell
        d = self.drug_emb(drug_idx)     # [batch, emb_dim]
        c = self.cell_emb(cell_idx)     # [batch, emb_dim]

        # concat & predict
        h = torch.cat([g, d, c], dim=1) # [batch, hidden+2*emb_dim]
        return torch.sigmoid(self.head(h).squeeze(1))  # [batch]


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm.auto import tqdm
from tqdm.auto import tqdm

def run_epoch(loader, model, optimizer=None, scheduler=None, train=True, desc=None):
    model.train() if train else model.eval()
    total_loss, count = 0.0, 0

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device)
        x, ei, bat = batch.x, batch.edge_index, batch.batch
        d_idx, c_idx = batch.drug_idx, batch.cell_idx
        y = batch.y

        if train:
            optimizer.zero_grad()

        # pass all five args
        out = model(x, ei, bat, d_idx, c_idx)
        loss = F.mse_loss(out, y)

        if train:
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

        total_loss += loss.item() * batch.num_graphs
        count      += batch.num_graphs

    return total_loss / count




# %%
# ── Cell: 10-Fold CV with OneCycleLR, Progress Bars, and Standardized Targets ─────────────
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, root_mean_squared_error
from tqdm.auto import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

EPOCHS = 30  # increase for full runs

# build group keys
data['drug_cell_group'] = data['drug'].astype(str) + '_' + data['depmap_id'].astype(str)
groups = data['drug_cell_group'].values

assert_df = pd.read_parquet('../results/kis_viability_filt.parquet')
assert_df['drug_cell_group'] = assert_df['drug'] + "_" + assert_df['depmap_id']
assert data.equals(assert_df)

gkf = GroupKFold(n_splits=10)
r2_scores = []
fold_train_histories = []
fold_val_histories   = []
train_idxes = []
test_idxes  = []
gnn_r2 = []
gnn_rmse = []

for fold, (train_idx, test_idx) in enumerate(
        gkf.split(data, data['viability'], groups=groups), start=1):
    print(f"\n=== Fold {fold}/10 ===")

    # split
    train_df = data.iloc[train_idx].reset_index(drop=True)
    test_df  = data.iloc[test_idx].reset_index(drop=True)

    # compute feature standardization params
    mu_feat = train_df[feature_cols].mean()
    sd_feat = train_df[feature_cols].std().replace(0,1)

    # compute target standardization params
    y_mean = train_df['viability'].mean()
    y_std  = train_df['viability'].std()
    if y_std == 0 or np.isnan(y_std):
        y_std = 1.0


    def prep(df):
        df = df.copy()
        df[feature_cols] = (df[feature_cols] - mu_feat) / sd_feat
        df['viability']  = (df['viability'] - y_mean) / y_std
        return df

    train_df, test_df = prep(train_df), prep(test_df)

    # DataLoaders
    train_ds = KinomeDataset(train_df, edge_index, feature_cols, drug2idx, cl2idx)
    test_ds  = KinomeDataset(test_df,  edge_index, feature_cols, drug2idx, cl2idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    # model + optimizer + OneCycleLR
    model     = KinomeGNN(hidden=32, p_drop=0.4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=100
    )

    train_hist, val_hist = [], []
    best_val = float('inf')

    # epoch loop
    for epoch in trange(1, EPOCHS+1, desc=f"Fold {fold}", leave=False):
        # Training
        model.train()
        tr_loss, tr_count = 0.0, 0
        for batch in tqdm(train_loader, desc=" train", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch,
                        batch.drug_idx, batch.cell_idx)
            loss = torch.nn.functional.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * batch.num_graphs
            tr_count += batch.num_graphs
        tr_mse = tr_loss / tr_count

        # Validation
        model.eval()
        va_loss, va_count = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=" val", leave=False):
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch,
                            batch.drug_idx, batch.cell_idx)
                loss = torch.nn.functional.mse_loss(out, batch.y)
                va_loss += loss.item() * batch.num_graphs
                va_count += batch.num_graphs
        va_mse = va_loss / va_count

        # best‐val checkpoint
        if va_mse < best_val:
            best_val = va_mse
            torch.save(model.state_dict(), f'../results/gnn_feat/best_fold{fold}.pt')

        train_hist.append(tr_mse)
        val_hist.append(va_mse)
        print(f"Fold {fold} Ep{epoch:02d} | train MSE: {tr_mse:.4f} | val MSE: {va_mse:.4f}")

    fold_train_histories.append(train_hist)
    fold_val_histories.append(val_hist)

    # final R² on test fold (de‐standardize predictions)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch,
                        batch.drug_idx, batch.cell_idx)
            y_pred.append(out.cpu().numpy())
            y_true.append(batch.y.cpu().numpy())
    y_true = np.concatenate(y_true) * y_std + y_mean
    y_pred = np.concatenate(y_pred) * y_std + y_mean

    fold_r2 = r2_score(y_true, y_pred)
    print(f"Fold {fold} R² = {fold_r2:.4f}")
    r2_scores.append(fold_r2)

    gnn_rmse.append(root_mean_squared_error(y_true, y_pred))
    gnn_r2.append(r2_score(y_true, y_pred))

# summarize CV performance
r2_scores = np.array(r2_scores)
print(f"\nGNN 10-Fold CV R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

# aggregated loss plot (on standardized scale)
train_arr = np.array(fold_train_histories)
val_arr   = np.array(fold_val_histories)
epochs    = np.arange(1, train_arr.shape[1]+1)
mean_tr   = train_arr.mean(0); std_tr = train_arr.std(0)
mean_va   = val_arr.mean(0);   std_va = val_arr.std(0)

plt.figure(figsize=(6,4))
plt.plot(epochs, mean_tr, label='Train MSE (std)')
plt.fill_between(epochs, mean_tr-std_tr, mean_tr+std_tr, alpha=0.3)
plt.plot(epochs, mean_va, label='Val MSE (std)')
plt.fill_between(epochs, mean_va-std_va, mean_va+std_va, alpha=0.3)
plt.xlabel('Epoch'); plt.ylabel('Standardized MSE')
plt.title('Aggregated Train/Val Loss Across 10-Fold CV')
plt.legend(); plt.tight_layout(); plt.show()


# %%
# aggregated loss plot
gnn_rmse = np.array(gnn_rmse)
gnn_r2 = np.array(gnn_r2)

train_arr = np.array(fold_train_histories)
val_arr   = np.array(fold_val_histories)


# save these to a results file using .pt
torch.save({
    'gnn_rmse': gnn_rmse,
    'gnn_r2': gnn_r2,
    'train_hist': train_arr,
    'val_hist': val_arr
}, '../results/gnn_feat/results.pt')


