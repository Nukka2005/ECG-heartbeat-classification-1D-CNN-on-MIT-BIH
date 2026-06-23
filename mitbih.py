# %%
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score

# pin seeds, otherwise auc drifts a few points each run
torch.manual_seed(42)
np.random.seed(42)

# %%
wfdb.dl_database('mitdb', dl_dir='./mitdb')

# %%
record = wfdb.rdrecord('mitdb/100', sampfrom=0, sampto=3600)
ann    = wfdb.rdann('mitdb/100', 'atr')
print(record.p_signal.shape)
print(ann.symbol[:20])

# %%
signal = record.p_signal[:, 0]   # MLII
print(signal.shape)
print(signal[0])
print(len(signal) / 360)

# %%
def extract_windows(record_name, data_dir='./mitdb'):
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    ann = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
    signal = record.p_signal[:, 0]

    windows = []
    labels = []

    for pos, sym in zip(ann.sample, ann.symbol):
        if sym not in ('N', 'V'):
            continue

        window = signal[pos - 90:pos + 90]   # 180 samples around the r-peak
        if len(window) != 180:
            continue   # beat sits too close to the start/end

        windows.append(window)
        labels.append(0 if sym == 'N' else 1)

    return np.array(windows), np.array(labels)

# %%
w, l = extract_windows('100')
print(w.shape)
print(l.shape)
print(np.bincount(l))

# %%
print(w[0])
print(l[0])

# %%
all_records = sorted([
    f.replace('.hea', '')
    for f in os.listdir('./mitdb')
    if f.endswith('.hea')
])
print(len(all_records))
print(all_records)

# %%
# split on record id, not on beats -> same patient never lands in both sets
train_records = [r for r in all_records if int(r) <= 200]
test_records  = [r for r in all_records if int(r) > 200]

print(len(train_records), len(test_records))
print(train_records)
print(test_records)

# %%
def build_dataset(record_list):
    all_windows = []
    all_labels = []

    for r in record_list:
        w, l = extract_windows(r)
        if len(w) == 0:
            # a couple of records have zero N/V beats; the empty array breaks concatenate
            print(f'{r}: 0 beats kept — skipping')
            continue
        all_windows.append(w)
        all_labels.append(l)
        print(f'{r}: {len(l)} beats kept')

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y

# %%
X_train, y_train = build_dataset(train_records)
X_test,  y_test  = build_dataset(test_records)

print('train:', X_train.shape, y_train.shape, np.bincount(y_train))
print('test: ', X_test.shape,  y_test.shape,  np.bincount(y_test))

# %%
class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(2)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, 1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # (B, 32, 90)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # (B, 64, 45)
        x = self.relu(self.conv3(x))                        # (B, 128, 45)
        x = self.gap(x)                                     # (B, 128, 1)
        x = x.flatten(1)                                    # (B, 128)
        x = self.fc(x)                                      # (B, 1) logit
        return x

# %%
model = ECGNet()
out = model(torch.randn(8, 1, 180))   # dummy batch, just checking shapes
print(out.shape)

# %%
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)   # -> (N, 1, 180)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)   # float (N,1) to match the logits
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

print(X_train_t.shape, y_train_t.shape)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

# X is sorted by patient, so shuffle or a batch ends up all one class
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# %%
xb, yb = next(iter(train_loader))
print(xb.shape)
print(yb.shape)
print(yb[:10].squeeze())

# %%
criterion = nn.BCEWithLogitsLoss()   # does the sigmoid itself, so give it logits
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch_losses = []

for epoch in range(20):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)   # *batch size, last batch is smaller

    epoch_loss = running_loss / len(train_ds)
    epoch_losses.append(epoch_loss)
    print(f'Epoch {epoch + 1:2d}/20   loss: {epoch_loss:.4f}')

# %%
plt.figure()
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title('Training loss per epoch')
plt.grid(True)
plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
model.eval()   # batchnorm switches to running stats here

all_probs = []
all_true  = []

with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        probs  = torch.sigmoid(logits)
        all_probs.append(probs)
        all_true.append(yb)

all_probs = torch.cat(all_probs).squeeze().numpy()
all_true  = torch.cat(all_true).squeeze().numpy()

auc = roc_auc_score(all_true, all_probs)   # needs probs, not the 0/1 preds
print(f'Test AUROC: {auc:.4f}')

# %%
fpr, tpr, thresholds = roc_curve(all_true, all_probs)
idx = (np.abs(thresholds - 0.5)).argmin()   # closest point to the 0.5 cutoff

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC  (AUROC = {auc:.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='random (0.5)')
plt.scatter(fpr[idx], tpr[idx], color='red', zorder=5, label='threshold = 0.5')
plt.xlabel('False Positive Rate  (1 − specificity)')
plt.ylabel('True Positive Rate  (sensitivity)')
plt.title('ROC curve — test set')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
preds = (all_probs >= 0.5).astype(int)
print(f'recall (sensitivity): {recall_score(all_true, preds):.3f}')
print(f'precision:            {precision_score(all_true, preds):.3f}')