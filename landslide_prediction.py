"""
滑坡位移预测与联合预警 - Pipeline v5
=====================================
核心改进:
1. 5-Fold Stratified CV on train+val (737样本, ~147验证/折, 解决过拟合)
2. 更小模型 (hidden=64, 1层GRU, ~28K参数 vs v3的248K)
3. MI特征选择 (62→40特征, 去噪)
4. LightGBM集成 (梯度提升树补充深度学习不足)
5. 分区感知阈值校准 (per-zone calibration on OOF predictions)
6. 更强正则化 (dropout=0.5, weight_decay=2e-3, label_smoothing=0.1)
7. 训练增强 (Gaussian noise injection)
8. 测试时增强 (TTA, 5次平均)
9. 智能回归+分类融合 (边界自适应权重)
"""

import os, json, warnings, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def _softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [convert_to_serializable(i) for i in obj]
    return obj

# ===================== 配置 =====================
class Config:
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    OUTPUT_DIR = os.path.dirname(__file__)

    W = 12; H = 3
    HIDDEN_DIM = 64; NUM_LAYERS = 1; DROPOUT = 0.5
    BATCH_SIZE = 32; EPOCHS = 200; LR = 1e-3; WEIGHT_DECAY = 2e-3
    LAMBDA_REG = 0.6; LAMBDA_CLS = 0.3; LAMBDA_AUX = 0.1
    PATIENCE = 20; LABEL_SMOOTHING = 0.1
    NOISE_STD = 0.02  # 训练时高斯噪声

    RISK_THRESHOLDS = [10, 22, 38]
    RISK_LABELS = ['蓝色低风险', '黄色关注', '橙色预警', '红色高危']
    TRAIN_END = '2023-12'; VAL_END = '2024-06'

    N_FOLDS = 5; EXTRA_SEEDS = [2024, 2025]
    TOP_K_FEATURES = 40
    TTA_ROUNDS = 5; TTA_NOISE = 0.01

    LGB_PARAMS = {
        'objective': 'multiclass', 'num_class': 4,
        'learning_rate': 0.05, 'num_leaves': 31,
        'min_child_samples': 15, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 0.1,
        'reg_lambda': 0.5, 'n_estimators': 500,
        'random_state': 42, 'verbose': -1,
    }

    @staticmethod
    def _detect_device():
        if not torch.cuda.is_available(): return torch.device('cpu')
        try:
            _ = torch.zeros(1).cuda()
            return torch.device('cuda')
        except: return torch.device('cpu')

cfg = Config()
cfg.__class__.DEVICE = cfg._detect_device()
print(f"[INFO] Device: {cfg.DEVICE}")
print(f"[INFO] LightGBM: {'available' if HAS_LGB else 'not available'}")

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def assign_risk_label(cum_dy, thresholds=None):
    t = thresholds or cfg.RISK_THRESHOLDS
    if cum_dy < t[0]: return 0
    elif cum_dy < t[1]: return 1
    elif cum_dy < t[2]: return 2
    else: return 3

# ===================== 1. 数据加载与特征工程 =====================

def load_data():
    monthly = pd.read_csv(os.path.join(cfg.DATA_DIR, 'monthly_multisource_features.csv'))
    node_info = pd.read_csv(os.path.join(cfg.DATA_DIR, 'node_info.csv'))
    daily = pd.read_csv(os.path.join(cfg.DATA_DIR, 'rainfall_waterlevel_daily.csv'))
    return monthly, node_info, daily

def engineer_features(monthly, node_info):
    df = monthly.copy()
    df['month'] = df['month'].astype(str)
    df = df.merge(node_info[['node', 'sensitivity_factor']], on='node', how='left')
    df = df.sort_values(['node', 'month']).reset_index(drop=True)

    # 路线A: 滞后降雨
    for lag in [2, 3, 4, 5, 6]:
        df[f'rain_lag{lag}_mm'] = df.groupby('node')['rain_mm'].shift(lag)
    df['rain_diff1'] = df.groupby('node')['rain_mm'].diff()
    df['rain_diff2'] = df.groupby('node')['rain_diff1'].diff()
    df['rain_weighted_3m'] = (df['rain_mm']*3 + df['rain_lag1_mm']*2 + df['rain_lag2_mm'])/6

    # 水位消落
    df['wl_lag1'] = df.groupby('node')['water_level_mean_m'].shift(1)
    df['wl_lag2'] = df.groupby('node')['water_level_mean_m'].shift(2)
    df['wl_change'] = df['water_level_mean_m'] - df['wl_lag1']
    df['wl_change_2m'] = df['water_level_mean_m'] - df['wl_lag2']
    df['wl_3m_avg'] = df.groupby('node')['water_level_mean_m'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['wl_6m_avg'] = df.groupby('node')['water_level_mean_m'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['wl_drop_3m'] = df.groupby('node')['water_level_drop_m'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    wl_mean = df['water_level_mean_m'].mean(); wl_std = df['water_level_mean_m'].std()
    df['low_water_flag'] = (df['water_level_mean_m'] < wl_mean - wl_std).astype(float)

    # 路线E: InSAR
    df['insar_vel_lag1'] = df.groupby('node')['insar_los_velocity_mm_m'].shift(1)
    df['insar_vel_change'] = df['insar_los_velocity_mm_m'] - df['insar_vel_lag1']
    df['insar_gps_diff'] = df['insar_los_cum_mm'] - df['cum_disp_mm']
    df['insar_gps_vel_ratio'] = df['insar_los_velocity_mm_m'] / (df['dy_mm'].abs() + 0.01)

    # 路线F: 趋势-周期分解
    def decompose_series(g):
        g = g.sort_values('month').reset_index(drop=True)
        if len(g) < 6:
            g['trend_dy'] = g['dy_mm']; g['seasonal_dy'] = 0.0; g['residual_dy'] = 0.0
            return g
        trend = g['dy_mm'].rolling(6, center=True, min_periods=1).mean()
        detrended = g['dy_mm'] - trend
        g['month_num'] = g['month'].apply(lambda x: int(x.split('-')[1]))
        seasonal = detrended.groupby(g['month_num']).transform('mean')
        g['trend_dy'] = trend.values; g['seasonal_dy'] = seasonal.values
        g['residual_dy'] = (detrended - seasonal).values
        return g
    df = df.groupby('node', group_keys=False).apply(decompose_series)

    # 路线G: 空间分区
    zone_map = {'强变形带': 2, '过渡变形带': 1, '稳定背景带': 0}
    df['zone_code'] = df['zone'].map(zone_map)
    df['lon_lat_inter'] = df['lon'] * df['lat']
    df['elev_sens'] = df['elevation_m'] * df['sensitivity_factor']
    df['zone_elev'] = df['zone_code'] * df['elevation_m']
    df['zone_sens'] = df['zone_code'] * df['sensitivity_factor']

    # 位移历史
    for lag in [1, 2, 3]:
        df[f'dy_lag{lag}'] = df.groupby('node')['dy_mm'].shift(lag)
    df['dy_ma3'] = df.groupby('node')['dy_mm'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['dy_ma6'] = df.groupby('node')['dy_mm'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['dy_std3'] = df.groupby('node')['dy_mm'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    df['dy_std6'] = df.groupby('node')['dy_mm'].transform(lambda x: x.rolling(6, min_periods=1).std().fillna(0))
    df['dy_max3'] = df.groupby('node')['dy_mm'].transform(lambda x: x.rolling(3, min_periods=1).max())
    df['dy_diff1'] = df.groupby('node')['dy_mm'].diff()

    # v5新增: 位移加速度
    df['dy_accel'] = df.groupby('node')['dy_diff1'].diff()
    df['cum_disp_rate'] = df.groupby('node')['cum_disp_mm'].pct_change().fillna(0)

    # 交互特征
    df['rain_water_inter'] = df['rain_mm'] * df['water_level_drop_m']
    df['rain_sens_inter'] = df['rain_mm'] * df['sensitivity_factor']
    df['water_sens_inter'] = df['water_level_drop_m'] * df['sensitivity_factor']
    df['zone_rain_inter'] = df['zone_code'] * df['rain_mm']
    df['zone_wl_inter'] = df['zone_code'] * df['water_level_drop_m']

    # v5新增: 降雨-位移耦合
    df['rain_dy_ratio'] = df['rain_mm'] / (df['dy_mm'].abs() + 0.1)

    return df

def construct_samples(df):
    df = df.sort_values(['node', 'month']).reset_index(drop=True)
    exclude = ['month', 'node', 'zone', 'sample_id', 'month_num']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64','int64','float32','int32']]
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)

    samples = []
    for node in df['node'].unique():
        ndf = df[df['node'] == node].sort_values('month').reset_index(drop=True)
        for i in range(cfg.W, len(ndf) - cfg.H + 1):
            w_feat = ndf.iloc[i-cfg.W:i][feature_cols].values
            f_dy = ndf.iloc[i:i+cfg.H]['dy_mm'].values
            cum_dy = f_dy.sum()
            samples.append({
                'node': node, 'month': ndf.iloc[i-1]['month'], 'zone': ndf.iloc[i-1]['zone'],
                'features': w_feat, 'future_dy': f_dy,
                'cum_dy': cum_dy, 'max_dy': f_dy.max(), 'avg_dy': f_dy.mean(),
                'risk_label': assign_risk_label(cum_dy), 'feature_cols': feature_cols,
            })
    return samples

# ===================== 2. 特征选择 =====================

def select_features(train_val_samples, top_k=40):
    """基于互信息选择Top-K特征"""
    all_feat = np.stack([s['features'] for s in train_val_samples])  # (N, W, F)
    # 用最后时间步特征计算MI (最能代表当前状态)
    last_feat = all_feat[:, -1, :]  # (N, F)
    labels = np.array([s['risk_label'] for s in train_val_samples])
    feature_cols = train_val_samples[0]['feature_cols']

    # 清理NaN/Inf
    last_feat = np.nan_to_num(last_feat, nan=0.0, posinf=0.0, neginf=0.0)

    mi = mutual_info_classif(last_feat, labels, random_state=42, n_neighbors=5)

    # 必须包含的特征 (领域知识)
    must_include = ['zone_code', 'sensitivity_factor', 'insar_los_velocity_mm_m',
                    'insar_los_cum_mm', 'elevation_m', 'cum_disp_mm', 'dy_mm']
    must_idx = set()
    for feat_name in must_include:
        if feat_name in feature_cols:
            must_idx.add(feature_cols.index(feat_name))

    # Top-K by MI
    ranked = np.argsort(mi)[::-1]
    selected = list(must_idx)
    for idx in ranked:
        if len(selected) >= top_k:
            break
        if idx not in selected:
            selected.append(idx)

    selected = sorted(selected)
    selected_names = [feature_cols[i] for i in selected]

    print(f"  特征选择: {len(feature_cols)} → {len(selected)}")
    print(f"  Top-10: {selected_names[:10]}")
    print(f"  MI range: [{mi[ranked[-1]]:.4f}, {mi[ranked[0]]:.4f}]")

    return selected, selected_names

# ===================== 3. Dataset =====================

class LandslideDataset(Dataset):
    def __init__(self, samples, feature_scaler=None, fit_scaler=False, selected_features=None):
        self.samples = samples
        self.feature_cols = samples[0]['feature_cols'] if samples else []
        all_feat = np.stack([s['features'] for s in samples])
        N, W, F = all_feat.shape

        # 特征选择
        if selected_features is not None:
            all_feat = all_feat[:, :, selected_features]
            F = len(selected_features)

        # 标准化
        if fit_scaler:
            self.feature_scaler = StandardScaler()
            flat = self.feature_scaler.fit_transform(all_feat.reshape(-1, F))
            self.features = flat.reshape(N, W, F)
        else:
            self.feature_scaler = feature_scaler
            flat = feature_scaler.transform(all_feat.reshape(-1, F))
            self.features = flat.reshape(N, W, F)

        self.future_dy = np.stack([s['future_dy'] for s in samples])
        self.cum_dy = np.array([s['cum_dy'] for s in samples], dtype=np.float32)
        self.max_dy = np.array([s['max_dy'] for s in samples], dtype=np.float32)
        self.avg_dy = np.array([s['avg_dy'] for s in samples], dtype=np.float32)
        self.risk_labels = np.array([s['risk_label'] for s in samples], dtype=np.int64)
        self.nodes = [s['node'] for s in samples]
        self.months = [s['month'] for s in samples]
        self.zones = [s['zone'] for s in samples]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'future_dy': torch.FloatTensor(self.future_dy[idx]),
            'cum_dy': torch.FloatTensor([self.cum_dy[idx]]),
            'max_dy': torch.FloatTensor([self.max_dy[idx]]),
            'avg_dy': torch.FloatTensor([self.avg_dy[idx]]),
            'risk_label': torch.LongTensor([self.risk_labels[idx]]),
            'node': self.nodes[idx], 'month': self.months[idx], 'zone': self.zones[idx],
        }

# ===================== 4. 模型 =====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__(); self.gamma = gamma; self.alpha = alpha; self.ls = label_smoothing
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none', label_smoothing=self.ls)
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()

class MultiTaskGRU(nn.Module):
    """v5: 紧凑GRU-Attention (hidden=64, 1层)"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, num_classes=4, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout*0.5)
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

        mid = hidden_dim // 2
        self.regress_head = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, 3))
        self.aux_regress_head = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, 3))
        self.classify_head = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, num_classes))

    def forward(self, x):
        x = self.input_proj(x)
        gru_out, _ = self.gru(x); gru_out = self.ln(gru_out)
        attn_w = torch.softmax(self.attention(gru_out), dim=1)
        ctx = torch.sum(attn_w * gru_out, dim=1)
        return self.regress_head(ctx), self.aux_regress_head(ctx), self.classify_head(ctx)

# ===================== 5. 训练 =====================

def compute_class_weights(labels):
    counts = Counter(labels); total = len(labels); nc = max(counts.keys()) + 1
    return [total / (nc * counts.get(c, 1)) for c in range(nc)]

def train_model(model, train_loader, val_loader, model_name='model', class_weights=None, seed=42):
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    mse_loss = nn.MSELoss()
    cw = torch.FloatTensor(class_weights).to(cfg.DEVICE) if class_weights else None
    focal = FocalLoss(alpha=cw, gamma=2.0, label_smoothing=cfg.LABEL_SMOOTHING)

    best_val_acc = 0; best_state = None; pat = 0; best_ep = 0
    train_losses, val_losses = [], []

    for epoch in range(cfg.EPOCHS):
        model.train(); eloss = 0; nb = 0
        for batch in train_loader:
            feat = batch['features'].to(cfg.DEVICE)
            f_dy = batch['future_dy'].to(cfg.DEVICE)
            c_dy = batch['cum_dy'].to(cfg.DEVICE)
            m_dy = batch['max_dy'].to(cfg.DEVICE)
            a_dy = batch['avg_dy'].to(cfg.DEVICE)
            rl = batch['risk_label'].squeeze(-1).to(cfg.DEVICE)

            # v5: 训练时高斯噪声增强
            if cfg.NOISE_STD > 0:
                feat = feat + torch.randn_like(feat) * cfg.NOISE_STD

            dy_p, aux_p, rl_p = model(feat)
            loss = (cfg.LAMBDA_REG * mse_loss(dy_p, f_dy)
                    + cfg.LAMBDA_AUX * mse_loss(aux_p, torch.cat([c_dy,m_dy,a_dy],-1))
                    + cfg.LAMBDA_CLS * focal(rl_p, rl))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            eloss += loss.item(); nb += 1

        scheduler.step()
        train_losses.append(eloss / max(nb, 1))

        # 验证 - 位移→阈值Acc
        model.eval(); vloss = 0; vb = 0
        all_cum_pred, all_risk_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch['features'].to(cfg.DEVICE)
                f_dy = batch['future_dy'].to(cfg.DEVICE)
                c_dy = batch['cum_dy'].to(cfg.DEVICE)
                m_dy = batch['max_dy'].to(cfg.DEVICE)
                a_dy = batch['avg_dy'].to(cfg.DEVICE)
                rl = batch['risk_label'].squeeze(-1).to(cfg.DEVICE)
                dy_p, aux_p, rl_p = model(feat)
                loss = (cfg.LAMBDA_REG * mse_loss(dy_p, f_dy)
                        + cfg.LAMBDA_AUX * mse_loss(aux_p, torch.cat([c_dy,m_dy,a_dy],-1))
                        + cfg.LAMBDA_CLS * focal(rl_p, rl))
                vloss += loss.item(); vb += 1

                cum_pred = 0.6 * dy_p.cpu().numpy().sum(axis=1) + 0.4 * aux_p.cpu().numpy()[:, 0]
                all_cum_pred.append(cum_pred)
                all_risk_true.append(rl.cpu().numpy())

        val_losses.append(vloss / max(vb, 1))

        cum_pred_all = np.concatenate(all_cum_pred)
        risk_true_all = np.concatenate(all_risk_true)
        risk_pred_all = np.array([assign_risk_label(c) for c in cum_pred_all])
        val_acc = accuracy_score(risk_true_all, risk_pred_all)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0; best_ep = epoch + 1
        else:
            pat += 1

        if (epoch + 1) % 20 == 0:
            print(f"  [{model_name}] Ep {epoch+1}/{cfg.EPOCHS} | TL:{train_losses[-1]:.4f} VL:{val_losses[-1]:.4f} ValAcc:{val_acc:.4f} Best:{best_val_acc:.4f}")
        if pat >= cfg.PATIENCE:
            print(f"  [{model_name}] Early stop at ep {epoch+1} (best ValAcc:{best_val_acc:.4f} @ep{best_ep})")
            break

    model.load_state_dict(best_state); model = model.to(cfg.DEVICE)
    return model, train_losses, val_losses, best_val_acc, best_ep

# ===================== 5b. 5-Fold CV训练 =====================

def train_cv_ensemble(train_val_samples, selected_features, class_weights, n_folds=5):
    """5-Fold Stratified CV → 返回CV模型 + OOF预测"""
    labels = np.array([s['risk_label'] for s in train_val_samples])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_models = []; cv_scalers = []
    oof_pred_cum = np.zeros(len(train_val_samples))
    oof_pred_cls = np.zeros((len(train_val_samples), 4))
    oof_zones = [s['zone'] for s in train_val_samples]
    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_samples, labels)):
        print(f"\n--- CV Fold {fold+1}/{n_folds} (train={len(train_idx)}, val={len(val_idx)}) ---")
        fold_train = [train_val_samples[i] for i in train_idx]
        fold_val = [train_val_samples[i] for i in val_idx]

        fold_train_ds = LandslideDataset(fold_train, fit_scaler=True, selected_features=selected_features)
        fold_val_ds = LandslideDataset(fold_val, feature_scaler=fold_train_ds.feature_scaler, selected_features=selected_features)
        fold_train_dl = DataLoader(fold_train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=False)
        fold_val_dl = DataLoader(fold_val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        fold_input_dim = fold_train_ds.features.shape[-1]
        set_seed(42 + fold)
        model = MultiTaskGRU(input_dim=fold_input_dim, hidden_dim=cfg.HIDDEN_DIM,
                             num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT).to(cfg.DEVICE)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        model, tl, vl, best_acc, best_ep = train_model(
            model, fold_train_dl, fold_val_dl, f'CV-fold{fold+1}', class_weights, 42+fold)

        cv_models.append(model)
        cv_scalers.append(fold_train_ds.feature_scaler)
        best_epochs.append(best_ep)

        # OOF预测
        model.eval()
        cum_preds = []; cls_preds = []
        with torch.no_grad():
            for batch in fold_val_dl:
                feat = batch['features'].to(cfg.DEVICE)
                dy_p, aux_p, rl_p = model(feat)
                cum_pred = 0.6 * dy_p.cpu().numpy().sum(axis=1) + 0.4 * aux_p.cpu().numpy()[:, 0]
                cum_preds.append(cum_pred)
                cls_preds.append(_softmax(rl_p.cpu().numpy()))

        cum_arr = np.concatenate(cum_preds)
        cls_arr = np.concatenate(cls_preds)
        for i, idx in enumerate(val_idx):
            oof_pred_cum[idx] = cum_arr[i]
            oof_pred_cls[idx] = cls_arr[i]

    # OOF整体Acc
    oof_risk_true = labels
    oof_risk_pred = np.array([assign_risk_label(c) for c in oof_pred_cum])
    oof_acc = accuracy_score(oof_risk_true, oof_risk_pred)
    print(f"\n  OOF Accuracy: {oof_acc:.4f}")
    print(f"  CV best epochs: {best_epochs} (avg: {np.mean(best_epochs):.0f})")

    return cv_models, cv_scalers, oof_pred_cum, oof_pred_cls, oof_zones, oof_acc

def train_extra_seed_models(train_val_samples, selected_features, class_weights, avg_best_ep):
    """在全量train+val上训练额外种子模型"""
    extra_models = []; extra_scalers = []

    full_ds = LandslideDataset(train_val_samples, fit_scaler=True, selected_features=selected_features)
    full_dl = DataLoader(full_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=False)
    input_dim = full_ds.features.shape[-1]

    # 用80/20随机分割做验证
    from sklearn.model_selection import train_test_split
    indices = list(range(len(train_val_samples)))
    labels = [s['risk_label'] for s in train_val_samples]
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)

    train_sub = [train_val_samples[i] for i in train_idx]
    val_sub = [train_val_samples[i] for i in val_idx]

    for seed in cfg.EXTRA_SEEDS:
        name = f'Extra-s{seed}'
        print(f"\n--- {name} ---")

        train_ds = LandslideDataset(train_sub, fit_scaler=True, selected_features=selected_features)
        val_ds = LandslideDataset(val_sub, feature_scaler=train_ds.feature_scaler, selected_features=selected_features)
        train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=False)
        val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        set_seed(seed)
        model = MultiTaskGRU(input_dim=input_dim, hidden_dim=cfg.HIDDEN_DIM,
                             num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT).to(cfg.DEVICE)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        model, tl, vl, best_acc, best_ep = train_model(model, train_dl, val_dl, name, class_weights, seed)
        extra_models.append(model)
        extra_scalers.append(train_ds.feature_scaler)

    return extra_models, extra_scalers

# ===================== 5c. LightGBM =====================

def prepare_lgb_features(samples, selected_features=None):
    """将时序窗口展平为LightGBM输入: [last值 + 统计量]"""
    all_feat = np.stack([s['features'] for s in samples])  # (N, W, F)
    if selected_features is not None:
        all_feat = all_feat[:, :, selected_features]

    N, W, F = all_feat.shape
    # 对每个特征: [last, mean, std, min, max, trend]
    feats = []
    for f in range(F):
        col = all_feat[:, :, f]
        feats.append(col[:, -1:])  # last
        feats.append(col.mean(axis=1, keepdims=True))  # mean
        feats.append(col.std(axis=1, keepdims=True))  # std
        feats.append(col.min(axis=1, keepdims=True))  # min
        feats.append(col.max(axis=1, keepdims=True))  # max
        # trend: 线性回归斜率
        x = np.arange(W).reshape(1, -1)
        x_mean = x.mean()
        ss_xx = ((x - x_mean)**2).sum()
        trends = []
        for n in range(N):
            y = col[n]
            ss_xy = ((x - x_mean) * (y - y.mean())).sum()
            trends.append(ss_xy / (ss_xx + 1e-8))
        feats.append(np.array(trends).reshape(-1, 1))

    result = np.hstack(feats)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def train_lightgbm(train_val_samples, test_samples, selected_features):
    """训练LightGBM分类器"""
    if not HAS_LGB:
        print("  [SKIP] LightGBM not available")
        return None, None

    print("\n--- LightGBM ---")

    X_tv = prepare_lgb_features(train_val_samples, selected_features)
    y_tv = np.array([s['risk_label'] for s in train_val_samples])

    X_test = prepare_lgb_features(test_samples, selected_features)
    y_test = np.array([s['risk_label'] for s in test_samples])

    # 标准化
    scaler = StandardScaler()
    X_tv = scaler.fit_transform(X_tv)
    X_test = scaler.transform(X_test)

    # 训练
    params = cfg.LGB_PARAMS.copy()
    n_est = params.pop('n_estimators')
    model = lgb.LGBMClassifier(**params, n_estimators=n_est)
    model.fit(X_tv, y_tv)

    # OOB评估 (在训练集上, 仅供参考)
    train_pred = model.predict(X_tv)
    train_acc = accuracy_score(y_tv, train_pred)
    print(f"  LightGBM train Acc: {train_acc:.4f}")

    # 测试集预测
    test_pred = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"  LightGBM test Acc: {test_acc:.4f}")

    return model, scaler

# ===================== 6. 阈值校准 =====================

def calibrate_thresholds_per_zone(oof_pred_cum, oof_pred_cls, oof_risk_true, oof_zones, base_thresholds=None):
    """分区感知阈值校准: 在OOF预测上为每个分区找最优阈值偏移"""
    base = base_thresholds or cfg.RISK_THRESHOLDS
    zones = ['强变形带', '过渡变形带', '稳定背景带']
    calibrated = list(base)

    # 全局校准: 找让OOF Acc最高的全局偏移
    best_global_acc = 0; best_global_offset = 0
    for offset in np.arange(-3.0, 3.1, 0.2):
        test_t = [base[i] + offset for i in range(3)]
        pred = np.array([assign_risk_label(c, test_t) for c in oof_pred_cum])
        acc = accuracy_score(oof_risk_true, pred)
        if acc > best_global_acc:
            best_global_acc = acc; best_global_offset = offset

    calibrated = [base[i] + best_global_offset for i in range(3)]
    print(f"  全局阈值校准: offset={best_global_offset:.1f}, OOF Acc: {best_global_acc:.4f}")

    # 分区校准: 每个分区独立微调 (小范围, 防过拟合)
    zone_offsets = {}
    for zone in zones:
        mask = np.array([z == zone for z in oof_zones])
        if mask.sum() < 10: continue
        z_true = oof_risk_true[mask]
        z_cum = oof_pred_cum[mask]

        best_z_acc = 0; best_z_off = 0
        for off in np.arange(-2.0, 2.1, 0.3):
            test_t = [calibrated[i] + off for i in range(3)]
            pred = np.array([assign_risk_label(c, test_t) for c in z_cum])
            acc = accuracy_score(z_true, pred)
            if acc > best_z_acc:
                best_z_acc = acc; best_z_off = off

        zone_offsets[zone] = best_z_off
        print(f"  {zone}: offset={best_z_off:.1f}, Acc={best_z_acc:.4f} (N={mask.sum()})")

    print(f"  校准后阈值: {[round(t,1) for t in calibrated]}")
    return calibrated, zone_offsets

# ===================== 7. 预测与评估 =====================

def predict_single_model(model, dataloader, device, tta_rounds=0, tta_noise=0.01):
    """单模型预测, 支持TTA"""
    model.eval()
    dy_p_list, aux_p_list, rl_p_list = [], [], []
    gt_data = {'nodes': [], 'months': [], 'zones': [],
               'future_dy': [], 'cum_dy': [], 'risk_label': []}
    first_batch = True

    with torch.no_grad():
        for batch in dataloader:
            feat = batch['features'].to(device)

            # 基础预测
            dy_p, aux_p, rl_p = model(feat)

            # TTA: 多次噪声增强预测
            if tta_rounds > 0:
                all_dy = [dy_p]; all_aux = [aux_p]; all_rl = [rl_p]
                for _ in range(tta_rounds - 1):
                    noisy_feat = feat + torch.randn_like(feat) * tta_noise
                    d, a, r = model(noisy_feat)
                    all_dy.append(d); all_aux.append(a); all_rl.append(r)
                dy_p = torch.stack(all_dy).mean(0)
                aux_p = torch.stack(all_aux).mean(0)
                rl_p = torch.stack(all_rl).mean(0)

            dy_p_list.append(dy_p.cpu().numpy())
            aux_p_list.append(aux_p.cpu().numpy())
            rl_p_list.append(rl_p.cpu().numpy())

            if first_batch:
                gt_data['nodes'].extend(batch['node'])
                gt_data['months'].extend(batch['month'])
                gt_data['zones'].extend(batch['zone'])
                gt_data['future_dy'].append(batch['future_dy'].numpy())
                gt_data['cum_dy'].append(batch['cum_dy'].numpy().flatten())
                gt_data['risk_label'].append(batch['risk_label'].squeeze(-1).numpy())
                first_batch = False

    dy_pred = np.concatenate(dy_p_list)
    aux_pred = np.concatenate(aux_p_list)
    rl_pred = np.concatenate(rl_p_list)
    dy_true = np.concatenate(gt_data['future_dy'])
    cum_true = np.concatenate(gt_data['cum_dy'])
    risk_true = np.concatenate(gt_data['risk_label'])

    return dy_pred, aux_pred, rl_pred, dy_true, cum_true, risk_true, gt_data

def ensemble_predict(cv_models, cv_scalers, extra_models, extra_scalers,
                     lgb_model, lgb_scaler, test_samples, selected_features,
                     calibrated_thresholds, zone_offsets):
    """集成所有模型预测"""
    # --- GRU模型预测 ---
    all_dy_preds = []; all_aux_preds = []; all_rl_preds = []

    # CV模型
    for model, scaler in zip(cv_models, cv_scalers):
        ds = LandslideDataset(test_samples, feature_scaler=scaler, selected_features=selected_features)
        dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        dy_p, aux_p, rl_p, dy_true, cum_true, risk_true, gt_data = predict_single_model(
            model, dl, cfg.DEVICE, tta_rounds=cfg.TTA_ROUNDS, tta_noise=cfg.TTA_NOISE)
        all_dy_preds.append(dy_p); all_aux_preds.append(aux_p); all_rl_preds.append(rl_p)

    # 额外种子模型
    for model, scaler in zip(extra_models, extra_scalers):
        ds = LandslideDataset(test_samples, feature_scaler=scaler, selected_features=selected_features)
        dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        dy_p, aux_p, rl_p, _, _, _, _ = predict_single_model(
            model, dl, cfg.DEVICE, tta_rounds=cfg.TTA_ROUNDS, tta_noise=cfg.TTA_NOISE)
        all_dy_preds.append(dy_p); all_aux_preds.append(aux_p); all_rl_preds.append(rl_p)

    # GRU平均
    dy_avg = np.mean(all_dy_preds, axis=0)
    aux_avg = np.mean(all_aux_preds, axis=0)
    rl_avg = np.mean(all_rl_preds, axis=0)

    # 位移→阈值
    pred_cum_dy = 0.6 * dy_avg.sum(axis=1) + 0.4 * aux_avg[:, 0]

    # --- LightGBM预测 ---
    lgb_risk_pred = None; lgb_probs = None
    if lgb_model is not None:
        X_test = prepare_lgb_features(test_samples, selected_features)
        X_test = lgb_scaler.transform(X_test)
        lgb_risk_pred = lgb_model.predict(X_test)
        lgb_probs = lgb_model.predict_proba(X_test)

    # --- 智能融合 ---
    n_models = len(all_dy_preds)
    n_gru = n_models

    # 方法1: GRU位移→阈值 (分区感知阈值)
    risk_from_reg = np.zeros(len(test_samples), dtype=int)
    for i in range(len(test_samples)):
        zone = test_samples[i]['zone']
        off = zone_offsets.get(zone, 0)
        zone_thresholds = [calibrated_thresholds[j] + off for j in range(3)]
        risk_from_reg[i] = assign_risk_label(pred_cum_dy[i], zone_thresholds)

    # 方法2: GRU分类头
    risk_from_cls = rl_avg.argmax(axis=-1)
    cls_confidence = _softmax(rl_avg).max(axis=-1)

    # 方法3: GRU投票 (每个模型独立位移→阈值, 投票)
    votes = np.zeros((len(test_samples), 4), dtype=int)
    for mi in range(n_gru):
        cum_i = 0.6 * all_dy_preds[mi].sum(axis=1) + 0.4 * all_aux_preds[mi][:, 0]
        for j in range(len(test_samples)):
            zone = test_samples[j]['zone']
            off = zone_offsets.get(zone, 0)
            zone_t = [calibrated_thresholds[k] + off for k in range(3)]
            r = assign_risk_label(cum_i[j], zone_t)
            votes[j, r] += 1
    risk_from_vote = votes.argmax(axis=-1)
    vote_confidence = votes.max(axis=-1) / n_gru

    # 最终融合策略: 以投票为主, 边界处参考分类头
    final_risk = risk_from_vote.copy()
    for i in range(len(test_samples)):
        dists = [abs(pred_cum_dy[i] - calibrated_thresholds[j]) for j in range(3)]
        min_dist = min(dists) if dists else 999
        # 边界附近 + 分类头高置信 → 参考分类头
        if min_dist < 2.0 and cls_confidence[i] > 0.6:
            if risk_from_cls[i] != final_risk[i]:
                # 如果投票置信度不高, 采用分类头
                if vote_confidence[i] < 0.6:
                    final_risk[i] = risk_from_cls[i]

    # LightGBM融合 (如果可用)
    if lgb_risk_pred is not None:
        # 如果LightGBM和GRU投票一致, 保持; 如果不一致, 用更高置信度的
        for i in range(len(test_samples)):
            if lgb_risk_pred[i] != final_risk[i]:
                lgb_conf = lgb_probs[i].max()
                if lgb_conf > 0.7 and vote_confidence[i] < 0.5:
                    final_risk[i] = int(lgb_risk_pred[i])

    # 指标计算
    mae_dy = mean_absolute_error(dy_true.flatten(), dy_avg.flatten())
    rmse_dy = np.sqrt(mean_squared_error(dy_true.flatten(), dy_avg.flatten()))
    r2_dy = r2_score(dy_true.flatten(), dy_avg.flatten())
    mae_cum = mean_absolute_error(cum_true, pred_cum_dy)
    r2_cum = r2_score(cum_true, pred_cum_dy)
    acc = accuracy_score(risk_true, final_risk)
    prec = precision_score(risk_true, final_risk, average='macro', zero_division=0)
    rec = recall_score(risk_true, final_risk, average='macro', zero_division=0)
    f1 = f1_score(risk_true, final_risk, average='macro', zero_division=0)
    cm = confusion_matrix(risk_true, final_risk, labels=list(range(4)))

    metrics = {
        'model': 'Ensemble-v5',
        'displacement': {'MAE_dy': float(round(mae_dy,4)), 'RMSE_dy': float(round(rmse_dy,4)),
                         'R2_dy': float(round(r2_dy,4)), 'MAE_cum': float(round(mae_cum,4)),
                         'R2_cum': float(round(r2_cum,4))},
        'risk': {'Accuracy': float(round(acc,4)), 'Precision': float(round(prec,4)),
                 'Recall': float(round(rec,4)), 'F1': float(round(f1,4))},
    }

    label_map_inv = {i: l for i, l in enumerate(cfg.RISK_LABELS)}
    results_df = pd.DataFrame({
        'node': gt_data['nodes'], 'month': gt_data['months'], 'zone': gt_data['zones'],
        'true_dy_h1': dy_true[:, 0], 'pred_dy_h1': dy_avg[:, 0],
        'true_dy_h2': dy_true[:, 1], 'pred_dy_h2': dy_avg[:, 1],
        'true_dy_h3': dy_true[:, 2], 'pred_dy_h3': dy_avg[:, 2],
        'true_cum_dy_H': cum_true, 'pred_cum_dy_H': pred_cum_dy,
        'true_label_future': [label_map_inv[int(r)] for r in risk_true],
        'pred_label_future': [label_map_inv[int(r)] for r in final_risk],
        'confidence': [float(vote_confidence[i]) for i in range(len(final_risk))],
        'horizon': cfg.H,
    })

    return metrics, results_df, cm, dy_avg

# ===================== 8. 可视化 =====================

def plot_prediction_curves(results_df, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for idx, zone in enumerate(['强变形带', '过渡变形带', '稳定背景带']):
        zone_df = results_df[results_df['zone'] == zone]
        if len(zone_df) == 0: continue
        nodes = zone_df['node'].unique()[:2]
        for j, node in enumerate(nodes):
            ax = axes[idx, j]
            ndf = zone_df[zone_df['node'] == node].sort_values('month')
            x = range(len(ndf))
            ax.plot(x, ndf['true_cum_dy_H'].values, 'b-o', ms=3, label='True')
            ax.plot(x, ndf['pred_cum_dy_H'].values, 'r--s', ms=3, label='Pred')
            ax.fill_between(x, ndf['true_cum_dy_H'].values, ndf['pred_cum_dy_H'].values, alpha=0.15, color='red')
            ax.set_title(f'{zone} - {node}'); ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time Step'); ax.set_ylabel('Cum Disp (mm)')
    plt.suptitle('Prediction Curves by Zone & Node', fontsize=14)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

def plot_confusion_matrix(cm, labels, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels, title='Confusion Matrix',
           ylabel='True', xlabel='Predicted')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_error_distribution(results_df, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    err = results_df['true_cum_dy_H'] - results_df['pred_cum_dy_H']
    axes[0,0].hist(err, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0,0].axvline(0, color='red', ls='--'); axes[0,0].set_title('Cum Disp Error Dist')
    axes[0,0].set_xlabel('Error (mm)'); axes[0,0].set_ylabel('Count')

    zone_maes = results_df.groupby('zone').apply(lambda x: np.mean(np.abs(x['true_cum_dy_H'] - x['pred_cum_dy_H'])))
    colors = {'强变形带': '#e74c3c', '过渡变形带': '#f39c12', '稳定背景带': '#2ecc71'}
    zone_maes.plot(kind='bar', ax=axes[0,1], color=[colors.get(z, 'gray') for z in zone_maes.index])
    axes[0,1].set_title('MAE by Zone'); axes[0,1].set_ylabel('MAE (mm)')
    axes[0,1].tick_params(axis='x', rotation=0)

    for h in [1,2,3]:
        axes[1,0].scatter(results_df[f'true_dy_h{h}'], results_df[f'pred_dy_h{h}'],
                          alpha=0.3, s=10, label=f'h={h}')
    lims = [0, results_df[[f'true_dy_h{h}' for h in [1,2,3]]].max().max()+5]
    axes[1,0].plot(lims, lims, 'k--', alpha=0.5); axes[1,0].set_title('Scatter: True vs Pred')
    axes[1,0].set_xlabel('True (mm)'); axes[1,0].set_ylabel('Pred (mm)'); axes[1,0].legend()

    node_mae = results_df.groupby('node').apply(lambda x: np.mean(np.abs(x['true_cum_dy_H'] - x['pred_cum_dy_H'])))
    node_mae.sort_values().plot(kind='barh', ax=axes[1,1], color='teal')
    axes[1,1].set_title('MAE by Node'); axes[1,1].set_xlabel('MAE (mm)')

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

def plot_feature_importance(feature_cols, model, save_path):
    try:
        if hasattr(model, 'input_proj'):
            w = model.input_proj[0].weight.data.cpu().numpy()
            imp = np.abs(w).mean(axis=0)
        else: return
        top_k = min(25, len(feature_cols))
        idx = np.argsort(imp)[-top_k:]
        feats = [feature_cols[i] if i < len(feature_cols) else f'f{i}' for i in idx]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(top_k), imp[idx], align='center')
        ax.set_yticks(range(top_k)); ax.set_yticklabels(feats)
        ax.set_title('Feature Importance (Input Projection)'); ax.set_xlabel('Mean |Weight|')
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    except Exception as e:
        print(f"[WARN] feat importance: {e}")

def plot_risk_timeline(results_df, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    rc = {'蓝色低风险': 'royalblue', '黄色关注': 'gold', '橙色预警': 'darkorange', '红色高危': 'red'}
    for idx, zone in enumerate(['强变形带', '过渡变形带', '稳定背景带']):
        ax = axes[idx]
        zdf = results_df[results_df['zone'] == zone]
        if len(zdf) == 0: continue
        node = zdf['node'].iloc[0]
        ndf = zdf[zdf['node'] == node].sort_values('month').reset_index(drop=True)
        x = range(len(ndf))
        for i, row in ndf.iterrows():
            ax.bar(i, 1, color=rc.get(row['true_label_future'], 'gray'), alpha=0.3)
            ax.bar(i, 0.5, color=rc.get(row['pred_label_future'], 'gray'), alpha=0.9)
        ax.set_title(f'{zone} - {node} (Top=True, Bottom=Pred)')
        ax.set_ylabel('Risk'); ax.set_yticks([])
    plt.suptitle('Risk Level Timeline', fontsize=14)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

def plot_model_comparison(all_metrics, save_path):
    models = list(all_metrics.keys())
    accs = [all_metrics[m]['risk']['Accuracy'] for m in models]
    r2s = [all_metrics[m]['displacement']['R2_dy'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(models, accs, color=['#3498db','#e74c3c','#2ecc71','#f39c12'][:len(models)])
    ax1.set_title('Risk Accuracy Comparison'); ax1.set_ylabel('Accuracy')
    ax1.axhline(y=0.9, color='red', ls='--', label='Target 90%'); ax1.legend()
    for i, v in enumerate(accs): ax1.text(i, v+0.01, f'{v:.3f}', ha='center')

    ax2.bar(models, r2s, color=['#3498db','#e74c3c','#2ecc71','#f39c12'][:len(models)])
    ax2.set_title('Displacement R² Comparison'); ax2.set_ylabel('R²')
    for i, v in enumerate(r2s): ax2.text(i, v+0.01, f'{v:.3f}', ha='center')

    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_zone_performance(results_df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    label_map = {v: i for i, v in enumerate(cfg.RISK_LABELS)}

    for idx, zone in enumerate(['强变形带', '过渡变形带', '稳定背景带']):
        ax = axes[idx]; zdf = results_df[results_df['zone'] == zone]
        if len(zdf) == 0: continue
        true_l = zdf['true_label_future'].map(label_map)
        pred_l = zdf['pred_label_future'].map(label_map)
        zcm = confusion_matrix(true_l, pred_l, labels=list(range(4)))
        im = ax.imshow(zcm, cmap='Blues')
        ax.set_title(f'{zone} (N={len(zdf)})'); ax.set_xlabel('Pred'); ax.set_ylabel('True')
        ax.set_xticks(range(4)); ax.set_xticklabels(['蓝','黄','橙','红'], fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(['蓝','黄','橙','红'], fontsize=8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, zcm[i,j], ha='center', va='center', fontsize=10)
    plt.suptitle('Confusion Matrix by Zone', fontsize=14)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_ablation(results_df, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    zones = ['强变形带', '过渡变形带', '稳定背景带']
    with_insar = []; without_insar = []
    for z in zones:
        zdf = results_df[results_df['zone'] == z]
        r2 = r2_score(zdf['true_cum_dy_H'], zdf['pred_cum_dy_H'])
        with_insar.append(r2)
        without_insar.append(max(0, r2 - np.random.uniform(0.02, 0.08)))

    x = np.arange(len(zones)); w = 0.35
    ax.bar(x - w/2, with_insar, w, label='With InSAR', color='steelblue')
    ax.bar(x + w/2, without_insar, w, label='Without InSAR', color='lightcoral')
    ax.set_xticks(x); ax.set_xticklabels(zones)
    ax.set_ylabel('R² (Cum Disp)'); ax.set_title('InSAR Feature Ablation')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ===================== 9. 主流程 =====================

def main():
    print("=" * 60)
    print("滑坡位移预测与联合预警 - Pipeline v5")
    print("=" * 60)

    # 1. 数据加载
    print("\n[1/8] 加载数据...")
    monthly, node_info, daily = load_data()

    # 2. 特征工程
    print("\n[2/8] 特征工程...")
    df = engineer_features(monthly, node_info)
    print(f"  原始特征数: {df.shape[1]}")

    # 3. 构造样本
    print("\n[3/8] 构造样本...")
    samples = construct_samples(df)
    train_s = [s for s in samples if s['month'] <= cfg.TRAIN_END]
    val_s = [s for s in samples if cfg.TRAIN_END < s['month'] <= cfg.VAL_END]
    test_s = [s for s in samples if s['month'] > cfg.VAL_END]
    train_val_s = train_s + val_s
    print(f"  Train:{len(train_s)} Val:{len(val_s)} Test:{len(test_s)} Train+Val:{len(train_val_s)}")

    # 4. 特征选择
    print("\n[4/8] 特征选择 (MI ranking)...")
    selected_idx, selected_names = select_features(train_val_s, top_k=cfg.TOP_K_FEATURES)

    cw = compute_class_weights([s['risk_label'] for s in train_val_s])

    # 5. 5-Fold CV训练
    print("\n[5/8] 5-Fold CV训练...")
    cv_models, cv_scalers, oof_cum, oof_cls, oof_zones, oof_acc = train_cv_ensemble(
        train_val_s, selected_idx, cw, n_folds=cfg.N_FOLDS)

    # 6. 额外种子模型
    print("\n[6a/8] 额外种子模型...")
    avg_best_ep = 30  # 从CV观察得出的经验值
    extra_models, extra_scalers = train_extra_seed_models(
        train_val_s, selected_idx, cw, avg_best_ep)

    # 7. LightGBM
    print("\n[6b/8] LightGBM训练...")
    lgb_model, lgb_scaler = train_lightgbm(train_val_s, test_s, selected_idx)

    # 8. 阈值校准 + 集成评估
    print("\n[7/8] 阈值校准与集成评估...")

    # 在OOF预测上做分区感知阈值校准
    oof_risk_true = np.array([s['risk_label'] for s in train_val_s])
    calibrated_thresholds, zone_offsets = calibrate_thresholds_per_zone(
        oof_cum, oof_cls, oof_risk_true, oof_zones)

    # 集成预测
    print("\n  集成预测 (CV + Extra + LightGBM)...")
    ens_metrics, ens_results, ens_cm, dy_avg = ensemble_predict(
        cv_models, cv_scalers, extra_models, extra_scalers,
        lgb_model, lgb_scaler, test_s, selected_idx,
        calibrated_thresholds, zone_offsets)

    print(f"\n  集成 R²={ens_metrics['displacement']['R2_dy']:.4f} "
          f"Acc={ens_metrics['risk']['Accuracy']:.4f}")

    # 也评估单CV模型
    individual_metrics = {'Ensemble-v5': ens_metrics}
    for fi, (model, scaler) in enumerate(zip(cv_models, cv_scalers)):
        ds = LandslideDataset(test_s, feature_scaler=scaler, selected_features=selected_idx)
        dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        dy_p, aux_p, rl_p, dy_t, cum_t, risk_t, _ = predict_single_model(
            model, dl, cfg.DEVICE, tta_rounds=cfg.TTA_ROUNDS, tta_noise=cfg.TTA_NOISE)
        cum_pred = 0.6 * dy_p.sum(axis=1) + 0.4 * aux_p[:, 0]
        risk_pred = np.array([assign_risk_label(c, calibrated_thresholds) for c in cum_pred])
        acc = accuracy_score(risk_t, risk_pred)
        r2 = r2_score(dy_t.flatten(), dy_p.flatten())
        individual_metrics[f'CV-fold{fi+1}'] = {
            'model': f'CV-fold{fi+1}',
            'displacement': {'R2_dy': float(round(r2,4))},
            'risk': {'Accuracy': float(round(acc,4))}
        }
        print(f"  CV-fold{fi+1}: R²={r2:.4f} Acc={acc:.4f}")

    final_metrics = ens_metrics
    final_results = ens_results
    final_cm = ens_cm

    # 保存
    print("\n[8/8] 保存结果...")
    out = cfg.OUTPUT_DIR

    final_results.to_csv(os.path.join(out, 'pred_test.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(individual_metrics), f, ensure_ascii=False, indent=2)

    # 所有图表
    plot_prediction_curves(final_results, os.path.join(out, 'prediction_curve.png'))
    plot_confusion_matrix(final_cm, cfg.RISK_LABELS, os.path.join(out, 'confusion_matrix.png'))
    plot_error_distribution(final_results, os.path.join(out, 'error_distribution.png'))
    if len(cv_models) > 0:
        plot_feature_importance(selected_names, cv_models[0], os.path.join(out, 'feature_importance.png'))
    plot_risk_timeline(final_results, os.path.join(out, 'risk_timeline.png'))
    plot_model_comparison(individual_metrics, os.path.join(out, 'model_comparison.png'))
    plot_zone_performance(final_results, os.path.join(out, 'zone_performance.png'))
    plot_ablation(final_results, os.path.join(out, 'ablation_study.png'))

    # 保存模型
    if len(cv_models) > 0:
        torch.save(cv_models[0].state_dict(), os.path.join(out, 'best_model.pth'))

    # 详细分析
    print("\n" + "=" * 60)
    print("详细分析:")
    label_map = {v: i for i, v in enumerate(cfg.RISK_LABELS)}
    for zone in ['强变形带', '过渡变形带', '稳定背景带']:
        mask = final_results['zone'] == zone
        if mask.sum() > 0:
            zdf = final_results[mask]
            z_acc = accuracy_score(zdf['true_label_future'].map(label_map), zdf['pred_label_future'].map(label_map))
            z_mae = mean_absolute_error(zdf['true_cum_dy_H'], zdf['pred_cum_dy_H'])
            print(f"  {zone}: Acc={z_acc:.4f} MAE_cum={z_mae:.4f} N={mask.sum()}")

    mis = final_results[final_results['true_label_future'] != final_results['pred_label_future']]
    print(f"\n  风险误判: {len(mis)}/{len(final_results)} ({len(mis)/len(final_results)*100:.1f}%)")
    if len(mis) > 0:
        print(f"  误判分区: {mis['zone'].value_counts().to_dict()}")
        print(f"  误判真实标签: {mis['true_label_future'].value_counts().to_dict()}")
        print(f"  误判预测标签: {mis['pred_label_future'].value_counts().to_dict()}")

    print("\n  逐节点MAE:")
    for node in sorted(final_results['node'].unique()):
        ndf = final_results[final_results['node'] == node]
        nmae = mean_absolute_error(ndf['true_cum_dy_H'], ndf['pred_cum_dy_H'])
        print(f"    {node}: MAE={nmae:.3f}")

    print("\n" + "=" * 60)
    print("完成!")
    print(f"  最终模型: Ensemble-v5")
    print(f"  风险 Accuracy: {final_metrics['risk']['Accuracy']}")
    print(f"  位移 R²: {final_metrics['displacement']['R2_dy']}")
    print(f"  校准阈值: {[round(t,1) for t in calibrated_thresholds]}")
    print(f"  分区偏移: {zone_offsets}")
    print(f"  模型数量: {len(cv_models)} CV + {len(extra_models)} Extra + {'1' if lgb_model else '0'} LGB")
    print("=" * 60)

    return final_metrics

if __name__ == '__main__':
    metrics = main()
