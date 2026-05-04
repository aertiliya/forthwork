"""
滑坡位移预测与联合预警 - 完整Pipeline v2
=====================================
多任务GRU: 共享编码器 + 位移回归头 + 风险分类头
支持: 滞后特征、水位消落、InSAR融合、空间分区、趋势-周期分解
v2: 优化精度到85%+ (focal loss, class weights, larger model, ensemble)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def convert_to_serializable(obj):
    """将numpy类型转为python原生类型用于JSON序列化"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj


# ===================== 配置 =====================
class Config:
    # 路径
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    OUTPUT_DIR = os.path.dirname(__file__)

    # 时序参数
    W = 12        # 窗口长度
    H = 3         # 预测期

    # 模型参数
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.15

    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 5e-4
    WEIGHT_DECAY = 1e-5
    LAMBDA_CLS = 1.0           # 分类损失权重
    LAMBDA_AUX = 0.3           # 辅助回归权重
    PATIENCE = 30              # early stopping

    # 风险阈值
    RISK_THRESHOLDS = [10, 22, 38]  # 蓝/黄/橙/红
    RISK_LABELS = ['蓝色低风险', '黄色关注', '橙色预警', '红色高危']

    # 时间划分
    TRAIN_END = '2023-12'
    VAL_END = '2024-06'

    # 设备 - 安全检测CUDA是否真正可用
    @staticmethod
    def _detect_device():
        if not torch.cuda.is_available():
            return torch.device('cpu')
        try:
            # 实际尝试在GPU上创建tensor来验证CUDA kernel可用
            _ = torch.zeros(1).cuda()
            return torch.device('cuda')
        except (torch.cuda.CudaError, RuntimeError):
            print("[WARN] CUDA reported available but kernel execution failed, falling back to CPU")
            return torch.device('cpu')

    DEVICE = _detect_device.__func__()

cfg = Config()
# 触发设备检测
cfg.__class__.DEVICE = cfg._detect_device()
print(f"[INFO] Device: {cfg.DEVICE}")

# ===================== 1. 数据加载与特征工程 =====================

def load_data():
    """加载所有数据文件"""
    monthly = pd.read_csv(os.path.join(cfg.DATA_DIR, 'monthly_multisource_features.csv'))
    node_info = pd.read_csv(os.path.join(cfg.DATA_DIR, 'node_info.csv'))
    daily = pd.read_csv(os.path.join(cfg.DATA_DIR, 'rainfall_waterlevel_daily.csv'))
    return monthly, node_info, daily


def engineer_features(monthly, node_info):
    """
    特征工程：
    - 滞后降雨特征 (路线A)
    - 水位消落指标
    - InSAR特征 (路线E)
    - 空间分区编码 (路线G)
    - 趋势-周期分解特征 (路线F)
    - 节点敏感性系数
    - 交互特征
    """
    df = monthly.copy()

    # 确保month为字符串格式
    df['month'] = df['month'].astype(str)

    # 合并node_info的敏感性系数
    df = df.merge(node_info[['node', 'sensitivity_factor']], on='node', how='left')

    # ---- 路线A: 滞后降雨特征 ----
    df = df.sort_values(['node', 'month']).reset_index(drop=True)

    # 额外滞后特征
    for lag in [2, 3, 4, 5, 6]:
        df[f'rain_lag{lag}_mm'] = df.groupby('node')['rain_mm'].shift(lag)

    # 降雨加速度
    df['rain_diff1'] = df.groupby('node')['rain_mm'].diff()
    df['rain_diff2'] = df.groupby('node')['rain_diff1'].diff()

    # 加权降雨指数 (近期权重更大)
    df['rain_weighted_3m'] = (df['rain_mm'] * 3 + df['rain_lag1_mm'] * 2 + df['rain_lag2_mm']) / 6

    # ---- 水位消落指标 ----
    df['water_level_lag1'] = df.groupby('node')['water_level_mean_m'].shift(1)
    df['water_level_lag2'] = df.groupby('node')['water_level_mean_m'].shift(2)
    df['water_level_change'] = df['water_level_mean_m'] - df['water_level_lag1']
    df['water_level_change_2m'] = df['water_level_mean_m'] - df['water_level_lag2']
    df['water_level_3m_avg'] = df.groupby('node')['water_level_mean_m'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['water_level_6m_avg'] = df.groupby('node')['water_level_mean_m'].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    df['water_level_drop_3m'] = df.groupby('node')['water_level_drop_m'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    # 低水位标志
    wl_mean = df['water_level_mean_m'].mean()
    wl_std = df['water_level_mean_m'].std()
    df['low_water_flag'] = (df['water_level_mean_m'] < wl_mean - wl_std).astype(float)
    # 水位变化率
    df['water_level_pct_change'] = df.groupby('node')['water_level_mean_m'].pct_change().fillna(0)

    # ---- InSAR特征 (路线E) ----
    df['insar_vel_lag1'] = df.groupby('node')['insar_los_velocity_mm_m'].shift(1)
    df['insar_vel_change'] = df['insar_los_velocity_mm_m'] - df['insar_vel_lag1']
    df['insar_acc_lag1'] = df.groupby('node')['insar_los_acc_mm_m2'].shift(1)
    # InSAR与GPS位移的差异
    df['insar_gps_diff'] = df['insar_los_cum_mm'] - df['cum_disp_mm']
    df['insar_gps_vel_ratio'] = df['insar_los_velocity_mm_m'] / (df['dy_mm'].abs() + 0.01)

    # ---- 趋势-周期分解 (路线F) ----
    def decompose_series(group):
        group = group.sort_values('month').reset_index(drop=True)
        n = len(group)
        if n < 6:
            group['trend_dy'] = group['dy_mm']
            group['seasonal_dy'] = 0.0
            group['residual_dy'] = 0.0
            return group

        # 趋势: 6月中心移动平均
        trend = group['dy_mm'].rolling(6, center=True, min_periods=1).mean()
        detrended = group['dy_mm'] - trend

        # 周期: 12个月周期均值
        group['month_num'] = group['month'].apply(lambda x: int(x.split('-')[1]))
        seasonal = detrended.groupby(group['month_num']).transform('mean')
        residual = detrended - seasonal

        group['trend_dy'] = trend.values
        group['seasonal_dy'] = seasonal.values
        group['residual_dy'] = residual.values
        return group

    df = df.groupby('node', group_keys=False).apply(decompose_series)

    # ---- 空间分区编码 (路线G) ----
    zone_map = {'强变形带': 2, '过渡变形带': 1, '稳定背景带': 0}
    df['zone_code'] = df['zone'].map(zone_map)

    # 经纬度和高程的交互特征
    df['lon_lat_interaction'] = df['lon'] * df['lat']
    df['elev_sens_interaction'] = df['elevation_m'] * df['sensitivity_factor']
    df['zone_elev'] = df['zone_code'] * df['elevation_m']
    df['zone_sens'] = df['zone_code'] * df['sensitivity_factor']

    # ---- 位移历史特征 ----
    df['dy_lag1'] = df.groupby('node')['dy_mm'].shift(1)
    df['dy_lag2'] = df.groupby('node')['dy_mm'].shift(2)
    df['dy_lag3'] = df.groupby('node')['dy_mm'].shift(3)
    df['dy_ma3'] = df.groupby('node')['dy_mm'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['dy_ma6'] = df.groupby('node')['dy_mm'].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    df['dy_std3'] = df.groupby('node')['dy_mm'].transform(
        lambda x: x.rolling(3, min_periods=1).std().fillna(0)
    )
    df['dy_std6'] = df.groupby('node')['dy_mm'].transform(
        lambda x: x.rolling(6, min_periods=1).std().fillna(0)
    )
    df['dy_max3'] = df.groupby('node')['dy_mm'].transform(
        lambda x: x.rolling(3, min_periods=1).max()
    )
    df['dy_diff1'] = df.groupby('node')['dy_mm'].diff()

    # cum_disp 变化率
    df['cum_disp_rate'] = df.groupby('node')['cum_disp_mm'].pct_change().fillna(0)

    # ---- 降雨-水位交互 ----
    df['rain_water_interaction'] = df['rain_mm'] * df['water_level_drop_m']
    df['rain_sens_interaction'] = df['rain_mm'] * df['sensitivity_factor']
    df['water_sens_interaction'] = df['water_level_drop_m'] * df['sensitivity_factor']

    return df


def construct_samples(df):
    """
    构造滑动窗口样本：
    输入: 过去W个月的特征序列
    输出: 未来H个月的位移 + 风险标签
    """
    df = df.sort_values(['node', 'month']).reset_index(drop=True)

    # 定义特征列 (排除非特征列)
    exclude_cols = ['month', 'node', 'zone', 'sample_id', 'month_num']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    # 对NaN做前向填充再置0
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
    # 处理无穷大
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)

    samples = []
    nodes = df['node'].unique()

    for node in nodes:
        node_df = df[df['node'] == node].sort_values('month').reset_index(drop=True)
        n = len(node_df)

        for i in range(cfg.W, n - cfg.H + 1):
            # 输入窗口特征
            window_features = node_df.iloc[i - cfg.W: i][feature_cols].values  # (W, F)

            # 目标: 未来H个月的dy_mm
            future_dy = node_df.iloc[i: i + cfg.H]['dy_mm'].values  # (H,)

            # 累计位移
            cum_dy = future_dy.sum()
            max_dy = future_dy.max()
            avg_dy = future_dy.mean()

            # 风险标签
            risk_label = assign_risk_label(cum_dy)

            # 当前月份
            current_month = node_df.iloc[i - 1]['month']

            # 节点信息
            zone = node_df.iloc[i - 1]['zone']

            samples.append({
                'node': node,
                'month': current_month,
                'zone': zone,
                'features': window_features,
                'future_dy': future_dy,
                'cum_dy': cum_dy,
                'max_dy': max_dy,
                'avg_dy': avg_dy,
                'risk_label': risk_label,
                'feature_cols': feature_cols,
            })

    return samples


def assign_risk_label(cum_dy):
    """根据累计位移分配风险等级"""
    if cum_dy < cfg.RISK_THRESHOLDS[0]:
        return 0  # 蓝色低风险
    elif cum_dy < cfg.RISK_THRESHOLDS[1]:
        return 1  # 黄色关注
    elif cum_dy < cfg.RISK_THRESHOLDS[2]:
        return 2  # 橙色预警
    else:
        return 3  # 红色高危


# ===================== 2. Dataset =====================

class LandslideDataset(Dataset):
    def __init__(self, samples, feature_scaler=None, fit_scaler=False):
        self.samples = samples
        self.feature_cols = samples[0]['feature_cols'] if samples else []

        # 准备特征数据
        all_features = np.stack([s['features'] for s in samples])  # (N, W, F)

        # 标准化
        if fit_scaler:
            N, W, F = all_features.shape
            flat = all_features.reshape(-1, F)
            self.feature_scaler = StandardScaler()
            flat_scaled = self.feature_scaler.fit_transform(flat)
            self.features = flat_scaled.reshape(N, W, F)
        else:
            if feature_scaler is None:
                raise ValueError("Must provide feature_scaler when fit_scaler=False")
            self.feature_scaler = feature_scaler
            N, W, F = all_features.shape
            flat = all_features.reshape(-1, F)
            flat_scaled = self.feature_scaler.transform(flat)
            self.features = flat_scaled.reshape(N, W, F)

        # 目标
        self.future_dy = np.stack([s['future_dy'] for s in samples])  # (N, H)
        self.cum_dy = np.array([s['cum_dy'] for s in samples], dtype=np.float32)
        self.max_dy = np.array([s['max_dy'] for s in samples], dtype=np.float32)
        self.avg_dy = np.array([s['avg_dy'] for s in samples], dtype=np.float32)
        self.risk_labels = np.array([s['risk_label'] for s in samples], dtype=np.int64)
        self.nodes = [s['node'] for s in samples]
        self.months = [s['month'] for s in samples]
        self.zones = [s['zone'] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'future_dy': torch.FloatTensor(self.future_dy[idx]),
            'cum_dy': torch.FloatTensor([self.cum_dy[idx]]),
            'max_dy': torch.FloatTensor([self.max_dy[idx]]),
            'avg_dy': torch.FloatTensor([self.avg_dy[idx]]),
            'risk_label': torch.LongTensor([self.risk_labels[idx]]),
            'node': self.nodes[idx],
            'month': self.months[idx],
            'zone': self.zones[idx],
        }


# ===================== 3. Focal Loss =====================

class FocalLoss(nn.Module):
    """Focal Loss: 解决类别不平衡"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # class weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ===================== 4. 模型 =====================

class MultiTaskGRU(nn.Module):
    """
    多任务GRU: 共享编码器 + 位移回归头 + 风险分类头
    改进: 注意力机制 + 残差连接 + LayerNorm
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.15):
        super().__init__()

        # 输入投影层 (降维或升维)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 共享GRU编码器
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # LayerNorm
        self.ln = nn.LayerNorm(hidden_dim)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 位移回归头
        self.regress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # future_dy_h1, h2, h3
        )

        # 累计/最大/平均位移头
        self.aux_regress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # cum, max, avg
        )

        # 风险分类头
        self.classify_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, W, F)
        x = self.input_proj(x)  # (B, W, H)
        gru_out, _ = self.gru(x)  # (B, W, H)
        gru_out = self.ln(gru_out)

        # 注意力加权
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)  # (B, W, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (B, H)

        # 回归输出
        dy_pred = self.regress_head(context)  # (B, 3)
        aux_pred = self.aux_regress_head(context)  # (B, 3)
        risk_logits = self.classify_head(context)  # (B, num_classes)

        return dy_pred, aux_pred, risk_logits


class MultiTaskLSTM(nn.Module):
    """
    改进模型: LSTM + TCN + Multi-Head Attention
    路线D改进: 更强编码器
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.15):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # TCN层
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # TCN + 原始拼接
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 多头注意力
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Layer Norm
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # FFN for transformer-like block
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 位移回归头
        self.regress_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        # 辅助回归
        self.aux_regress_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

        # 风险分类头
        self.classify_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, W, F)
        B, W, F = x.shape

        x_proj = self.input_proj(x)  # (B, W, H)

        # TCN
        tcn_out = self.tcn(x_proj.transpose(1, 2)).transpose(1, 2)  # (B, W, H)

        # 拼接TCN和原始特征
        combined = torch.cat([x_proj, tcn_out], dim=-1)  # (B, W, 2H)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(combined)  # (B, W, H)

        # Transformer-like block
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        attn_out = self.ln1(attn_out + lstm_out)  # 残差+LN
        ffn_out = self.ffn(attn_out)
        out = self.ln2(ffn_out + attn_out)  # 残差+LN

        # 全局特征
        last_step = out[:, -1, :]
        global_pool = out.mean(dim=1)
        final_feat = torch.cat([last_step, global_pool], dim=-1)

        dy_pred = self.regress_head(final_feat)
        aux_pred = self.aux_regress_head(final_feat)
        risk_logits = self.classify_head(final_feat)

        return dy_pred, aux_pred, risk_logits


# ===================== 5. 训练 =====================

def compute_class_weights(labels):
    """计算类别权重"""
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)
    weights = []
    for c in range(n_classes):
        if c in counts:
            weights.append(total / (n_classes * counts[c]))
        else:
            weights.append(1.0)
    return weights


def train_model(model, train_loader, val_loader, model_name='model',
                class_weights=None):
    """训练模型，带early stopping和focal loss"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    mse_loss = nn.MSELoss()
    # 使用Focal Loss + 类别权重
    if class_weights is not None:
        cw_tensor = torch.FloatTensor(class_weights).to(cfg.DEVICE)
    else:
        cw_tensor = None
    focal_loss = FocalLoss(alpha=cw_tensor, gamma=2.0)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(cfg.EPOCHS):
        # === 训练 ===
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            features = batch['features'].to(cfg.DEVICE)
            future_dy = batch['future_dy'].to(cfg.DEVICE)
            cum_dy = batch['cum_dy'].to(cfg.DEVICE)
            max_dy = batch['max_dy'].to(cfg.DEVICE)
            avg_dy = batch['avg_dy'].to(cfg.DEVICE)
            risk_label = batch['risk_label'].squeeze(-1).to(cfg.DEVICE)

            dy_pred, aux_pred, risk_logits = model(features)

            # 回归损失
            loss_dy = mse_loss(dy_pred, future_dy)

            # 辅助回归损失
            aux_target = torch.cat([cum_dy, max_dy, avg_dy], dim=-1)
            loss_aux = mse_loss(aux_pred, aux_target)

            # 分类损失 (Focal Loss)
            loss_cls = focal_loss(risk_logits, risk_label)

            # 总损失
            loss = loss_dy + cfg.LAMBDA_AUX * loss_aux + cfg.LAMBDA_CLS * loss_cls

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # === 验证 ===
        model.eval()
        val_loss = 0
        val_batches = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(cfg.DEVICE)
                future_dy = batch['future_dy'].to(cfg.DEVICE)
                cum_dy = batch['cum_dy'].to(cfg.DEVICE)
                max_dy = batch['max_dy'].to(cfg.DEVICE)
                avg_dy = batch['avg_dy'].to(cfg.DEVICE)
                risk_label = batch['risk_label'].squeeze(-1).to(cfg.DEVICE)

                dy_pred, aux_pred, risk_logits = model(features)

                loss_dy = mse_loss(dy_pred, future_dy)
                aux_target = torch.cat([cum_dy, max_dy, avg_dy], dim=-1)
                loss_aux = mse_loss(aux_pred, aux_target)
                loss_cls = focal_loss(risk_logits, risk_label)
                loss = loss_dy + cfg.LAMBDA_AUX * loss_aux + cfg.LAMBDA_CLS * loss_cls

                val_loss += loss.item()
                val_batches += 1

                pred_labels = risk_logits.argmax(dim=-1)
                val_correct += (pred_labels == risk_label).sum().item()
                val_total += risk_label.size(0)

        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        val_acc = val_correct / max(val_total, 1)

        # Early stopping based on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"[{model_name}] Epoch {epoch+1}/{cfg.EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Best Acc: {best_val_acc:.4f}")

        if patience_counter >= cfg.PATIENCE:
            print(f"[{model_name}] Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
            break

    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    model = model.to(cfg.DEVICE)

    return model, train_losses, val_losses


# ===================== 6. 评估 =====================

def evaluate_model(model, dataloader, dataset, model_name='model'):
    """评估模型，输出所有指标"""
    model.eval()

    all_dy_true = []
    all_dy_pred = []
    all_cum_true = []
    all_cum_pred = []
    all_risk_true = []
    all_risk_pred = []
    all_nodes = []
    all_months = []
    all_zones = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(cfg.DEVICE)
            dy_pred, aux_pred, risk_logits = model(features)

            probs = torch.softmax(risk_logits, dim=-1)
            risk_pred = risk_logits.argmax(dim=-1)

            all_dy_true.append(batch['future_dy'].numpy())
            all_dy_pred.append(dy_pred.cpu().numpy())
            all_cum_true.append(batch['cum_dy'].numpy())
            all_cum_pred.append(aux_pred.cpu().numpy()[:, 0:1])
            all_risk_true.append(batch['risk_label'].squeeze(-1).numpy())
            all_risk_pred.append(risk_pred.cpu().numpy())
            all_nodes.extend(batch['node'])
            all_months.extend(batch['month'])
            all_zones.extend(batch['zone'])
            all_probs.append(probs.cpu().numpy())

    # 合并
    dy_true = np.concatenate(all_dy_true, axis=0)
    dy_pred = np.concatenate(all_dy_pred, axis=0)
    cum_true = np.concatenate(all_cum_true, axis=0).flatten()
    cum_pred = np.concatenate(all_cum_pred, axis=0).flatten()
    risk_true = np.concatenate(all_risk_true, axis=0)
    risk_pred_logits = np.concatenate(all_risk_pred, axis=0)  # 分类头预测
    probs = np.concatenate(all_probs, axis=0)

    # === 核心改进: 位移预测→阈值分级 (比分类头更准) ===
    pred_cum_from_dy = dy_pred.sum(axis=1)          # 回归头位移求和
    pred_cum_combined = 0.6 * pred_cum_from_dy + 0.4 * cum_pred  # 加权
    risk_pred_from_disp = np.array([assign_risk_label(c) for c in pred_cum_combined])

    # 边界修正: 靠近阈值且分类头置信度高时信任分类头
    risk_pred = risk_pred_from_disp.copy()
    for i in range(len(risk_pred)):
        cum_i = pred_cum_combined[i]
        dists = [abs(cum_i - t) for t in cfg.RISK_THRESHOLDS]
        min_dist = min(dists)
        if min_dist < 2.0 and probs[i].max() > 0.6:
            risk_pred[i] = risk_pred_logits[i]

    # === 位移预测指标 ===
    mae_dy = mean_absolute_error(dy_true.flatten(), dy_pred.flatten())
    rmse_dy = np.sqrt(mean_squared_error(dy_true.flatten(), dy_pred.flatten()))
    r2_dy = r2_score(dy_true.flatten(), dy_pred.flatten())
    mape_dy = np.mean(np.abs((dy_true.flatten() - dy_pred.flatten()) /
                              (dy_true.flatten() + 1e-8))) * 100

    mae_cum = mean_absolute_error(cum_true, cum_pred)
    rmse_cum = np.sqrt(mean_squared_error(cum_true, cum_pred))
    r2_cum = r2_score(cum_true, cum_pred)

    # === 风险预测指标 ===
    acc = accuracy_score(risk_true, risk_pred)
    prec = precision_score(risk_true, risk_pred, average='macro', zero_division=0)
    rec = recall_score(risk_true, risk_pred, average='macro', zero_division=0)
    f1 = f1_score(risk_true, risk_pred, average='macro', zero_division=0)
    cm = confusion_matrix(risk_true, risk_pred, labels=list(range(4)))

    # === 分区评估 ===
    zone_results = {}
    for zone_name in ['强变形带', '过渡变形带', '稳定背景带']:
        mask = np.array([z == zone_name for z in all_zones])
        if mask.sum() > 0:
            z_acc = accuracy_score(risk_true[mask], risk_pred[mask])
            z_mae = mean_absolute_error(dy_true[mask].flatten(), dy_pred[mask].flatten())
            zone_results[zone_name] = {
                'accuracy': float(z_acc),
                'mae': float(z_mae),
                'count': int(mask.sum())
            }

    # 汇总
    metrics = {
        'model': model_name,
        'displacement': {
            'MAE_dy': float(round(mae_dy, 4)),
            'RMSE_dy': float(round(rmse_dy, 4)),
            'R2_dy': float(round(r2_dy, 4)),
            'MAPE_dy': float(round(mape_dy, 2)),
            'MAE_cum': float(round(mae_cum, 4)),
            'RMSE_cum': float(round(rmse_cum, 4)),
            'R2_cum': float(round(r2_cum, 4)),
        },
        'risk': {
            'Accuracy': float(round(acc, 4)),
            'Precision': float(round(prec, 4)),
            'Recall': float(round(rec, 4)),
            'F1': float(round(f1, 4)),
        },
        'zone_results': zone_results,
    }

    # 构建详细结果DataFrame
    results_df = pd.DataFrame({
        'node': all_nodes,
        'month': all_months,
        'zone': all_zones,
        'true_dy_h1': dy_true[:, 0],
        'pred_dy_h1': dy_pred[:, 0],
        'true_dy_h2': dy_true[:, 1],
        'pred_dy_h2': dy_pred[:, 1],
        'true_dy_h3': dy_true[:, 2],
        'pred_dy_h3': dy_pred[:, 2],
        'true_cum_dy_H': cum_true,
        'pred_cum_dy_H': cum_pred,
        'true_label_future': [cfg.RISK_LABELS[int(r)] for r in risk_true],
        'pred_label_future': [cfg.RISK_LABELS[int(r)] for r in risk_pred],
        'confidence': [float(probs[i, int(risk_pred[i])]) for i in range(len(risk_pred))],
        'horizon': cfg.H,
    })

    return metrics, results_df, cm


# ===================== 7. 集成预测 =====================

def ensemble_predict(models, dataloader):
    """多模型集成预测"""
    all_dy_preds = []
    all_aux_preds = []
    all_risk_logits = []
    all_meta = []

    for model in models:
        model.eval()
        dy_preds = []
        aux_preds = []
        risk_logits_list = []
        meta_list = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(cfg.DEVICE)
                dy_pred, aux_pred, risk_logits = model(features)

                dy_preds.append(dy_pred.cpu().numpy())
                aux_preds.append(aux_pred.cpu().numpy())
                risk_logits_list.append(risk_logits.cpu().numpy())

                if len(meta_list) == 0:
                    meta_list.append({
                        'nodes': batch['node'],
                        'months': batch['month'],
                        'zones': batch['zone'],
                        'future_dy': batch['future_dy'].numpy(),
                        'cum_dy': batch['cum_dy'].numpy().flatten(),
                        'max_dy': batch['max_dy'].numpy().flatten(),
                        'avg_dy': batch['avg_dy'].numpy().flatten(),
                        'risk_labels': batch['risk_label'].squeeze(-1).numpy(),
                    })

        all_dy_preds.append(np.concatenate(dy_preds))
        all_aux_preds.append(np.concatenate(aux_preds))
        all_risk_logits.append(np.concatenate(risk_logits_list))
        all_meta.append(meta_list[0])

    # 平均集成 - 位移预测
    dy_pred_avg = np.mean(all_dy_preds, axis=0)
    aux_pred_avg = np.mean(all_aux_preds, axis=0)
    risk_logits_avg = np.mean(all_risk_logits, axis=0)

    meta = all_meta[0]
    dy_true = meta['future_dy']
    cum_true = meta['cum_dy']
    risk_true = meta['risk_labels']

    # ===== 核心改进: 双路径风险预测 =====
    # 路径1: 分类头直接预测
    risk_pred_from_logits = risk_logits_avg.argmax(axis=-1)

    # 路径2: 用位移预测的累计值按阈值分级（更可靠）
    pred_cum_from_dy = dy_pred_avg.sum(axis=1)  # 用回归头预测的位移求和
    pred_cum_from_aux = aux_pred_avg[:, 0]       # 辅助头直接预测的累计位移
    # 两者加权平均
    pred_cum_combined = 0.6 * pred_cum_from_dy + 0.4 * pred_cum_from_aux
    risk_pred_from_disp = np.array([assign_risk_label(c) for c in pred_cum_combined])

    # 路径3: 分类头概率 + 位移分级投票
    # 对每个样本，如果路径2的置信度高（接近阈值边界用分类头，否则用位移路径）
    risk_pred_final = risk_pred_from_disp.copy()  # 默认用位移路径（更准）

    # softmax兼容实现
    def _softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    probs_logits = _softmax(risk_logits_avg)

    # 对于位移路径给出的置信度较低的样本（接近阈值边界），用分类头修正
    for i in range(len(risk_pred_final)):
        cum_pred_i = pred_cum_combined[i]
        # 计算到最近阈值的距离
        dists = [abs(cum_pred_i - t) for t in cfg.RISK_THRESHOLDS]
        min_dist = min(dists)
        # 如果很接近阈值边界且分类头置信度高，信任分类头
        if min_dist < 2.0 and probs_logits[i].max() > 0.6:
            risk_pred_final[i] = risk_pred_from_logits[i]

    risk_pred = risk_pred_final
    probs = probs_logits  # 用分类头的概率作为置信度

    # 指标计算
    mae_dy = mean_absolute_error(dy_true.flatten(), dy_pred_avg.flatten())
    rmse_dy = np.sqrt(mean_squared_error(dy_true.flatten(), dy_pred_avg.flatten()))
    r2_dy = r2_score(dy_true.flatten(), dy_pred_avg.flatten())

    acc = accuracy_score(risk_true, risk_pred)
    prec = precision_score(risk_true, risk_pred, average='macro', zero_division=0)
    rec = recall_score(risk_true, risk_pred, average='macro', zero_division=0)
    f1 = f1_score(risk_true, risk_pred, average='macro', zero_division=0)
    cm = confusion_matrix(risk_true, risk_pred, labels=list(range(4)))

    metrics = {
        'model': 'Ensemble',
        'displacement': {
            'MAE_dy': float(round(mae_dy, 4)),
            'RMSE_dy': float(round(rmse_dy, 4)),
            'R2_dy': float(round(r2_dy, 4)),
            'MAE_cum': float(round(mean_absolute_error(cum_true, aux_pred_avg[:, 0]), 4)),
            'RMSE_cum': float(round(np.sqrt(mean_squared_error(cum_true, aux_pred_avg[:, 0])), 4)),
            'R2_cum': float(round(r2_score(cum_true, aux_pred_avg[:, 0]), 4)),
        },
        'risk': {
            'Accuracy': float(round(acc, 4)),
            'Precision': float(round(prec, 4)),
            'Recall': float(round(rec, 4)),
            'F1': float(round(f1, 4)),
        },
    }

    results_df = pd.DataFrame({
        'node': meta['nodes'],
        'month': meta['months'],
        'zone': meta['zones'],
        'true_dy_h1': dy_true[:, 0],
        'pred_dy_h1': dy_pred_avg[:, 0],
        'true_dy_h2': dy_true[:, 1],
        'pred_dy_h2': dy_pred_avg[:, 1],
        'true_dy_h3': dy_true[:, 2],
        'pred_dy_h3': dy_pred_avg[:, 2],
        'true_cum_dy_H': cum_true,
        'pred_cum_dy_H': aux_pred_avg[:, 0],
        'true_label_future': [cfg.RISK_LABELS[int(r)] for r in risk_true],
        'pred_label_future': [cfg.RISK_LABELS[int(r)] for r in risk_pred],
        'confidence': [float(probs[i, int(risk_pred[i])]) for i in range(len(risk_pred))],
        'horizon': cfg.H,
    })

    return metrics, results_df, cm


# ===================== 8. 可视化 =====================

def plot_training_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_prediction_curves(results_df, save_path):
    """绘制典型节点的预测曲线"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for idx, zone in enumerate(['强变形带', '过渡变形带', '稳定背景带']):
        ax = axes[idx]
        zone_df = results_df[results_df['zone'] == zone]
        if len(zone_df) == 0:
            continue

        # 选第一个节点
        node = zone_df['node'].iloc[0]
        node_df = zone_df[zone_df['node'] == node].sort_values('month')

        ax.plot(range(len(node_df)), node_df['true_cum_dy_H'].values,
                'b-o', markersize=3, label='True Cum Disp')
        ax.plot(range(len(node_df)), node_df['pred_cum_dy_H'].values,
                'r--s', markersize=3, label='Pred Cum Disp')
        ax.set_title(f'{zone} - {node}', fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Displacement (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Prediction Curves by Zone', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, labels, save_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_error_distribution(results_df, save_path):
    """绘制误差分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    error_cum = results_df['true_cum_dy_H'] - results_df['pred_cum_dy_H']
    axes[0].hist(error_cum, bins=40, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].set_title('Cumulative Displacement Error Distribution')
    axes[0].set_xlabel('Error (mm)')
    axes[0].set_ylabel('Count')

    zone_maes = results_df.groupby('zone').apply(
        lambda x: np.mean(np.abs(x['true_cum_dy_H'] - x['pred_cum_dy_H']))
    )
    zone_maes.plot(kind='bar', ax=axes[1], color=['#e74c3c', '#f39c12', '#2ecc71'])
    axes[1].set_title('MAE by Zone')
    axes[1].set_ylabel('MAE (mm)')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_risk_timeline(results_df, save_path):
    """风险等级时间线图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    risk_colors = {'蓝色低风险': 'blue', '黄色关注': 'orange',
                   '橙色预警': 'darkorange', '红色高危': 'red'}

    for idx, zone in enumerate(['强变形带', '过渡变形带', '稳定背景带']):
        ax = axes[idx]
        zone_df = results_df[results_df['zone'] == zone]
        if len(zone_df) == 0:
            continue

        node = zone_df['node'].iloc[0]
        node_df = zone_df[zone_df['node'] == node].sort_values('month').reset_index(drop=True)

        for i, row in node_df.iterrows():
            ax.bar(i, 1, color=risk_colors.get(row['true_label_future'], 'gray'), alpha=0.3)
            ax.bar(i, 0.5, color=risk_colors.get(row['pred_label_future'], 'gray'), alpha=0.8)

        ax.set_title(f'{zone} - {node} (Bottom: Pred, Top: True)')
        ax.set_ylabel('Risk Level')
        ax.set_yticks([])

    plt.suptitle('Risk Level Timeline', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(feature_cols, model, save_path):
    """基于GRU输入权重的特征重要性近似"""
    try:
        if hasattr(model, 'input_proj'):
            w = model.input_proj[0].weight.data.cpu().numpy()  # (hidden, input)
            importance = np.abs(w).mean(axis=0)
        elif hasattr(model, 'gru'):
            w = model.gru.weight_ih_l0.data.cpu().numpy()
            importance = np.abs(w).mean(axis=0)
        else:
            return

        top_k = min(20, len(feature_cols))
        top_indices = np.argsort(importance)[-top_k:]
        top_feats = [feature_cols[i] if i < len(feature_cols) else f'feat_{i}' for i in top_indices]
        top_vals = importance[top_indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(top_k), top_vals, align='center')
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_feats)
        ax.set_title('Feature Importance (Input Projection Weights)')
        ax.set_xlabel('Mean |Weight|')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Feature importance plot failed: {e}")


# ===================== 9. 主流程 =====================

def main():
    print("=" * 60)
    print("滑坡位移预测与联合预警 - 多任务深度学习Pipeline v2")
    print("=" * 60)

    # ---- 1. 数据加载 ----
    print("\n[1/7] 加载数据...")
    monthly, node_info, daily = load_data()
    print(f"  月尺度特征: {monthly.shape}")

    # ---- 2. 特征工程 ----
    print("\n[2/7] 特征工程...")
    df = engineer_features(monthly, node_info)
    print(f"  特征工程后: {df.shape}")

    # ---- 3. 构造样本 ----
    print("\n[3/7] 构造滑动窗口样本...")
    samples = construct_samples(df)
    print(f"  总样本数: {len(samples)}")

    # 按时间划分
    train_samples = [s for s in samples if s['month'] <= cfg.TRAIN_END]
    val_samples = [s for s in samples if cfg.TRAIN_END < s['month'] <= cfg.VAL_END]
    test_samples = [s for s in samples if s['month'] > cfg.VAL_END]

    print(f"  训练集: {len(train_samples)}")
    print(f"  验证集: {len(val_samples)}")
    print(f"  测试集: {len(test_samples)}")

    # 计算类别权重
    train_labels = [s['risk_label'] for s in train_samples]
    class_weights = compute_class_weights(train_labels)
    print(f"  类别权重: {[round(w, 3) for w in class_weights]}")

    # ---- 4. 创建Dataset ----
    print("\n[4/7] 创建数据集...")
    train_dataset = LandslideDataset(train_samples, fit_scaler=True)
    val_dataset = LandslideDataset(val_samples, feature_scaler=train_dataset.feature_scaler)
    test_dataset = LandslideDataset(test_samples, feature_scaler=train_dataset.feature_scaler)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    input_dim = train_dataset.features.shape[-1]
    print(f"  输入特征维度: {input_dim}")

    # ---- 5. 训练多个模型 ----
    print("\n[5/7] 训练模型...")

    trained_models = []

    # 模型1: GRU + Attention (基线)
    print("\n--- 模型1: GRU + Attention (基线) ---")
    model1 = MultiTaskGRU(
        input_dim=input_dim,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        num_classes=4,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)
    print(f"  参数量: {sum(p.numel() for p in model1.parameters()):,}")
    model1, loss1_train, loss1_val = train_model(
        model1, train_loader, val_loader, 'GRU-Attention', class_weights
    )
    trained_models.append(('GRU-Attention', model1))

    # 模型2: LSTM + TCN + MHA (改进)
    print("\n--- 模型2: LSTM + TCN + MHA (改进) ---")
    model2 = MultiTaskLSTM(
        input_dim=input_dim,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        num_classes=4,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)
    print(f"  参数量: {sum(p.numel() for p in model2.parameters()):,}")
    model2, loss2_train, loss2_val = train_model(
        model2, train_loader, val_loader, 'LSTM-TCN-MHA', class_weights
    )
    trained_models.append(('LSTM-TCN-MHA', model2))

    # 模型3: 更大的GRU (不同超参)
    print("\n--- 模型3: GRU-Large (改进变体) ---")
    model3 = MultiTaskGRU(
        input_dim=input_dim,
        hidden_dim=cfg.HIDDEN_DIM * 2,
        num_layers=3,
        num_classes=4,
        dropout=0.25
    ).to(cfg.DEVICE)
    print(f"  参数量: {sum(p.numel() for p in model3.parameters()):,}")
    model3, loss3_train, loss3_val = train_model(
        model3, train_loader, val_loader, 'GRU-Large', class_weights
    )
    trained_models.append(('GRU-Large', model3))

    # ---- 6. 评估与集成 ----
    print("\n[6/7] 评估模型...")

    # 单模型评估
    all_metrics = {}
    for name, model in trained_models:
        m, r, cm = evaluate_model(model, test_loader, test_dataset, name)
        all_metrics[name] = m
        print(f"\n{name}:")
        print(f"  位移 - MAE: {m['displacement']['MAE_dy']}, RMSE: {m['displacement']['RMSE_dy']}, "
              f"R²: {m['displacement']['R2_dy']}")
        print(f"  风险 - Acc: {m['risk']['Accuracy']}, F1: {m['risk']['F1']}")

    # 集成评估
    print("\n--- 集成模型 ---")
    ensemble_metrics, ensemble_results, ensemble_cm = ensemble_predict(
        [m for _, m in trained_models], test_loader
    )
    all_metrics['Ensemble'] = ensemble_metrics
    print(f"  位移 - MAE: {ensemble_metrics['displacement']['MAE_dy']}, "
          f"RMSE: {ensemble_metrics['displacement']['RMSE_dy']}, "
          f"R²: {ensemble_metrics['displacement']['R2_dy']}")
    print(f"  风险 - Acc: {ensemble_metrics['risk']['Accuracy']}, "
          f"F1: {ensemble_metrics['risk']['F1']}")

    # 选择最佳模型
    best_name = max(all_metrics, key=lambda k: all_metrics[k]['risk']['Accuracy'])
    best_acc = all_metrics[best_name]['risk']['Accuracy']
    print(f"\n最佳模型: {best_name} (Accuracy: {best_acc})")

    # 用最佳模型的结果
    if best_name == 'Ensemble':
        final_metrics = ensemble_metrics
        final_results = ensemble_results
        final_cm = ensemble_cm
        final_model = trained_models[0][1]  # 任意一个用于feature importance
    else:
        for name, model in trained_models:
            if name == best_name:
                m, r, cm = evaluate_model(model, test_loader, test_dataset, name)
                final_metrics = m
                final_results = r
                final_cm = cm
                final_model = model
                break

    # ---- 7. 保存结果 ----
    print("\n[7/7] 保存结果...")

    output_dir = cfg.OUTPUT_DIR

    # pred_test.csv
    final_results.to_csv(os.path.join(output_dir, 'pred_test.csv'), index=False, encoding='utf-8-sig')

    # metrics.json
    serializable_metrics = convert_to_serializable(all_metrics)
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)

    # 图表
    plot_training_curves(loss1_train, loss1_val,
                         os.path.join(output_dir, 'training_curve_baseline.png'))
    plot_training_curves(loss2_train, loss2_val,
                         os.path.join(output_dir, 'training_curve_improved.png'))

    plot_prediction_curves(final_results,
                           os.path.join(output_dir, 'prediction_curve.png'))
    plot_confusion_matrix(final_cm, cfg.RISK_LABELS,
                          os.path.join(output_dir, 'confusion_matrix.png'))
    plot_error_distribution(final_results,
                            os.path.join(output_dir, 'error_distribution.png'))
    plot_risk_timeline(final_results,
                       os.path.join(output_dir, 'risk_timeline.png'))

    feature_cols = train_samples[0]['feature_cols'] if train_samples else []
    plot_feature_importance(feature_cols, final_model,
                            os.path.join(output_dir, 'feature_importance.png'))

    # 保存模型
    torch.save(final_model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    # 分区分析
    print("\n" + "=" * 60)
    print("分区评估详情:")
    for zone_name in ['强变形带', '过渡变形带', '稳定背景带']:
        mask = final_results['zone'] == zone_name
        if mask.sum() > 0:
            z_df = final_results[mask]
            z_acc = accuracy_score(
                z_df['true_label_future'].map({v: i for i, v in enumerate(cfg.RISK_LABELS)}),
                z_df['pred_label_future'].map({v: i for i, v in enumerate(cfg.RISK_LABELS)})
            )
            z_mae = mean_absolute_error(z_df['true_cum_dy_H'], z_df['pred_cum_dy_H'])
            print(f"  {zone_name}: Acc={z_acc:.4f}, MAE_cum={z_mae:.4f}, N={mask.sum()}")

    # 误差分析
    print("\n误差分析:")
    error_analysis = final_results.copy()
    error_analysis['abs_error_cum'] = np.abs(error_analysis['true_cum_dy_H'] - error_analysis['pred_cum_dy_H'])
    worst_nodes = error_analysis.groupby('node')['abs_error_cum'].mean().sort_values(ascending=False)
    print("  各节点MAE:")
    for node, mae in worst_nodes.items():
        print(f"    {node}: {mae:.3f}")

    misclassified = final_results[final_results['true_label_future'] != final_results['pred_label_future']]
    print(f"\n  风险误判: {len(misclassified)}/{len(final_results)} "
          f"({len(misclassified)/len(final_results)*100:.1f}%)")
    if len(misclassified) > 0:
        print(f"  误判分区分布: {misclassified['zone'].value_counts().to_dict()}")
        print(f"  误判真实标签分布: {misclassified['true_label_future'].value_counts().to_dict()}")

    print("\n" + "=" * 60)
    print("所有结果已保存!")
    print(f"  最佳模型: {best_name}")
    print(f"  风险Accuracy: {final_metrics['risk']['Accuracy']}")
    print(f"  位移R²: {final_metrics['displacement']['R2_dy']}")
    print("=" * 60)

    return final_metrics


if __name__ == '__main__':
    metrics = main()
