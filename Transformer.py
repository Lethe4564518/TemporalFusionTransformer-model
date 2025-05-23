## ============================= 導入套件 ============================= ##
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import Rbf

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss, NaNLabelEncoder
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.metrics import RMSE

from sklearn.decomposition import PCA

from multiprocessing import freeze_support
import dill
import torch.serialization




# ============================= 超參數設定 =============================
# TFT 模型超參數
SEQ_LEN = 24                        # 編碼器序列長度
PRED_LEN = 24                       # 預測長度（預測下一期）
BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 0.001
N_SPLITS = 10                        # GroupKFold 的 fold 數
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZE = 256                   # 隱藏層大小
ATTENTION_HEAD_SIZE = 16            # attention head 大小
DROPOUT = 0.03                      # 隨機丟棄神經元機率
HIDDEN_CONTINUOUS_SIZE = 128        # 連續特徵的隱藏層大小
OUTPUT_SIZE = 1                     # 輸出大小 (僅預測下一期報酬率)
LOG_INTERVAL = 10                   # 記錄間隔
WEIGHT_DECAY = 1e-6                 # 降低權重衰減以允許更極端的權重值
REDUCE_ON_PLATEAU_PATIENCE = 5      # 當驗證集的 loss 不再下降時，減少學習率
LOSS_ALPHA = 0.4                    # RMSE 損失權重
LOSS_BETA = 0.3                     # MAE 損失權重
LOSS_GAMMA = 0.15                    # 極值鼓勵權重
LOSS_DELTA = 0.15                    # 分布懲罰權重 (過於集中時會增加損失)
EXTREME_THRESHOLD = 0.08             # 極值閾值


# Trainer 超參數
MAX_EPOCHS = NUM_EPOCHS
ACCELERATOR = "gpu" if DEVICE=="cuda" else "auto"
DEVICES = 1 if DEVICE=="cuda" else "auto"
GRADIENT_CLIP_VAL = 0.1
LIMIT_TRAIN_BATCHES = 1.0
LIMIT_VAL_BATCHES = 1.0
ACCUMULATE_GRAD_BATCHES = 2
PRECISION = 32
ENABLE_PROGRESS_BAR = True




# ============================= 設置隨機種子 ============================= ##
SEED = 123

# 設置 Python 內建的隨機種子
random.seed(SEED)

# 設置 NumPy 的隨機種子
np.random.seed(SEED)

# 設置 PyTorch 的隨機種子
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)            # 如果使用多個 GPU
    torch.backends.cudnn.deterministic = True   # 確保每次返回的卷積算法是確定的
    torch.backends.cudnn.benchmark = False      # True 可提高計算速度，但是使用隨機算法

# 設置 Python 的 hash seed
os.environ['PYTHONHASHSEED'] = str(SEED)




# ============================= 資料讀取與前處理 =============================
# FIXME: 調整順序提高易讀性

start_time = time.time()

print(f'----- 當前使用設備 -----\n目前使用: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}\n')

try:
    current_folder = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_folder, "data.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"\n找不到文件: {data_path}，請確認 {data_path} 是否與 ML.py 在同一資料夾\n")

    with open(data_path, 'r', encoding='utf-8') as file:
        df = pd.read_csv(file)
    print(f'----- 資料讀取成功 -----\n資料總筆數: {len(df)} 筆\n')

except Exception as e:
    print(f'讀取資料時出錯: {str(e)}')
    sys.exit(1)


# 轉換年月並排序
df['年月'] = pd.to_datetime(df['年月'])
df = df.sort_values(['股票代碼', '年月']).reset_index(drop=True)


# ============================= 資料處理 =============================
def augment_extreme_samples(df, threshold, noise_scale=0.1):
    """
    對極值樣本進行擴張
    1. 識別極值樣本
    2. 增加噪聲以生成新樣本
    3. 保持時間序列的連續性
    """
    # 複製原始數據
    augmented_df = df.copy()
    
    # 找出極值樣本
    extreme_mask = abs(df['月報酬率']) > threshold
    extreme_samples = df[extreme_mask].copy()
    
    if len(extreme_samples) > 0:
        # 為極值樣本添加噪聲
        noise = np.random.normal(0, noise_scale, size=len(extreme_samples))
        extreme_samples['月報酬率'] = extreme_samples['月報酬率'] * (1 + noise)
        
        # 為相關特徵也添加適當的噪聲
        for col in ['return_1m', 'return_3m', 'return_6m']:
            if col in extreme_samples.columns:
                noise = np.random.normal(0, noise_scale, size=len(extreme_samples))
                extreme_samples[col] = extreme_samples[col] * (1 + noise)
        
        # 將增強後的樣本添加到原始數據中
        augmented_df = pd.concat([augmented_df, extreme_samples])
    
    return augmented_df


def process_data(data):
    """ 資料處理 """
    print('----- 檢查資料中是否有缺失值 -----')
    try:
        result = data.copy()    # 直接使用 CSV 中的數據
        result = result.sort_values(by=['股票代碼', '年月'])    # 確保數據按股票代碼和年月排序
        
        # 檢查資料中是否有缺失值
        missing_values = result.isna().sum()
        if missing_values.any():
            print('資料中仍有缺失值存在')
            print(missing_values[missing_values > 0])
        else:
            print('已確認資料中沒有缺失值')
        
        # 添加數據增強
        print('\n----- 進行極值樣本增強 -----')
        result = augment_extreme_samples(result, 
                                       threshold=EXTREME_THRESHOLD, 
                                       noise_scale=0.1)
        print(f'增強後的總數據量: {len(result)}')
        
        return result
    except Exception as e:
        print(f'資料處理過程中發生錯誤: {str(e)}')
        return None


# 1. 添加技術指標
def add_technical_features(df):
    # 價格動量特徵
    df['return_1m'] = df.groupby('股票代碼')['收盤價'].pct_change(1).fillna(0)    # 使用0填充
    df['return_3m'] = df.groupby('股票代碼')['收盤價'].pct_change(3).fillna(0)
    df['return_6m'] = df.groupby('股票代碼')['收盤價'].pct_change(6).fillna(0)
    
    # 波動率特徵 - 使用rolling計算前添加最小期數限制
    df['volatility_3m'] = df.groupby('股票代碼')['月報酬率'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)
    
    df['volatility_6m'] = df.groupby('股票代碼')['月報酬率'].transform(
        lambda x: x.rolling(6, min_periods=1).std()
    ).fillna(0)
    
    # 相對強度指標 - 使用transform確保不會產生NA
    df['rel_strength'] = df.groupby('股票代碼')['月報酬率'].transform(
        lambda x: x - df.groupby('年月')['月報酬率'].transform('mean')
    ).fillna(0)
    
    # 趨勢特徵 - 添加最小期數限制
    # FIXME: Use obj.ffill() or obj.bfill() ?
    df['ma5'] = df.groupby('股票代碼')['收盤價'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    ).fillna(method='ffill').fillna(method='bfill')
    
    df['ma20'] = df.groupby('股票代碼')['收盤價'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    ).fillna(method='ffill').fillna(method='bfill')
    
    # 避免除以零
    df['price_trend'] = (df['ma5'] / df['ma20'] - 1).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


# 2. 添加時間特徵
def add_time_features(df):
    # 基本時間特徵
    df['month'] = df['年月'].dt.month
    df['quarter'] = df['年月'].dt.quarter
    
    # 週期性編碼
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # 是否為季末月
    df['is_quarter_end'] = df['month'].isin([3,6,9,12]).astype(int)
    
    return df


# # 3. 添加產業/市場特徵 ?
# def add_market_features(df):
#     # 計算市場整體報酬
#     df['market_return'] = df.groupby('年月')['月報酬率'].transform('mean')
#     return df


# # 2. 處理時間序列的季節性
# def add_seasonal_features(df):
#     # 移除整體趨勢
#     df['detrended_return'] = df.groupby('股票代碼')['月報酬率'].transform(
#         lambda x: x - x.rolling(12).mean()
#     )
#     return df


# 執行數據處理並檢查是否有缺失值
df = add_technical_features(df)
df = add_time_features(df)
processed_data = process_data(df)

# 設定要用的特徵（排除識別與標籤）
exclude_cols = ['股票代碼', '股票名稱', '年月', '月報酬率']
features = [col for col in df.columns if col not in exclude_cols]
target = '月報酬率'


# FIXME: 此處因應 panel data 使用 robust expanding minmax 進行標準化，並針對運算效能優化
# 但因為預測效果不如預期，模型會傾向於預測 0 附近的值，故暫不使用
# def robust_expanding_minmax(group, min_periods=5):
#     scaled = group.copy()

#     for col in features:                                            # 對每個特徵進行標準化
#         values = group[col]
#         min_vals = values.expanding(min_periods=1).min()            # 計算累積最小值
#         max_vals = values.expanding(min_periods=1).max()            # 計算累積最大值

#         denom = (max_vals - min_vals).replace(0, np.nan) + 1e-8     # 避免分母為 0
#         norm = (values - min_vals) / denom

#         # fallback: 若時間點小於 min_periods 或全為 NaN，則設為 0.5
#         fallback_mask = (values.expanding().count() < min_periods) | norm.isna()     # 若時間點小於 min_periods 或全為 NaN，則設為 0.5
#         norm = norm.mask(fallback_mask, 0.5)

#         scaled[col] = norm

#     return scaled

# # 每支股票獨立標準化且沒有 data leakage 的問題
# df = df.groupby('股票代碼', group_keys=False).apply(robust_expanding_minmax)


# 建立時間索引（每個股票代碼的資料按時間順序編號）
df['time_idx'] = df.groupby('股票代碼').cumcount()

# 將股票代碼轉換為字符串類型
df['股票代碼'] = df['股票代碼'].astype(str)




# ============================= 自定義損失函數 =============================
class ExtremePredictionLoss(RMSE):
    """ 
    改進的損失函數，結合多個目標：
    1. RMSE 作為基礎
    2. MAE 損失（對極值更敏感）
    3. 極值鼓勵項（當預測接近實際極值時減少損失）
    4. 分布懲罰項（當預測過於集中時增加損失）
    """
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1, extreme_threshold=0.1):
        super().__init__()
        self.alpha = alpha          # RMSE 權重
        self.beta = beta            # MAE 權重
        self.gamma = gamma          # 極值鼓勵權重
        self.delta = delta          # 分布懲罰權重
        self.threshold = extreme_threshold
        self.eps = 1e-8            # 數值穩定性的小常數
    
    def loss(self, y_pred, target):
        # 確保預測值和目標值的維度匹配
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]  # 取第一個元素，通常是預測值
        
        # 處理維度不匹配
        if y_pred.shape != target.shape:
            # 如果是最後一個維度不同，取最後一個時間步的預測
            if len(y_pred.shape) > len(target.shape):
                y_pred = y_pred[..., -1]
            # 確保形狀完全匹配
            y_pred = y_pred.view(target.shape)
        
        # 檢查並處理無效值
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=-1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            # 1. RMSE 損失
            rmse_loss = torch.sqrt(torch.mean((y_pred - target) ** 2) + self.eps)
            
            # 2. MAE 損失
            mae_loss = torch.mean(torch.abs(y_pred - target) + self.eps)
            
            # 3. 極值鼓勵項
            extreme_mask = torch.abs(target) > self.threshold
            if torch.any(extreme_mask):
                extreme_loss = torch.mean(torch.abs(y_pred[extreme_mask] - target[extreme_mask]) + self.eps)
            else:
                extreme_loss = torch.tensor(0.0, device=y_pred.device)
            
            # 4. 分布懲罰項（添加數值穩定性）
            pred_std = torch.std(y_pred) + self.eps
            target_std = torch.std(target) + self.eps
            distribution_penalty = torch.abs(pred_std - target_std)
            
            # 確保所有損失項都是有效的數值
            if torch.isnan(rmse_loss) or torch.isinf(rmse_loss):
                rmse_loss = torch.tensor(0.0, device=y_pred.device)
            if torch.isnan(mae_loss) or torch.isinf(mae_loss):
                mae_loss = torch.tensor(0.0, device=y_pred.device)
            if torch.isnan(extreme_loss) or torch.isinf(extreme_loss):
                extreme_loss = torch.tensor(0.0, device=y_pred.device)
            if torch.isnan(distribution_penalty) or torch.isinf(distribution_penalty):
                distribution_penalty = torch.tensor(0.0, device=y_pred.device)
            
            # 組合所有損失（確保不會產生 NaN）
            total_loss = (self.alpha * rmse_loss + 
                         self.beta * mae_loss - 
                         self.gamma * extreme_loss + 
                         self.delta * distribution_penalty)
            
            # 最後的安全檢查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("警告：損失計算出現 NaN 或 Inf, 改使用原本的 RMSE 作為 Loss Function")
                return super().loss(y_pred, target)
            
            return total_loss
            
        except Exception as e:
            print(f"損失計算出現錯誤：{str(e)}")
            # 如果出現任何錯誤，回退到基礎的 RMSE 損失
            return super().loss(y_pred, target)




# ============================= 自訂 Callback 用於記錄 loss 與權重演變 =============================
callback_list = []
class RecordCallback(Callback):
    """ 記錄訓練與驗證 loss 與權重演變 """
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.weight_history = []    # 記錄每個 epoch 結束時模型最後輸出層的權重
        self.full_weights = []      # 每個 epoch 的全模型權重
        self.loss_traj = []         # 損失值軌跡

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.detach().cpu().item())
        else:
            print("錯誤: 訓練集 loss 為 None\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            loss_value = val_loss.detach().cpu().item()
            self.val_losses.append(loss_value)
            self.loss_traj.append(loss_value)
        else:
            print("錯誤: 驗證集 loss 為 None\n")

        # 嘗試取得 TFT 模型輸出層權重
        # 根據 pytorch-forecasting 的版本，此層名稱可能為 output_layer 或 final_fc
        try:
            weight = pl_module.model.output_layer.weight.detach().cpu().numpy().flatten()
        except AttributeError:
            try:
                weight = pl_module.model.final_fc.weight.detach().cpu().numpy().flatten()
            except AttributeError as e:
                weight = None
                print(f"取得權重時發生錯誤: {str(e)}\n")
        if weight is not None:
            self.weight_history.append(weight)
        
        # 記錄全模型權重的向量
        try:
            full_vector = torch.nn.utils.parameters_to_vector(pl_module.parameters()).detach().cpu().numpy()
            self.full_weights.append(full_vector)
        except Exception as e:
            print(f"記錄全模型權重時發生錯誤: {str(e)}\n")




# ============================= Debug: 檢查模型中所有組件的設備狀態 =============================
# def check_model_devices(model):
#     """ 檢查模型中所有組件的設備狀態, Debug 用 """
#     print("\n模型組件設備狀態:")
#     for name, param in model.named_parameters():
#         print(f"{name}: {param.device}")
#     for name, buffer in model.named_buffers():
#         print(f"{name} (buffer): {buffer.device}")




# ============================= 自訂 LightningModule 用於 TFT 模型 =============================
class TFTLightningWrapper(pl.LightningModule):
    """
    自訂 TFTLightningWrapper 類別，
    修復直接使用 TemporalFusionTransformer.from_dataset() 建構 LightningModule 類別的模型時，
    training_step 和 validation_step 中的 batch 類型不匹配問題。
    """
    def __init__(self, model: BaseModelWithCovariates) -> None:
        super().__init__()
        self.model = model   # 初始化模型並封裝
        # 確保在初始化後立即將模型移到與wrapper相同的設備
        self.device_type = DEVICE   # 記錄當前使用的設備類型
        # 此時self.device可能還不可用，所以先使用DEVICE
        self.model = self.model.to(DEVICE)
        print(f"\n模型初始化時的設備: {DEVICE}")
        print(f"模型參數設備: {next(self.model.parameters()).device}")

    def on_post_move_to_device(self):
        """ 當 LightningModule 移動到新設備時自動調用 """
        device = next(self.parameters()).device
        self.model = self.model.to(device)
        print(f"模型已被移動到設備: {device}")
        return super().on_post_move_to_device()

    def forward(self, *args, **kwargs):
        """ 確保調用前模型是在正確的設備上 """
        current_device = next(self.parameters()).device
        self.model = self.model.to(current_device)
        return self.model(*args, **kwargs)

    def _move_batch_to_device(self, batch):
        """ 將 batch 移到正確的設備 """
        # 獲取當前模型所在設備
        current_device = next(self.parameters()).device
        
        if isinstance(batch, (tuple, list)):
            # 如果 batch 是列表或元組，將其轉換為字典
            if len(batch) >= 2:
                x, y = batch
                if isinstance(x, dict):
                    batch_dict = {k: v.to(current_device) if torch.is_tensor(v) else v for k, v in x.items()}
                else:
                    # 如果 x 不是字典，創建一個新的字典
                    batch_dict = {"x": x.to(current_device) if torch.is_tensor(x) else x}
                if torch.is_tensor(y):
                    batch_dict["y"] = y.to(current_device)
                batch = batch_dict
            else:
                batch = [b.to(current_device) if torch.is_tensor(b) else b for b in batch]
        elif isinstance(batch, dict):
            batch = {k: v.to(current_device) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif torch.is_tensor(batch):
            # NOTE: 使用 current_device 而不是 self.device 避免設備不匹配
            batch = batch.to(current_device)
        return batch

    def training_step(self, batch, batch_idx):
        batch = self._move_batch_to_device(batch)
        outputs = self.model(batch)
        
        # 確保 outputs 和 batch 都是正確的格式
        if hasattr(outputs, 'prediction'):
            predictions = outputs.prediction
        elif hasattr(outputs, 'output'):
            predictions = outputs.output
        else:
            predictions = outputs
            
        if isinstance(batch, dict):
            target = batch.get('decoder_target', batch.get('encoder_target'))
        else:
            target = batch[1] if len(batch) > 1 else None
            
        loss = self.model.loss(predictions, target)     # 計算損失
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=BATCH_SIZE)     # Log loss
        return loss
        
    def validation_step(self, batch, batch_idx):
        batch = self._move_batch_to_device(batch)
        outputs = self.model(batch)
        
        # 確保 outputs 和 batch 都是正確的格式
        if hasattr(outputs, 'prediction'):
            predictions = outputs.prediction
        elif hasattr(outputs, 'output'):
            predictions = outputs.output
        else:
            predictions = outputs
            
        if isinstance(batch, dict):
            target = batch.get('decoder_target', batch.get('encoder_target'))
        else:
            target = batch[1] if len(batch) > 1 else None
            
        loss = self.model.loss(predictions, target)     # 計算損失
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)     # Log loss
        return loss
        
    def configure_optimizers(self):
        # Return the model's optimizer configuration
        return self.model.configure_optimizers()




# ============================= 評估指標函數 =============================
def rmse_metric(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# FIXME: 需修改sharpe_ratio，樣本數過少時可能出錯
def sharpe_ratio(returns):
    # 假設 returns 為一系列報酬，無風險利率設為 0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    return mean_ret / std_ret if std_ret != 0 else np.nan




# ============================= 繪製 loss surface 上的權重軌跡 =============================
def plot_best_loss_contour_trajectory(callback_list, all_val_losses, ax):
    """
    繪製最佳 fold 的模型在降維後 loss surface 上的權重演化軌跡，
    使用 2D 等高線圖 + 權重漸變軌跡線條 + 起點圓圈 + 終點X 來繪製。
    """
    best_fold_idx = np.argmin([val_loss[-1] for val_loss in all_val_losses])
    callback = callback_list[best_fold_idx]

    weights = np.array(callback.full_weights)
    losses = np.array(callback.loss_traj)

    if len(weights) == 0 or len(losses) == 0:
        print("尚未記錄任何模型權重或損失")
        return

    # 使用 PCA 降維
    pca = PCA(n_components=2)
    weights_2d = pca.fit_transform(weights)

    # 計算坐標軸範圍，增加 20% 的邊距
    x_min = weights_2d[:, 0].min() - 0.2 * abs(weights_2d[:, 0].max() - weights_2d[:, 0].min())
    x_max = weights_2d[:, 0].max() + 0.2 * abs(weights_2d[:, 0].max() - weights_2d[:, 0].min())
    y_min = weights_2d[:, 1].min() - 0.2 * abs(weights_2d[:, 1].max() - weights_2d[:, 1].min())
    y_max = weights_2d[:, 1].max() + 0.2 * abs(weights_2d[:, 1].max() - weights_2d[:, 1].min())

    # RBF 插值來產生 loss contour
    rbf = Rbf(weights_2d[:, 0], weights_2d[:, 1], losses, function='multiquadric', smooth=0.1)
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = rbf(X, Y)

    # 等高線圖
    ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)

    # 漸變軌跡線條
    points = weights_2d.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(losses.min(), losses.max())
    lc = LineCollection(segments, cmap='autumn', norm=norm)
    lc.set_array(losses)
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # 起點：圓圈 + label
    ax.scatter(weights_2d[0, 0], weights_2d[0, 1], color='blue', s=50, zorder=3)

    # 終點：箭頭 + X + label
    ax.annotate('', xy=weights_2d[-1], xytext=weights_2d[-2],
                arrowprops=dict(arrowstyle='->', color='red', lw=2), zorder=3)
    ax.scatter(weights_2d[-1, 0], weights_2d[-1, 1], marker='x', color='green', s=60, zorder=3)

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Best Fold Trajectory on Loss Contour (Fold {best_fold_idx + 1})")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)




# ============================= GroupKFold 交叉驗證 =============================
# # 以股票代碼作為分組依據，但可能導致每個 fold 的數據表現相差過大，暫不使用
# unique_companies = df['股票代碼'].unique()
# gkf = GroupKFold(n_splits=N_SPLITS)
# fold_results = []

# # 將各股票資料依股票代碼分組，建立 group 字串陣列
# company_groups = df.groupby('股票代碼').groups.keys()

# # 以股票代碼作為分組依據，取出各 fold 的股票代碼
# folds = list(gkf.split(unique_companies, groups=unique_companies))




# ============================= 時間感知型分組交叉驗證 =============================
def time_aware_group_split(df, n_splits=N_SPLITS):
    """ 時間感知的分組交叉驗證 """
    # 按時間排序並分段
    df = df.sort_values('年月')
    unique_dates = df['年月'].unique()
    date_segments = np.array_split(unique_dates, n_splits)
    
    # 對每個時間段，再按股票分組
    folds = []
    for i in range(n_splits):
        # 訓練集時間段
        train_dates = np.concatenate(date_segments[:i] + date_segments[i+1:])
        # 驗證集時間段
        val_dates = date_segments[i]
        
        # 建立訓練集和驗證集的索引
        train_idx = df[df['年月'].isin(train_dates)].index
        val_idx = df[df['年月'].isin(val_dates)].index
        
        folds.append((train_idx, val_idx))
    
    return folds


# ============================= 滾動時間窗口交叉驗證 =============================
# def rolling_window_split(df, train_size=24, val_size=6, step=3):
#     """ 滾動時間窗口的交叉驗證 """
#     folds = []
#     unique_dates = sorted(df['年月'].unique())
    
#     for i in range(0, len(unique_dates) - train_size - val_size, step):
#         # 訓練窗口
#         train_dates = unique_dates[i:i+train_size]
#         # 驗證窗口
#         val_dates = unique_dates[i+train_size:i+train_size+val_size]
        
#         train_idx = df[df['年月'].isin(train_dates)].index
#         val_idx = df[df['年月'].isin(val_dates)].index
        
#         folds.append((train_idx, val_idx))
    
#     return folds


# ============================= 分層時間交叉驗證 =============================
# def stratified_time_split(df, n_splits=5):
#     """ 分層時間的交叉驗證 """
#     # 按產業分組
#     industry_groups = df.groupby('產業類別')
    
#     # 對每個產業進行時間序列分割
#     all_folds = []
#     for industry, group in industry_groups:
#         dates = sorted(group['年月'].unique())
#         date_segments = np.array_split(dates, n_splits)
        
#         for i in range(n_splits):
#             val_dates = date_segments[i]
#             train_dates = np.concatenate([seg for j, seg in enumerate(date_segments) if j != i])
            
#             train_idx = group[group['年月'].isin(train_dates)].index
#             val_idx = group[group['年月'].isin(val_dates)].index
            
#             all_folds.append((train_idx, val_idx))
    
#     return all_folds




# ============================= 開始交叉驗證並訓練模型 =============================
fold_results = []
if __name__ == '__main__':
    freeze_support()
    
    # 使用時間感知型的分組交叉驗證
    for fold, (train_comp_idx, val_comp_idx) in enumerate(time_aware_group_split(df, n_splits=N_SPLITS)):
        print(f"\n============ Fold {fold+1} ============")
        # 取得該 fold 的股票代碼
        train_companies = df['股票代碼'].iloc[train_comp_idx]
        val_companies = df['股票代碼'].iloc[val_comp_idx]
        
        # 建立訓練與驗證 DataFrame（根據股票代碼篩選）
        train_df = df[df['股票代碼'].isin(train_companies)].copy()
        val_df = df[df['股票代碼'].isin(val_companies)].copy()
        
        # FIXME: 嘗試添加時間分割
        # for company in train_companies:
        #     company_data = train_df[train_df['股票代碼'] == company]
        #     time_points = company_data['年月'].unique()
        #     split_idx = int(len(time_points) * 0.8)
            
        #     # 僅使用前 80% 的時間點作為訓練
        #     train_times = time_points[:split_idx]
        #     train_df = train_df[~((train_df['股票代碼'] == company) & 
        #                          (~train_df['年月'].isin(train_times)))]
        
        ## --------------------------- ##
        # 建立 TimeSeriesDataSet (TFT model用)：
        # 訓練集使用所有可觀測歷史資料，encoder 長度 = SEQ_LEN，並預測未來 PRED_LEN
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=target,
            group_ids=["股票代碼"],
            min_encoder_length=SEQ_LEN,                     # 最小 encoder 序列長度
            max_encoder_length=SEQ_LEN,                     # 固定 encoder 長度
            min_prediction_length=PRED_LEN,
            max_prediction_length=PRED_LEN,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=features + [target],
            static_categoricals=["股票代碼"],
            target_normalizer=None,                         # 此例中直接預測原始標籤（也可標準化 ex. StandardScaler）
            categorical_encoders={"股票代碼": NaNLabelEncoder(add_nan=True)},  # 添加 NaNLabelEncoder
            allow_missing_timesteps=True,                   # 允許缺失的時間步
            add_relative_time_idx=True,                     # 添加相對時間索引
        )
        
        # 驗證集：使用與訓練集相同的設定，但預測期間為最後一段
        validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
        
        # 建立 dataloader 
        train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
        val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
        

        ## --------------------------- ##
        # 建立 TFT 模型
        tft_model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=LR,
            hidden_size=HIDDEN_SIZE,                                        # 隱藏層大小
            attention_head_size=ATTENTION_HEAD_SIZE,                        # attention head 大小
            dropout=DROPOUT,                                                # 隨機丟棄神經元機率
            hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,                  # 連續特徵的隱藏層大小
            output_size=OUTPUT_SIZE,                                        # 輸出維度 (此例中為1，因為只預測1期)
            loss=ExtremePredictionLoss(alpha=LOSS_ALPHA, beta=LOSS_BETA, gamma=LOSS_GAMMA, delta=LOSS_DELTA, extreme_threshold=EXTREME_THRESHOLD),   # 使用自定義損失函數
            log_interval=LOG_INTERVAL,                                      # 每 X 個 batch 記錄一次 loss
            weight_decay=WEIGHT_DECAY,                                      # L2 regularization
            reduce_on_plateau_patience=REDUCE_ON_PLATEAU_PATIENCE,          # 當驗證集 loss 不再下降時，減少學習率
        )

        # 將模型移到指定設備
        tft_model = tft_model.to(DEVICE)
        # print(f"\n模型創建後的設備: {next(tft_model.parameters()).device}")

        # 創建 TFTLightningWrapper 實例
        tft = TFTLightningWrapper(tft_model)

        # 確保 wrapper 也在正確的設備上
        tft = tft.to(DEVICE)
        # print(f"\nWrapper 創建後的設備: {next(tft.parameters()).device}")

        # 使用自訂 Callback 記錄 loss 與權重演變
        record_cb = RecordCallback()
        callback_list.append(record_cb)

        # Early stopping 與模型 checkpoint
        early_stop_cb = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss", 
            save_top_k=1, 
            mode="min",
            dirpath="D:/git-repos/TemporalFusionTransformer-model/checkpoints",  # 指定檢查點保存路徑
            filename=f"best_model_fold{fold+1}"  # 指定文件名格式
        )
        
        # 建立 Trainer
        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator=ACCELERATOR,
            devices=DEVICES,
            callbacks=[record_cb, early_stop_cb, checkpoint_cb],
            gradient_clip_val=GRADIENT_CLIP_VAL,
            limit_train_batches=LIMIT_TRAIN_BATCHES,
            limit_val_batches=LIMIT_VAL_BATCHES,
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
            precision=PRECISION,
            enable_progress_bar=ENABLE_PROGRESS_BAR,
            default_root_dir="D:/git-repos/TemporalFusionTransformer-model/training_logs" # 指定日誌保存路徑
        )
        

        ## ------------ DEBUG 區塊 ------------ ##
        # NOTE: 使用 QuantileLoss 時需檢查一個 batch 的 forward 與 loss 計算是否正常
        # debug_batch = next(iter(train_dataloader))
        # debug_batch = tft._move_batch_to_device(debug_batch)  # 使用 wrapper 中的函數將 batch 移到正確設備

        # # 檢查 debug_batch 的鍵值
        # print("Debug batch keys:", debug_batch.keys())

        # # 呼叫模型的 forward，獲取模型輸出
        # outputs = tft.model(debug_batch)  # 直接呼叫內部模型的 forward

        # # 如果輸出是自定義的 Output 類型，我們需要從中提取 'prediction'
        # if isinstance(outputs, tuple):
        #     # 檢查並提取預測結果
        #     predictions = outputs[0]  # 假設預測是輸出的第一個元素
        #     print(f"Debugging predictions shape: {predictions.shape}")
        # elif isinstance(outputs, dict):
        #     predictions = outputs.get('prediction', None)
        #     if predictions is None:
        #         raise KeyError("Output dictionary does not contain 'prediction' key.")
        # else:
        #     predictions = outputs  # 如果不是 tuple 或 dict，直接當作預測處理

        # # 確認 predictions 是 Tensor
        # if not isinstance(predictions, torch.Tensor):
        #     raise TypeError(f"Expected predictions to be a tensor, but got {type(predictions)}")
        # predictions = predictions.squeeze(-1)                       # 因已知 output_size=1，故直接壓縮維度
        # print(f"Debugging predictions shape: {predictions.shape}")  # 打印預測形狀，確保格式正確

        # # 確保 debug_batch 中包含 'encoder_target' 並且它是 Tensor
        # target = debug_batch['encoder_target']  # 假設我們使用 encoder_target 作為目標
        # if not isinstance(target, torch.Tensor):
        #     raise TypeError(f"Expected target to be a tensor, but got {type(target)}")
        # print(f"Debugging target shape: {target.shape}")  # 打印 target 的形狀，確保格式正確

        # # 確保 target 和 predictions 的形狀一致
        # # NOTE: 如果使用 QuantileLoss，predictions 的形狀應該是 (batch_size, prediction_length, output_size)，其中 output_size 是 quantile 的數量
        # # if target.shape != predictions.shape:
        # #     raise ValueError(f"Shape mismatch between target and predictions: {target.shape} != {predictions.shape}")

        # # 計算損失
        # loss = tft.model.loss(predictions, target)
        # print("Debug Loss:", loss.item())
        # print("Debug Loss grad_fn:", loss.grad_fn)
        # 如果沒有錯誤，再進行後續步驟
        ## -------------------------------------- ##

        print(f'\nfold {fold+1} 訓練中...\n')
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        ## --------------------------- ##
        # FIXME: 使用驗證集來評估，但驗證集的資料量過少可能導致評估結果不準確
        try:
            print("\n開始評估...")
            tft = tft.to(DEVICE)
            tft.model.eval()  # 設置模型為評估模式
            tft.model = tft.model.to(DEVICE)

            # 檢查模型設備狀態
            # check_model_devices(tft.model)
            
            # 添加詳細的設備信息調試
            # print("\n設備信息:")
            # print(f"模型設備: {next(tft.model.parameters()).device}")
            # print(f"模型模式: {'eval' if tft.model.training is False else 'train'}")
            
            # 初始化預測值和實際值
            all_predictions = []
            all_targets = []
            
            # 使用 val_dataloader 進行預測
            ## ------------ DEBUG 區塊 (移除會報錯) ------------ ##
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    # print(f"\n處理 batch {batch_idx + 1}:")
                    
                    # 檢查 batch 中每個張量的設備
                    if isinstance(batch, tuple):
                        # print("batch 的類型: 元組")
                        x = batch[0]
                        if isinstance(x, dict):
                            # print("batch 輸入字典中的張量設備:")
                            # for k, v in x.items():
                            #     if torch.is_tensor(v):
                            #         print(f"{k}: {v.device}")
                            pass
                        y = batch[1]
                        if isinstance(y, tuple):
                            # print("batch 目標值元組中的張量設備:")
                            # for i, tensor in enumerate(y):
                            #     if torch.is_tensor(tensor):
                            #         print(f"batch 的目標值 {i}: {tensor.device}")
                            pass
                    else:
                        #print("batch 的類型: 字典")
                        # print("batch 中的張量設備:")
                        # for k, v in batch.items():
                        #     if torch.is_tensor(v):
                        #         print(f"{k}: {v.device}")
                        pass
                    
                    # 顯示模型當前設備
                    model_device = next(tft.model.parameters()).device
                    # print(f"模型當前設備: {model_device}")
                    
                    # 將整個 batch 移到正確的設備上
                    batch = tft._move_batch_to_device(batch)
                    
                    # 進行預測
                    # print(f"\n開始預測...")
                    
                    # 檢查 batch 是否已正確移至模型設備
                    # for k, v in batch.items():
                    #     if torch.is_tensor(v):
                    #         print(f"預測前確認 - {k} 張量設備: {v.device}")
                    #         if v.device != model_device:
                    #             print(f"警告: {k} 張量設備不匹配! 將其移至 {model_device}")
                    #             batch[k] = v.to(model_device)
                    #         break
                    
                    try:
                        predictions = tft.model(batch)
                        # print(f"預測結果設備: {predictions.device if torch.is_tensor(predictions) else '非張量'}")
                        
                        # 確保預測和目標都已移至CPU再轉換為numpy
                        try:
                            # 檢查預測張量類型並轉換
                            # print(f"預測結果類型: {type(predictions)}")
                            
                            # 從 Output 對象中提取實際的張量
                            if hasattr(predictions, 'prediction'):
                                prediction_tensor = predictions.prediction
                                # print(f"使用 predictions.prediction")
                            elif hasattr(predictions, 'output'):
                                prediction_tensor = predictions.output
                                # print(f"使用 predictions.output")
                            elif hasattr(predictions, 'to_tensor'):
                                prediction_tensor = predictions.to_tensor()
                                # print(f"使用 predictions.to_tensor()")
                            else:
                                # 如果以上都不行，print出所有可能有用的屬性
                                # print(f"所有屬性: {dir(predictions)}")
                                # 尋找第一個張量屬性
                                for attr_name in dir(predictions):
                                    if attr_name.startswith('__'):
                                        continue
                                    attr = getattr(predictions, attr_name)
                                    if torch.is_tensor(attr):
                                        prediction_tensor = attr
                                        # print(f"發現張量屬性: {attr_name}")
                                        break
                        
                            # 處理找到的張量
                            if torch.is_tensor(prediction_tensor):
                                prediction_tensor_cpu = prediction_tensor.cpu().detach()
                                
                                # 獲取目標值，處理不同格式的 batch
                                if isinstance(batch, tuple) and len(batch) > 1:
                                    # 如果 batch 是元組格式(x, y)
                                    target_tensor = batch[1]
                                    if isinstance(target_tensor, tuple):
                                        target_tensor = target_tensor[0]
                                else:
                                    # 如果 batch 是字典格式
                                    target_keys = ['decoder_target', 'encoder_target', 'y', 'target']
                                    for key in target_keys:
                                        if key in batch and torch.is_tensor(batch[key]):
                                            target_tensor = batch[key]
                                            # print(f"使用目標鍵: {key}")
                                            break
                                    else:  # 如果沒有找到適合的鍵
                                        # print(f"可用的鍵: {batch.keys()}")
                                        raise ValueError("找不到適合的目標值張量")
                                
                                targets_cpu = target_tensor.cpu().detach()
                                
                                # print(f"預測形狀: {prediction_tensor_cpu.shape}, 目標形狀: {targets_cpu.shape}")
                                
                                # 添加到列表
                                all_predictions.append(prediction_tensor_cpu.numpy())
                                all_targets.append(targets_cpu.numpy())
                            else:
                                raise TypeError(f"找不到可用的預測張量")
                        except Exception as e:
                            print(f"轉換預測結果為numpy時出錯: {str(e)}")
                            # print(f"預測結果類型: {type(predictions)}")
                            # if 'prediction_tensor' in locals():
                            #     print(f"張量類型: {type(prediction_tensor)}")
                    except Exception as e:
                        print(f"預測時發生錯誤: {str(e)}")
                        # print("錯誤發生時的模型狀態:")
                        # print(f"模型設備: {next(tft.model.parameters()).device}")
                        # print(f"輸入設備: {[v.device for v in batch.values() if torch.is_tensor(v)]}")
                        
                        # 嘗試詳細檢查 batch 和模型的設備狀態
                        # print("\n詳細設備診斷:")
                        # print("模型參數:")
                        # for name, param in tft.model.named_parameters():
                        #     if param.requires_grad:
                        #         print(f"{name}: {param.device}")
                        #         break  # 只打印第一個參數作為示例
                        
                        # print("\n輸入張量:")
                        # for k, v in batch.items():
                        #     if torch.is_tensor(v):
                        #         print(f"{k}: {v.device}")
                        
                        # 再次嘗試將整個 batch 移到正確的設備上
                        # print("\n嘗試再次將 batch 移動到模型設備...")
                        # model_device = next(tft.model.parameters()).device
                        # batch = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    
                    # 獲取目標值
                    try:
                        if isinstance(batch, tuple):
                            targets = batch[1][0].to(DEVICE) if isinstance(batch[1], tuple) else batch[1].to(DEVICE)
                        else:
                            targets = batch.get('decoder_target', batch.get('encoder_target', batch.get('y')))
                            if targets is None:
                                raise ValueError("找不到目標值在batch中")
                        # print(f"目標值設備: {targets.device}")
                    except Exception as e:
                        print(f"處理目標值時發生錯誤: {str(e)}")
                        raise e
            ## -------------------------------------- ##
            

            # 將所有預測和目標值拼接成一個大張量
            predictions = np.concatenate(all_predictions, axis=0)
            actuals = np.concatenate(all_targets, axis=0)
            
            # print 預測和目標值的形狀
            print(f"Predictions shape: {predictions.shape}")
            print(f"Actuals shape: {actuals.shape}")

            # 移除預測中多餘的維度（如果存在）
            if len(predictions.shape) > len(actuals.shape):
                predictions = predictions.squeeze(-1)  # 移除最後一個維度
                print(f"調整後的 Predictions 形狀: {predictions.shape}")
            
            # 計算指標
            fold_rmse = rmse_metric(actuals, predictions)
            fold_sharpe = sharpe_ratio(predictions - actuals)
            
            print(f"\nFold {fold+1} Results:")
            print(f"RMSE: {fold_rmse:.4f}")
            print(f"Sharpe Ratio: {fold_sharpe:.4f}")
            
        except Exception as e:
            print(f"\n評估過程發生錯誤: {str(e)}")
            print("\nDebug information:")
            
            # 檢查模型狀態
            print("Model device:", next(tft.model.parameters()).device)
            print("Model mode:", "eval" if tft.model.training is False else "train")
            
            # 檢查數據加載器細節，Debug 用
            # print("\n數據加載器:")
            # for i, batch in enumerate(val_dataloader):
            #     if i >= 3:  # 只打印前3個 batch
            #         break
            #     print(f"\nbatch 的第 {i+1} 個元素:")
            #     if isinstance(batch, tuple):
            #         print("batch 的類型: 元組")
            #         print("batch 的長度:", len(batch))
            #         if isinstance(batch[0], dict):
            #             print("batch 的輸入字典鍵:", batch[0].keys())
            #         if isinstance(batch[1], tuple):
            #             print("目標值類型: 元組")
            #             print("目標值長度:", len(batch[1]))
            #     else:
            #         print("batch 的類型: 字典")
            #         print("batch 的 keys:", batch.keys())
            
            # 設置默認值
            print("\n沒有抓取到預測結果")
            fold_rmse = float('nan')
            fold_sharpe = float('nan')
            actuals = np.array([])
            predictions = np.array([])

        fold_results.append({
            "fold": fold+1,
            "rmse": fold_rmse,
            "sharpe": fold_sharpe,
            "train_losses": record_cb.train_losses,
            "val_losses": record_cb.val_losses,
            "weight_history": record_cb.weight_history,
            "actuals": actuals,
            "predictions": predictions,
            "model": tft.model
        })

    print("\n模型訓練和評估完成!")
    print(f'總運行時間: {time.time() - start_time:.4f} 秒')




# ============================= 創建集成模型 =============================
print("\n========== 創建集成模型 ==========")
class WeightedEnsemble:
    """ 基於驗證集表現的加權集成 """
    def __init__(self, models, val_scores):
        self.models = models
        self.weights = 1 / np.array(val_scores)     # 使用驗證集 RMSE 的倒數作為權重
        self.weights = self.weights / np.sum(self.weights)
    
    def predict(self, X):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred * weight)
        return torch.sum(torch.stack(predictions), dim=0)

# 使用方法如下
def create_weighted_ensemble(fold_results):
    # 收集所有 fold 的模型和驗證集 RMSE
    models = []
    val_scores = []
    for result in fold_results:
        model = result['model']
        model.eval()                # 設置為評估模式
        models.append(model)
        val_scores.append(result['rmse'])
    
    return WeightedEnsemble(models, val_scores)

# 創建最終的集成模型
final_ensemble = create_weighted_ensemble(fold_results)


# 確保目錄存在
if not os.path.exists('D:/git-repos/TemporalFusionTransformer-model/ensemble_models'):
    os.makedirs('D:/git-repos/TemporalFusionTransformer-model/ensemble_models')

# 使用 dill 保存完整的集成模型
with open('D:/git-repos/TemporalFusionTransformer-model/ensemble_models/final_ensemble_model.pt', 'wb') as f:
    dill.dump(final_ensemble, f)

print("集成模型已完整保存成 final_ensemble_model.pt")


# 額外保存模型配置
model_config = {
    'model_class': WeightedEnsemble.__name__,
    'fold_results': fold_results,
    'device': DEVICE,
    'SEQ_LEN': SEQ_LEN,
    'PRED_LEN': PRED_LEN,
    'BATCH_SIZE': BATCH_SIZE,
    'NUM_EPOCHS': NUM_EPOCHS,
    'LR': LR,
    'N_SPLITS': N_SPLITS,
    'HIDDEN_SIZE': HIDDEN_SIZE,
    'HIDDEN_CONTINUOUS_SIZE': HIDDEN_CONTINUOUS_SIZE,
    'OUTPUT_SIZE': OUTPUT_SIZE,
    'LOG_INTERVAL': LOG_INTERVAL,
    'WEIGHT_DECAY': WEIGHT_DECAY,
    'DROPOUT': DROPOUT,
    'HIDDEN_CONTINUOUS_SIZE': HIDDEN_CONTINUOUS_SIZE,
    'OUTPUT_SIZE': OUTPUT_SIZE,
    'ATTENTION_HEAD_SIZE': ATTENTION_HEAD_SIZE,
    'LOSS_ALPHA': LOSS_ALPHA,
    'LOSS_BETA': LOSS_BETA,
    'LOSS_GAMMA': LOSS_GAMMA,
    'LOSS_DELTA': LOSS_DELTA,
    'EXTREME_THRESHOLD': EXTREME_THRESHOLD
}
with open('D:/git-repos/TemporalFusionTransformer-model/ensemble_models/model_config.pt', 'wb') as f:
    dill.dump(model_config, f)

print("模型配置已保存成 model_config.pt")
print("保存後的模型已調整為評估模式")




# ============================= 載入集成模型 =============================
# FIXME: 嘗試載入模型
print("\n========== 載入集成模型 ==========")
# 載入集成模型
with open('D:/git-repos/TemporalFusionTransformer-model/ensemble_models/final_ensemble_model.pt', 'rb') as f:
    model = dill.load(f)

# 載入模型配置
with open('D:/git-repos/TemporalFusionTransformer-model/ensemble_models/model_config.pt', 'rb') as f:
    model_config = dill.load(f)

print("集成模型已成功載入")




# ============================= 整體結果彙整與視覺化 =============================
print("\n========== 結果彙整 ==========")
all_rmse = []
all_sharpe = []
all_train_losses = []
all_val_losses = []
all_predictions = []
all_actuals = []

# 收集所有 fold 的結果
for res in fold_results:
    if not np.isnan(res["rmse"]):
        all_rmse.append(res["rmse"])
    if not np.isnan(res["sharpe"]):
        all_sharpe.append(res["sharpe"])
    if len(res["train_losses"]) > 0:
        all_train_losses.append(res["train_losses"])
    if len(res["val_losses"]) > 0:
        all_val_losses.append(res["val_losses"])
    if len(res["predictions"]) > 0:
        all_predictions.append(res["predictions"])
    if len(res["actuals"]) > 0:
        all_actuals.append(res["actuals"])

# 計算平均指標
avg_rmse = np.mean(all_rmse) if all_rmse else np.nan
avg_sharpe = np.mean(all_sharpe) if all_sharpe else np.nan

print(f"總平均 RMSE: {avg_rmse:.4f}")
print(f"總平均 Sharpe Ratio: {avg_sharpe:.4f}")


# 一併繪製所有 fold 的結果
fig = plt.figure(figsize=(18, 10))

# 1. 繪製訓練與驗證 loss 曲線
ax1 = fig.add_subplot(221)
for i, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
    ax1.plot(range(1, len(train_loss)+1), train_loss, label=f"Fold {i+1} Train")
    ax1.plot(range(1, len(val_loss)+1), val_loss, label=f"Fold {i+1} Val", linestyle='--')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss Curves for All Folds")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# 2. 繪製集成模型的預測值 vs 實際值的散佈圖
ax2 = fig.add_subplot(222)
# 計算集成模型的預測
ensemble_predictions = []
ensemble_actuals = []
for i, (pred, act) in enumerate(zip(all_predictions, all_actuals)):
    ensemble_predictions.append(pred * final_ensemble.weights[i])
    ensemble_actuals.append(act)

ensemble_predictions = np.sum(ensemble_predictions, axis=0)
ensemble_actuals = np.mean(ensemble_actuals, axis=0)

ax2.scatter(ensemble_actuals.flatten(), ensemble_predictions.flatten(), 
           alpha=0.6, label="Ensemble Model", s=10)
ax2.plot([-0.2, 0.2], [-0.2, 0.2], "r--", label="Perfect Prediction")
ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Predicted Values")
ax2.set_title("Ensemble Model: Prediction vs Actual")
ax2.set_xlim(-0.2, 0.2)
ax2.set_ylim(-0.2, 0.2)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 繪製權重演變的3D軌跡圖
ax3 = fig.add_subplot(223)
for i, cb in enumerate(callback_list):
    plot_best_loss_contour_trajectory(callback_list, all_val_losses, ax=ax3)

# 4. 繪製 RMSE 和 Sharpe Ratio 的箱型圖
ax4 = fig.add_subplot(224)
data = [all_rmse, all_sharpe]
boxplot = ax4.boxplot(data, tick_labels=["RMSE", "Sharpe Ratio"])
ax4.set_title("Distribution of Evaluation Metrics")

for i, d in enumerate(data):    # 增加數值標籤
    if d:
        y = np.mean(d)
        ax4.text(i+1, y, f'Mean: {y:.4f}', 
                horizontalalignment='center', 
                verticalalignment='bottom')
ax4.grid(True, alpha=0.3)

# 調整子圖之間的間距
plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("evaluation_results.png 保存完成\n")

# 單獨生成過擬合分析圖表
if all_train_losses and all_val_losses:
    overfitting_fig = plt.figure(figsize=(10, 6))
    ax_over = overfitting_fig.add_subplot(111)
    
    # 計算過擬合比率
    all_ratios = []
    for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
        # 記錄每個epoch的比率
        epoch_ratios = []
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            if train_loss > 1e-6:               # 避免除以太小的數
                ratio = val_loss / train_loss
                epoch_ratios.append(ratio)
            else:
                epoch_ratios.append(np.nan)
        
        # 繪製每個 fold 的過擬合比率並隨 epoch 變化
        epochs = np.arange(1, len(epoch_ratios) + 1)
        ax_over.plot(epochs, epoch_ratios, 'o-', label=f'Fold {i+1}')
        all_ratios.extend(epoch_ratios)
    
    # 增加參考線
    ax_over.axhline(y=1.0, color='g', linestyle='--', label='Perfect Fit Line (Val/Train=1.0)')
    ax_over.axhline(y=1.3, color='r', linestyle='--', label='Overfitting Warning Line (Val/Train=1.3)')
    
    # 設置圖表樣式
    ax_over.set_xlabel('Epoch')
    ax_over.set_ylabel('Val/Train Loss Ratio')
    ax_over.set_title('Overfitting Indicator Changes with Epoch')
    ax_over.grid(True, alpha=0.3)
    ax_over.legend()
    
    # 計算平均過擬合指標
    valid_ratios = [r for r in all_ratios if not np.isnan(r)]
    if valid_ratios:
        avg_ratio = np.mean(valid_ratios)
        ax_over.text(0.5, 0.02, f'Average Overfitting Ratio: {avg_ratio:.4f}', 
                transform=ax_over.transAxes, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # 輸出結果
        print("Overfitting Analysis:")
        print(f"Average Overfitting Ratio: {avg_ratio:.4f}")
        if avg_ratio > 1.3:
            print("Model may be overfitting\n")
        elif avg_ratio < 1.1:
            print("Model fitting is good\n")
        else:
            print("Model has a mild overfitting trend\n")
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(overfitting_fig)
    print("overfitting_analysis.png 保存完成\n")


# ============================= 列出所有超參數 =============================
print("========== TFT 模型超參數 ==========")
print(f"SEQ_LEN: {SEQ_LEN}")
print(f"PRED_LEN: {PRED_LEN}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"NUM_EPOCHS: {NUM_EPOCHS}")
print(f"LR: {LR}")
print(f"N_SPLITS: {N_SPLITS}")
print(f"DEVICE: {DEVICE}")
print(f"HIDDEN_SIZE: {HIDDEN_SIZE}")
print(f"ATTENTION_HEAD_SIZE: {ATTENTION_HEAD_SIZE}")
print(f"DROPOUT: {DROPOUT}")
print(f"HIDDEN_CONTINUOUS_SIZE: {HIDDEN_CONTINUOUS_SIZE}")
print(f"OUTPUT_SIZE: {OUTPUT_SIZE}")
print(f"LOG_INTERVAL: {LOG_INTERVAL}")
print(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
print(f"REDUCE_ON_PLATEAU_PATIENCE: {REDUCE_ON_PLATEAU_PATIENCE}")
print(f"LOSS_ALPHA: {LOSS_ALPHA}")
print(f"LOSS_BETA: {LOSS_BETA}")
print(f"LOSS_GAMMA: {LOSS_GAMMA}")
print(f"LOSS_DELTA: {LOSS_DELTA}")
print(f"EXTREME_THRESHOLD: {EXTREME_THRESHOLD}")

print("\n========== Trainer 超參數 ==========")
print(f"MAX_EPOCHS: {MAX_EPOCHS}")
print(f"ACCELERATOR: {ACCELERATOR}")
print(f"DEVICES: {DEVICES}")
print(f"GRADIENT_CLIP_VAL: {GRADIENT_CLIP_VAL}")
print(f"LIMIT_TRAIN_BATCHES: {LIMIT_TRAIN_BATCHES}")
print(f"LIMIT_VAL_BATCHES: {LIMIT_VAL_BATCHES}")
print(f"ACCUMULATE_GRAD_BATCHES: {ACCUMULATE_GRAD_BATCHES}")
print(f"PRECISION: {PRECISION}")
print(f"ENABLE_PROGRESS_BAR: {ENABLE_PROGRESS_BAR}")




