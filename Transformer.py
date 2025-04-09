## ============================= 導入套件 ============================= ##
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss, NaNLabelEncoder
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.metrics import RMSE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA




# ============================= 超參數設定 =============================
SEQ_LEN = 12                # 編碼器序列長度
PRED_LEN = 12                # 預測長度（預測下一期）
BATCH_SIZE = 64
NUM_EPOCHS = 1
LR = 0.001
N_SPLITS = 2                # GroupKFold 的 fold 數
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# sys.setrecursionlimit(10000)  # 增加遞歸限制




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

def process_data(data):
    """ 資料處理 """
    print('----- 檢查資料中是否有缺失值 -----')
    try:
        # 直接使用 CSV 中的數據，不需要再計算技術指標
        result = data.copy()
        
        # 確保數據按股票代碼和年月排序
        result = result.sort_values(by=['股票代碼', '年月'])
        
        # 檢查是否有缺失值
        missing_values = result.isna().sum()
        if missing_values.any():
            print('資料中仍有缺失值存在')
            print(missing_values[missing_values > 0])
        else:
            print('已確認資料中沒有缺失值')
        
        return result
    
    except Exception as e:
        print(f'資料處理過程中發生錯誤: {str(e)}')
        return None

# 執行數據處理並檢查是否有缺失值
processed_data = process_data(df)

# 設定要用的特徵（排除識別與標籤）
exclude_cols = ['股票代碼', '股票名稱', '年月', '月報酬率']
features = [col for col in df.columns if col not in exclude_cols]
target = '月報酬率'

# 標準化特徵
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 建立時間索引（每個股票代碼的資料按時間順序編號）
df['time_idx'] = df.groupby('股票代碼').cumcount()

# 將股票代碼轉換為字符串類型
df['股票代碼'] = df['股票代碼'].astype(str)




# ============================= 自訂 Callback 用於記錄 loss 與權重演變 =============================
class RecordCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.weight_history = []  # 記錄每個 epoch 結束時模型最後輸出層的權重

    def on_epoch_end(self, trainer, pl_module):
        # 讀取 Lightning 記錄的 train 與 val loss
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())
        # 這裡以 TFT 模型的 output_layer 權重作為記錄對象
        # 根據 pytorch-forecasting 的版本，此層名稱可能為 output_layer 或 final_fc
        try:
            weight = pl_module.output_layer.weight.detach().cpu().numpy().flatten()
        except AttributeError:
            weight = pl_module.final_fc.weight.detach().cpu().numpy().flatten()
        self.weight_history.append(weight)




# ============================= 檢查模型中所有組件的設備狀態 =============================
def check_model_devices(model):
    """檢查模型中所有組件的設備狀態"""
    print("\n模型組件設備狀態:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
    for name, buffer in model.named_buffers():
        print(f"{name} (buffer): {buffer.device}")




# ============================= 自訂 LightningModule 用於 TFT 模型 =============================
# 自訂 TFTLightningWrapper 類別，修復 training_step 和 validation_step 中的 batch 類型不匹配問題
class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, model: BaseModelWithCovariates):
        super().__init__()
        self.model = model
        # 確保在初始化後立即將模型移到與wrapper相同的設備
        self.device_type = DEVICE  # 記錄當前使用的設備類型
        # 注意：此時self.device可能還不可用，所以我們先使用DEVICE
        self.model = self.model.to(DEVICE)
        print(f"\n模型初始化時的設備: {DEVICE}")
        print(f"模型參數設備: {next(self.model.parameters()).device}")

    # 每次設備改變時確保模型也移動到新設備
    def on_post_move_to_device(self):
        # 當LightningModule移動到新設備時自動調用
        device = next(self.parameters()).device
        self.model = self.model.to(device)
        print(f"模型已被移動到設備: {device}")
        return super().on_post_move_to_device()

    def forward(self, *args, **kwargs):
        # 確保調用前模型在正確的設備上
        current_device = next(self.parameters()).device
        self.model = self.model.to(current_device)
        return self.model(*args, **kwargs)

    def _move_batch_to_device(self, batch):
        """
        Helper function to move the batch to the correct device.
        Handles both dictionary and tensor batch formats.
        """
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
            # FIXME: 使用 current_device 而不是 self.device
            batch = batch.to(current_device)
        return batch

    def training_step(self, batch, batch_idx):
        # Move batch to the correct device
        batch = self._move_batch_to_device(batch)
        
        # Call the model's forward method
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
            
        # 計算損失
        loss = self.model.loss(predictions, target)
        
        # Log loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=BATCH_SIZE)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # Move batch to the correct device
        batch = self._move_batch_to_device(batch)
        
        # Call the model's forward method
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
            
        # 計算損失
        loss = self.model.loss(predictions, target)
        
        # Log loss
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
        
    def configure_optimizers(self):
        # Return the model's optimizer configuration
        return self.model.configure_optimizers()




# ============================= 評估指標函數 =============================
def rmse_metric(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# FIXME: 需修改sharpe_ratio，樣本數過少時可能出錯?
def sharpe_ratio(returns):
    # 假設 returns 為一系列報酬，無風險利率設為 0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    return mean_ret / std_ret if std_ret != 0 else np.nan




# ============================= GroupKFold 交叉驗證 =============================
# 以股票代碼作為分組依據
unique_companies = df['股票代碼'].unique()
gkf = GroupKFold(n_splits=N_SPLITS)
fold_results = []

# 將各股票資料依股票代碼分組，建立 group 字串陣列
company_groups = df.groupby('股票代碼').groups.keys()

# 以股票代碼作為分組依據，取出各 fold 的股票代碼
folds = list(gkf.split(unique_companies, groups=unique_companies))




# ============================= 開始交叉驗證 =============================
for fold, (train_comp_idx, val_comp_idx) in enumerate(folds):
    print(f"\n===== Fold {fold+1} =====")
    # 取得該 fold 的股票代碼
    train_companies = unique_companies[train_comp_idx]
    val_companies = unique_companies[val_comp_idx]
    
    # 建立訓練與驗證 DataFrame（根據股票代碼篩選）
    train_df = df[df['股票代碼'].isin(train_companies)].copy()
    val_df = df[df['股票代碼'].isin(val_companies)].copy()
    
    ## --------------------------- ##
    # 建立 TimeSeriesDataSet (TFT model用)：
    # 訓練集：使用所有可觀測歷史資料，encoder 長度 = SEQ_LEN，並預測未來 PRED_LEN
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target,
        group_ids=["股票代碼"],
        min_encoder_length=SEQ_LEN,         # 最小 encoder 序列長度
        max_encoder_length=SEQ_LEN,         # 固定 encoder 長度
        min_prediction_length=PRED_LEN,
        max_prediction_length=PRED_LEN,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=features + [target],
        # 設定靜態變數 static_categoricals，例如股票代碼
        static_categoricals=["股票代碼"],
        target_normalizer=None,  # 此例中直接預測原始標籤（也可標準化 ex. StandardScaler）
        categorical_encoders={"股票代碼": NaNLabelEncoder(add_nan=True)}  # 添加 NaNLabelEncoder
    )
    
    # 驗證集：使用與訓練集相同的設定，但預測期間為最後一段
    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
    
    # 建立 dataloader 並確保數據在正確的設備上
    train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    
    ## --------------------------- ##
    # 建立 TFT 模型
    # 使用 RMSE 作為 loss，可改用其他 loss 比如 QuantileLoss
    tft_model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LR,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=1,
        loss=RMSE(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    # 將模型移到指定設備
    tft_model = tft_model.to(DEVICE)
    print(f"\n模型創建後的設備: {next(tft_model.parameters()).device}")

    # 創建 TFTLightningWrapper 實例
    tft = TFTLightningWrapper(tft_model)

    # 確保 wrapper 也在正確的設備上
    tft = tft.to(DEVICE)
    print(f"\nWrapper 創建後的設備: {next(tft.parameters()).device}")

    # 使用自訂 Callback 記錄 loss 與權重演變
    record_cb = RecordCallback()
    
    # Early stopping 與模型 checkpoint
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    
    # 建立 Trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu" if DEVICE=="cuda" else "auto",
        devices=1 if DEVICE=="cuda" else "auto",
        callbacks=[record_cb, early_stop_cb, checkpoint_cb],
        gradient_clip_val=0.1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        enable_progress_bar=True
    )
    
    print(f'\nfold {fold+1} 訓練中...\n')
    

    ## ------------ DEBUG 區塊 ------------ ##
    # BUG: 使用 QuantileLoss 時需檢查一個 batch 的 forward 與 loss 計算是否正常
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
    # -------------------------------------- #


    
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    ## --------------------------- ##
    # FIXME: 預測與評估 (在驗證集上)
    try:
        print("\n開始評估...")
        tft.model.eval()  # 設置模型為評估模式
        
        # 確保模型在正確的設備上
        tft.model = tft.model.to(DEVICE)
        tft = tft.to(DEVICE)
        
        # 檢查模型設備狀態
        # check_model_devices(tft.model)
        
        # 添加詳細的設備信息調試
        print("\n設備信息:")
        print(f"模型設備: {next(tft.model.parameters()).device}")
        print(f"模型模式: {'eval' if tft.model.training is False else 'train'}")
        
        # 初始化預測值和實際值
        all_predictions = []
        all_targets = []
        
        # 使用驗證數據加載器進行預測
        # BUG: 解決 batch 裡的各種 tensor 設備問題
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # print(f"\n處理批次 {batch_idx + 1}:")
                
                # 檢查批次中每個張量的設備
                if isinstance(batch, tuple):
                    # print("批次類型: 元組")
                    x = batch[0]
                    if isinstance(x, dict):
                        # print("輸入字典中的張量設備:")
                        # for k, v in x.items():
                        #     if torch.is_tensor(v):
                        #         print(f"{k}: {v.device}")
                        pass
                    y = batch[1]
                    if isinstance(y, tuple):
                        # print("目標值元組中的張量設備:")
                        # for i, tensor in enumerate(y):
                        #     if torch.is_tensor(tensor):
                        #         print(f"目標 {i}: {tensor.device}")
                        pass
                else:
                    #print("批次類型: 字典")
                    # print("字典中的張量設備:")
                    # for k, v in batch.items():
                    #     if torch.is_tensor(v):
                    #         print(f"{k}: {v.device}")
                    pass
                
                # 顯示模型當前設備
                model_device = next(tft.model.parameters()).device
                # print(f"模型當前設備: {model_device}")
                
                # 將整個批次移到正確的設備上
                batch = tft._move_batch_to_device(batch)
                
                # 進行預測
                # print(f"\n開始預測...")
                
                # 檢查批次是否已正確移至模型設備
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
                            # 如果以上都不行，打印所有可能有用的屬性
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
                        
                        # 現在處理找到的張量
                        if torch.is_tensor(prediction_tensor):
                            prediction_tensor_cpu = prediction_tensor.cpu().detach()
                            
                            # 獲取目標值，處理不同格式的批次
                            if isinstance(batch, tuple) and len(batch) > 1:
                                # 如果批次是元組格式(x, y)
                                target_tensor = batch[1]
                                if isinstance(target_tensor, tuple):
                                    target_tensor = target_tensor[0]
                            else:
                                # 如果批次是字典格式
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
                    
                    # 再次嘗試將整個批次移到正確的設備上
                    # print("\n嘗試再次將批次移動到模型設備...")
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
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        actuals = np.concatenate(all_targets, axis=0)
        
        # Print shapes
        print(f"Predictions shape: {predictions.shape}")
        print(f"Actuals shape: {actuals.shape}")

        # 移除預測中多餘的維度（如果存在）
        if len(predictions.shape) > len(actuals.shape):
            predictions = predictions.squeeze(-1)  # 移除最後一個維度
            print(f"調整後的 Predictions 形狀: {predictions.shape}")
        
        # Calculate metrics
        fold_rmse = rmse_metric(actuals, predictions)
        fold_sharpe = sharpe_ratio(predictions - actuals)
        
        print(f"\nFold {fold+1} Results:")
        print(f"RMSE: {fold_rmse:.4f}")
        print(f"Sharpe Ratio: {fold_sharpe:.4f}")
        
    except Exception as e:
        print(f"\n評估過程發生錯誤: {str(e)}")
        print("\nDebug information:")
        
        # Check model state
        print("Model device:", next(tft.model.parameters()).device)
        print("Model mode:", "eval" if tft.model.training is False else "train")
        
        # 檢查數據加載器細節
        # print("\n數據加載器:")
        # for i, batch in enumerate(val_dataloader):
        #     if i >= 3:  # 只打印前3個批次
        #         break
        #     print(f"\n批次 {i+1}:")
        #     if isinstance(batch, tuple):
        #         print("批次類型: 元組")
        #         print("批次長度:", len(batch))
        #         if isinstance(batch[0], dict):
        #             print("輸入字典鍵:", batch[0].keys())
        #         if isinstance(batch[1], tuple):
        #             print("目標值類型: 元組")
        #             print("目標值長度:", len(batch[1]))
        #     else:
        #         print("批次類型: 字典")
        #         print("批次鍵:", batch.keys())
        
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
    })

print("\n模型訓練和評估完成!")
print(f'總運行時間: {time.time() - start_time:.4f} 秒')


# ============================= 整體結果彙整與視覺化 =============================
print("\n===== 結果彙整 =====")
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
plt.figure(figsize=(15, 10))

# 1. 繪製訓練與驗證 loss 曲線
plt.subplot(2, 2, 1)
for i, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
    plt.plot(range(1, len(train_loss)+1), train_loss, label=f"Fold {i+1} Train")
    plt.plot(range(1, len(val_loss)+1), val_loss, label=f"Fold {i+1} Val", linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves for All Folds")
plt.legend()
plt.grid(True, alpha=0.3)
# 確保即使損失值很小也能顯示
plt.ylim(bottom=0)  # 從0開始
if len(all_train_losses) > 0 and len(all_train_losses[0]) > 0:
    max_loss = max([max(loss) for loss in all_train_losses + all_val_losses])
    plt.ylim(top=max_loss * 1.2)  # 設置上限為最大損失的1.2倍

# 2. 繪製預測值 vs 實際值的散佈圖
plt.subplot(2, 2, 2)
for i, (pred, act) in enumerate(zip(all_predictions, all_actuals)):
    plt.scatter(act.flatten(), pred.flatten(), alpha=0.6, label=f"Fold {i+1}")
plt.plot([-0.2, 0.2], [-0.2, 0.2], "r--", label="Perfect Prediction")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Prediction vs Actual Scatter Plot")
plt.xlim(-0.2, 0.2)  # 限制x軸範圍為更合理的值
plt.ylim(-0.2, 0.2)  # 限制y軸範圍為更合理的值
plt.legend()
plt.grid(True, alpha=0.3)

# 3. 繪製權重演變的2D軌跡圖
plt.subplot(2, 2, 3)
for i, res in enumerate(fold_results):
    if len(res["weight_history"]) > 0:
        weight_history = np.array(res["weight_history"])
        # 確保有足夠的數據來擬合PCA
        print(f"weight_history shape: {weight_history.shape}")
        if weight_history.shape[0] >= 2:
            pca = PCA(n_components=2)
            weight_2d = pca.fit_transform(weight_history)
            plt.plot(weight_2d[:,0], weight_2d[:,1], marker="o", label=f"Fold {i+1}")
            # 添加箭頭指示方向
            plt.arrow(weight_2d[0,0], weight_2d[0,1], 
                    weight_2d[-1,0]-weight_2d[0,0], weight_2d[-1,1]-weight_2d[0,1], 
                    head_width=0.1, head_length=0.1, fc='k', ec='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Weight Evolution Trajectory")
plt.legend()
plt.grid(True, alpha=0.3)

# 4. 繪製 RMSE 和 Sharpe Ratio 的箱型圖
plt.subplot(2, 2, 4)
data = [all_rmse, all_sharpe]
boxplot = plt.boxplot(data, labels=["RMSE", "Sharpe Ratio"])
plt.title("Distribution of Evaluation Metrics")
# 添加數值標籤
for i, d in enumerate(data):
    if d:  # 確保有數據
        y = np.mean(d)
        plt.text(i+1, y, f'Mean: {y:.4f}', 
                horizontalalignment='center', 
                verticalalignment='bottom')
plt.grid(True, alpha=0.3)

# 調整子圖之間的間距
plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300)
plt.show()
