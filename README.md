# Temporal Fusion Transformer 時間序列預測模型實作


## 專案說明
此專案實作了 Temporal Fusion Transformer (TFT) 模型，用於台灣股市月報酬率的預測。
資料來源為 TEJ 中台灣市值前 96 的上市公司，並涵蓋了 10 年的各式月資料。


## 核心功能
* 使用 Temporal Fusion Transformer 架構進行多公司的時間序列預測
* 實現交叉驗證評估模型性能 (GroupKFold)
* 具備數據前處理流程，包括特徵工程及缺失值處理
* 針對股票資料進行技術指標計算
* 盡可能地利用 GPU 加速
* 模型評估及結果視覺化
* 方便調整的超參數區塊


## 技術特點
* **深度學習框架**：PyTorch + PyTorch Lightning
* **預測模型**：Temporal Fusion Transformer (使用 pytorch-forecasting )
* **關鍵技術**：
  - 注意力機制
  - GroupKFold 交叉驗證
  - 梯度累積
  - 早停機制


## 模型架構
TFT 模型結合了 LSTM 和 Transformer 的優點，特別適合 panel data 預測：
* **變量選擇**：識別重要特徵
* **解釋性**：注意力機制提供結果可解釋性
* **多變量輸入**：同時處理分類和連續特徵


## 環境要求
* Python 3.8+
* PyTorch 1.10+
* pytorch-forecasting
* PyTorch Lightning
* pandas, numpy, matplotlib, scikit-learn
* 與 GPU 相符之 CUDA 版本


## 實驗結果
模型在交叉驗證評估中取得了良好的表現：
* 平均 RMSE: 約 0.0922
* 預測結果散佈圖顯示全部 Fold 綜合來看模型能初步捕捉月報酬率趨勢
* 極端值預測、特徵選擇及超參數設置上仍有許多改進空間


## 使用說明
1. 安裝所需的套件
2. 檢查資料集 (data.csv) 是否與執行檔 (Transformer.py) 在同一資料夾
3. 運行 Transformer.py 開始訓練
4. 查看 evaluation_results.png 了解模型表現


## 未來改進方向
* 嘗試不同的損失函數組合
* 增加更多金融領域的特徵工程
* 深入調整模型內部架構以更好地捕捉報酬率的特性
* 使用集成學習以提高預測穩定性
* 利用分散式訓練提高訓練效率
* 添加更多預測結果及模型表現可視化
