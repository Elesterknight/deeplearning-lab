# Deep Learning Lab (2025)

國立陽明交通大學 2025 深度學習課程 Lab 2–8 的整合 repo，原本分散在七個獨立 repository，此處合併為單一專案，每個 lab 各自一個子目錄。

## Labs

| 目錄 | 主題 | 主要內容 |
|---|---|---|
| [`lab2/`](lab2/) | Binary Classification | `lab2.ipynb`、課程筆記（Introduction and ML Basics） |
| [`lab3/`](lab3/) | Semi-Supervised Flower Classification | `lab3.ipynb`、`result.csv` |
| [`lab4/`](lab4/) | Semantic Segmentation on BCSS | `lab4.ipynb`、`kaggle_metric.py` |
| [`lab5/`](lab5/) | Object Detection on Pascal VOC | `lab5.ipynb`、`src/`（YOLO 模型、dataset、VOC 評估）、`yolo_loss.py`、`predict_test.py` |
| [`lab6/`](lab6/) | Text Summarization with Seq2Seq | `HW6_314834014.ipynb`、`data_utils.py`、`inference.py`、`eval.py`、TIFU 資料集、[`README_REPRODUCE.md`](lab6/README_REPRODUCE.md) |
| [`lab7/`](lab7/) | Image Generation (DCGAN / WGAN) | `Lab7_314834014.ipynb`、報告 PDF、生成結果圖、`regenerate_images.py` |
| [`lab8/`](lab8/) | Reinforcement Learning — Super Mario World | `Lab8.ipynb`、`custom_policy.py`、`wrappers.py`、`eval.py` |

## 關於模型權重

為控制 repo 體積，**模型權重、大型產物與課程提供的原始資料集未納入本 repo**。原始檔案仍保留在各 lab 的原始 repository：

| 檔案 | 大小 | 原始位置 |
|---|---|---|
| `submission_tta.csv` | 385 MB | `Elesterknight617/deeplearning-lab4` |
| `checkpoints/*.pth`（6 個） | 各 221 MB | `Elesterknight617/deeplearning-lab5` |
| `best_model_checkpoint.pt` | 119 MB | `Elesterknight617/deeplearning-lab4` |
| `supurvised.pt`、`self_training.pt` | 各 104 MB | `Elesterknight617/deeplearning-lab3` |
| `policy/policy.optimizer.pth`、`policy.pth` | 86 MB / 43 MB | `Elesterknight617/deeplearning-lab8` |
| `dataset/tifu/tifu_train.jsonl` | 64 MB | `Elesterknight617/deeplearning-lab6` |
| `model_weight/*.pth`（GAN） | 14 MB / 11 MB | `Elesterknight617/deeplearning-lab7` |
| `Lab3_data_flower_2025/`（4,295 張 JPG） | 227 MB | `Elesterknight617/deeplearning-lab3` |

lab3、lab4、lab5 的權重原先以 Git LFS 儲存，合計約 2.1 GB，超出 GitHub 免費 LFS 額度（1 GB），因此未一併遷移。各 lab 的 notebook 均包含完整訓練流程，可自行重新訓練產生權重。

## 來源

| Lab | 原始 repository |
|---|---|
| lab2 | https://github.com/Elesterknight617/deeplearning-lab2 |
| lab3 | https://github.com/Elesterknight617/deeplearning-lab3 |
| lab4 | https://github.com/Elesterknight617/deeplearning-lab4 |
| lab5 | https://github.com/Elesterknight617/deeplearning-lab5 |
| lab6 | https://github.com/Elesterknight617/deeplearning-lab6 |
| lab7 | https://github.com/Elesterknight617/deeplearning-lab7 |
| lab8 | https://github.com/Elesterknight617/deeplearning-lab8 |
