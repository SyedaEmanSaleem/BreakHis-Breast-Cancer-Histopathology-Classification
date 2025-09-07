# 🩺 BreakHis Classification with DenseNet201

This project applies **DenseNet201 with a custom multi-input pipeline** (histopathology images + magnification metadata) for **breast cancer classification** on the **BreakHis dataset**. The dataset consists of **microscopic biopsy images at 40x, 100x, 200x, and 400x magnification**.

Using advanced training strategies (dropout, batch normalization, ELU activations, and learning rate scheduling), the model achieved:

* **Accuracy**: 97.05%
* **AUC**: 0.9942
* **Precision**: 98.49%
* **Recall**: 96.62%
* **Validation Accuracy**: 93.18%
* **Validation AUC**: 0.9520

---

## 📂 Contents

* `BreakHis_DenseNet201.ipynb` – Main notebook with preprocessing, training, and evaluation
* `requirements.txt` – Dependencies used in the project
* `training_logs/` – Training history logs and metrics

---

## ⚙️ Requirements

* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Download the **BreakHis dataset**.
2. Update dataset paths in the notebook (`BreakHis_DenseNet201.ipynb`).
3. Train the model:

```bash
jupyter notebook BreakHis_DenseNet201.ipynb
```

---

## 📊 Results

| Metric    | Train  | Validation |
| --------- | ------ | ---------- |
| Accuracy  | 97.05% | 93.18%     |
| AUC       | 0.9942 | 0.9520     |
| Precision | 98.49% | 93.58%     |
| Recall    | 96.62% | 98.39%     |
| Loss      | 0.0982 | 0.2104     |

---

## 🏗️ Model Architecture (DenseNet201 + Metadata)

* **Backbone**: DenseNet201 (ImageNet weights, partially frozen)
* **Image branch**: Global Average Pooling → Dense (712, ELU) → BN → Dropout
* **Magnification branch**: Dense (128, ELU) → BN → Dropout → Dense (64, ELU)
* **Concatenation** of image + magnification features
* Dense layers with L2 regularization + Dropout
* Output: Sigmoid activation (binary classification)

---

## 🔑 Key Features

* **Magnification-aware multi-input model**
* **Advanced regularization**: Dropout, BatchNorm, L2
* **Adaptive learning rate** with `ReduceLROnPlateau`
* **Early stopping & checkpointing** for robust training


