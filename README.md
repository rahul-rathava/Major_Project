# AI Image Detection Using Custom CNN

Hybrid CNN based on InceptionResNetV2, achieving ~96.6% accuracy.  
Includes data augmentation, L2 regularization, and fine-tuning.

## 🗂️ Structure
```
ai_image_detector/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

## 🚀 Training
```bash
pip install -r requirements.txt
python src/train.py
```

## 📊 Evaluation
```bash
python src/evaluate.py
```
