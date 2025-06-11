# Fine-Tuning Object Detection Models on PASCAL VOC 2012

This repository contains a comprehensive study and implementation of fine-tuning three powerful object detection architectures—**YOLOv8**, **DETR**, and **Faster R-CNN**—on the [PASCAL VOC 2012 dataset](https://universe.roboflow.com/jacob-solawetz/pascal-voc-2012/dataset/13). We also explored additional methods like CLIP and OWL-ViT for vision-language alignment and open-vocabulary object detection.

---

## 📌 Project Overview

- 📊 **Dataset**: PASCAL VOC 2012 (via Roboflow), 17,112 annotated images, 20 object classes  
- 🔍 **Models Used**:  
  - YOLOv8 (n, s, m, l, x) – single-stage, fast and efficient  
  - DETR (ResNet-50, ResNet-101) – end-to-end transformer-based detector  
  - Faster R-CNN (ResNet-50 with FPN) – two-stage, accurate on small objects  
- 📈 **Evaluation Metric**: mAP@0.5, Precision, Recall

---

## 📊 Results Summary

| Model           | mAP@0.5 | Precision | Recall |
|----------------|---------|-----------|--------|
| YOLOv8n        | 67      | 68        | 68     |
| YOLOv8s        | 66      | 68        | 66     |
| YOLOv8m        | 69      | 71        | 70     |
| YOLOv8l        | 70      | 78        | 79     |
| YOLOv8x        | 76      | 79        | 80     |
| DETR R50       | 66      | 68        | 72     |
| DETR R101      | 67      | 68        | 73     |
| Faster R-CNN   | 69      | 74        | 75     |

---

## 🧠 Key Findings

- **YOLOv8x** outperformed all other models in terms of detection accuracy.
- **DETR** was effective in cluttered scenes, but slower to train.
- **Faster R-CNN** remains a solid baseline, especially for detailed localization.

---

## 🧪 Fine-Tuned Models

We provide fine-tuned weights for all major model architectures:

- 🔗 [YOLOv8 (All variants)](https://example.com/yolo-models](https://www.dropbox.com/scl/fi/jpbedyfmg4w5hipksw15o/8mwithoutbest.pt?rlkey=tdpx00ije3p2z1vxi57i306oj&st=myvk0183&dl=0))
- 🔗 [DETR (R50 and R101)](https://example.com/detr-models](https://www.dropbox.com/scl/fi/jpbedyfmg4w5hipksw15o/8mwithoutbest.pt?rlkey=tdpx00ije3p2z1vxi57i306oj&st=myvk0183&dl=0))
- 🔗 [Faster R-CNN](https://example.com/fasterrcnn-models](https://www.dropbox.com/scl/fi/jpbedyfmg4w5hipksw15o/8mwithoutbest.pt?rlkey=tdpx00ije3p2z1vxi57i306oj&st=myvk0183&dl=0))


> Replace these links with your actual model download URLs.

---

## 📄 Report

For full technical details, see the PDF report in [`report.pdf`](./report.pdf), which includes:

- Detailed architecture explanations
- Evaluation metrics and figures
- Dataset annotation and preprocessing details
- CLIP and OWL-ViT analysis for open-vocabulary detection

---

## 🧑‍💻 Author

**MohammadKazem Rajabi**  
✉️ mohammadkazem.rajabi@studenti.unipd.it

---

## 📜 License

This project is licensed for academic and educational use. For commercial use, please contact the author.
