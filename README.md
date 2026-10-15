# SafePlaceScan

Computer vision model that flags safety hazards around playgrounds: **potholes, trash, graffiti, and tents/encampments**.

**Live demo:** _coming soon_

## Background

My team built the original SafePlaceScan at the SJSU Cybersecurity & AI Hackathon (3rd of 30). It had to run on an NVIDIA Jetson Nano. I owned the dataset and the on-device deployment.

This repo is my rebuild of the project from scratch, without the Jetson requirement, so it runs on any machine and in the browser. The core challenge is the same: no single public dataset covers all four hazards, so most of the work is merging several inconsistent datasets into one clean training set.

Original team: mMaya and Sanjana 

## How it works

1. **Merge datasets** (`scripts/merge.py`): pulls four public datasets from Roboflow Universe and:
   - remaps each dataset's class names onto 4 unified classes, dropping irrelevant ones (e.g. "Efflorescence" wall stains in the graffiti set)
   - converts TACO's polygon segmentation labels into bounding boxes
   - caps each source (max 1,500 images) so trash doesn't dominate the other classes
   - pools everything and re-splits 80/10/10 with a fixed seed for reproducibility
2. **Train** (`scripts/train.py`): fine-tunes a pretrained YOLO11n detector (2.6M parameters) for 50 epochs.
3. **Evaluate** (`scripts/evaluate.py`): per-class precision, recall, and mAP on a held-out test set the model never saw during training.
4. **Demo** (`app.py`): Gradio web app. Upload a photo and see detected hazards.

## Dataset

| Source | Hazard | Images used |
|---|---|---|
| TACO (Trash Annotations in Context) | trash | 1,500 (capped from 3,597) |
| Pothole dataset | pothole | 665 |
| Graffiti dataset | graffiti | 836 (of 1,213, after dropping non-graffiti labels) |
| Tent dataset | tent | 212 |
| **Total** | | **3,213** (2,570 train / 321 val / 322 test) |

## Results

Test set (322 images, never seen during training):

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Pothole | 0.844 | 0.663 | 0.772 | 0.481 |
| Trash | 0.763 | 0.381 | 0.469 | 0.325 |
| Graffiti | 0.859 | 0.881 | 0.923 | 0.718 |
| Tent | 0.961 | 1.000 | 0.993 | 0.769 |
| **Overall** | **0.857** | **0.731** | **0.789** | **0.573** |

Inference runs at about 5 ms per image on a T4 GPU.

![Confusion matrix](results/confusion_matrix_normalized.png)

**What the numbers say:**
- **Graffiti is the strongest class.** It's large and visually distinctive.
- **Trash is the weakest, mainly on recall (0.38).** TACO contains many very small objects (cigarette butts, bottle caps), which are hard to detect at 640px input. When the model does flag trash, it's usually right (precision 0.76), but it misses a lot.
- **The tent score should be read with caution.** The test set has only 28 tents, and the source dataset is small, so near-duplicate images may appear across splits. A larger, more varied tent dataset would give a more reliable estimate.

**Next steps:** add more tent data, try higher input resolution or a larger model for small trash, and test on real playground photos.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env              # add your Roboflow API key
python scripts/download.py
python scripts/merge.py
python scripts/train.py           # needs a GPU (Colab or Kaggle)
python scripts/evaluate.py
python app.py                     # launches the demo locally
```

## Datasets & licenses

- TACO: universe.roboflow.com/narwhal/taco-trash-annotations-in-context-vpdro (CC BY 4.0)
- Potholes: universe.roboflow.com/itech-area/pothole-ndjrq (ODbL v1.0)
- Graffiti: universe.roboflow.com/valentijn/graffiti-nmltp (_check license on page_)
- Tents: universe.roboflow.com/traininggaebo/tent-model (_check license on page_)

## Repo layout

```
app.py                  Gradio demo
configs/datasets.yaml   sources + class mapping
scripts/                download, merge, train, evaluate
weights/best.pt         trained model
results/                metrics + plots
```
