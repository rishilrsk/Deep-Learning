# 🖼️ Image Caption Generator

> **Turn any picture into a description — powered by AI.**

Upload a photo, click a button, and get a clear, human-readable caption describing what's in the image. That's it. No complicated setup, no technical knowledge required.

---

## ✨ What Does This Do?

This app uses **artificial intelligence** to look at a photo and write a short description of what it sees — just like a person would.

For example:
- 📸 Upload a photo of a dog on a beach → *"a dog running on the beach near the ocean"*
- 🏔️ Upload a landscape photo → *"a mountain with snow on top and a lake in front"*
- 👨‍👩‍👧 Upload a family photo → *"a group of people standing together and smiling"*

---

## 🚀 How to Use

### Step 1 — Install the requirements

Make sure you have **Python 3.10 or newer** installed on your computer. Then open a terminal in this folder and run:

```
pip install -r requirements.txt
```

### Step 2 — Launch the app

```
python app.py
```

### Step 3 — Open in your browser

After a few seconds, a local web address will appear in the terminal (something like `http://127.0.0.1:7860`). Open that link in your browser.

### Step 4 — Upload & generate

1. **Drag & drop** a photo (or click to browse)
2. Click **✨ Generate Caption**
3. Read your AI-generated description!

> 💡 **First launch may be slow** — the app downloads the AI model once on its first run (~1 GB). After that, it starts much faster.

---

## 📁 Project Files

| File | Purpose |
|---|---|
| `app.py` | The main web application |
| `caption_model.py` | The custom deep learning model architecture |
| `train.py` | Script to train the model from scratch |
| `download_dataset.py` | Helper to download the training dataset |
| `requirements.txt` | List of software dependencies |
| `PROJECT_REPORT.md` | Detailed technical report on the model |
| `CNN_LSTM_Project_Report.md` | In-depth architecture & training analysis |

---

## 🧠 How It Works (Simple Version)

The app uses a pre-trained AI model called **BLIP** (by Salesforce) that has been trained on millions of images and their descriptions. When you upload a photo:

1. The AI **looks at the image** and identifies objects, people, actions, and scenery
2. It **generates a natural sentence** describing what it sees
3. The description appears on your screen

The project also includes a **custom-built model** (CNN + LSTM with Attention) that was trained on the Flickr8k dataset as part of our deep learning coursework.

---

## 🎓 About

Built as a **Deep Learning course project** at **Prasad V Potluri Siddhartha Institute of Technology**.

---

## 🛠️ For Developers

<details>
<summary>Click to expand technical details</summary>

### Training the custom model

```bash
python download_dataset.py
python train.py --images "data/flickr8k/Images" --captions "data/flickr8k/captions.txt" --epochs 20
```

### Architecture
- **Encoder:** ResNet-50 (pretrained on ImageNet) extracts spatial features
- **Decoder:** LSTM with Bahdanau additive attention generates captions word-by-word
- **Web Demo:** Uses Salesforce BLIP for production-quality captions

### Deployment
Upload `app.py` and `requirements.txt` to a Gradio Space or any Python-capable hosting platform. The model downloads automatically on first start.

</details>
