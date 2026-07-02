\# 😊 Real-Time Emotion Detection System



<div align="center">



!\[Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python\&logoColor=white)

!\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge\&logo=tensorflow\&logoColor=white)

!\[OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge\&logo=opencv\&logoColor=white)

!\[Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge\&logo=streamlit\&logoColor=white)

!\[Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge\&logo=keras\&logoColor=white)



\*\*A real-time facial emotion recognition system powered by a Convolutional Neural Network (CNN) trained on the FER-2013 dataset. Detects 7 human emotions live from webcam with animated confidence bars in a Streamlit web app.\*\*



\[Features](#-features) • \[Demo](#-demo) • \[Tech Stack](#-tech-stack) • \[Project Structure](#-project-structure) • \[Installation](#-installation) • \[How It Works](#-how-it-works) • \[Results](#-results) • \[Author](#-author)



</div>



\---



\## 🎯 Features



\- 🎥 \*\*Real-time webcam inference\*\* — detects emotions live from your camera

\- 🧠 \*\*CNN trained from scratch\*\* — 3-block convolutional architecture on FER-2013

\- 😊 \*\*7 emotion classes\*\* — Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral

\- 📊 \*\*Live confidence bars\*\* — animated probability bars for all 7 emotions simultaneously

\- 🎯 \*\*65.55% test accuracy\*\* — competitive with published benchmarks on FER-2013

\- 🖥️ \*\*Streamlit web app\*\* — clean dark dashboard UI, no setup needed to run

\- 👤 \*\*Multi-face support\*\* — detects and classifies multiple faces in one frame

\- ⚡ \*\*Optimized inference\*\* — processes every 3rd frame for smooth real-time performance



\---



\## 🎬 Demo 
[e8ef35d8-4f13-4337-97e8-01ac90791489.webm](https://github.com/user-attachments/assets/3c707c69-097d-4d30-b333-254c5cb843c1)

\---



\## 🛠️ Tech Stack



| Technology | Version | Purpose |

|---|---|---|

| Python | 3.10+ | Core programming language |

| TensorFlow | 2.x | Deep learning framework |

| Keras | Built-in | Neural network API |

| OpenCV | 4.x | Webcam access + face detection |

| Streamlit | 1.x | Web app dashboard |

| NumPy | Latest | Array operations + preprocessing |

| Matplotlib | Latest | Training history visualization |

| Scikit-learn | Latest | Class weights + evaluation |

| Pillow | Latest | Image file handling |

| FER-2013 | — | Training dataset (Kaggle) |



\---



\## 📁 Project Structure

\---



\## ⚙️ Installation



\### Prerequisites

\- Python 3.10 or higher

\- Webcam connected to your system

\- Kaggle account (to download dataset)



\### Step 1 — Clone the repository

```bash

git clone https://github.com/himanshunath007/emotion-detection.git

cd emotion-detection

```



\### Step 2 — Create virtual environment

```bash

python -m venv venv



\# Windows

venv\\Scripts\\activate



\# Mac/Linux

source venv/bin/activate

```



\### Step 3 — Install dependencies

```bash

pip install -r requirements.txt

```



\### Step 4 — Download FER-2013 dataset

Set up your Kaggle API credentials then run:

```bash

kaggle datasets download -d msambare/fer2013

Expand-Archive -Path fer2013.zip -DestinationPath dataset

```



\### Step 5 — Preprocess the dataset

```bash

python preprocess.py

```



\### Step 6 — Train the model

```bash

python train\_model.py

```

> Training takes approximately 5-8 hours on CPU. Final accuracy: \~65.55%



\### Step 7 — Run the Streamlit app

```bash

streamlit run app.py

```



Open your browser at `http://localhost:8501`



\---



\## 🧠 How It Works



\### Pipeline Overview

\*\*Total parameters:\*\* \~5.2 million



\### Training Configuration



| Parameter | Value |

|---|---|

| Optimizer | Adam (lr=0.001) |

| Loss function | Categorical Crossentropy |

| Batch size | 64 |

| Epochs | 50 (EarlyStopping) |

| Best epoch | 45 |

| Class weighting | Balanced (handles class imbalance) |

| LR scheduler | ReduceLROnPlateau (factor=0.5) |



\---



\## 📊 Results



\### Final Performance



| Metric | Value |

|---|---|

| Test Accuracy | \*\*65.55%\*\* |

| Test Loss | 1.9401 |

| Training Accuracy | 98.25% |

| Best Epoch | 45 / 50 |

| Total Training Time | \~7 hours (CPU) |



\### Dataset Distribution



| Emotion | Train Images | Test Images |

|---|---|---|

| Happy | 7,215 | 1,774 |

| Neutral | 4,965 | 1,233 |

| Sad | 4,830 | 1,247 |

| Fear | 4,097 | 1,024 |

| Angry | 3,995 | 958 |

| Surprise | 3,171 | 831 |

| Disgust | 436 | 111 |

| \*\*Total\*\* | \*\*28,709\*\* | \*\*7,178\*\* |



\### Benchmark Comparison



| Model | Accuracy |

|---|---|

| This project (CNN from scratch) | \*\*65.55%\*\* |

| Published FER-2013 benchmarks | 65–72% |

| Human performance on FER-2013 | \~65% |



> Note: FER-2013 is known to be a challenging dataset with noisy labels. Human-level performance is estimated at \~65%, making this model competitive.



\---



\## 🚀 Skills Demonstrated



\- \*\*Deep Learning\*\* — CNN design, training, evaluation from scratch

\- \*\*Computer Vision\*\* — Real-time face detection, image preprocessing

\- \*\*Data Engineering\*\* — Dataset download, normalization, augmentation handling

\- \*\*Model Deployment\*\* — Streamlit web app with live inference

\- \*\*Python\*\* — OOP, file I/O, NumPy, OpenCV, TensorFlow/Keras

\- \*\*MLOps basics\*\* — Model checkpointing, early stopping, LR scheduling

\- \*\*Version Control\*\* — Git workflow, GitHub, meaningful commit history



\---



\## 🔮 Future Improvements



\- \[ ] Add data augmentation to improve generalization

\- \[ ] Implement transfer learning with VGG16 or ResNet50

\- \[ ] Deploy to cloud (AWS EC2 / Streamlit Cloud)

\- \[ ] Add emotion history graph over time

\- \[ ] Support video file input in addition to webcam

\- \[ ] Mobile-friendly UI with responsive design

\- \[ ] Real-time emotion logging to CSV



\---



\## 👨‍💻 Author



\*\*Himanshu\*\*

\- 🎓 B.Tech Computer Science Engineering (Cloud Computing)

\- 🏫 VIT Bhopal University — Batch 2024–2028

\- 💻 GitHub: \[@himanshunath007](https://github.com/himanshunath007)

\- 📧 himanshu2005nath@gmail.com



\---



\## 📄 License



This project is open source and available under the \[MIT License](LICENSE).



\---



<div align="center">



\*\*⭐ If you found this project helpful, please give it a star on GitHub! ⭐\*\*



\*Built with ❤️ by Himanshu — VIT Bhopal\*



</div>

