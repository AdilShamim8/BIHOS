<div align="center">
  <img src="https://img.icons8.com/color/96/000000/medical-doctor.png" alt="Logo" width="80" height="80">
  <h1 align="center">BIHOS</h1>
  <p align="center">
    <strong>CUET AI Health Screening System</strong>
    <br />
    <em>An AI-powered, multi-modal diagnostic assistant for screening dermatological, ocular, and ungual conditions using EfficientNetB2 FP16.</em>
    <br />
    <br />
    <a href="#-about-the-project">About</a>
    ·
    <a href="#-ai-models--capabilities">Models</a>
    ·
    <a href="#-getting-started">Installation</a>
    ·
    <a href="#-tech-stack">Tech Stack</a>
  </p>

  <!-- Badges -->
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
    <img src="https://img.shields.io/badge/TensorFlow-Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow Lite" />
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/Status-Hackathon_Submission-brightgreen?style=for-the-badge" alt="Hackathon" />
  </p>
</div>

---

## 📖 About The Project

**BIHOS (CUET AI Health Screening System)** is a comprehensive, AI-driven diagnostic screening tool designed to provide rapid, accessible, and high-accuracy preliminary analysis for various health conditions. Developed as a **Hackathon Project Submission**, BIHOS leverages state-of-the-art Deep Learning techniques to analyze medical imagery across four distinct domains: **Eye, Hair/Scalp, Nail, and Skin**.

By integrating **EfficientNetB2** models optimized with **FP16 precision** via TensorFlow Lite, the system ensures real-time inference without compromising accuracy, making it lightweight enough to run efficiently on standard CPUs.

> **Note:** This project is built with a stunning **Premium Dark Glassmorphism Theme** to provide a seamless, modern, and engaging user experience.

---

## ✨ Key Features

- **Multi-Modal Screening:** Analyzes 4 different health domains (Eye, Hair, Nail, Skin) using dedicated AI models.
- **High Performance:** Utilizes EfficientNetB2 architecture compressed to TFLite (FP16) for fast, resource-efficient CPU inference.
- **Granular Classification:** Capable of identifying over **35 distinct disease classes** with high confidence.
- **Risk Severity Mapping:** Automatically categorizes predictions into severity levels (Low, Medium, High, Critical) with intuitive visual indicators (🟡, 🟠, 🔴, 🆘).
- **Beautiful UI/UX:** Built with Streamlit but heavily customized using vanilla CSS to feature a dark glassmorphism aesthetic, smooth micro-animations, and responsive design.
- **Privacy First:** Entirely local inference. No medical imagery is sent to external servers, ensuring complete data privacy.

---

## 🧠 AI Models & Capabilities

BIHOS incorporates four independent, highly specialized neural networks:

### 👁️ Eye Disease AI
- **Description:** Screens retinal fundus images or clear frontal eye photographs.
- **Classes:** Cataract, Diabetic Retinopathy, Glaucoma, Normal / Healthy Eye.

### 💇 Hair & Scalp Disease AI
- **Description:** Identifies various scalp and hair conditions from close-up photos.
- **Classes:** Alopecia Areata, Androgenetic Alopecia, Telogen Effluvium, Dandruff, Tinea Capitis, Trichotillomania, Folliculitis, Lichen Planopilaris, etc.

### 💅 Nail Disease AI
- **Description:** Detects ungual conditions from fingernail or toenail close-ups.
- **Classes:** Onychomycosis (Fungal), Nail Psoriasis, Paronychia, Melanonychia, Onycholysis, Normal.

### 🧴 Skin Disease AI
- **Description:** Classifies dermatological conditions from photos of rashes, lesions, and discolorations.
- **Classes:** Melanoma, Acne, Eczema, Psoriasis, Lupus, Cellulitis, Vasculitis, Herpes HPV, and many more (20 classes total).

---

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/) + Custom CSS (Glassmorphism, Google Fonts: Inter & Space Grotesk)
- **Deep Learning Framework:** [TensorFlow](https://www.tensorflow.org/) / TensorFlow Lite
- **Model Architecture:** EfficientNetB2 (FP16 Quantized)
- **Image Processing:** [Pillow (PIL)](https://python-pillow.org/) & [NumPy](https://numpy.org/)

---

## 🚀 Getting Started

Follow these instructions to get a local copy of the project up and running on your machine.

### Prerequisites

Ensure you have **Python 3.10, 3.11, or 3.12** installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bihos-cuet-ai.git
   cd bihos-cuet-ai
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv .venv
   ```
   - On Windows: `.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Models are Placed Correctly**
   Ensure that the `Models/` directory exists in the root folder and contains the following TFLite files:
   - `eye_efficientnetb2_fp16.tflite`
   - `hair_efficientnetb2_fp16.tflite`
   - `nail_efficientnetb2_fp16.tflite`
   - `skin_efficientnetb2_fp16.tflite`

### Running the Application

Launch the Streamlit server using the following command:

```bash
streamlit run streamlit_app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

---

## 📸 Screenshots

*(Replace these placeholders with actual screenshots of your application prior to hackathon submission)*

| Home Dashboard | Inference & Results |
| :---: | :---: |
| <img src="https://via.placeholder.com/600x350/0f1320/ffffff?text=Home+Screen" width="400" /> | <img src="https://via.placeholder.com/600x350/0f1320/ffffff?text=Inference+Results" width="400" /> |

---

## ⚠️ Disclaimer

**Research & Educational Use Only.**

BIHOS is developed at CUET for research and hackathon demonstration purposes. It is **NOT** a certified medical device and must **NOT** be used for clinical diagnosis, medical triage, or as a substitute for professional medical advice. Always consult a licensed and qualified healthcare provider for medical diagnosis and treatment.

---

## 🤝 Team / Contributors

- **Your Name / Team Name** - [GitHub Profile](https://github.com/yourusername) - Role/Contribution

If you'd like to contribute, feel free to fork the repository and submit a pull request!

---

## 📜 Attributions & Acknowledgements

In compliance with SciBlitz AI Challenge 2026 rules (Section 10.2), we attribute the following third-party resources used in this project:
- **Models:** Built upon the [EfficientNetB2](https://arxiv.org/abs/1905.11946) architecture.
- **Datasets:** *(Please fill in the names/links of the datasets used for training Eye, Hair, Nail, and Skin models, e.g., ISIC, ODIR, etc.)*
- **Frameworks & Libraries:** 
  - [TensorFlow / TFLite](https://www.tensorflow.org/) (Apache License 2.0)
  - [Streamlit](https://streamlit.io/) (Apache License 2.0)
  - [Pillow (PIL)](https://python-pillow.org/) (HPND License)
  - [NumPy](https://numpy.org/) (BSD 3-Clause)
- **Assets:** Icons provided by [Icons8](https://icons8.com).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<p align="center">
  <i>Crafted with ❤️ for the AI Hackathon.</i>
</p>
