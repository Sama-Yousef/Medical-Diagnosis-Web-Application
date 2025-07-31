# 🩺 Comprehensive Medical Diagnosis Web Application

![App Screenshot](./Screenshot%202025-07-16%20042919.png)

An intelligent, AI-powered web application that empowers users with accessible medical tools for early diagnosis, medication insights, chest X-ray analysis, personalized diet tips, and health tracking. Built with **Flask**, **transformer-based models**, and **computer vision**, and deployed using **Docker** on **Microsoft Azure** for scalable and secure access.

---

## 🚀 Key Features

### ✅ Symptom Checker
- Free-text input processed using a Hugging Face transformer model.
- Returns ICD-9 codes and disease names via Groq’s **LLaMA 3** model.

### 💊 Medication Info
- Fetches detailed drug data from the **OpenFDA API** (dosage, side effects, warnings).

### 🌐 X-Ray Diagnosis
- Upload chest radiographs.
- Uses a **Vision Transformer** model to detect conditions like pneumonia.

### 🥗 Diet Recommendations
- Personalized nutritional advice for common chronic conditions.
- Returns local (Egyptian) food examples using Groq API.

### 📊 Health Tips Based on Trends
- Scrapes top health news using **NewsAPI**.
- Converts trends into actionable health advice.

---

## 🧠 Technologies Used

| Category        | Stack                                                   |
|----------------|----------------------------------------------------------|
| Backend         | Flask, Python                                            |
| ML/NLP          | Hugging Face Transformers, PyTorch, PIL                 |
| LLM             | Groq API (LLaMA 3)                                       |
| Computer Vision | ViT (`vit-xray-pneumonia-classification`)               |
| APIs            | OpenFDA API, NewsAPI                                     |
| Frontend        | HTML, CSS, Jinja2                                        |
| Auth/File Upload| Secure file uploads using `Werkzeug`                    |
| **Deployment**  | **Docker + Azure Web App for Containers**               |
