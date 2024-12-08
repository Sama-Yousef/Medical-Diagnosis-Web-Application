from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoImageProcessor, AutoModelForImageClassification
import torch
from groq import Groq

app = Flask(__name__)

# Symptoms checker class
class Symptoms:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ashishkgpian/betty_icd9_classifier_ehr_symptoms_text_icd9_150_epochs")
        self.model = AutoModelForSequenceClassification.from_pretrained("ashishkgpian/betty_icd9_classifier_ehr_symptoms_text_icd9_150_epochs")

    def symptoms_checker(self, symptoms):
        inputs = self.tokenizer(symptoms, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        predicted_label = logits.argmax(dim=-1).item()
        predicted_icd9_code = self.model.config.id2label[predicted_label]

        # Groq integration for additional details
        API_KEY = "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"
        client = Groq(api_key=API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"ICD-9 code {predicted_icd9_code}, provide the disease name only."
            }],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=50
        )
        return chat_completion.choices[0].message.content

# Chest X-ray checker class
class ChestXrayChecker:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("lxyuan/vit-xray-pneumonia-classification")
        self.model = AutoModelForImageClassification.from_pretrained("lxyuan/vit-xray-pneumonia-classification")

    def check_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_idx = logits.argmax(-1).item()
        return self.model.config.id2label[predicted_class_idx]
    


from groq import Groq  # Ensure this is the correct import

class DietRecommend:
    def __init__(self):
        # Initialize with the API key, which can be provided when creating an instance
        self.api_key = "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"
        self.client = Groq(api_key=self.api_key)

    def recommend_diet(self):
        # Ensure diagnosis is passed as a parameter
        diagnosis = ['Type 2 diabetes', 'hypertension', 'high cholesterol']
        
        # Convert the diagnosis list to a nicely formatted string
        diagnosis_str = "\n".join([f"- {item}" for item in diagnosis])
        
        # API call to Groq for diet recommendation
        chat_completion = self.client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Imagine you are an expert in nutrition and dietetics. Based on the following list of diagnoses, provide a tailored dietary recommendation that focuses on promoting healing, maintaining optimal health, and managing the specific conditions. Be sure to explain the reasoning behind each food choice and avoid suggesting foods that may worsen the conditions listed. Please provide the list of valid Egyptian foods only in Arabic.  The diagnoses are as follows:\n{diagnosis_str}"
            }],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=150  # Adjusted to give a longer response
        )

        # Debugging step: Print the structure of chat_completion to inspect it
        return chat_completion.choices[0].message.content





symptoms_checker = Symptoms()
chest_checker = ChestXrayChecker()
diet_recommendation = DietRecommend()




@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    if "symptoms" in request.form:
        symptoms = request.form["symptoms"]
        result = symptoms_checker.symptoms_checker(symptoms)
        return render_template("result.html", result=result, type="Symptoms Diagnosis")
    
    elif "file" in request.files:
        file = request.files["file"]
        if file:
            image = Image.open(file.stream).convert("RGB")
            result = chest_checker.check_image(image)
            return render_template("result.html", result=result, type="Chest X-Ray Diagnosis")

    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
