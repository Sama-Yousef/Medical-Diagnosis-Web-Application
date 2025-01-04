from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoImageProcessor, AutoModelForImageClassification
import torch
from groq import Groq
import requests
import os

from werkzeug.utils import secure_filename
from PIL import Image

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



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

class MedicationInfo:
    @staticmethod
    def fetch_medication_info(medication_name):
        try:
            api_url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{medication_name}"
            response = requests.get(api_url)

            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    result = data["results"][0]
                    return {
                        "generic_name": result.get("openfda", {}).get("generic_name", ["N/A"])[0],
                        "brand_name": result.get("openfda", {}).get("brand_name", ["N/A"])[0],
                        "dosage": result.get("dosage_and_administration", ["N/A"])[0],
                        "side_effects": result.get("adverse_reactions", ["N/A"])[0],
                        "warnings": result.get("warnings", ["N/A"])[0],
                    }
                return {"error": "Medication not found in the OpenFDA database."}
            elif response.status_code == 404:
                return {"error": "Medication not found. Try using the generic name instead of the brand name."}
            else:
                return {"error": f"API error: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

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


class DietRecommend:
    def __init__(self):
        self.api_key = "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"
        self.client = Groq(api_key=self.api_key)

    def recommend_diet(self):
        diagnosis = ['Type 2 diabetes', 'hypertension', 'high cholesterol']
        diagnosis_str = "\n".join([f"- {item}" for item in diagnosis])
        
        chat_completion = self.client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Imagine you are an expert in nutrition and dietetics. Based on the following list of diagnoses, provide a tailored dietary recommendation that focuses on promoting healing, maintaining optimal health, and managing the specific conditions. Please provide the list of valid Egyptian foods . The diagnoses are as follows:\n{diagnosis_str}. give me the response in organized and readable with clear structure in points each point in new line in the begin of each new line write -  "
            }],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=150
        )
        sections = chat_completion.choices[0].message.content.split("\n\n")
        organized_text = ""
        for section in sections:
            # Add new line between items in the list
            section = section.replace("-", "\n")
            section = section.replace(":", ":\n")
            section = section.replace("**", ":\n")
            # Add new lines before section titles

            organized_text += section + "\n\n"

        return organized_text
    


    
class TodayTips:
    def __init__(self):
        self.api_key = "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"
        self.client = Groq(api_key=self.api_key)

    def Today_Tips(self):
        # Replace with a real health trends source or news API, like 'https://newsapi.org/'
        # Here we'll use a sample public API for health news, like NewsAPI (you would need an API key)
        NewsAPI_KEY = "f3ec3d7d9d4e425da6adf62e7b675945"  # Replace with your NewsAPI key
        url = f'https://newsapi.org/v2/top-headlines?category=health&apiKey={NewsAPI_KEY}'

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Extract the titles of the latest health trends
            health_trends = [article['title'] for article in data['articles']]
        else:
            health_trends= "Unable to fetch health trends"

        # Retrieve the API key for Groq from environment variable
        Groq_API_KEY = "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"  # Ensure your API_KEY is set as an environment variable



        # # Print the fetched health trends
        # print("Today's health trends:")
        # for trend in health_trends:
        #     print(f"- {trend}")

        # Initialize the Groq client
        client = Groq(api_key=Groq_API_KEY)

        # Use the health trends in the Groq API request
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Here are some of today's health trends: {', '.join(health_trends)}. based on it Can you tell me tips to avoide the deseas and be healthy?"
            }],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=100
        )

        # Print the response from Groq API
        return(chat_completion.choices[0].message.content)

       
        
        


        

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/check')
def index():
    return render_template('index.html')

@app.route('/xray')
def xray():
    return render_template('index - Copy.html')


@app.route('/medication')
def medication():
    return render_template('index - Copy (2).html')


@app.route('/check_symptoms', methods=['POST'])
def check_symptoms():
    symptoms = request.form['symptoms']
    print(symptoms)
    symptoms_instance = Symptoms() 
    result = symptoms_instance.symptoms_checker(symptoms)
    return render_template('result.html', result=result, symptom=symptoms)

# @app.route("/", methods=["GET", "POST"])
# def check_symptoms():
#     if request.method == "POST":
#         symptoms = request.form["symptoms"]
#         result = Symptoms.symptoms_checker(symptoms)
#         return render_template("result.html", result=result)
#     return render_template("index.html")



@app.route('/check_medication', methods=['POST'])
def check_medication():
    medication_name = request.form['medication_name']
    result = MedicationInfo.fetch_medication_info(medication_name)  # Corrected here
    return render_template('medication_result.html', result=result)



@app.route('/check_xray', methods=['POST'])
def check_xray():
    if 'xray_image' not in request.files:
        return "No file part in the request", 400
    file = request.files['xray_image']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        # Secure and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Create the upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        try:
            file.save(file_path)  # Save the file
            # Open the saved file as a PIL Image
            image = Image.open(file)

            # Diagnose the X-ray using ChestXrayChecker
            chest_xray_checker = ChestXrayChecker()
            predicted_label = chest_xray_checker.check_image(image)
            print("Predicted Label: ", predicted_label)

            # Render the result template with the image and label
            return render_template(
                'xray_result.html', 
                image_path=f'uploads/{filename}', 
                predicted_label=predicted_label
            )
        except Exception as e:
            print(f"Error processing the image: {e}")
            return "An error occurred while processing the image. Please try again.", 500
    else:
        return "Invalid file type", 400

        

@app.route('/get_diet_recommendation', methods=['POST'])
def get_diet_recommendation():
    Diet_Recommend=DietRecommend()
    diet = Diet_Recommend.recommend_diet()
    return render_template('diet_result.html', diet=diet)


@app.route('/tips', methods=['GET'])
def trend_tips():
    Today_Tips=TodayTips()
    tips = Today_Tips.Today_Tips()
    return render_template('trend_tips.html', tips=tips)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

