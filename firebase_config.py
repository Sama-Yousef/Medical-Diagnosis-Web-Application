import firebase_admin
from firebase_admin import credentials, firestore, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate("path_to_your_firebase_credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-project-id.appspot.com'
})

# Firestore client
db = firestore.client()

# Storage bucket
bucket = storage.bucket()
