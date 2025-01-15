# DEVELOPMENT
if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\code\backend-sampahmas-deteksi\serviceAccount.json'

service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if service_account_path is None:
    raise ValueError("Environment variable GOOGLE_APPLICATION_CREDENTIALS not set")

cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'sampahmas-3a4f0.appspot.com'
})