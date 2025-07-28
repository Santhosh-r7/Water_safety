from flask import Flask, jsonify, request
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Change to your MongoDB URI
db = client["mydatabase"]                           # Change to your database name
collection = db["mycollection"]

@app.route('/api/data/<string:id>', methods=['GET'])
def fetch_from_mongo(id):
    try:
        document = collection.find_one({'_id': ObjectId(id)})
        if document:
            return dumps(document)
        else:
            return jsonify({"error": "Not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#get model
MODEL_FOLDER = './models'
ALLOWED_EXTENSIONS = ['.pt', '.pkl', '.h5']  # Adjust to your model types

def get_latest_model_file(folder_path, extensions):
    model_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1] in extensions
    ]

    if not model_files:
        return None

    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

@app.route('/api/latest-model', methods=['GET'])
def latest_model():
    latest_file = get_latest_model_file(MODEL_FOLDER, ALLOWED_EXTENSIONS)
    if latest_file:
        return jsonify({
            "latest_model": os.path.basename(latest_file),
            "full_path": os.path.abspath(latest_file)
        })
    else:
        return jsonify({"error": "No model files found"}), 404
    
def fetch_model():
    return models #list of models

if __name__ == '__main__':
    app.run(debug=True)