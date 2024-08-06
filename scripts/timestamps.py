from firebase_admin import initialize_app, credentials, firestore
import os
from google.cloud.firestore import FieldFilter

script_dir = os.path.dirname(os.path.realpath(__file__))
# cert = os.path.join(script_dir, "../private/dev-admin-key.json")
cd = credentials.Certificate("../private/dev-admin-key.json")
app = initialize_app(cd)

db = firestore.Client()


def find_em_timestamps(transcriptID):
    sentence_predictions_ref = db.collection("sentencePredictionSnapshots")
    query = sentence_predictions_ref.where(filter=FieldFilter("__transcriptId", "==", transcriptID)).stream()

    timestamps = {}
    for doc in query:
        timestamps[doc.to_dict()["__id"]] = doc.to_dict()["_createdAt"]
    
    return timestamps

def find_ann_timestamps(transcriptID):
    ann_ref = db.collection("annotations")
    query = ann_ref.where(filter=FieldFilter("__transcriptId", "==", transcriptID)).stream()

    timestamps = {}
    for doc in query:
        timestamps[doc.to_dict()["__id"]] = doc.to_dict()["_updatedAt"]
    
    return timestamps