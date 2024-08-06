from firebase_admin import initialize_app, credentials, firestore
import os
from google.cloud.firestore import FieldFilter

script_dir = os.path.dirname(os.path.realpath(__file__))
# cert = os.path.join(script_dir, "../private/dev-admin-key.json")
cd = credentials.Certificate("../private/dev-admin-key.json")
app = initialize_app(cd)

db = firestore.Client()

def assign_line_scores(transcriptID):

    annotations_ref = db.collection("annotations")
    query = annotations_ref.where(filter=FieldFilter("__transcriptId", "==", transcriptID)).stream()

    line_scores = {"before":{}, "after":{}}
    users = []
    for doc in query:
        date = doc.to_dict()["_updatedAt"]
        
        if (date.year, date.month, date.day) < (2024, 4, 8):
            for i in range(doc.to_dict()["_indices"]["startLine"], doc.to_dict()["_indices"]["endLine"] + 1):
                if i not in line_scores["before"]:
                    line_scores["before"][i] = 1
                else:
                    line_scores["before"][i] += 1
            users.append(doc.to_dict()["__userId"])
    
    annotations_ref = db.collection("annotations")
    query = annotations_ref.where(filter=FieldFilter("__transcriptId", "==", transcriptID)).stream()
    for doc in query:
        date = doc.to_dict()["_updatedAt"]
        if (date.year, date.month, date.day) > (2024, 4, 8):
            if doc.to_dict()["__userId"] in users:
                for i in range(doc.to_dict()["_indices"]["startLine"], doc.to_dict()["_indices"]["endLine"] + 1):
                    if i not in line_scores["after"]:
                        line_scores["after"][i] = 1
                    else:
                        line_scores["after"][i] += 1
    
    return line_scores