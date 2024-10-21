from firebase_admin import initialize_app, credentials, firestore
import os
from google.cloud.firestore import FieldFilter

script_dir = os.path.dirname(os.path.realpath(__file__))
# cert = os.path.join(script_dir, "../private/dev-admin-key.json")
cd = credentials.Certificate("../private/dev-admin-key.json")
app = initialize_app(cd)

db = firestore.Client()

# input transcriptID and binary relevance array that marks relevant sentences as 1 and 0 for everything else (not relevant)
def print_relevant(transcriptID, relevance):
    line_data = db.collection("transcriptlineDatas")
    query = line_data.where(filter=FieldFilter("__transcriptId", "==", transcriptID)).stream()
    for doc in query:
        for line in doc.to_dict()["lines"]:
            if relevance[line["__lineNumber"]] == 1:
                print(str(line["__lineNumber"]) + ": " + line["text"] + "\n")

