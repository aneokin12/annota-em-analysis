from firebase_admin import initialize_app, credentials, firestore
import os
from google.cloud.firestore import FieldFilter

script_dir = os.path.dirname(os.path.realpath(__file__))
# cert = os.path.join(script_dir, "../private/dev-admin-key.json")
cd = credentials.Certificate("../private/dev-admin-key.json")
app = initialize_app(cd)

db = firestore.Client()

def most_recent_user_annotations(transcriptID, rqID, sentence_prediction_snapshot_id):
    annotations_ref = db.collection("annotations")
    query = annotations_ref.where(filter=FieldFilter("__transcriptId", "==", transcriptID)).where(filter=FieldFilter("__researchQuestionId", "==", rqID)).where(filter=FieldFilter("isDeleted", "==", False)).where(filter=FieldFilter("round", "==","FINAL")).stream()

    annotations = []

    # docs = list(query)
    # print(docs)
    for doc in query:
        annotations.append(doc.to_dict())
    
    # sorted_annotations = sorted(annotations, key=lambda x: x["_updatedAt"], reverse=True)

    sentence_prediction = db.collection("sentencePredictionSnapshots").document(sentence_prediction_snapshot_id).get()
    num_lines = sentence_prediction.to_dict()["_sentenceChunks"][-1]["end"]

    majority_vote = [0] * num_lines
    visited = set()
    students = set()

    for annotation in annotations:
        user_id = annotation["__userId"]
        start = annotation["_indices"]["startLine"]
        end = annotation["_indices"]["endLine"]
        print("recent timestamp =", annotation["_updatedAt"])
        students.add(user_id)
        for i in range(start, end + 1):
            if (user_id, i) not in visited:
                majority_vote[i] += 1
                visited.add(i)
    
    print("num_students =", len(students))
    print("vote count (most recent) =", majority_vote)
    majority = len(students) // 2
    for i in range(num_lines):
        if majority_vote[i] > majority:
            majority_vote[i] = 1
        else:
            majority_vote[i] = 0

    return majority_vote

def earliest_user_annotations(transcriptID, rqID, sentence_prediction_snapshot_id):
    annotations_ref = db.collection("annotations")
    query = annotations_ref.where(filter=FieldFilter("__transcriptId", "==", transcriptID)).where(filter=FieldFilter("__researchQuestionId", "==", rqID)).where(filter=FieldFilter("isDeleted", "==", False)).stream()

    annotations = []

    # docs = list(query)
    # print(docs)
    for doc in query:
        annotations.append(doc.to_dict())
    
    sorted_annotations = sorted(annotations, key=lambda x: x["_createdAt"])

    sentence_prediction = db.collection("sentencePredictionSnapshots").document(sentence_prediction_snapshot_id).get()
    num_lines = sentence_prediction.to_dict()["_sentenceChunks"][-1]["end"]

    majority_vote = [0] * num_lines
    visited = set()
    students = set()

    for annotation in sorted_annotations:
        user_id = annotation["__userId"]
        start = annotation["_indices"]["startLine"]
        end = annotation["_indices"]["endLine"]
        print("earliest timestamp =", annotation["_createdAt"])
        students.add(user_id)
        for i in range(start, end + 1):
            if (user_id, i) not in visited:
                majority_vote[i] += 1
                visited.add(i)
    
    print("num_students =", len(students))
    print("vote count (earliest) =", majority_vote)
    majority = len(students) // 2
    for i in range(num_lines):
        if majority_vote[i] > majority:
            majority_vote[i] = 1
        else:
            majority_vote[i] = 0

    return majority_vote


most_recent = most_recent_user_annotations("RGcUaOIhqwnFjeDXHUH2", "CYHeKZCMMfwQrlUjAGdC", "0RFnsurqtfCZZuZS8mS5")
# earliest = earliest_user_annotations("RGcUaOIhqwnFjeDXHUH2", "CYHeKZCMMfwQrlUjAGdC", "0RFnsurqtfCZZuZS8mS5")

# print(most_recent == earliest)
print("binary votes (most recent) =", most_recent)
# print("binary votes (earliest) =", earliest)

    