import warnings
import numpy as np
import firebase_admin
from firebase_admin import initialize_app, credentials, firestore
from firebase_functions import firestore_fn, options
import os
import google.cloud.firestore
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client
from google.cloud.firestore import FieldFilter
from collections import defaultdict
from datetime import timedelta
import pprint

cd = credentials.Certificate("../private/dev-admin-key.json")
app = initialize_app(cd)
db = firestore.Client()

def sort_annotations(timestamps):
    timestamps_sorted = sorted(timestamps, key=lambda x: x[1])
    earliest_datetime = timestamps_sorted[0][1]
    limit_datetime = earliest_datetime + timedelta(days=1)

    first_24_hours = []
    later_annotations = []

    for annotation_id, datetime_obj in timestamps_sorted:
        if datetime_obj < limit_datetime:
            first_24_hours.append((annotation_id, datetime_obj))
        else:
            later_annotations.append((annotation_id, datetime_obj))

    return [first_24_hours, later_annotations]

def user_annotations(transcriptID):
    ann_ref = db.collection("annotations")
    annotations = {}
    query = ann_ref.where(filter=FieldFilter("__transcriptId", "==", transcriptID)).stream()
    for doc in query:
        doc_data = doc.to_dict()
        user_id = doc_data["__userId"]
        research_question_id = doc_data["__researchQuestionId"]
        annotation_id = doc_data["__id"]
        updated_at = doc_data["_updatedAt"]
        
        if user_id not in annotations:
            annotations[user_id] = {research_question_id: [(annotation_id, updated_at)]}
        else:
            if research_question_id not in annotations[user_id]:
                annotations[user_id][research_question_id] = [(annotation_id, updated_at)]
            else:
                annotations[user_id][research_question_id].append((annotation_id, updated_at))
    for user in annotations:
        for research_question in annotations[user]:
            annotations[user][research_question] = sort_annotations(annotations[user][research_question])

    return annotations

pprint.pp(user_annotations("owpnUQpMArcWOcBgOhHM"))