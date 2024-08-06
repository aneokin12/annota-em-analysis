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
import pprint
import sys
import json

def extract_RQT_sentence_prediction_ids(research_question_id, transcript_id, logging=False, log_filepath=None):
    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"\nExtracting Sentence Prediction IDs for Research Question ID: {research_question_id}, Transcript ID: {transcript_id}\n")
    
    sentence_prediction_snapshots = db.collection("sentencePredictionSnapshots").where(filter=FieldFilter("__researchQuestionId", "==", 
research_question_id)).where(filter=FieldFilter("__transcriptId", "==", transcript_id)).stream()
    sentence_prediction_snapshots_list = []

    if sentence_prediction_snapshots is None:
        print(f"No sentence prediction snapshots found for research question ID: {research_question_id} and transcript ID: {transcript_id}")
        sys.exit()

    for sentence_prediction_snapshot in sentence_prediction_snapshots:
        snapshot_data = sentence_prediction_snapshot.to_dict()
        sentence_prediction_snapshots_list.append(snapshot_data)
    
    # Sort the list of dictionaries by creation date
    sorted_snapshots = sorted(sentence_prediction_snapshots_list, key=lambda x: x['_createdAt'])
    
    # Extract the IDs into a separate list
    sorted_snapshot_ids = [snapshot['__id'] for snapshot in sorted_snapshots]

    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"Sorted Sentence Prediction Snapshot IDs: {sorted_snapshot_ids}\n")
    
    if len(sorted_snapshot_ids) == 0:
        print(f"No sentence prediction snapshots found for research question ID: {research_question_id} and transcript ID: {transcript_id}")
        sys.exit()

    return sorted_snapshot_ids

def extract_sentence_prediction_snapshot_relevance(sentence_prediction_snapshot_id, logging=False, log_filepath=None):
    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"\nExtracting Sentence Prediction Relevance for Sentence Prediction ID: {sentence_prediction_snapshot_id}\n")
    
    sentence_prediction = db.collection("sentencePredictionSnapshots").document(sentence_prediction_snapshot_id).get()
    
    if not sentence_prediction.exists:
        print(f"No such sentence prediction snapshot document: {sentence_prediction_snapshot_id}")
        sys.exit()
    
    sentence_prediction_data = sentence_prediction.to_dict()
    sentence_chunks = sentence_prediction_data["_sentenceChunks"]
    num_lines = sentence_chunks[-1]["end"]
    # binary array to store relevance of each sentence
    relevance = [0] * num_lines
    for sentence in sentence_chunks:
        start = sentence["start"]
        end = sentence["end"]
        is_relevant = sentence["isRelevant"]
        if logging:
            with open(log_filepath, 'a') as log_file:
                log_file.write(f"Sentence Chunk: {sentence}, start: {start}, end: {end}, relevance: {is_relevant}\n")
        # loop through the sentence chunk and apply its relevancy status to all lines in the chunk
        # start - 1 because the start index is 1-based?
        for i in range(start - 1, end):
            relevance[i] = 1 if is_relevant else 0
            if logging:
                with open(log_filepath, 'a') as log_file:
                    log_file.write(f"relevance[{i}]: {relevance[i]}\n")
    
    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"Sentence Prediction Snapshot Relevance Array: {relevance}\n")
    return relevance

if __name__ == "__main__":
	# initialize db
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cert = os.path.join(script_dir, "../private/dev-admin-key.json")
    cd = credentials.Certificate(cert)
    app = initialize_app(cd)
    db = firestore.client()

    log_filepath = os.path.join(script_dir, 'logging.txt')
    result_file = os.path.join(script_dir, 'relevance_array.txt')
    
    sentence_prediction_snapshot_id = "uPGKlEduD3kl64r9bVl4"
    research_question_id = "CYHeKZCMMfwQrlUjAGdC"
    transcript_id = "owpnUQpMArcWOcBgOhHM"

    # clear log file
    with open(log_filepath, 'w') as log_file:
        log_file.write("Starting Extract Sentence Prediction Relevance Script\n")
    
    relevance_array = []

    # get relevance array for all sentence prediction snapshots
    sentence_prediction_snapshot_ids = extract_RQT_sentence_prediction_ids(research_question_id, transcript_id, logging=True, log_filepath=log_filepath)

    for sentence_prediction_snapshot_id in sentence_prediction_snapshot_ids:
        relevance_array.append(extract_sentence_prediction_snapshot_relevance(sentence_prediction_snapshot_id, logging=True, log_filepath=log_filepath))

    # get ground truth relevance array (last object/em run)
    # relevance_array = extract_sentence_prediction_snapshot_relevance("cT1DcR4fT7AGXlhDlpl8", logging=True, log_filepath=log_filepath)

    with open(result_file, 'w') as outfile:
        json.dump(relevance_array, outfile)