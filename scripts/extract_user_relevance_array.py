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

"""
Extracts the relevancy array from a user's annotations for a RQT pair.

Args:
    user_id (str): The user ID of the user.
    research_question_id (str): The research question ID of the research question.
    transcript_id (str): The transcript ID of the transcript.
    logging (bool): A boolean to determine if the function should print out the results.

Returns:
    relevancy_array (int array): A binary (0/1) array of the user's relevancy annotations.
"""
def extract_user_relevance_array(user_id, research_question_id, transcript_id, logging=False, log_filepath=None):
    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write("Starting extract_user_relevance_array function\n")
    
    # get transcript number of lines
    transcript_data = db.collection("transcriptlineDatas").where(filter=FieldFilter("__transcriptId", "==", transcript_id)).stream()

    for data in transcript_data: # this should only execute once but firebase makes you do it this way ?
        transcript_line_data = data.to_dict()
        transcript_line_data = transcript_line_data["lines"]
        num_lines = transcript_line_data[-1]["__lineNumber"]
        user_relevancy_array = [0] * num_lines
    
        if logging:
            with open(log_filepath, 'a') as log_file:
                log_file.write(f"Number of Lines in Transcript {transcript_id}: {num_lines}\n")
                log_file.write(f"\nExtracting User Annotations from user id {user_id}\n")
    
        # Get all annotations for the user
        user_annotations = db.collection("annotations").where(filter=FieldFilter("__userId", "==", user_id)).where(filter=FieldFilter(
            "__researchQuestionId", "==", research_question_id)).where(filter=FieldFilter("__transcriptId", "==", transcript_id)).where(filter=FieldFilter("isDeleted", "==", False)).stream()

        for user_annotation in user_annotations:
            annotation = user_annotation.to_dict()
            annotation_indices = annotation["_indices"]
            start_line = annotation_indices["startLine"]
            end_line = annotation_indices["endLine"]
            if logging:
                with open(log_filepath, 'a') as log_file:
                    log_file.write(f"Parsing annotation {annotation["__id"]} from line {start_line} to line {end_line}: {annotation["quote"]}\n")
            user_relevancy_array[start_line:end_line+1] = [1] * (end_line - start_line + 1)

        if logging:
            with open(log_filepath, 'a') as log_file:
                log_file.write(f"\nUser Relevancy Array: {user_relevancy_array}\n")
                log_file.write("Ending extract_user_relevance_array function\n")

        return user_relevancy_array

if __name__ == "__main__":
    # initialize db
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cert = os.path.join(script_dir, "../private/dev-admin-key.json")
    cd = credentials.Certificate(cert)
    app = initialize_app(cd)
    db = firestore.client()

    log_filepath = os.path.join(script_dir, 'logging.txt')
    result_file = os.path.join(script_dir, 'user_relevance_array.txt')

    with open(log_filepath, 'w') as log_file:
        log_file.write("Starting extract_user_relevance_array script\n")

    user_id = "VU5I4r8u60VB3NE731mfUp9x7BG3" # neo's user id
    research_question_id = "1aIg3CodFAOoxD73ozzl" # rq 2 challenges and barriers
    transcript_id = "S3TdL6mSCuOFCNG87tMS" # david ayers transcript
    
    user_relevancy_array = extract_user_relevance_array(user_id, research_question_id, transcript_id, logging=True, log_filepath=log_filepath)
    
    print(user_relevancy_array)
    
    with open(result_file, 'w') as result_file:
        result_file.write(str(user_relevancy_array))

