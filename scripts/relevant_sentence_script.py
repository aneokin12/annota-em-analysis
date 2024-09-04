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
Categorize relevant and nonrelevant sentences in a sentence prediction.

Args:
    sentence_prediction_id (str): The id of the sentence prediction to categorize.
    logging (bool): A boolean to determine if the function should print out the results.

Returns:
    relevant_sentence_predictions (list): A list of tuples containing the start and end indices of relevant sentences.
    nonrelevant_sentence_predictions (list): A list of tuples containing the start and end indices of nonrelevant sentences.
"""


def categorize_relevant_sentences(sentence_prediction_id, logging=False, log_filepath=None):
    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"\nCategorizing Relevant Sentences for Sentence Prediction ID: {sentence_prediction_id}\n")

    sentence_prediction = db.collection(
        "sentencePredictions").document(sentence_prediction_id).get()

    if not sentence_prediction.exists:
        print(f"No such sentence prediction document: {sentence_prediction_id}")
        sys.exit()

    relevant_sentence_predictions = []
    nonrelevant_sentence_predictions = []

    sentence_chunks = sentence_prediction.to_dict()["_sentenceChunks"]

    for sentence in sentence_chunks:
        sentence_indices = (sentence["start"], sentence["end"])
        if sentence["isRelevant"]:
            relevant_sentence_predictions.append(sentence_indices)
        else:
            nonrelevant_sentence_predictions.append(sentence_indices)

    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"""
Relevant Sentences: {relevant_sentence_predictions}
Nonrelevant Sentences: {nonrelevant_sentence_predictions}\n""")

    return relevant_sentence_predictions, nonrelevant_sentence_predictions


"""
Extract the start and end lines of a sentence from an annotation.

Args:
    annotation_id (str): The id of the annotation to extract the sentence from.
    logging (bool): A boolean to determine if the function should print out the results.

Returns:
    sentence_start (int): The start line of the sentence.
    sentence_end (int): The end line of the sentence.
"""


def extract_annotation_sentence(annotation_id, logging=False, log_filepath=None):
    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"\nExtracting Annotation Interval from Annotation ID: {annotation_id}\n")

    annotation = db.collection("annotations").document(annotation_id).get()

    if not annotation.exists:
        print(f"No such annotation document: {annotation_id}")
        sys.exit()

    annotation_data = annotation.to_dict()
    sentence = annotation_data["_indices"]
    sentence_start, sentence_end = sentence["startLine"], sentence["endLine"]

    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"Annotation Start Line: {sentence_start}, End Line: {sentence_end}\n")

    return sentence_start, sentence_end


"""
Compare the relevance of annotations to the relevance of sentences.

Args:
    annotation_list (list): A list of annotation ids to compare.
    relevant_sentence_predictions (list): A list of tuples containing the start and end indices of relevant sentences.
    logging (bool): A boolean to determine if the function should print out the results.

Returns:
    relevance (int): The relevance score of the annotations.
"""


def compare_annotations_relevance(annotation_list, relevant_sentence_predictions, logging=False, log_filepath=None):
    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"\nMatching Relevant Sentences to {str(len(annotation_list))} Annotations: {annotation_list}\n")

    relevance = 0
    for annotation_id in annotation_list:
        annotation_start, annotation_end = extract_annotation_sentence(
            annotation_id, logging=True, log_filepath=log_filepath)
        for relevant_sentence in relevant_sentence_predictions:
            interval_end = min(annotation_end, relevant_sentence[1])
            interval_start = max(annotation_start, relevant_sentence[0])
            # + 1 to account for ending index being inclusive, i.e. (90, 91) is 2 lines of relevancy, not 91 - 90 = 1 line of relevancy
            overlap = interval_end - interval_start + 1
            if overlap > 0:
                relevance += overlap
                if logging:
                    with open(log_filepath, 'a') as log_file:
                        log_file.write(f"Matched Annotation: {annotation_id} Overlap: {overlap}, Relevance: {relevance}\n")
    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"\n===========\nFinal Relevance Score: {relevance} Lines of Relevance from {str(len(annotation_list))} Annotations\n===========\n")
    return relevance


"""
Compare the relevance score change of user's annotations before and after a day.

Args:
    user_annotations (dict): A dictionary containing the user annotations for each research question, in the format {'userId': {'researchQuestion': [[(priorAnnotationId, annotationTimestamp), ...], [(postAnnotationId, annotationTimestamp), ...]]}}.
    relevant_sentence_predictions (list): A list of tuples containing the start and end indices of relevant sentences.
    logging (bool): A boolean to determine if the function should print out the results.

Returns:
    relevance_changes (list): A list of tuples containing the prior and post relevancy scores for each user.
"""


def compare_relevance_change(user_annotations, relevant_sentence_predictions, logging=True, log_filepath=None):
    if logging:
        with open(log_filepath, 'a') as log_file:
            log_file.write(f"\nComparing Relevance Change for User Annotations: {user_annotations}\n")

    relevance_changes = []

    for user in user_annotations.keys():
        prior_relevance = 0
        post_relevance = 0

        for research_question in user_annotations[user].keys():
            prior_annotations = [annotation_data[0]
                                 for annotation_data in user_annotations[user][research_question][0]]
            post_annotations = [annotation_data[0]
                                for annotation_data in user_annotations[user][research_question][1]]

            if len(prior_annotations) == 0 or len(post_annotations) == 0:
                if logging:
                    with open(log_filepath, 'a') as log_file:
                        log_file.write(f"No annotations for user {user} in research question {research_question} for either prior or post day. Skipping...\n")
                        continue

            prior_relevance += compare_annotations_relevance(
                prior_annotations, relevant_sentence_predictions, logging=True, log_filepath=log_filepath)
            post_relevance += compare_annotations_relevance(
                post_annotations, relevant_sentence_predictions, logging=True, log_filepath=log_filepath)

        # the prior and post relevancy scores for the user
        user_relevancy_change = (prior_relevance, post_relevance)
        if logging:
            with open(log_filepath, 'a') as log_file:
                log_file.write(f"User: {user}, Relevance Change: {user_relevancy_change}, Net Change: {post_relevance - prior_relevance}\n")
        relevance_changes.append(user_relevancy_change)
    return relevance_changes


if __name__ == "__main__":
    # initialize db
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cert = os.path.join(script_dir, "../private/dev-admin-key.json")
    cd = credentials.Certificate(cert)
    app = initialize_app(cd)
    db = firestore.client()

    user_data = os.path.join(script_dir, 'db-query.json')
    log_filepath = os.path.join(script_dir, 'logging.txt')
    result_file = os.path.join(script_dir, 'relevance_changes.txt')

    with open(user_data, 'r') as data_file:
        user_annotations = json.load(data_file)

    # find relevant and non relevant sentences
    sentence_prediction_id = "0R2nbNA2zn5Xd43foBKX" # sentencePrediction ID for David Ayers RQ/T pairing

    # set logging to True to print out the results and for debugging

    relevant_sentence_predictions, nonrelevant_sentence_predictions = categorize_relevant_sentences(
        sentence_prediction_id, logging=True, log_filepath=log_filepath)

    relevance_changes = compare_relevance_change(
        user_annotations, relevant_sentence_predictions, logging=True, log_filepath=log_filepath)

    with open(result_file, 'w') as outfile:
        json.dump(relevance_changes, outfile)

    lengths = {len(inner_array) for inner_array in relevance_changes}

    # verify that the inner arrays are of the same length
    # If the set has only one element, all inner arrays are of the same length
    if len(lengths) == 1: print("All inner arrays are of the same length") 
    else: print("Inner arrays are of different lengths")
