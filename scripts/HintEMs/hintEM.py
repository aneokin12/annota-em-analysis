from model import dawid_skene
import numpy as np


def run_inference(N):
    predictions = dawid_skene(N)

    return predictions



def determine_label(hint, prediction):
    #hint at task is NOT_RELEVANT
    if hint == 0:
        #agreeing is correct -> final label is not relevant
        if prediction == 1:
            return 0
        #disagreeing is correct -> final label is relevant
        if prediction == 0:
            return 1
    #hint at task is MISSED, ie RELEVANT
    if hint == 1:
        #agreeing is correct, -> final label is relevant
        if prediction == 1:
            return 1
        #disagreeing is correct -> final label is not relevant
        if prediction == 0:
            return 0
        

def ensemble_predictions(original_preds, new_preds, task_mapping):

    preds = np.array([])

    for i in range(len(preds)):
        #if the current task is covered by the hint em, set the final label as that 
        if i in task_mapping:
            original_hint = task_mapping[i][1]
            final_label = determine_label(original_hint, new_preds[i])
            preds[i] = final_label
        else:
            preds[i] = original_preds[i]
    return preds


def run_hint_em(original_preds, hint_data, task_map):


    #generate predictions for whether to agree or disagree per task
    raw_predictions = run_inference(hint_data)


    #mesh agreement predictions with original hint to generate relevancy labels, use vanilla em predictions to fill in uncovered tasks
    predictions = ensemble_predictions(original_preds, raw_predictions, task_map)

    np.save('hintpreds.npy', predictions)





if __name__ == '__main__':

    inputTensor = np.load('hints.npy', allow_pickle = True)
    # total_ann_mat = inputTensor[:, ~np.all(np.isnan(inputTensor), axis=0)]
    preds = run_inference(inputTensor)

    print(len(preds))



