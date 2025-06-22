

import pandas as pd

def extract_ground_truths(csv_path):
    """
    Reads a CSV where "David's annotations" marks the start of a block of rows (with labels '1'–'5')
    containing 'x' in the column corresponding to the ground-truth score for each task.
    Returns a list of integers (1–5) or None for each task-column, in the same order they appear.
    """
    df = pd.read_csv(csv_path)
    
    # Find the row index where the first column equals "David's annotations"
    ann_row = df[df.iloc[:, 0] == "David's annotations"].index
    if len(ann_row) == 0:
        raise ValueError("Could not find a row labeled \"David's annotations\" in the first column.")
    ann_idx = ann_row[0]
    
    # Take all rows below that; keep only those whose first-column value is a string '1'–'5'
    block = df.iloc[ann_idx + 1 :].copy()
    valid_rows = block[block.iloc[:, 0].isin([str(i) for i in range(1, 6)])]
    
    # Build a mapping {column_name: ground_truth_score_or_None}
    ground_truths = {}
    for col in valid_rows.columns[1:]:
        # Look for the row in this block where the cell equals 'x'
        hit = valid_rows[valid_rows[col] == 'x']
        if not hit.empty:
            # The first-column of that row is the score (string '1'–'5')
            score_str = hit.iloc[0, 0]
            ground_truths[col] = int(score_str)
        else:
            # If no 'x' found, set to None (or choose a default)
            ground_truths[col] = None
    
    # Return as a list in the same column order (excluding the first column)
    return [ground_truths[col] for col in valid_rows.columns[1:]]




if __name__ == "__main__":
    hw4_path = "hw4labels.csv"
    hw5_path = "hw5labels.csv"
    
    hw4_ground_truths = extract_ground_truths(hw4_path)
    hw5_ground_truths = extract_ground_truths(hw5_path)
    
    print("HW4 ground truths:", hw4_ground_truths)
    print("HW5 ground truths:", hw5_ground_truths)