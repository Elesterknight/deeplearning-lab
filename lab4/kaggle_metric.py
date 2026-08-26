"""
Kaggle metric for semantic segmentation: Calculate mIoU per image, then average across all test set.
This version is optimized for Kaggle submission (no tqdm, clean doctests).
"""
import numpy as np
import pandas as pd
import ast


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    pass


def calculate_miou_per_image(pred_mask, gt_mask, ignore_class=None, n_classes=3):
    """
    Calculate mIoU for a single image.

    Args:
        pred_mask: Predicted mask (H, W) numpy array
        gt_mask: Ground truth mask (H, W) numpy array
        ignore_class: Class to ignore in mIoU calculation (set to None to include all classes)
        n_classes: Number of classes (default: 3)

    Returns:
        mIoU for this image (float or NaN if no valid classes)
    """
    pred_mask = np.array(pred_mask)
    gt_mask = np.array(gt_mask)

    iou_list = []

    for cls in range(n_classes):
        pred_c = (pred_mask == cls)
        gt_c = (gt_mask == cls)

        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()

        # Skip if union is 0 (class doesn't exist in both GT and Pred)
        if union == 0:
            continue

        # Calculate IoU
        iou = intersection / union

        # Only add to list if not the ignored class
        if ignore_class is None or cls != ignore_class:
            iou_list.append(iou)

    # Return mIoU for this image
    if len(iou_list) > 0:
        return np.mean(iou_list)
    else:
        return np.nan


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, ignore_class: int = 0) -> float:
    """
    Calculate mIoU metric for semantic segmentation.

    This metric:
    1. Calculates mIoU for each image (ignoring class 0)
    2. Averages the mIoU across all images in the test set

    Args:
        solution: Ground truth dataframe with columns [row_id_column_name, 'pred_mask']
        submission: Prediction dataframe with columns [row_id_column_name, 'pred_mask']
        row_id_column_name: Name of the row ID column (e.g., 'index')
        ignore_class: Class to ignore in mIoU calculation (default: 0)

    Returns:
        Mean of per-image mIoU scores (float between 0 and 1)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> solution = pd.DataFrame({'index': [0, 1], 'pred_mask': ['[[0,1],[1,2]]', '[[0,1],[2,2]]']})
        >>> submission = pd.DataFrame({'index': [0, 1], 'pred_mask': ['[[0,1],[1,2]]', '[[0,1],[2,1]]']})
        >>> result = score(solution, submission, 'index')
        >>> 0 <= result <= 1
        True
    """
    # Sort by row ID to ensure alignment
    solution = solution.sort_values(row_id_column_name).reset_index(drop=True)
    submission = submission.sort_values(row_id_column_name).reset_index(drop=True)

    # Check that row IDs match
    if not solution[row_id_column_name].equals(submission[row_id_column_name]):
        raise ParticipantVisibleError("Row IDs in solution and submission do not match")

    # Remove row ID column
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Check that 'pred_mask' column exists
    if 'pred_mask' not in solution.columns:
        raise ParticipantVisibleError("Solution must have 'pred_mask' column")
    if 'pred_mask' not in submission.columns:
        raise ParticipantVisibleError("Submission must have 'pred_mask' column")

    n_samples = len(solution)
    if n_samples != len(submission):
        raise ParticipantVisibleError(f"Solution has {n_samples} samples but submission has {len(submission)} samples")

    miou_scores = []

    # Calculate mIoU for each image
    for i in range(n_samples):
        try:
            # Parse pred_mask from string to array
            gt_mask_str = solution.iloc[i]['pred_mask']
            pred_mask_str = submission.iloc[i]['pred_mask']

            # Convert string representation to actual array
            gt_mask = np.array(ast.literal_eval(gt_mask_str))
            pred_mask = np.array(ast.literal_eval(pred_mask_str))

            # Check dimensions match
            if gt_mask.shape != pred_mask.shape:
                raise ParticipantVisibleError(f"Sample {i}: GT shape {gt_mask.shape} != Pred shape {pred_mask.shape}")

            # Calculate mIoU for this image
            miou = calculate_miou_per_image(pred_mask, gt_mask, ignore_class=ignore_class)

            # Only add non-NaN scores
            if not np.isnan(miou):
                miou_scores.append(miou)

        except Exception as e:
            raise ParticipantVisibleError(f"Error processing sample {i}: {str(e)}")

    # Return mean of all per-image mIoU scores
    if len(miou_scores) > 0:
        final_score = np.mean(miou_scores)
        return float(final_score)
    else:
        raise ParticipantVisibleError("No valid mIoU scores calculated")
