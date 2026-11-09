import numpy as np
import argparse
import random
import os
from code.config.args import add_missing_generation_arguments
from code.config.dataset_profiles import DATASET_CATEGORY_NO_REAL_MISSING
from code.core.data_io import build_missing_data_path, load_non_missing_data
from code.core.path_config import apply_workspace_paths


parser = argparse.ArgumentParser(description="classifying models setting")
add_missing_generation_arguments(parser)

args = parser.parse_args()
apply_workspace_paths(args)

np.random.seed(args.seed + args.iter)
random.seed(args.seed + args.iter)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def standardize(values):
    std = np.std(values)
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - np.mean(values)) / std


def get_pc1_anchor(org_data):
    feature_matrix = np.asarray(org_data[:, 1:], dtype=float)
    if feature_matrix.shape[1] == 0:
        raise ValueError("PC1-based MAR requires at least one non-label feature column.")
    centered = feature_matrix - np.mean(feature_matrix, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1_direction = vt[0]
    pc1_scores = centered @ pc1_direction
    return standardize(pc1_scores)


def get_feature_anchor(org_data, attr):
    return standardize(np.asarray(org_data[:, attr], dtype=float))


def calibrate_logistic_probabilities(anchor_scores, target_rate, beta, clip_value):
    clipped_scores = np.clip(anchor_scores, -clip_value, clip_value)
    low = -20.0
    high = 20.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        probabilities = sigmoid(midpoint + beta * clipped_scores)
        if probabilities.mean() > target_rate:
            high = midpoint
        else:
            low = midpoint
    return sigmoid((low + high) / 2.0 + beta * clipped_scores)


def choose_start_index(start_probabilities, attr_mask):
    weighted = start_probabilities * (attr_mask == 0)
    total = weighted.sum()
    if total <= 0:
        return None
    normalized = weighted / total
    return int(np.random.choice(np.arange(attr_mask.shape[0]), p=normalized))


def canmask(i, j, len, mask_flag):
    if i-1 >= 0:
        if mask_flag[i-1, j] == 1:
            return False
    if i+len < mask_flag.shape[0]:
        if mask_flag[i+len, j] == 1:
            return False
    return True

def get_MCAR_mask_flag(args, org_data):
    mask_flag = np.zeros(org_data.shape)

    time_step_num = org_data.shape[0]

    missing_num_for_attr = int(time_step_num * args.missing_percent) 

    for attr in range(1, org_data.shape[1]):
        while np.sum(mask_flag[:, attr]) < missing_num_for_attr:
            mask_len = random.randint(1, args.max_missing_lenth)
            x = random.randint(0, time_step_num-1)
    
            if canmask(x, attr, mask_len, mask_flag) == False:
                continue
            mask_flag[x, attr] = 1

            if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
            for i in range(x+1, min(time_step_num, x+mask_len)):
                mask_flag[i, attr] = 1
                if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                    break    
            if np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
    return mask_flag

def get_MAR_mask_flag(args, org_data):
    mask_flag = np.zeros(org_data.shape)

    time_step_num = org_data.shape[0]
    missing_num_for_attr = int(time_step_num * args.missing_percent) 
    anchor_scores = get_pc1_anchor(org_data)
    start_probabilities = calibrate_logistic_probabilities(
        anchor_scores,
        args.missing_percent,
        args.mar_logistic_beta,
        args.mar_logistic_clip,
    )

    for attr in range(1, org_data.shape[1]):
        while np.sum(mask_flag[:, attr]) < missing_num_for_attr:
            mask_len = random.randint(1, args.max_missing_lenth)
            x = choose_start_index(start_probabilities, mask_flag[:, attr])
            if x is None:
                break
    
            if canmask(x, attr, mask_len, mask_flag) == False:
                continue
            mask_flag[x, attr] = 1

            if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
            for i in range(x+1, min(time_step_num, x+mask_len)):
                mask_flag[i, attr] = 1
                if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                    break    
            if np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
    return mask_flag

def get_MNAR_mask_flag(args, org_data):
    mask_flag = np.zeros(org_data.shape)

    time_step_num = org_data.shape[0]
    missing_num_for_attr = int(time_step_num * args.missing_percent) 

    for attr in range(1, org_data.shape[1]):
        anchor_scores = get_feature_anchor(org_data, attr)
        start_probabilities = calibrate_logistic_probabilities(
            anchor_scores,
            args.missing_percent,
            args.mar_logistic_beta,
            args.mar_logistic_clip,
        )
        while np.sum(mask_flag[:, attr]) < missing_num_for_attr:
            mask_len = random.randint(1, args.max_missing_lenth)
            x = choose_start_index(start_probabilities, mask_flag[:, attr])
            if x is None:
                break
    
            if canmask(x, attr, mask_len, mask_flag) == False:
                continue
            mask_flag[x, attr] = 1

            if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
            for i in range(x+1, min(time_step_num, x+mask_len)):
                mask_flag[i, attr] = 1
                if  np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                    break    
            if np.sum(mask_flag[:, attr]) >= missing_num_for_attr:
                break
    return mask_flag

if __name__ == '__main__':
    if args.dataset_category != DATASET_CATEGORY_NO_REAL_MISSING:
        raise ValueError(
            "generate_missing_data.py only supports dataset_category='no_real_missing'. "
            "Real-missing datasets should provide their own missing files."
        )
    org_data = load_non_missing_data(args)
    
    if args.missing_mechanism == "MCAR":
        mask_flag = get_MCAR_mask_flag(args, org_data)
    if args.missing_mechanism == "MAR":
        mask_flag = get_MAR_mask_flag(args, org_data)
    if args.missing_mechanism == "MNAR":
        mask_flag = get_MNAR_mask_flag(args, org_data)

    org_data[np.where(mask_flag == 1)] = np.nan

    
    # Save missing data under data/Missing_data; reads remain backward-compatible
    # with the legacy lowercase directory name.
    missing_datasetname = build_missing_data_path(args, ensure_parent=True)
    np.savetxt(missing_datasetname, org_data, delimiter=",", fmt='%.2f')
