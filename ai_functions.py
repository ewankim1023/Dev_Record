#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# © SnapDragon Monitoring Ltd
# ==============================================================================
import pandas as pd
import json
# import boto3
# import sklearn.metrics as metrics
from io import StringIO
from io import BytesIO
# import joblib
from tqdm import tqdm
import re
import os
import multiprocessing
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import PIL

# import utils

# import utils.general as general
# import utils.storage as storage
# import utils.image_model_constants as constants

# from definitions import AI_BUCKET, IMAGE_BUCKET, URGENCY_WEIGHTING, ENVIRONMENT
# from utils.image_model_constants import INFRINGING_TRAIN_IMAGE_COUNT, INNOCENT_TRAIN_IMAGE_COUNT, IRRELEVANT_TRAIN_IMAGE_COUNT

def convert_json_to_dataframe(data):
    dataframe = pd.DataFrame()
    id_list = []
    description_list = []
    image_list = []
    images_list = []
    tags_list = []
    class_list = []
    exported_json = data

    infringing_tags = ["suspect_design",
                     "suspect_trademark",
                     "suspect_copyright",
                     "design",
                     "trademark",
                     "copyright",
                     "counterfeit",
                     "patent", 
                     ]

    for product_offer in exported_json["product_offers"]:
        id_list.append(general.return_empty_if_no_value_exists(product_offer["id"]))
        description_list.append(general.return_empty_if_no_value_exists(product_offer["description"].replace("#", "")))
        image_list.append(general.return_empty_if_no_value_exists(product_offer["image"]))
        images_list.append(general.return_empty_if_no_value_exists(product_offer["images"]))
        tags_list.append(general.return_empty_if_no_value_exists(product_offer["tags"]))

        if len(general.return_empty_if_no_value_exists(product_offer["tags"])) > 0:
            if "irrelevant" in general.return_empty_if_no_value_exists(product_offer["tags"]):
                class_list.append("irrelevant")
            elif "innocent" in general.return_empty_if_no_value_exists(product_offer["tags"]):
                class_list.append("innocent")
            elif len(set(infringing_tags).intersection(set(general.return_empty_if_no_value_exists(product_offer["tags"])))) > 0:
                class_list.append("infringing")
            else:
                class_list.append(None)
        else:
            class_list.append(None)

    dataframe["id"         ] = id_list
    dataframe["description"] = description_list
    dataframe["image"      ] = image_list
    dataframe["images"     ] = images_list
    dataframe["tags"       ] = tags_list
    dataframe["class"      ] = class_list

    product = exported_json["product"]
    product_id = general.return_empty_if_no_value_exists(product["id"])
    product_name = [general.return_empty_if_no_value_exists(product["name"].lower())]

    return dataframe, product_id, product_name

def check_whether_there_are_enough_classifications_to_train(data):
    data_infringing = remove_non_standard_image_hashes(data.loc[data["class"] == "infringing"])
    data_infringing = data_infringing[~data_infringing["image"].isin(["empty"])]

    data_innocent = remove_non_standard_image_hashes(data.loc[data["class"] == "innocent"])
    data_innocent = data_innocent[~data_innocent["image"].isin(["empty"])]

    data_irrelevant = remove_non_standard_image_hashes(data.loc[data["class"] == "irrelevant"])
    data_irrelevant = data_irrelevant[~data_irrelevant["image"].isin(["empty"])]

    return{"infringing_links": len(data_infringing),
           "innocent_links": len(data_innocent),
           "irrelevant_links": len(data_irrelevant),
           }

def process_exported_data(action, source, logger):
    s3_client = boto3.client("s3")
    s3_resource = boto3.resource("s3")

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=AI_BUCKET, Prefix=source)

    data = pd.DataFrame()

    for page in pages:
        if page['KeyCount'] == 0:
            error_message = "--- No data exported from Swoop ---"
            return {"passed": False, "error_message": error_message}
        for obj in page["Contents"]:
            file_name = str(obj["Key"])
            if any(file_type in file_name for file_type in [".json"]):
                content_object = s3_resource.Object(AI_BUCKET, file_name)
                file_content = content_object.get()["Body"].read().decode("utf-8")
                json_content = json.loads(file_content)
                data_batch, product_id, product_name = convert_json_to_dataframe(json_content)
                data = pd.concat([data, data_batch])
                if action == "train":
                    data = data.dropna(subset=["class"])

    data = data.reset_index()

    logger.info("Exported json data for {} (id - {}) converted to DataFrame - {} rows".format(product_name[0], product_id, len(data)))
    return {"passed": True, "data": data, "product_id": product_id, "product_name": product_name}

def check_models_exist(product_id):
    return {"passed": storage.does_s3_key_exist(
            AI_BUCKET,
            "%s/image_model/saved_model.pb" % product_id
            ),
            "error_message": "--- Model not found. Train a model for this product. ---"
            }

def get_threshold_comparison_dataframe_and_roc_auc_score(relevant, score):
    fpr, tpr, threshold = metrics.roc_curve(relevant, score)

    """ roc_auc score is 'Receiver Operator Curve Area Under Curve - a single metric describing quality of model
    """
    roc_auc_score = metrics.auc(fpr, tpr)

    threshold_comparison_dataframe = pd.DataFrame() # dataframe storing true and false positive rates at different thresholds
    threshold_comparison_dataframe["threshold"] = threshold.tolist()
    threshold_comparison_dataframe["tpr"      ] = tpr.tolist()
    threshold_comparison_dataframe["fpr"      ] = fpr.tolist()
    threshold_comparison_dataframe = threshold_comparison_dataframe[threshold_comparison_dataframe["threshold"] <= 1]

    return threshold_comparison_dataframe, roc_auc_score

def calculate_reduction(threshold, results, metric):
    return (len(results[results[metric] <= threshold]) / len(results)) * 100

def get_optimum_threshold(metric, min_tpr, tpr_fpr_ratio, threshold_comparison_dataframe, results, logger):
    df = threshold_comparison_dataframe[threshold_comparison_dataframe["tpr"] >= min_tpr].copy()
    df.loc[:, "loss"] = (1 - df["tpr"]) * 100 # loss is false negative rate as a percentage

    # reduction is what percentage of links are below threshold
    df.loc[:, "reduction"] = df["threshold"].apply(lambda x: calculate_reduction(x, results, metric))

    # Difference between actual loss and maximum acceptable loss multiplied by tpr_fpr_ratio i.e. the higher this score,
    # the better the result
    df.loc[:, "loss_score"] = (df["loss"].max() - df["loss"]) * tpr_fpr_ratio

    # The difference bewtween actual reduction and largest possible reduction at max acceptable loss multiplied by -1.
    # i.e. the higher this score, the better the result.
    df.loc[:, "reduction_score"] = (df["reduction"].max() - df["reduction"]) * -1

    # Loss and Reduction scores added together. The higher this score, the better the result achieved at this threshold.
    df.loc[:, "score"] = (df["loss_score"] + df["reduction_score"])

    df_find_optimum = df.sort_values(by=["score"], ascending=False).head(1)
    TPR = (df_find_optimum["tpr"].sum()) * 100
    FPR = (df_find_optimum["fpr"].sum()) * 100
    TNR = (1 - df_find_optimum["fpr"].sum()) * 100
    FNR = (1 - df_find_optimum["tpr"].sum()) * 100

    logger.info("------------")
    logger.info("Optimum threshold = {}".format(round(df_find_optimum["threshold"].sum(), 4)))
    logger.info("True Positive Rate = {}%".format(round(TPR, 2)))
    logger.info("False Negative Rate = {}%".format(round(FNR, 2)))
    logger.info("True Negative Rate = {}%".format(round(TNR, 2)))
    logger.info("False Positive Rate = {}%".format(round(FPR, 2)))
    logger.info("------------")

    return df_find_optimum["threshold"].sum(), TPR, FPR, TNR, FNR

def calculate_metrics_and_threshold(data, relevant, score, metric, min_tpr, tpr_fpr_ratio, logger):
    threshold_comparison_dataframe, roc_auc_score = get_threshold_comparison_dataframe_and_roc_auc_score(relevant, score)
    optimum_threshold, TPR, FPR, TNR, FNR = get_optimum_threshold(
        metric, min_tpr, tpr_fpr_ratio, threshold_comparison_dataframe, data, logger)

    metrics_summary_dataframe = pd.DataFrame()
    metrics_summary_dataframe["optimum_threshold"] = [optimum_threshold]
    metrics_summary_dataframe["roc_auc_score"    ] = [roc_auc_score]
    metrics_summary_dataframe["TPR"              ] = [TPR]
    metrics_summary_dataframe["FPR"              ] = [FPR]
    metrics_summary_dataframe["TNR"              ] = [TNR]
    metrics_summary_dataframe["FNR"              ] = [FNR]

    return metrics_summary_dataframe

def urgency_calculation(row):
    if row["relevant"] == 0:
        return int(int(row["score"] * 100) * URGENCY_WEIGHTING)
    else:
        return int(100 - int(row["score"] * 100))

def produce_training_score_rankings(data):
    urgency_df = data[["id", "text_score", "relevant"]].rename(columns={"text_score": "score"})
    urgency_df["urgency"] = urgency_df.apply(urgency_calculation, axis=1)
    urgency_df = urgency_df[["id", "urgency"]]
    urgency_df = urgency_df[urgency_df["urgency"] >= 50]

    return urgency_df

def save_training_score_rankings(urgency_df, product_id):
    s3_key = "%s/review_urgency.json" % product_id
    write_dataframe_to_json_on_s3(urgency_df, s3_key) # TODO: Handle exceptions

def produce_and_save_training_score_rankings(data, product_id, logger):
    urgency_df = produce_training_score_rankings(data)
    save_training_score_rankings(urgency_df, product_id)
    logger.info("Training score rankings compiled and saved")

def pull_dataframe_from_s3(filename):
    return pd.read_csv(
        BytesIO(get_raw_data_from_s3(filename)),
        encoding="utf8",
    )

def write_to_s3(key, data):
    boto3.resource("s3").Object(AI_BUCKET, key).put(Body=data)

def write_dataframe_to_csv_on_s3(dataframe, filename):
    print("Writing %d records to %s" % (len(dataframe), filename))

    with StringIO() as csv_buffer:
        dataframe.to_csv(csv_buffer, sep=",", index=False)
        write_to_s3(filename, csv_buffer.getvalue())

def write_dataframe_to_json_on_s3(dataframe, filename):
    json_buffer = StringIO()
    dataframe.to_json(json_buffer, orient="records")

    write_to_s3(filename, json_buffer.getvalue())

def get_raw_data_from_s3(filename):
    return boto3.client("s3").get_object(Bucket=AI_BUCKET, Key=filename)["Body"].read()

def pull_dataframe_from_s3_as_json(filename):
    return pd.read_json(
        BytesIO(get_raw_data_from_s3(filename)),
        encoding="utf8",
    )

def write_model_to_s3(model_object, filename):
    with BytesIO() as file:
        joblib.dump(model_object, file)
        file.seek(0)
        boto3.client("s3").upload_fileobj(Bucket=AI_BUCKET, Key=filename, Fileobj=file)

def download_model_from_s3(filename):
    with BytesIO() as file:
        boto3.client("s3").download_fileobj(Bucket=AI_BUCKET, Key=filename, Fileobj=file)
        file.seek(0)
        return joblib.load(file)

def download_images_from_s3(s3_client, hash_list, path):
    for image_hash in tqdm(hash_list, desc="Images downloaded"):
        if storage.does_s3_key_exist(IMAGE_BUCKET, image_hash):
            storage.download_image_from_s3_to_temporary_directory(s3_client, image_hash, path)

def zip_hash_list_and_path_list(hash_list, path):
    path_list = [path] * len(hash_list)
    return list(zip(hash_list, path_list))

def download_images_using_starmap(hash_path_list):
    pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), len(hash_path_list)))
    pool.starmap(storage.download_image_from_s3_to_temporary_directory_using_cdn, hash_path_list, chunksize=10)
    pool.close()

def parallel_download_images_from_s3(hash_list, path):
    hash_path_list = zip_hash_list_and_path_list(hash_list, path)
    download_images_using_starmap(hash_path_list)

def save_keras_model_to_s3(s3_client, product_id, temp_dir):
    origin = "{}/keras_model/saved_model.pb".format(temp_dir)
    destination = "{}/image_model/saved_model.pb".format(product_id)
    storage.put_object_in_s3(s3_client, origin, destination)

    origin = "{}/keras_model/keras_metadata.pb".format(temp_dir)
    destination = "{}/image_model/keras_metadata.pb".format(product_id)
    storage.put_object_in_s3(s3_client, origin, destination)

    origin = "{}/keras_model/variables/*".format(temp_dir)
    destination = "{}/image_model/variables/".format(product_id)
    storage.put_objects_from_directory_in_s3(s3_client, origin, destination)

    origin = "{}/keras_model/assets/*".format(temp_dir)
    destination = "{}/image_model/assets/".format(product_id)
    storage.put_objects_from_directory_in_s3(s3_client, origin, destination)

def download_saved_keras_model_from_s3(product_id, s3_client, s3_bucket, temp_dir):
    key = "{}/image_model/saved_model.pb".format(product_id)
    filename = "{}/keras_model/saved_model.pb".format(temp_dir)
    storage.download_file_from_s3(s3_client, key, filename)

    key = "{}/image_model/keras_metadata.pb".format(product_id)
    filename = "{}/keras_model/keras_metadata.pb".format(temp_dir)
    storage.download_file_from_s3(s3_client, key, filename)

    folder = "{}/image_model/variables".format(product_id)
    file_location = "{}/keras_model/variables/".format(temp_dir)
    storage.download_file_from_directory_in_s3(s3_client, s3_bucket, folder, file_location)

    folder = "{}/image_model/assets".format(product_id)
    file_location = "{}/keras_model/assets/".format(temp_dir)
    storage.download_file_from_directory_in_s3(s3_client, s3_bucket, folder, file_location)

def check_if_custom_image_training_set_exists(product_id):
    return storage.does_s3_key_exist(
        AI_BUCKET,
        "{}/image_training_data/image_training_dataset_custom".format(product_id)
    )

def get_image_hash_from_key(row):
    if len(row["image"]) == 40 and row["image"].isalnum():
        return row["image"]
    else:
        return "empty"

def remove_non_standard_image_hashes(data):
    data["image_hash"] = data.apply(get_image_hash_from_key, axis=1)
    data["image"] = data["image_hash"].copy()
    data = data.drop(["image_hash"], axis=1)
    return data

def build_image_hash_list(data, required_image_count):
    data = data.sample(n=min(required_image_count, len(data)))
    return data["image"].tolist()

def build_default_image_training_set(product_id, data):
    data = data[data["image"].notna()]
    data = data[~data["image"].isin(["empty"])]

    infringing_hashes = build_image_hash_list(data[data["class"] == "infringing"], INFRINGING_TRAIN_IMAGE_COUNT)
    innocent_hashes = build_image_hash_list(data[data["class"] == "innocent"], INNOCENT_TRAIN_IMAGE_COUNT)
    irrelevant_hashes = build_image_hash_list(data[data["class"] == "irrelevant"], IRRELEVANT_TRAIN_IMAGE_COUNT)

    json_data = {"product_id": product_id,
                 "infringing_hashes": infringing_hashes,
                 "innocent_hashes": innocent_hashes,
                 "irrelevant_hashes": irrelevant_hashes}

    return json_data

def build_default_image_training_set_if_custom_does_not_exist(product_id, data, logger):
    if check_if_custom_image_training_set_exists(product_id):
        logger.info("Custom image training dataset exists")
        return "custom"
    else:
        json_data = build_default_image_training_set(product_id, data)
        storage.save_json_file_to_s3(product_id, json_data)
        logger.info("No custom image training dataset exists - default set created.")
        return "default"

def get_json_file_from_s3(json_data_key):
    return storage.get_json_file_from_s3(json_data_key)

def count_test_images(df, class_label, list_of_keys):
    df = df[df["class"] == class_label]
    df = df[~df["image"].isin(list_of_keys)]
    df = df[~df["image"].isin(["empty"])]
    return len(df)

def get_number_of_training_and_testing_images(image_json_data, data):
    number_of_images = {}

    number_of_images["train_infringing"] = len(image_json_data["infringing_hashes"])
    number_of_images["train_innocent"] = len(image_json_data["innocent_hashes"])
    number_of_images["train_irrelevant"] = len(image_json_data["irrelevant_hashes"])

    number_of_images["test_infringing"] = count_test_images(data, "infringing", image_json_data["infringing_hashes"])
    number_of_images["test_innocent"] = count_test_images(data, "innocent", image_json_data["innocent_hashes"])
    number_of_images["test_irrelevant"] = count_test_images(data, "irrelevant", image_json_data["irrelevant_hashes"])

    return number_of_images

def check_if_can_train_and_test(number_of_images):
    if (number_of_images["train_infringing"] >= constants.MIN_INFRINGING_TRAIN_IMAGE_COUNT and number_of_images["train_innocent"] >= constants.MIN_INNOCENT_TRAIN_IMAGE_COUNT and number_of_images["train_irrelevant"] >= constants.MIN_IRRELEVANT_TRAIN_IMAGE_COUNT):
        can_train = True
    else:
        can_train = False

    if (number_of_images["test_infringing"] >= constants.MIN_INFRINGING_TEST_IMAGE_COUNT and number_of_images["test_innocent"] >= constants.MIN_INNOCENT_TEST_IMAGE_COUNT and number_of_images["test_irrelevant"] >= constants.MIN_IRRELEVANT_TEST_IMAGE_COUNT):
        can_test = True
    else:
        can_test = False

    return {"passed": can_train, "can_test": can_test}

def delete_corrupt_images(directories, logger):
    for directory in directories:
        for file in os.listdir(directory):
            try:
                Image.open(directory+"/"+file).verify()
            except PIL.UnidentifiedImageError:
                logger.info("File {} is not valid".format(file))
                os.remove(directory+"/"+file)
            #Truncated image error handled this way as PIL throws a generic OSError
            except OSError as e:
                print(directory+"/"+file)
                if any(x in str(e) for x in ["image file is truncated", "Truncated"]):
                    logger.info("File {} is truncated".format(file))
                    os.remove(directory+"/"+file)
                else:
                    raise e
            except Exception as e:
                if "decompression bomb" in str(e):
                    logger.info("File {} is too large".format(file))
                    os.remove(directory+"/"+file)
                else:
                    raise e

def check_for_logo_detector(product_id, logo_detection_model_json):
    with open(logo_detection_model_json) as f:
        logo_detector_models = json.load(f)
    for model in logo_detector_models[ENVIRONMENT]:
        if model["product_id"] == product_id:
            return {
                "passed": True,
                "model_exists": True,
                "project_arn": model["project_arn"],
                "model_arn": model["model_arn"],
                "version_name": model["version_name"],
                }
    return {"passed": True, "model_exists": False}