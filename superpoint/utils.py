import numpy as np
import pickle
import os
import time


def compute_matches(retrieved_all, ground_truth_info):
    matches = []
    itr = 0
    failed_cases = []
    print("\n==> Failed cases:")
    for retr in retrieved_all:
        if retr in ground_truth_info[itr][1]:
            # if (retr in ground_truth_info[itr]):
            matches.append(1)
        else:
            print(
                "==> Query "
                + str(itr)
                + " retrieved the wrong reference "
                + str(retr)
                + ", the ground truth should be "
                + str(ground_truth_info[itr][1])
            )
            failed_cases.append([itr, retr, ground_truth_info[itr][1]])
            matches.append(0)
        itr = itr + 1
    return matches, failed_cases


def compute_recall_rate_at_n_range(n, similarity_matrix, ground_truth_info):
    n_range = range(n)
    recall_rate_at_n_range = np.zeros(len(n_range))

    itr = 0
    for cnt in n_range:
        recall_rate_at_n_range[itr] = compute_recall_rate_at_n(
            cnt + 1, similarity_matrix, ground_truth_info
        )
        itr += 1

    return recall_rate_at_n_range


def compute_recall_rate_at_n(n, similarity_matrix, ground_truth_info):
    matches = []
    total_queries = len(similarity_matrix)
    match_found = 0

    for query in range(total_queries):
        top_n_retrieved = np.argpartition(similarity_matrix[query], -1 * n)[-1 * n :]
        for retr in top_n_retrieved:
            if retr in ground_truth_info[query][1]:
                match_found = 1
                break

        if match_found == 1:
            matches.append(1)
            match_found = 0
        else:
            matches.append(0)
            match_found = 0

    recall_rate_at_n = float(np.sum(matches)) / float(total_queries)

    return recall_rate_at_n


def get_query_img_name(dataset, cnt):
    if dataset == "Gardens":
        name = str(cnt).zfill(3)
        return "Image" + name + ".jpg"
    elif dataset == "Test":
        return str(cnt) + ".png"
    elif (
        dataset == "ESSEX3IN1" or "SPED" or "CrossSeasons" or "Pittsburgh" or "Nordland"
    ):
        name = str(cnt).zfill(7)
        return name + ".jpg"


def get_refer_img_name(dataset, cnt):
    if dataset == "Gardens":
        name = str(cnt).zfill(3)
        return "Image" + name + ".jpg"
    elif dataset == "Test":
        return str(cnt) + ".png"
    elif (
        dataset == "ESSEX3IN1" or "SPED" or "CrossSeasons" or "Pittsburgh" or "Nordland"
    ):
        name = str(cnt).zfill(7)
        return name + ".jpg"


def print_and_store_result(
    config,
    opt,
    total_time,
    refer_encoding_time,
    matching_time,
    similarity,
    method,
    strategy,
    parameters,
    recall_rate_n_range,
):
    total_query_imgs = config["Dataset"][opt.dataset]["total_query_imgs"]
    total_refer_imgs = config["Dataset"][opt.dataset]["total_refer_imgs"]
    dataset_dir = config["Dataset"]["Root"] + config["Dataset"][opt.dataset]["path"]
    result_root = config["output_path"]
    dataset = opt.dataset

    if config["print_runtime"] is True:
        print("\n==> Average time per query: ", total_time / total_query_imgs)
        print("==> Average encoding time: ", refer_encoding_time / total_refer_imgs)
        print("==> Average matching time per query: ", matching_time / total_query_imgs)

    if config["print_precision"] is True:
        predictions = np.argmax(np.array(similarity), axis=1)
        ground_truth_info = np.load(
            dataset_dir + "ground_truth_new.npy", allow_pickle=True
        )
        matches, failed_cases = compute_matches(predictions, ground_truth_info)
        print(
            "\n==> Precision @ 100% Recall: "
            + str(np.sum(matches) / total_query_imgs)
            + "("
            + str(np.sum(matches))
            + "/"
            + str(total_query_imgs)
            + ")"
        )
        recall_rate_at_n = compute_recall_rate_at_n_range(
            recall_rate_n_range, similarity, ground_truth_info
        )
        print("==> Recall Rate @ [1, 20]: " + str(recall_rate_at_n))

    if config["store_result"] is True:
        if os.path.exists(result_root + dataset + "/" + method) is False:
            os.makedirs(result_root + dataset + "/" + method)

        if (
            os.path.exists(
                result_root + dataset + "/" + method + "/" + strategy
            )
            is False
        ):
            os.makedirs(
                result_root + dataset + "/" + method + "/" + strategy
            )

        with open(
            result_root
            + dataset
            + "/"
            + method
            + "/"
            + strategy
            + "/similarity",
            "wb",
        ) as fp:
            pickle.dump(similarity, fp)
        fp.close()
        print("==> Similarity matrix saved.")

        with open(
            result_root
            + dataset
            + "/"
            + method
            + "/"
            + strategy
            + "/sorted_similarity",
            "wb",
        ) as fp:
            pickle.dump(np.sort(similarity), fp)
        fp.close()
        print("==> Sorted similarity matrix saved.")

        with open(
            result_root
            + dataset
            + "/"
            + method
            + "/"
            + strategy
            + "/sorted_prediction",
            "wb",
        ) as fp:
            pickle.dump(np.argsort(similarity), fp)
        fp.close()
        print("==> Sorted prediction saved.")

        with open(
            result_root
            + dataset
            + "/"
            + method
            + "/"
            + strategy
            + "/failed_cases",
            "wb",
        ) as fp:
            pickle.dump(failed_cases, fp)
        fp.close()
        print("==> Failed cases saved.")

        with open(
            result_root
            + dataset
            + "/"
            + method
            + "/"
            + strategy
            + "/result.txt",
            "w",
        ) as f:
            f.writelines("Dataset: " + dataset + "\n")
            if dataset == "Gardens":
                f.writelines(
                    "Sequence: "
                    + config["Dataset"][opt.dataset]["query_dir"].split("/")[2]
                    + " vs "
                    + config["Dataset"][opt.dataset]["refer_dir"].split("/")[2]
                    + "\n"
                )
            f.writelines("Pipeline: SuperPoint + " + method + " + RANSAC" + "\n")

            if method == "AttnPatch":
                f.writelines("Threshold: " + str(parameters[0]) + "\n")
                f.writelines("Re-projection error: " + str(parameters[1]) + "\n")
            elif method == "All":
                f.writelines("Re-projection error: " + str(parameters) + "\n")

            f.writelines(
                "\n+++++++++++++++ Place Recognition Performance +++++++++++++++\n"
            )
            f.writelines(
                "Precision @ 100% Recall: "
                + str(np.sum(matches) / total_query_imgs)
                + "("
                + str(np.sum(matches))
                + "/"
                + str(total_query_imgs)
                + ")"
                + "\n"
            )
            f.writelines("Recall Rate @ [1, 20]: " + str(recall_rate_at_n) + "\n")

            f.writelines("\n+++++++++++++++ Runtime Performance +++++++++++++++\n")
            f.writelines(
                "Average time per query: " + str(total_time / total_query_imgs) + "\n"
            )
            f.writelines(
                "Average encoding time: "
                + str(refer_encoding_time / total_refer_imgs)
                + "\n"
            )
            f.writelines(
                "Average matching time per query: "
                + str(matching_time / total_query_imgs)
                + "\n"
            )

            f.writelines("\n+++++++++++++++ Failed cases +++++++++++++++\n")
            for cnt in range(len(failed_cases)):
                f.writelines(
                    "Query "
                    + str(failed_cases[cnt][0])
                    + " retrieved the wrong reference "
                    + str(failed_cases[cnt][1])
                    + ", the ground truth should be "
                    + str(failed_cases[cnt][2])
                    + "\n"
                )
        f.close()
        print("==> Runtime performance saved.")

        print(
            "==> Results are saved at "
            + result_root
            + dataset
            + "/"
            + method
            + "/"
            + strategy
        )


def print_and_store_result_ii(
    config,
    total_query_imgs,
    predictions,
    dataset_dir,
    sp_root,
    dataset_name,
    method,
    recall_rate_n_range,
):
    if config["print_precision"] is True:
        ground_truth_info = np.load(
            dataset_dir + "ground_truth_new.npy", allow_pickle=True
        )
        matches, _ = compute_matches(predictions.T[0], ground_truth_info)
        print(
            "Precision @ 100% Recall: "
            + str(np.sum(matches) / total_query_imgs)
            + "("
            + str(np.sum(matches))
            + "/"
            + str(total_query_imgs)
            + ")"
        )
        recall_rate_at_n = compute_recall_rate_at_n_range(
            recall_rate_n_range, predictions, ground_truth_info
        )
        print("Recall Rate @ [1, 20]: " + str(recall_rate_at_n))

    if config["store_result"] is True:
        if os.path.exists(sp_root + dataset_name) is False:
            os.mkdir(sp_root + dataset_name)

        with open(
            sp_root + dataset_name + "/result_" + method + ".txt", "w"
        ) as f:
            f.writelines(
                "Precision @ 100% Recall: "
                + str(np.sum(matches) / total_query_imgs)
                + "("
                + str(np.sum(matches))
                + "/"
                + str(total_query_imgs)
                + ")"
                + "\n"
            )
            f.writelines("Recall Rate @ [1, 20]: " + str(recall_rate_at_n) + "\n")
        f.close()
        print("==> Runtime information saved.")
