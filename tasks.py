# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import re
import itertools
import json

from settings import Parameters, EraseItem
import decorator
from tools import *


@decorator.logging
@decorator.ignore_warning
def select_targets(info, store, dim=Parameters.dim_targets):
    """
    Select targets and give them embedding position
    :param info:
    :param store:
    :param dim:
    :return:
    """
    info = info[["tag", "target", "grade", "frequency"]]
    abs_freq = info.groupby(["tag", "target"]).apply(sum_freq)\
        .sort_values(["frequency"], ascending=False).drop(columns=["grade"]).reset_index(drop=True)

    # neg_list = info["grade"] == -1
    # info.loc[neg_list, ["frequency"]] = -info.loc[neg_list, ["frequency"]]
    # zero_list = info["grade"] == 0
    # info.loc[zero_list, ["frequency"]] = 0
    # sign_freq = info.groupby(["tag", "target"]).apply(sum_freq) \
    #     .sort_values(["frequency"], ascending=False).drop(columns=["grade"]).reset_index(drop=True)

    # need some delete and merge operation
    # print(abs_freq[:dim])
    targets, hot_targets, serial = dict(), dict(), 0
    for k, v in abs_freq.iterrows():
        key = "%s-%s" % (v["tag"], v["target"])
        if key in EraseItem.erase_target:
            continue
        targets[key] = serial
        if k < dim:
            hot_targets["%s-%s" % (v["tag"], v["target"])] = serial
        serial += 1
    store.append("targets", levels=1, dim=dim, data=targets, layout="dict", val_type="str")
    return hot_targets


@decorator.logging
def map_model_urls(urls, store):
    print(urls.columns)
    map_urls = dict()
    for k, v in urls.iterrows():
        map_urls["%s-%s" % (no_blank(v["brand"]), no_blank(v["model"]))] = v["imageurl"]
    store.append("map_urls", levels=1, data=map_urls, layout="dict", val_type="str")
    return map_urls


@decorator.logging
def select_submarkets(info, store):
    """
    Select submarkets
    :param info:
    :param store:
    :return:
    """
    sm_jsons = info["submarket"].values
    sm_map, sm_hot = dict(), dict()
    trantab = make_trantab("{} '")
    for line in sm_jsons:
        items = str(line).translate(trantab).split(",")
        for item in items:
            sm_id, sm_name = item.split(":")
            if sm_name in EraseItem.erase_submarket:
                continue
            sm_map[sm_name] = sm_id
            sm_hot.setdefault(sm_name, 1)
            sm_hot[sm_name] += 1
    sm_hot = sorted(sm_hot.items(), key=lambda x: x[1], reverse=True)

    # print("hot submarkets", sm_hot)
    all_sm = {sm_name: num for sm_name, num in sm_hot}
    sm_hot = {sm_name: num for sm_name, num in sm_hot if num > Parameters.floor_submarkets}
    store.append("sm_map", levels=1, data=sm_map, layout="dict", val_type="str")
    store.append("submarkets", levels=1, floor=Parameters.floor_submarkets, data=all_sm, layout="dict", val_type="int")
    return sm_map, sm_hot


@decorator.logging
@decorator.ignore_warning
def reconstruct_info(info_target, info_submarket, target_hot, sm_hot):
    """
    Unfolding useful information by product and left merge
    :param info_target:
    :param info_submarket:
    :param target_hot:
    :param sm_hot:
    :return:
    """
    neg_list = info_target["grade"] == -1
    info_target.loc[neg_list, ["frequency"]] = -info_target.loc[neg_list, ["frequency"]]
    zero_list = info_target["grade"] == 0
    info_target.loc[zero_list, ["frequency"]] = 0
    df_score = info_target[["brand", "model", "tag", "target", "frequency"]]\
        .groupby(["brand", "model", "tag", "target"]).apply(sum_score).reset_index(drop=True)

    dict_score = dict()
    for k, v in df_score.iterrows():
        model = "%s-%s" % (no_blank(v["brand"]), no_blank(v["model"]))
        target = v["target"]
        try:
            type(dict_score[model])
        except KeyError:
            dict_score[model] = dict()
        finally:
            dict_score[model][target] = v["score"]

    columns = ["brand", "model", "biz", "ts_price", "submarket", "title"]
    df_res = pd.DataFrame([], columns=columns)
    for k, v in info_submarket.iterrows():
        # maybe can merge brand and model as one column
        brand, model = [no_blank(v["brand"])], [no_blank(v["model"])]
        biz, ts_price = [v["biz30day"]], [v["total_sold_price"]]
        try:
            title = [re.search(r"(?<='书名': ').*?(?=', ')", str(v["sku"])).group()]
        except AttributeError:
            title = [""]
        trantab = make_trantab("{} '")
        sm_list = [item.split(":")[1] for item in str(v["submarket"]).translate(trantab).split(",")]
        submarkets = [x for x in sm_list if x in sm_hot.keys()]

        values = list(itertools.product(brand, model, biz, ts_price, submarkets, title))
        df_sku = pd.DataFrame(values, columns=columns)

        # df_target_score = df_score[(df_score["brand"] == v["brand"]) & (df_score["model"] == v["model"])]
        values = list()
        try:
            model = "%s-%s" % (no_blank(v["brand"]), no_blank(v["model"]))
            for key, value in dict_score[model].items():
                if key in target_hot.keys():
                    values.append([key, int(dict_score[model][key])])
        except KeyError:
            values.append(["", 0])
        df_target = pd.DataFrame(values, columns=["target", "score"])
        df_target["brand"], df_target["model"] = v["brand"], v["model"]
        df_merge = pd.merge(df_sku, df_target,
                            how="left", on=["brand", "model"])
        df_res = df_res.append(df_merge, ignore_index=True)

    df_res["target"] = df_res["target"].fillna("")
    df_res["score"] = df_res["score"].fillna(0)
    return df_res


@decorator.logging
def embedding(info, targets, store, normal="DEFAULT"):
    """
    Embedding scores and transform scores
    :param info:
    :param targets:
    :return:
    """
    mat = list()
    map_line = dict()
    if "DEFAULT" == normal:
        func = embedding_one
    else:
        func = embedding_one_normal
    info.groupby(["brand", "model"]).apply(func, targets, mat, map_line)
    store.append("map_line", levels=1, data=map_line, layout="dict", val_type="int")
    return mat, map_line


@decorator.logging
@decorator.ignore_warning
def pca(mat, store, topNfeat=Parameters.top_N_feature):
    """
    Dimension reduction
    :param mat:
    :param store:
    :param topNfeat:
    :return:
    """
    # judge topNfeat
    mat = np.array(mat).astype(float)  # M * N
    meanVals = np.mean(mat, axis=0)  # N * 1
    meanRemoved = mat - meanVals  # M * N

    covMat = np.cov(meanRemoved, rowvar=False)  # N * N

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # N, N * N

    eigValInd = np.argsort(eigVals)  # 200
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # n
    # redEigVects = eigVects[eigValInd]  # n * N
    redEigVects = eigVects[:, eigValInd]  # N * n

    lowDDataMat = meanRemoved * redEigVects  # M * n
    lowDDataMat = lowDDataMat.astype(float).tolist()  # M * n
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # M * N

    store.append("mat", data=lowDDataMat, layout="np.array", val_type="float")
    store.append("mat_org", data=mat, layout="np.array", val_type="float")
    store.append("mat_restore", data=reconMat, layout="np.array", val_type="float")
    store.append("mat_trans", data=redEigVects, layout="np.array", val_type="float")
    return lowDDataMat, reconMat


@decorator.logging
@decorator.ignore_warning
def lda(c1, c2, topNfeat=Parameters.top_N_feature):
    """
    Dimension reduction
    :param c1:
    :param c2:
    :param topNfeat:
    :return:
    """
    m1 = np.mean(c1, axis=0)
    m2 = np.mean(c2, axis=0)
    c = np.vstack((c1, c2))
    m = np.mean(c, axis=0)

    n1 = c1.shape[0]
    n2 = c2.shape[0]

    s1 = 0
    for i in range(0, n1):
        s1 = s1 + (c1[i, :] - m1).T * (c1[i, :] - m1)
    s2 = 0
    for i in range(0, n2):
        s2 = s2 + (c2[i, :] - m2).T * (c2[i, :] - m2)
    Sw = (n1 * s1 + n2 * s2) / (n1 + n2)
    Sb = (n1 * (m - m1).T * (m - m1) + n2 * (m - m2).T * (m - m2)) / (n1 + n2)

    eigVals, eigVects = np.linalg.eig(np.mat(Sw).I * Sb)

    eigValInd = np.argsort(eigVals)  # 200
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # n
    # redEigVects = eigVects[eigValInd]  # n * N
    redEigVects = eigVects[:, eigValInd]  # N * n

    lowDDataMat = c * redEigVects  # M * n
    lowDDataMat = lowDDataMat.astype(float).tolist()  # M * n
    reconMat = (lowDDataMat * redEigVects.T) + m  # M * N
    reconMat = reconMat.astype(float).tolist()  # M * N

    indexVec = np.argsort(-eigVals)
    nLargestIndex = indexVec[:1]
    maxW = eigVects[:, nLargestIndex]
    return lowDDataMat, reconMat, redEigVects, maxW


@decorator.logging
def select_top_model(info, sm_hot, store):
    """
    Select the standard model to describe submarket
    :param info:
    :param sm_hot:
    :return:
    """
    info = info.drop_duplicates(["brand", "model", "submarket"])
    models = dict()
    for submarket, _ in sm_hot.items():
        info_bak = info[info["submarket"] == submarket].sort_values([Parameters.submarkets_sort_key], ascending=False)[
                   : Parameters.top_num_submarkets_as_standard]
        for k, v in info_bak.iterrows():
            models.setdefault(submarket, []).append("%s-%s" % (v["brand"], v["model"]))
    store.append("top_models", levels=2, data=models, layout="dict-list", val_type="str")
    return models


@decorator.logging
def model_map_to_line(models, map_line):
    """
    Map the standard model to line No.
    :param models:
    :param map_line:
    :return:
    """
    lines = dict()
    for submarket, model_list in models.items():
        lines[submarket] = [map_line[model] for model in model_list]
    return lines


@decorator.logging
def establish_portraits_sm(lines, mat, store):
    """
    Find the center coordinate as the vector of submarkets
    :param lines:
    :param mat:
    :return:
    """
    portraits = dict()
    for submarket, line_nos in lines.items():
        cluster = cluster_vec()
        for no in line_nos:
            cluster.send(mat[no])
        try:
            cluster.send(None)
        except StopIteration as e:
            portraits[submarket] = e.value
        except Exception as e:
            raise e
    store.append("portraits_sm", levels=2, data=portraits, layout="dict-list", val_type="float")
    return portraits


@decorator.logging
def establish_portraits_sm_by_lda(lines, mat, store):
    """
    Find the center coordinate as the vector of submarkets by lda
    :param line:
    :param mat:
    :param store:
    :return:
    """
    portraits = dict()
    mat = np.array(mat).astype(float)
    store.append("mat_org", data=mat, layout="np.array", val_type="float")
    for submarket, line_nos in lines.items():
        other_nos = [x for x in range(len(mat)) if x not in line_nos]
        lowDDataMat, reconMat, redEigVects, maxW = lda(c1=mat[other_nos, :], c2=mat[line_nos, :])

        store.append("mat_"+submarket, data=lowDDataMat, layout="np.array", val_type="float")
        store.append("mat_restore_"+submarket, data=reconMat, layout="np.array", val_type="float")
        store.append("mat_trans_"+submarket, data=redEigVects, layout="np.array", val_type="float")
        store.append("maxW_"+submarket, data=maxW, layout="np.array", val_type="float")

        cluster = cluster_vec()
        for no in line_nos:
            cluster.send(reconMat[no])
        try:
            cluster.send(None)
        except StopIteration as e:
            portraits[submarket] = e.value
        except Exception as e:
            raise e
    store.append("portraits_sm", levels=2, data=portraits, layout="dict-list", val_type="float")
    return portraits


@decorator.logging
def find_position(mat, map_line, portraits_sm, store):
    distances, similarity = dict(), dict()
    for model, vec_no in map_line.items():
        distances[model], similarity[model] = dict(), dict()
        vec, total = mat[vec_no], 0
        for submarket, portrait in portraits_sm.items():
            measure = cosine(vec, portrait)
            distances[model][submarket] = measure
            total += measure
        for submarket, measure in distances[model].items():
            try:
                similarity[model][submarket] = measure * 100 / total
            except ZeroDivisionError:
                similarity[model][submarket] = 0
    store.append("distances", levels=2, data=distances, layout="dict-dict", val_type="float")
    store.append("similarity", levels=2, data=similarity, layout="dict-dict", val_type="float")
    return distances, similarity


@decorator.logging
def append_method(mat, append_mat):
    for line in range(len(mat)):
        mat[line].extend(append_mat[line])
    return mat


@decorator.logging
def load_sku_json(sku_info):
    for k, v in sku_info.iterrows():
        sku = eval(v["sku"])
        for key, val in sku.items():
            sku_info.at[k, key] = val
    del sku_info["sku"]
    return sku_info


@decorator.logging
@decorator.ignore_warning
def map_feature_info(info, map_line, store, feature):
    info["serial"] = -1
    for k, v in info.iterrows():
        model = "%s-%s" % (no_blank(v["brand"]), no_blank(v["model"]))
        try:
            serial = map_line[model]
            info.at[k, "serial"] = serial
        except KeyError:
            # print("useless", v["brand"], v["model"])
            info.drop(k, axis=0, inplace=False)
    info = info[info["serial"] != -1]

    info.set_index(["serial"], inplace=True)
    all_index = set(map_line.values())
    has_index = set(info.index)
    loss_index = all_index - has_index
    print("Warning:")
    print("loss size", len(loss_index))
    print("loss rate", len(loss_index) / len(all_index))
    # for k, v in map_line.items():
    #     if v in loss_index:
    #         print(k)
    for inx in loss_index:
        info.loc[inx] = [-1] * len(info.columns)
    info = info.sort_index(ascending=True)

    mat = [[] for _ in range(len(map_line))]
    items = [col for col in info.columns if col not in ["brand", "model"]]
    for item in items:
        method = map_feature_func(item)
        vals = method(info[item].values)
        for vec, ele in zip(mat, vals):
            try:
                vec.extend(ele)
            except TypeError:
                vec.append(ele)
    store.append(feature, data=mat, layout="np.array", val_type="float")
    return mat


# ======================================================================================


def query_position(store):
    store.add_path_suffix("pca")
    map_line = store.load("map_line")
    mat = store.load("mat")
    portraits_sm = store.load("portraits_sm")
    map_urls = store.load("map_urls")
    similarity = store.load("similarity")
    while True:
        brand = input("\ninput brand:\n")
        model = input("\ninput model:\n")
        correct = input("\nconfirm input(input '#' to re input, input any other chars to confirm):\n")
        if "#" in correct:
            continue

        try:
            line_no = map_line["%s-%s" % (brand, model)]
        except Exception as e:
            print("error brand or model！")
            continue

        total, sim = 0, dict()
        print("\nDistance:")
        for submarket, portrait in portraits_sm.items():
            distince = cosine(mat[line_no], portrait)
            print(submarket, distince)
            total += distince
            sim[submarket] = distince

        print("\nSimilarity:")
        for submarket, distince in sim.items():
            print(submarket, round(distince*100/total, 2), "%")

        print("\nimage url:", map_urls["%s-%s" % (brand, model)])


def query_position_lda(store):
    map_line = store.load("map_line")
    mat = store.load("mat_org")
    portraits_sm = store.load("portraits_sm")
    map_urls = store.load("map_urls")
    similarity = store.load("similarity")
    while True:
        brand = input("\ninput brand:\n")
        model = input("\ninput model:\n")
        correct = input("\nconfirm input(input '#' to re input, input any other chars to confirm):\n")
        if "#" in correct:
            continue

        try:
            line_no = map_line["%s-%s" % (brand, model)]
        except Exception as e:
            print("error brand or model！")
            continue

        total, sim = 0, dict()
        print("\nDistance:")
        for submarket, portrait in portraits_sm.items():
            distince = cosine(mat[line_no], portrait)
            print(submarket, distince)
            total += distince
            sim[submarket] = distince

        print("\nSimilarity:")
        for submarket, distince in sim.items():
            print(submarket, round(distince * 100 / total, 2), "%")

        print("\nimage url:", map_urls["%s-%s" % (brand, model)])


def arrange_sim(pcid, cid, suffix=None):
    from store import Store
    store = Store(pcid=pcid, cid=cid)
    if suffix is not None:
        store.add_path_suffix(suffix)
    similarity = store.load("similarity")
    map_urls = store.load("map_urls")

    data = list()
    for key, sub_dict in similarity.items():
        line = list()
        for sub_key, val in sub_dict.items():
            line.append([sub_key, val])
        line.sort(key=lambda x: x[1], reverse=True)
        record = [key]
        for item in line:
            record.append(item[0])
            record.append(str(round(item[1], 2)) + "%")
        try:
            record.append(map_urls[key])
            data.append(record)
        except KeyError:
            print("map url KeyError", key)

    columns = ["brand-model", "1st sm", "1st share", "2nd sm", "2nd share", "3rd sm", "3rd share",
               "4th sm", "4th share", "5th sm", "5th share", "image_url"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(store.path.format(name="arrange_result").replace(".txt", ".csv"), encoding="utf-8")
