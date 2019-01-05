# -*- coding:utf-8 -*-

import inspect

from settings import *
import decorator
import map_method


def sum_freq(df):
    total = df["frequency"].sum()
    df["frequency"] = total
    return df[:1]


def make_trantab(src, dst=""):
    trantab = dict()
    for item in src:
        trantab[ord(item)] = dst
    return trantab


def sum_score(df):
    total = df["frequency"].sum()
    tag, target = df["tag"].values[0], df["target"].values[0]
    df["score"], df["target"] = total, "%s-%s" % (tag, target)
    return df[:1]


def embedding_one(df, targets, mat, map_line):
    vector = [0] * len(targets)
    for k, v in df.iterrows():
        if "" == v["target"]:
            continue
        position = targets[v["target"]]
        vector[position] = int(v["score"])

    key = "%s-%s" % (no_blank(df["brand"].values[0]), no_blank(df["model"].values[0]))
    if key not in map_line.keys():
        map_line[key] = len(mat)
        mat.append(vector)


def embedding_one_normal(df, targets, mat, map_line):
    vector = [0] * len(targets)
    for k, v in df.iterrows():
        if "" == v["target"]:
            continue
        position = targets[v["target"]]
        vector[position] = int(v["score"])

    total = sum(vector)
    try:
        vector = [x/total for x in vector]
    except ZeroDivisionError:
        pass

    key = "%s-%s" % (no_blank(df["brand"].values[0]), no_blank(df["model"].values[0]))
    if key not in map_line.keys():
        map_line[key] = len(mat)
        mat.append(vector)


@decorator.coroutine
def cluster_vec():
    # total = [0] * Parameters.top_N_feature
    total = [0] * Parameters.dim_targets
    count = 0
    while True:
        vec = yield
        if vec is None:
            return [x/count for x in total]
        total = [x+y for x, y in zip(total, vec)]
        count += 1


def cosine(vec1, vec2):
    dot_product = sum([x * y for x, y in zip(vec1, vec2)])
    module1 = sum([x*x for x in vec1]) ** 0.5
    module2 = sum([x*x for x in vec2]) ** 0.5
    try:
        return abs(dot_product / (module1 * module2))
    except ZeroDivisionError:
        return 0


def no_blank(s):
    s.replace("\\xa", "")
    return s.strip()


def map_feature_func(item):
    print(item, end=" ")
    funcs = {name: func for name, func in inspect.getmembers(map_method, inspect.isfunction)}
    if item in funcs.keys():
        return funcs[item]
    elif item in FeatureType.sale_type:
        return funcs["map_feature_sale"]
    elif item in FeatureType.continuous_type:
        return funcs["map_feature_continuous"]
    elif item in FeatureType.mul_con_type:
        return funcs["map_feature_mut_con"]
    elif item in FeatureType.discrete_type:
        return funcs["map_feature_discrete"]
    elif item in FeatureType.mul_dis_type:
        return funcs["map_feature_mul_dis"]
    else:
        return funcs["map_feature_discrete"]


if __name__ == '__main__':
    items = ["price", "continuous", "discrete", "test", "other", "产品功能"]
    for item in items:
        func = map_feature_func(item)
        print(item, func)
        func(1)
