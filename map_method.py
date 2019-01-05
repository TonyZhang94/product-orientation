# -*- coding:utf-8 -*-


def map_feature_sale(vals, div=10, celling=1):
    sort_vals = sorted([x for x in vals if x != -1], reverse=False)
    serial, size, prev = dict(), 0, -1
    for num in sort_vals:
        if num != prev:
            size += 1
            prev = num
        serial[num] = size

    rep = list()
    block = celling / div
    for val in vals:
        if -1 == val:
            rep.append(0.0)
        else:
            mul = serial[val] * div // size
            if mul * size < serial[val] * div:
                mul += 1
            rep.append(mul*block)
    return rep


def map_feature_continuous(vals):
    print("continuous")
    return [0] * len(vals)


def map_feature_mul_con(vals):
    print("mul continuous")
    return [0] * len(vals)


def map_feature_discrete(vals):
    print("discrete")
    return [0] * len(vals)


def map_feature_mul_dis(vals):
    print("mul discrete")
    vec_items = list()
    count = dict()
    for val in vals:
        try:
            if "" == val:
                raise Exception
            val = val.replace("\xa0", " ").replace(",", " ").replace("，", " ").replace("、", " ")
            vec_item = val.split()
        except Exception as e:
            print("mul discrete split error", val)
            print(e)
            continue
        vec_items.append(set(vec_item))
        for ele in vec_item:
            try:
                count[ele] += 1
            except KeyError:
                count[ele] = 1
    items = sorted(count.items(), key=lambda d: d[1], reverse=True)
    map_pos = {k[0]: rank for k, rank in zip(items[: 10], range(10))}
    useful_item = set(map_pos.keys())

    mat = list()
    for vec_item in vec_items:
        vec = [0] * 10
        vec_item &= useful_item
        for item in vec_item:
            vec[map_pos[item]] = 1
        mat.append(list(vec))
    print(map_pos)
    return mat


def attr_name(vals):
    """identify attr method"""


if __name__ == '__main__':
    vals = [2, 3, 4, 5, 6, 7, 8, -1, 1, 9, -1, 9, 5, 10]
    print("len", len(vals))
    ret = map_feature_sale(vals)
    print(ret)

    s = None
    s.split()
