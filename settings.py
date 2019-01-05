# -*- coding:utf-8 -*-

from DBparam import *


DBDefault = Outer99


class Entrance(object):
    def __init__(self, pcid="4", cid="50012097", datamonth="201805"):
        self.__pcid = str(pcid)
        self.__cid = str(cid)
        self.__datamonth = str(datamonth)

    @property
    def pcid(self):
        return self.__pcid

    @property
    def cid(self):
        return self.__cid

    @property
    def datamonth(self):
        return self.__datamonth


class Parameters(object):
    dim_targets = 200
    floor_submarkets = 20
    top_num_submarkets_as_standard = 20
    top_per_submarkets_as_standard = 0.2
    top_N_feature = 50
    submarkets_sort_key = "biz"


class FileBase(object):
    info = "data/info_{name}_pcid{pcid}cid{cid}.csv"
    result = "result/pcid{pcid}cid{cid}/"


class Split(object):
    split_line = "*********************\n"
    key_split = " =+= "
    key_sub_split = ", "
    key_value_split = ": "
    list_split = " "


class EraseItem(object):
    erase_target = dict()
    erase_submarket = dict()

    wrong_all = []

    items = []
    items.extend(wrong_all)
    erase_target["50012097"] = set(items)

    items = []
    items.extend(wrong_all)
    erase_submarket["50012097"] = set(items)


class FeatureType(object):
    sale_type = ("price", "aver_price", "biz30day", "total_sold_price")
    sale_type = set(sale_type)

    continuous_type = []
    continuous_type = set(continuous_type)

    mul_con_type = []
    mul_con_type = set(mul_con_type)

    # default type
    discrete_type = []
    discrete_type = set(discrete_type)

    mul_dis_type = ["功能", "使用功能", "产品功能", "颜色", "颜色分类", "适用人数"]
    mul_dis_type = set(mul_dis_type)
