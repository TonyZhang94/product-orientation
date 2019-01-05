# -*- coding:utf-8 -*-

import sqlalchemy as sa
import numpy as np
import pandas as pd

import decorator
from settings import DBDefault
from settings import FileBase


class ReadDBException(Exception):
    """Choose Get Data From DB"""


@decorator.connect
def get_engine(info, db):
    return sa.create_engine("postgresql://{USER}:{PASS}@{HOST}:{PORT}/{DB}".
                            format(USER=info.user,
                                   PASS=info.password,
                                   HOST=info.host,
                                   PORT=info.port,
                                   DB=info.DB[db]))


def read_data(src, fname, sql, db):
    try:
        if "DEFAULT" == src:
            raise ReadDBException
        df = pd.read_csv(fname, encoding="utf-8", index_col=0)
    except (FileExistsError, FileNotFoundError, ReadDBException) as flag:
        try:
            engine = get_engine(DBDefault, db)
        except Exception as e:
            raise e

        try:
            print(sql)
            df = pd.read_sql_query(sql, engine)
        except Exception as e:
            raise e
        else:
            if flag.__doc__ != "Choose Get Data From DB":
                df.to_csv(fname, encoding="utf-8")
    except Exception as e:
        raise e
    return df


@decorator.logging
def get_targets_info(pcid, cid, datamonth=None, src="DEFAULT"):
    fname = FileBase.info.format(name="targets", pcid=pcid, cid=cid)
    fields = ["brand", "model", "tag", "target", "grade", "frequency"]
    field, table = ", ".join(fields), "comment.review_analysis_pcid{pcid}_cid{cid}".format(pcid=pcid, cid=cid)
    if datamonth is not None:
        sql = "SELECT {field} FROM {table} WHERE datamonth='{datamonth}';".format(
            field=field, table=table, datamonth=datamonth)
    else:
        sql = "SELECT {field} FROM {table};".format(field=field, table=table)
    return read_data(src, fname=fname, sql=sql, db="report_dg")


@decorator.logging
def get_urls(pcid, cid, datamonth=None, src="DEFAULT"):
    fname = FileBase.info.format(name="model-urls", pcid=pcid, cid=cid)
    fields = ["brand", "model", "imageurl"]
    field, table = ", ".join(fields), "product_brain.product_brain_pcid{pcid}".format(pcid=pcid, cid=cid)
    if datamonth is not None:
        sql = "SELECT {field} FROM {table} WHERE cid='{cid}' and datamonth='{datamonth}';".format(
            field=field, table=table, cid=cid, datamonth=datamonth)
    else:
        sql = "SELECT {field} FROM {table} WHERE cid='{cid}';".format(field=field, table=table, cid=cid)
    return read_data(src, fname=fname, sql=sql, db="report_dg")


@decorator.logging
def get_submarket_info(pcid, cid, datamonth=None, src="DEFAULT"):
    fname = FileBase.info.format(name="submarkets", pcid=pcid, cid=cid)
    fields = ["brand", "model", "biz30day", "total_sold_price", "submarket", "target_score"]
    field, table = ", ".join(fields), "product_brain.product_brain_pcid{pcid}".format(pcid=pcid, cid=cid)
    if datamonth is not None:
        sql = "SELECT {field} FROM {table} WHERE cid='{cid}' and datamonth='{datamonth}';".format(
            field=field, table=table, cid=cid, datamonth=datamonth)
    else:
        sql = "SELECT {field} FROM {table} WHERE cid='{cid}';".format(field=field, table=table, cid=cid)
    return read_data(src, fname=fname, sql=sql, db="report_dg")


# @decorator.logging
# def get_sku_info(pcid, cid, src="DEFAULT"):
#     fname = FileBase.info.format(name="sku", pcid=pcid, cid=cid)
#     fields = ["*"]
#     field, table = ", ".join(fields), "sd_model_pcid{pcid}.cid{cid}".format(pcid=pcid, cid=cid)
#     sql = "SELECT {field} FROM {table};".format(field=field, table=table)
#     return read_data(src, fname=fname, sql=sql, db="standard_library")

@decorator.logging
def get_sku_info(pcid, cid, datamonth, src="DEFAULT"):
    fname = FileBase.info.format(name="sku", pcid=pcid, cid=cid)
    fields = ["brand", "model", "sku"]
    field, table = ", ".join(fields), "product_brain.product_brain_pcid{pcid}".format(pcid=pcid, cid=cid)
    if datamonth is not None:
        sql = "SELECT {field} FROM {table} WHERE cid='{cid}' and datamonth='{datamonth}';".format(
            field=field, table=table, cid=cid, datamonth=datamonth)
    else:
        sql = "SELECT {field} FROM {table} WHERE cid='{cid}';".format(field=field, table=table, cid=cid)
    return read_data(src, fname=fname, sql=sql, db="report_dg")


@decorator.logging
def get_sale_info(pcid, cid, datamonth=None, src="DEFAULT"):
    fname = FileBase.info.format(name="sale", pcid=pcid, cid=cid)
    fields = ["brand", "model", "price", "biz30day", "total_sold_price"]
    field, table = ", ".join(fields), "product_brain.product_brain_pcid{pcid}".format(pcid=pcid, cid=cid)
    if datamonth is not None:
        sql = "SELECT {field} FROM {table} WHERE cid='{cid}' and datamonth='{datamonth}';".format(
            field=field, table=table, cid=cid, datamonth=datamonth)
    else:
        sql = "SELECT {field} FROM {table} WHERE cid='{cid}';".format(field=field, table=table, cid=cid)
    return read_data(src, fname=fname, sql=sql, db="report_dg")


if __name__ == '__main__':
    from settings import Entrance
    enter = Entrance()
    pcid = enter.pcid
    cid = enter.cid
    datamonth = enter.datamonth
    sku = get_sku_info(pcid, cid, datamonth)
    print(sku)
    sale = get_sale_info(pcid, cid, datamonth)
    print(sale)
