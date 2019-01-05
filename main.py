# -*- coding:utf-8 -*-

from get_data import *
from tasks import *
from store import Store


class Manager(object):
    def __init__(self, pcid, cid, datamonth):
        self.pcid = pcid
        self.cid = cid
        self.datamonth = datamonth
        
        self.save = Store(pcid, cid)

    @decorator.process
    def make_mat(self):
        """
        Select the useful targets and submarkets
        Organize raw data
        Embedding the scores and make the matrix
        Construct the relationship between line No. and model
        PCA dimension reduction
        :return:
        """
        info_target = get_targets_info(self.pcid, self.cid, self.datamonth, src="LOCAL")
        targets = select_targets(info_target, store=self.save)

        urls = get_urls(self.pcid, self.cid, self.datamonth, src="LOCAL")
        _ = map_model_urls(urls, self.save)

        info_submarket = get_submarket_info(self.pcid, self.cid, self.datamonth, src="LOCAL")
        _, sm_hot, = select_submarkets(info_submarket, store=self.save)

        info = reconstruct_info(info_target, info_submarket, targets, sm_hot)
        mat, map_line = embedding(info, targets, self.save)
        pca_mat, _ = pca(mat, self.save)

        # mat, map_line = embedding(info, targets, self.save, normal="NORMAL")
        # pca_mat = mat

        return pca_mat, map_line, info, sm_hot

    @decorator.process
    def describe_submarket(self, mat, map_line, info, sm_hot):
        """
        Describe submarkets as vector
        :param mat:
        :param map_line:
        :param info:
        :param sm_hot:
        :return:
        """
        models = select_top_model(info, sm_hot, self.save)
        lines = model_map_to_line(models, map_line)

        portraits_sm = establish_portraits_sm(lines, mat, self.save)
        # portraits_sm = establish_portraits_sm_by_lda(lines, mat, self.save)

        return portraits_sm

    @decorator.process
    def position_models(self, mat, map_line, portraits_sm):
        """
        Calculate distance between every model and every submarket
        :param mat:
        :param map_line:
        :param portraits_sm:
        :return:
        """
        distances, similarity = find_position(mat, map_line, portraits_sm, self.save)
        return distances, similarity

    @decorator.process
    def append_feature(self, mat, map_line):
        sale_info = get_sale_info(self.pcid, self.cid, self.datamonth, src="LOCAL")
        sale_mat = map_feature_info(sale_info, map_line, self.save, "sale")
        mat = append_method(mat, sale_mat)

        sku_info = get_sku_info(self.pcid, self.cid, self.datamonth, src="LOCAL")
        # getting sku from table product_brain need loading json
        sku_info = load_sku_json(sku_info)
        sku_mat = map_feature_info(sku_info, map_line, self.save, "sku")
        mat = append_method(mat, sku_mat)

        self.save.append("mat_plus", data=mat, layout="np.array", val_type="float")
        return mat

    @decorator.process
    def run(self):
        """
        Process the tasks and save datum and results
        :return:
        """
        mat, map_line, info, sm_hot = self.make_mat()
        mat = self.append_feature(mat, map_line)
        portraits_sm = self.describe_submarket(mat, map_line, info, sm_hot)
        _, _ = self.position_models(mat, map_line, portraits_sm)
        self.save.save()

    @decorator.process
    def query(self, method="PCA"):
        """
        Query the distance and similarity between model and hot submarkets
        : demo: brand 1号宝贝 model OUYOU0011
        : demo: brand fronton/弗朗顿 model CB-666D
        :return:
        """
        if "LDA" == method:
            query_position_lda(self.save)
        else:
            query_position(self.save)

    def main(self):
        choice = input("\ninput 0 to generate result，input 1 to query result:\n")
        if "0" == str(choice):
            self.run()
        else:
            method = input("\ninput suffix:\n")
            self.query(method)


if __name__ == '__main__':
    enter = Entrance(pcid="4", cid="50012097", datamonth="201805")
    pcid = enter.pcid
    cid = enter.cid
    datamonth = enter.datamonth

    # save = Store(pcid, cid)
    # mat = save.load("mat")
    # print(mat)
    # exit()

    obj = Manager(pcid=pcid, cid=cid, datamonth=datamonth)
    obj.run()
    # obj.query()
    # main()

    # arrange_sim(pcid=pcid, cid=cid, suffix="pca")
    arrange_sim(pcid=pcid, cid=cid)
