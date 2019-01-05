# -*- coding:utf-8 -*-

import os
import numpy as np

from settings import FileBase, Split


class Store(object):
    def __init__(self, pcid, cid):
        self.pcid = pcid
        self.cid = cid
        self.path = FileBase.result.format(pcid=pcid, cid=cid)

        self.store = dict()
        try:
            os.makedirs(self.path)
        except (FileExistsError, FileNotFoundError) as e:
            print(self.path, "is exist")
        except Exception as e:
            raise e
        else:
            print("create", self.path, "success")
        finally:
            self.path += "{name}.txt"

        self.params = self.path.format(name="settings")
        self.params_dict = None

    def add_path_suffix(self, suffix):
        self.path = "%s_%s/%s" % (self.path[:-11], suffix, "{name}.txt")
        self.params = self.path.format(name="settings")

    def append(self, key, **kwargs):
        self.store[key] = {k: v for k, v in kwargs.items()}

    def save_dict_dict(self, name, data):
        fname = self.path.format(name=name)
        with open(fname, mode="w", encoding="utf-8") as fp:
            for key, val in data["data"].items():
                line = key + Split.key_split
                line = line + Split.key_sub_split.join(k+Split.key_value_split+str(v) for k, v in val.items()) + "\n"
                fp.write(line)

    def save_dict_list(self, name, data):
        fname = self.path.format(name=name)
        with open(fname, mode="w", encoding="utf-8") as fp:
            for key, val in data["data"].items():
                line = key + Split.key_split
                line = line + Split.list_split.join(map(str, val)) + "\n"
                fp.write(line)

    def save_np_array(self, name, data):
        fname = self.path.format(name=name).replace(".txt", "")
        np.save(fname, data["data"])

    def save_dict(self, name, data):
        fname = self.path.format(name=name)
        with open(fname, mode="w", encoding="utf-8") as fp:
            for key, val in data["data"].items():
                line = key + Split.key_split + str(val) + "\n"
                fp.write(line)

    def save(self):
        with open(self.params, mode="w", encoding="utf-8") as params_file:
            for name, val in self.store.items():
                params = ", ".join("%s: %r" % (k, v) for k, v in val.items() if k != "data")
                params_file.write("{name} = {params}\n".format(name=name, params=params))

                layout = val["layout"]
                if "sub_name" in val.keys():
                    name = "%s-%s" % (name, val["sub_name"])
                if "dict-dict" == layout:
                    self.save_dict_dict(name, val)
                elif "dict-list" == layout:
                    self.save_dict_list(name, val)
                elif "np.array" == layout:
                    self.save_np_array(name, val)
                elif "dict" == layout:
                    self.save_dict(name, val)

    def load_settings(self):
        print("read settings...")
        self.params_dict = dict()
        with open(self.params, mode="r", encoding="utf-8") as params_file:
            for line in iter(params_file.readline, ""):
                try:
                    name, params = line.strip().split(" = ")
                except ValueError:
                    pass
                except Exception as e:
                    raise e
                else:
                    params = params.strip().split(", ")
                    self.params_dict[name] = dict()
                    for param in params:
                        key, val = param.split(": ")
                        try:
                            self.params_dict[name][key] = int(val)
                        except ValueError:
                            self.params_dict[name][key] = val

    def load_dict_dict(self, name, val_type):
        fname = self.path.format(name=name)
        data = dict()
        with open(fname, mode="r", encoding="utf-8") as fp:
            for line in iter(fp.readline, ""):
                name, line = line.strip().split(Split.key_split)
                data[name] = dict()
                key_val_list = line.split(Split.key_sub_split)
                for key_val in key_val_list:
                    key, val = key_val.split(Split.key_value_split)
                    if "float" == val_type:
                        data[name][key] = float(val)
                    elif "int" == val_type:
                        data[name][key] = int(val)
                    else:
                        data[name][key] = val
        return data

    def load_dict_list(self, name, val_type):
        fname = self.path.format(name=name)
        data = dict()
        with open(fname, mode="r", encoding="utf-8") as fp:
            for line in iter(fp.readline, ""):
                name, line = line.strip().split(Split.key_split)
                if "float" == val_type:
                    data[name] = list(map(float, line.split()))
                elif "int" == val_type:
                    data[name] = list(map(int, line.split()))
                else:
                    data[name] = line.split()
        return data

    def load_np_array(self, name, val_type):
        fname = self.path.format(name=name).replace(".txt", ".npy")
        data = np.load(fname)
        if "float" == val_type:
            data = data.astype(float)
        return data.tolist()

    def load_dict(self, name, val_type):
        fname = self.path.format(name=name)
        data = dict()
        with open(fname, mode="r", encoding="utf-8") as fp:
            for line in iter(fp.readline, ""):
                name, val = line.strip().split(Split.key_split)
                if "float" == val_type:
                    data[name] = float(val)
                elif "int" == val_type:
                    data[name] = int(val)
                else:
                    data[name] = val
        return data

    def load(self, name):
        if self.params_dict is None:
            self.load_settings()

        params = self.params_dict[name]
        layout = params["layout"].replace("'", "")
        val_type = params["val_type"].replace("'", "")
        if "dict-dict" == layout:
            data = self.load_dict_dict(name, val_type)
        elif "dict-list" == layout:
            data = self.load_dict_list(name, val_type)
        elif "np.array" == layout:
            data = self.load_np_array(name, val_type)
        elif "dict" == layout:
            data = self.load_dict(name, val_type)
        else:
            print("file", name, "doesn't exist!")
            data = None
        return data
