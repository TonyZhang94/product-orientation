# -*- coding:utf-8 -*-


class Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def func(self, raw):
        ret = dict(raw)
        for key, value in raw.items():
            record_type = key
            for record in value:
                print(record)
                key = "%s.%s" % (record_type, record['serial'])
                record['serial'] = key
                ret[key] = Record(**record)
                print(record)
        return ret


class LineItem(object):
    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price

    def get_weight(self):
        return self.__weight

    def set_weight(self, value):
        if value > 0:
            self.__weight = value
        else:
            raise ValueError("value must be > 0")

    weight = property(fget=get_weight, fset=set_weight)


class Quantity(object):
    __counter = 0

    def __init__(self):
        cls = self.__class__
        prefix = cls.__name__
        index = cls.__counter
        self.storage_name = "_{}#{}".format(prefix, index)
        cls.__counter += 1

    def __get__(self, instance, owner):
        print("get", self.storage_name)
        return getattr(instance, self.storage_name)

    def __set__(self, instance, value):
        print("set", self.storage_name)
        if value > 0:
            setattr(instance, self.storage_name, value)
        else:
            raise ValueError("value must be > 0")


class LineItem1(object):
    weight = Quantity()
    price = Quantity()

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price


if __name__ == '__main__':
    obj1 = LineItem1("rbsml", 1, 20)
    obj2 = LineItem1("rbsml", 1, 20)
    print(obj1.__dict__)
    print(obj1.weight)
    print(type(obj1.weight))

    print(LineItem1.__dict__)

    obj2 = LineItem1("rbsml", 1, 20)
    print(obj2.__dict__)

    exit(0)
    obj = LineItem("rbsml", 1, 20)
    print(obj.subtotal())

    obj = Record(pcid="50012097", cid="4", datamonth="201805")
    print(obj.pcid, obj.cid, obj.datamonth)

    raw = dict()
    d = dict()
    d["serial"] = "1"
    d["key1"] = 2
    d["key2"] = [1, 2, "3", "4"]
    d["dict"] = dict()
    d["dict"]["abc"] = "abc"
    raw["d"] = [d]
    ret = obj.func(raw)
    print(ret)
