class MemoryAllocation:
    _store_name = "address"
    
    def __init__(self):
        self._address_map = {}

    def __setattr__(self,att,value):
        self._address_map[att] = value
    
    def __getattr__(self, att):
        return self._address_map[att]

class DataManager:
    InfoCenterMap = {
        MemoryAllocation.store_name: MemoryAllocation
    }
    def __init__(self, datas):
        self.StorageCenter = {}
        for data_name, kwargs in datas:
            data_cls = self.InfoCenterMap.get(data_name, None)
            if data_cls is None:
                raise ValueError("Unknown transformation method: {}".format(data_name))
            datastorage = data_cls(**kwargs)
            self.StorageCenter.update({data_name: datastorage})

    @classmethod
    def register_datamap(cls, data_cls, overwrite=False):
        cls.InfoCenterMap[data_cls.name] = data_cls

    def __getattr__(self,attr):
        cls_instance = self.StorageCenter[attr]
        return cls_instance

    def __setattr__(self,attr,value):
        cls_instance = self.StorageCenter[attr]
        k, v= value
        cls_instance.__setattr__(self, k, v)

    def group(self, op):
        ret = {}
        for key, cls_object in self.StorageCenter.items():
            ans = cls_object.__getattr__(op)
            ret.update({cls_object._store_name:  ans})
        return ret



