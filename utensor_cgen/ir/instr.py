class MemoryAllocation:
    store_name = "MemoryInfo"
    
    def __init__(self):
        self._address_map = {}

    def __setattr__(self,att,value):
        self._address_map[att] = value

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
        return getattr(cls_instance, attr)

    def __setattr__(self,attr,value):
        cls_instance = self.StorageCenter[attr]
        k, v= value
        cls_instance.__setattr__(self, k, v)




