class MemoryAllocation:
    _store_name = "address"
    
    def __init__(self):
        self._address_map = {}

    def __setattr__(self, attr, value):
        if attr == '_address_map':
            return super().__setattr__(attr, value)
        self._address_map[attr] = value
    
    def __getattr__(self, att):
        return self._address_map[att]

class DataManager:
    InfoCenterMap = {
        MemoryAllocation._store_name: MemoryAllocation
    }
    def __init__(self, datas):
        self.StorageCenter = {}
        for data_name in datas:
            data_cls = self.InfoCenterMap.get(data_name, None)
            if data_cls is None:
                raise ValueError("Unknown transformation method: {}".format(data_name))
            datastorage = data_cls()
            self.StorageCenter.update({data_name: datastorage})

    @classmethod
    def register_datamap(cls, data_cls, overwrite=False):
        cls.InfoCenterMap[data_cls.name] = data_cls

    def __getattr__(self,attr):
        if attr == 'StorageCenter':
            raise AttributeError('StorageCenter')
        cls_instance = self.StorageCenter[attr]
        return cls_instance

    def __setattr__(self,attr,value):
        if attr == 'StorageCenter':
            return super().__setattr__(attr, value)
        cls_instance = self.StorageCenter[attr]
        k, v= value
        cls_instance.__setattr__(k, v)

    def group(self, tensor):
        ret = {}
        for key, cls_object in self.StorageCenter.items():
            ans = cls_object.__getattr__(tensor)
            ret.update({cls_object._store_name:  ans})
        return ret



