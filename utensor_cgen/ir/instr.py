class MemoryAllocation(object):
  _store_name = "address"
  
  def __init__(self):
    self._address_map = {}

  def __setattr__(self, attr, value):
    if attr == '_address_map':
      return super(MemoryAllocation, self).__setattr__(attr, value)
    self._address_map[attr] = value
  
  def __getattr__(self, att):
    return self._address_map[att]

  def __deepcopy__(self, memo):
    new_obj = MemoryAllocation()
    new_obj._address_map = {k: v for k, v in self._address_map.items()}
    return new_obj

class DataManager(object):

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

  def __getattr__(self, attr):
    if attr == 'StorageCenter':
      raise AttributeError('StorageCenter')
    elif attr.startswith('__'):
      return super(DataManager, self).__getattr__(attr)
    cls_instance = self.StorageCenter[attr]
    return cls_instance

  def __setattr__(self, attr, value):
    if attr == 'StorageCenter':
      return super(DataManager, self).__setattr__(attr, value)
    cls_instance = self.StorageCenter[attr]
    k, v = value
    cls_instance.__setattr__(k, v)

  def group(self, tensor):
    ret = {}
    for cls_object in self.StorageCenter.values():
      ans = cls_object.__getattr__(tensor)
      ret.update({cls_object._store_name:  ans})
    return ret
