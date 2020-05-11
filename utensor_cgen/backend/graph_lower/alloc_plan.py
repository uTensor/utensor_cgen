import attr
import six
from attr.validators import instance_of

from utensor_cgen.logger import logger

__all__ = ['TimeslotAllocation', 'SpaceAllocation', 'TimeSpaceAllocation', 'AllocationPlan']

@attr.s
class TimeslotAllocation(object):
  time_slot_start = attr.ib(validator=instance_of(int))
  # if time_slot_end is None, it's a time sapn with no end
  time_slot_end = attr.ib(validator=instance_of((int, type(None))))

  def __attrs_post_init__(self):
    assert self.time_slot_start >= 0, \
      'invalid time_slot_start: %s' % self.time_slot_start
    if self.time_slot_end is not None:
      assert self.time_slot_end >= self.time_slot_start, \
        'invalid time_slot_end: %s ~ %s' % (self.time_slot_start, self.time_slot_end)
  
  def __contains__(self, slot):
    assert isinstance(slot, int), 'incorrect slot type: %s' % type(slot)
    is_in = self.time_slot_start <= slot
    if self.time_slot_end is not None:
      is_in = is_in and slot <= self.time_slot_end
    return is_in


@attr.s
class SpaceAllocation(object):
  offset_start = attr.ib(validator=instance_of(int))
  size = attr.ib(validator=instance_of(int))
  data_alignment = attr.ib(validator=instance_of(int))
  offset_end = attr.ib(init=False)

  def __attrs_post_init__(self):
    assert self.offset_start >= 0, \
      'invalid offset_start: %s' % self.offset_start
    assert self.size > 0, \
      'invalid size: %s' % self.size
    errmsg = ( self.data_alignment > 1 and
      'the memory offset is not aligned: %s (not %ss aligned)' or
      'the memory offset is not aligned: %s (not %s aligned)'
    )
    assert self.size % self.data_alignment == 0, \
      errmsg  % (self.size, self.data_alignment)
    self.offset_end = self.offset_start + self.size - 1
  
  def __contains__(self, offset):
    return self.offset_start <= offset <= self.offset_end


@attr.s
class TimeSpaceAllocation(object):
  entity_name = attr.ib(validator=instance_of(six.string_types))
  _time_alloc = attr.ib(validator=instance_of(TimeslotAllocation), repr=False)
  _space_alloc = attr.ib(validator=instance_of(SpaceAllocation), repr=False)
  time_slot_start = attr.ib(init=False)
  time_slot_end = attr.ib(init=False)
  offset_start = attr.ib(init=False)
  offset_end = attr.ib(init=False)
  size = attr.ib(init=False)

  def __attrs_post_init__(self):
    self.time_slot_start = self._time_alloc.time_slot_start
    self.time_slot_end = self._time_alloc.time_slot_end
    self.offset_start = self._space_alloc.offset_start
    self.offset_end = self._space_alloc.offset_end
    self.size = self._space_alloc.size

  @classmethod
  def init(cls, entity_name, time_slot_start, time_slot_end, offset_start, size):
    time_alloc = TimeslotAllocation(time_slot_start, time_slot_end)
    space_alloc = SpaceAllocation(offset_start, size)
    return cls(
      entity_name=entity_name,
      time_alloc=time_alloc,
      space_alloc=space_alloc
    )
  
  def is_alive_in_timeslot(self, time_slot):
    return time_slot in self._time_alloc
  
  def is_occupied(self, offset):
    return offset in self._space_alloc


class AllocationPlan(object):

  def __init__(self, allocs, total_size):
    for alloc in allocs:
      if not isinstance(alloc, TimeSpaceAllocation):
        raise ValueError(
          'expecting value of {} of type {}, get {}'.format(k, TimeSpaceAllocation, type(v))
        )
    self.plan = {alloc.entity_name: alloc for alloc in allocs}
    self.total_size = total_size

  def __setitem__(self, entity_name, alloc):
    if not isinstance(alloc, TimeSpaceAllocation):
      raise ValueError(
        'the value should be of type {}, get {}'.format(TimeSpaceAllocation, type(alloc))
      )
    if entity_name in self._plan:
      logger.warning(
        'duplicate entity_name detected: {}'.format(entity_name)
      )
    self._plan[entity_name] = alloc

  def __getitem__(self, entity_name):
    if entity_name not in self.plan:
      raise KeyError('%s not found' % entity_name)
    return self.plan[entity_name]
  
  def __contains__(self, entity_name):
    return entity_name in self.plan
  
  def __delitem__(self, entity_name):
    del self.plan[entity_name]
  
  def __getattr__(self, attr_name):
    return getattr(self.plan, attr_name)
