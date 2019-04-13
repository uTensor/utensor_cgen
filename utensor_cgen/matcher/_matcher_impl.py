from collections import deque
from itertools import product

import attr
from attr.validators import instance_of

from utensor_cgen.ir import MetaOperationInfo, OperationInfo, uTensorGraph
from utensor_cgen.matcher._morphism import IdentityMorphism, Morphism
from utensor_cgen.utils import ops_bfs_queue

__all__ = ["uTensorGraphMatcher"]

@attr.s(frozen=True, slots=True)
class OpEqualityDelegate(object):

  # op_type -> list[tuple] (permutations)
  _association_map = {}
  # op_type -> dict[op_type] -> morphism
  _compatibility_map = {}

  @classmethod
  def is_associative(cls, permutations):
    def deco(op):
      if op.op_type in cls._association_map:
        raise ValueError(
          "duplicate associativity definition found for {}".format(op.op_type)
        )
      assert (
        isinstance(permutations, tuple) and 
        all([isinstance(perm, tuple) for perm in permutations])
      ), "`permutations` should be tuple of int tuples"
      cls._association_map[op.op_type] = permutations
      return op
    return deco

  @classmethod
  def is_compatible_with(cls, other_op_type, morphism_type, **kwargs):
    if not issubclass(morphism_type, Morphism):
      raise ValueError(
        'expecting Morphism for `morphism`, get {}'.format(morphism_type)
      )
    def deco(op):
      if op.op_type not in cls._compatibility_map:
        cls._compatibility_map[op.op_type] = {}
      if other_op_type in cls._compatibility_map[op.op_type]:
        raise RuntimeError(
          "Multiple morphisms from {} to {} detected".format(op.op_type, other_op_type)
        )
      if not other_op_type in cls._compatibility_map[op.op_type]:
        cls._compatibility_map[op.op_type][other_op_type] = morphism_type(**kwargs)
      return op
    return deco

  @classmethod
  def query(cls, sub_op, patrn_op):
    """
    Parameters
    ----------
    sub_op : OperationInfo
      the op in the subject ugraph
    patrn_op : OperationInfo
      the op in the pattern ugraph to match with

    Return
    ------
    is_eq : bool
      these two ops are equivalent if True, False o.w.
    equivalent_ops : List[OperationInfo]
      a list of equivalent ops derieved from `sub_op`
    """
    # to activate all configurations
    import utensor_cgen.backend.operators as _

    is_eq = False
    equivalent_ops = []
    if sub_op.op_type == patrn_op.op_type and sub_op.op_type not in cls._association_map:
      is_eq = True
      equivalent_ops = [patrn_op]
    elif sub_op.op_type == patrn_op.op_type:
      is_eq = True
      equivalent_ops = []
      for perm in cls._association_map[sub_op.op_type]:
        equivalent_ops.append(
          OperationInfo(
            name=sub_op.name,
            backend=sub_op.backend,
            ugraph=sub_op.ugraph,
            input_tensors=[sub_op.input_tensors[j] for j in perm],
            output_tensors=sub_op.output_tensors,
            op_type=sub_op.op_type,
            op_attr=sub_op.op_attr,
          )
        )
    elif patrn_op.op_type in cls._compatibility_map.get(sub_op.op_type, []):
      is_eq = True
      morphism = cls._compatibility_map[sub_op.op_type][patrn_op.op_type]
      equivalent_ops = [MetaOperationInfo(op_info=sub_op, morphism=morphism)]
    elif not patrn_op.input_tensors and \
      patrn_op.op_type == 'Placeholder':
      # match input node which is a placeholder anyway
      is_eq = True
      equivalent_ops = [sub_op]

    return is_eq, equivalent_ops

@attr.s
class uTensorGraphMatcher(object):

  pattern_ugraph = attr.ib(validator=instance_of(uTensorGraph))

  def _match(self, other_ugraph):
    outputs_pool = []
    for op in self.pattern_ugraph.output_ops:
      same_ops = other_ugraph.get_ops_by_type(op.op_type)
      if not same_ops:
        # there are missing output(s)
        # no way to match, return empty list
        return []
      outputs_pool.append(same_ops)
    output_candidates = product(*outputs_pool)
    for outputs in output_candidates:
      states = [
        _MatchState(
          match=uTensorGraphMatch(
            pattern_ugraph=self.pattern_ugraph,
            subject_ugraph=other_ugraph
          ),
          sub_bfs_queue=ops_bfs_queue(other_ugraph, init_nodes=outputs),
          patrn_bfs_queue=ops_bfs_queue(self.pattern_ugraph),
        )
      ]
      import pdb; pdb.set_trace()
      while True:
        states = self._visit(states)
        if not states:
          break
        for state in states:
          if state.is_done:
            yield state.match

  def match(self, other_ugraph, n=1):
    match_gen = self._match(other_ugraph)
    matches = []
    try:
      for _ in range(n):
        matches.append(next(match_gen))
    except StopIteration:
      pass
    return matches

  def match_all(self, other_ugraph):
    return list(self._match(other_ugraph))

  def _visit(self, states):
    # visit the state with a top-down bfs fashion
    # return the states that are still matched
    new_states = []
    for state in states:
      match = state.match
      sub_op = state.sub_bfs_queue.popleft()
      patrn_op = state.patrn_bfs_queue.popleft()
      is_eq, eq_ops = OpEqualityDelegate.query(sub_op, patrn_op)
      if is_eq:
        for eq_op in eq_ops:
          new_state = _MatchState(
            match=uTensorGraphMatch(
              pattern_ugraph=match.pattern_ugraph,
              subject_ugraph=match.subject_ugraph,
              patrn2subj_op_map={k: v for k, v in match.patrn2subj_op_map.items()},
              subj2patrn_op_map={k: v for k, v in match.subj2patrn_op_map.items()},
              patrn2subj_tensor_map={k: v for k, v in match.patrn2subj_tensor_map.items()},
              subj2patrn_tensor_map={k: v for k, v in match.subj2patrn_tensor_map.items()}
            ),
            sub_bfs_queue=deque(state.sub_bfs_queue),
            patrn_bfs_queue=deque(state.patrn_bfs_queue),
          )
          new_state.match.update_op_map(patrn_op, eq_op)
          new_states.append(new_state)
    return new_states

@attr.s
class uTensorGraphMatch(object):

  pattern_ugraph = attr.ib(type=uTensorGraph)
  subject_ugraph = attr.ib(type=uTensorGraph)

  # map from op_name to op_info
  patrn2subj_op_map = attr.ib(factory=dict)
  subj2patrn_op_map = attr.ib(factory=dict)
  # tensor in pattern -> tensor in target
  patrn2subj_tensor_map = attr.ib(factory=dict)
  # tensor in target -> tensor in pattern
  subj2patrn_tensor_map = attr.ib(factory=dict)

  def update_op_map(self, pattern_op, subj_op):
    self.patrn2subj_op_map[pattern_op.name] = subj_op
    self.subj2patrn_op_map[subj_op.name] = pattern_op
    for pattern_tensor, target_tensor in zip(pattern_op.input_tensors, subj_op.input_tensors):
      self.patrn2subj_tensor_map[pattern_tensor.name] = target_tensor
      self.subj2patrn_tensor_map[target_tensor.name] = pattern_tensor

@attr.s
class _MatchState(object):
  match = attr.ib()
  @match.validator
  def check(self, attrib, value):
    if not isinstance(value, uTensorGraphMatch):
      raise ValueError(
        'expecting a uTensorGraphMatch, get {}'.format(type(value))
      )
  # sub_bfs_queue is a queue for BFS of the subject ugraph
  sub_bfs_queue = attr.ib(validator=instance_of(deque))
  # consume_queue is a queue defines the matching order of pattern ugraph
  patrn_bfs_queue = attr.ib(validator=instance_of(deque))
  visited = attr.ib(init=False, factory=set)

  @property
  def is_done(self):
    """
    a state is done, if
    1. the patrn_bfs_queue is empty
    2. the sub_bfs_queue is empty
    """
    return not self.patrn_bfs_queue
