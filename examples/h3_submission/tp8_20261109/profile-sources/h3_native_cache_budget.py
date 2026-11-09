"""Right-size static conditioner KV storage; preserve native model computation."""
from contextlib import contextmanager
import inspect
import math
import time

from h3_competition_checks import run_single_entry

INITIAL_TOKENS = 256
ALIGNMENT = 256


def cache_shapes(engine, expected_capacity):
    """Inspect native references only; do not copy weights or cache to Torch/CPU."""
    ranks = engine.get_kv_cache()
    shapes = [[list(tensor.shape) for tensor in layers] for layers in ranks]
    if len(shapes) != 4 or any(len(layers) != 50 for layers in shapes):
        raise ValueError('Expected four ranks of 50 conditioner KV layers')
    if any(len(shape) != 5 or shape[0:2] != [2, 1]
           or shape[3] != expected_capacity
           for layers in shapes for shape in layers):
        raise ValueError('Unexpected native static KV layout or capacity')
    # The pinned conditioner uses BF16 cache storage. These are allocation sizes,
    # not measured whole-device memory peaks.
    return {
        'rank_first_layer_shapes': [layers[0] for layers in shapes],
        'rank_layer_counts': [len(layers) for layers in shapes],
        'bf16_allocated_bytes_by_rank': [sum(math.prod(s) * 2 for s in layers) for layers in shapes],
    }


@contextmanager
def budget_conditioner(encoder_class, cache_config_class, state):
    original_init = encoder_class.__init__
    original_encode = encoder_class.encode
    signature = inspect.signature(original_init)

    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        limit = int(bound.arguments['max_cache_len'])
        if limit <= 0:
            raise ValueError('Explicit positive conditioner cache limit required')
        capacity = min(INITIAL_TOKENS, limit)
        bound.arguments['max_cache_len'] = capacity
        started = time.perf_counter()
        original_init(*bound.args, **bound.kwargs)
        self._h3_cache_capacity = capacity
        self._h3_cache_limit = limit
        allocation = cache_shapes(self.engine, capacity)
        old_bytes = [size * limit // capacity for size in allocation['bf16_allocated_bytes_by_rank']]
        state['instances'].append({
            'original_limit_tokens': limit, 'initial_capacity_tokens': capacity,
            'create_and_load_wall_seconds': time.perf_counter() - started,
            'initial_allocation': allocation,
            'original_capacity_bf16_bytes_by_rank': old_bytes,
            'growth': [], 'input_lengths': [],
        })
        self._h3_cache_record = state['instances'][-1]

    def encode(self, model_inputs):
        shape = tuple(model_inputs['input_ids'].shape)
        if len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
            raise ValueError('Conditioner cache budget requires one nonempty input')
        length = int(shape[1])
        record = self._h3_cache_record
        record['input_lengths'].append(length)
        if length > self._h3_cache_limit:
            raise ValueError('Input exceeds original conditioner cache limit')
        if length > self._h3_cache_capacity:
            capacity = min(self._h3_cache_limit,
                           math.ceil(length / ALIGNMENT) * ALIGNMENT)
            started = time.perf_counter()
            # Native reset waits for rank workers. This encoder always performs
            # full prefill with past_kv_lengths=0, so no past tokens are retained.
            self.engine.reset_cache(cache_config_class(1, capacity))
            allocation = cache_shapes(self.engine, capacity)
            self._h3_cache_capacity = capacity
            record['growth'].append({'capacity_tokens': capacity,
                                     'wall_seconds': time.perf_counter() - started,
                                     'allocation': allocation})
        return original_encode(self, model_inputs)

    encoder_class.__init__ = initialize
    encoder_class.encode = encode
    try:
        yield
    finally:
        encoder_class.__init__ = original_init
        encoder_class.encode = original_encode


def run_cache_entry(namespace, report, output):
    from infinilm.cache import StaticKVCacheConfig

    live = namespace['main'].__globals__
    state = {'policy': 'initial_256_grow_before_full_prefill_to_original_limit',
             'instances': [], 'native_weights_and_operations_unchanged': True,
             'memory_scope': 'Verified native static KV tensor shapes, BF16 allocation arithmetic; not whole-device peaks.',
             'speedup_proven': False}
    report['conditioner_cache_budget'] = state
    with budget_conditioner(live['MiniMaxH3ConditionEncoder'], StaticKVCacheConfig, state):
        return run_single_entry(namespace, report, output)
