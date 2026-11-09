"""Local validation gates; these do not implement the organizer's similarity score."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROMPT = 'A dog falls asleep'
PROFILE = {
    'prompt': PROMPT, 'seed': 42, 'steps': 10, 'height': 384, 'width': 576,
    'num_frames': 124, 'fps': 24, 'video_codec': 'h264', 'audio_required': True,
    'profile_status': 'existing_local_comparison_profile_not_officially_confirmed',
}


def evaluation_status(basic_passed=False):
    return {
        'basic_validation_passed': bool(basic_passed),
        'official_similarity_status': 'not_evaluated',
        'official_similarity_passed': None,
        'preliminary_qualification_passed': None,
        'official_performance_score': None,
        'formal_score_passed': False,
        'note': 'Local decode/specification checks do not establish official qualification or score.',
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_media(path, profile=None):
    import av
    import numpy as np

    expected = dict(PROFILE if profile is None else profile)
    result = {'passed': False, 'expected': expected, 'path': str(path),
              'scope': 'Local full decode and basic gates; no official similarity metric.'}
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Expected exactly one video stream')
        stream = container.streams.video[0]
        actual = {'width': stream.width, 'height': stream.height,
                  'fps': float(stream.average_rate) if stream.average_rate else None,
                  'video_codec': stream.codec_context.name,
                  'audio_streams': len(container.streams.audio),
                  'frames': 0, 'all_black_frames': 0, 'wrong_size_frames': 0}
        timestamps_valid = True
        previous = None
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format='rgb24')
            actual['wrong_size_frames'] += rgb.shape != (expected['height'], expected['width'], 3)
            actual['all_black_frames'] += not bool(rgb.any())
            timestamp = frame.time
            if timestamp is None or (previous is not None and timestamp <= previous):
                timestamps_valid = False
            previous = timestamp
            actual['frames'] += 1
    actual['timestamps_strictly_increasing'] = timestamps_valid
    audio = {'decoded_samples': 0, 'all_finite': True, 'absmax': 0.0}
    if actual['audio_streams']:
        with av.open(str(path)) as container:
            for frame in container.decode(audio=0):
                values = frame.to_ndarray()
                audio['decoded_samples'] += frame.samples
                finite = bool(np.isfinite(values).all())
                audio['all_finite'] = audio['all_finite'] and finite
                if finite and values.size:
                    audio['absmax'] = max(audio['absmax'], float(np.abs(values).max()))
    actual['audio'] = audio
    checks = {
        'width': actual['width'] == expected['width'],
        'height': actual['height'] == expected['height'],
        'frames': actual['frames'] == expected['num_frames'],
        'fps': actual['fps'] == expected['fps'],
        'codec': actual['video_codec'] == expected['video_codec'],
        'frame_dimensions': actual['wrong_size_frames'] == 0,
        'timestamps': timestamps_valid,
        # A strict local puppy regression gate, not an organizer-defined black-frame rule.
        'no_pure_black_frames': actual['all_black_frames'] == 0,
        'audio_present': not expected['audio_required'] or actual['audio_streams'] > 0,
        'audio_nonempty': not expected['audio_required'] or audio['decoded_samples'] > 0,
        'audio_finite': audio['all_finite'],
        'audio_nonzero': not expected['audio_required'] or audio['absmax'] > 0,
    }
    result.update(actual=actual, checks=checks, passed=all(checks.values()),
                  bytes=Path(path).stat().st_size, sha256=sha256(path))
    return result


def validate_generation_record(record):
    metrics = record.get('metrics', record)
    parameters = metrics.get('parameters', {})
    checks = {
        'prompt': parameters.get('prompt') == PROMPT,
        'configuration': all(parameters.get(key) == PROFILE[key] for key in
                             ('seed', 'steps', 'height', 'width', 'num_frames', 'fps')),
        'native_backend': parameters.get('legacy') is False,
        'eight_devices': parameters.get('gpus') == list(range(8)),
        'single_request_calls': metrics.get('calls', {}) == {
            'video_encode': 1, 'transformer': 9, 'video_scheduler': 9,
            'audio_scheduler': 9, 'video_decode': 1, 'audio_decode': 1},
        'no_experimental_kernels': metrics.get('experimental_mha_rope_enabled') is False,
        'execution_succeeded': record.get('passed') is True and metrics.get('passed') is True,
    }
    if 'generation_exit_code' in record:
        checks['exit_zero'] = record['generation_exit_code'] == 0
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'Generation-record consistency only; media provenance requires the media SHA check.'}


def summarize_raw_media(values, torch, np):
    """Reduce Torch media on its current device; keep export arguments untouched."""
    if torch.is_tensor(values):
        tensor = values.detach()
        return {'shape': list(tensor.shape), 'size': tensor.numel(),
                'finite': bool(torch.isfinite(tensor).all().item()),
                'nonzero': bool(tensor.any().item())}
    array = np.asarray(values)
    return {'shape': list(array.shape), 'size': array.size,
            'finite': bool(np.isfinite(array).all()), 'nonzero': bool(array.any())}


def run_single_entry(namespace, report, output):
    """Invoke the existing entry exactly once, retaining the source-device bridge."""
    import numpy as np
    import torch

    live = namespace['main'].__globals__
    timing_class = live['TimingReport']
    original_init = timing_class.__init__
    original_encode = live['encode_video']
    timing_objects = []
    encode_calls = 0
    report['evaluation'] = evaluation_status()
    report['evaluation_profile'] = PROFILE
    report['execution_mode'] = 'single_request_fresh_process_no_warmup'

    def capture(self, devices):
        original_init(self, devices)
        timing_objects.append(self)

    def encode(video, *args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        if encode_calls != 1:
            raise ValueError('Competition entry must export exactly one request')
        if report.get('calls') != {'video_encode': 1, 'transformer': 9,
                                  'video_scheduler': 9, 'audio_scheduler': 9,
                                  'video_decode': 1, 'audio_decode': 1}:
            raise ValueError('Single-request model call counts differ')
        video_values = summarize_raw_media(video, torch, np)
        audio_values = summarize_raw_media(kwargs.get('audio'), torch, np)
        raw = {
            'video_shape': video_values['shape'], 'audio_shape': audio_values['shape'],
            'video_finite': video_values['finite'], 'audio_finite': audio_values['finite'],
            'video_nonzero': video_values['nonzero'], 'audio_nonzero': audio_values['nonzero'],
        }
        report['raw_media_gate'] = raw
        if (raw['video_shape'] != [124, 384, 576, 3] or audio_values['size'] == 0
                or not all(raw[key] for key in ('video_finite', 'audio_finite',
                                               'video_nonzero', 'audio_nonzero'))):
            raise ValueError('Invalid raw media before encoding')
        return original_encode(video, *args, **kwargs)

    timing_class.__init__ = capture
    live['encode_video'] = encode
    started = time.perf_counter()
    try:
        for device in range(8):
            torch.cuda.reset_peak_memory_stats(device)
        namespace['main']()
        if len(timing_objects) != 1 or encode_calls != 1:
            raise ValueError('Expected one TimingReport and one media export')
    finally:
        timing_class.__init__ = original_init
        live['encode_video'] = original_encode
        report['single_entry_wall_seconds'] = time.perf_counter() - started
        if timing_objects:
            report['timings'] = {key: list(values) for key, values in timing_objects[-1].values.items()}
        report['timing_contract'] = {
            'total.end_to_end': 'Existing entry: input preparation, model loading, generation, media export and final device synchronization.',
            'single_entry_wall_seconds': 'Wrapper wall clock around one existing entry call; excludes runtime/dependency setup and post-generation decoding.',
            'generation.pipeline_total': 'Generation stage only; does not replace end-to-end timing.',
            'official_timing_protocol_confirmed': False,
        }
        try:
            report['memory'] = {
                'torch_peak_allocated_bytes': [torch.cuda.max_memory_allocated(i) for i in range(8)],
                'torch_peak_reserved_bytes': [torch.cuda.max_memory_reserved(i) for i in range(8)],
                'scope': 'Torch allocator peaks during this entry only; native InfiniCore allocations are excluded. No whole-device peak claim.',
            }
        except Exception as error:
            report['memory'] = {'error': str(error)}


def annotate_result(value):
    metrics = value.get('metrics', value)
    basic = value.get('passed') is True and metrics.get('basic_validation', {}).get('passed', False)
    value['evaluation'] = evaluation_status(basic)
    value['evaluation_profile'] = dict(PROFILE)
    value['pass_scope'] = 'Local single-request execution and basic media gates only.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    record = json.loads(args.metrics.read_text(encoding='utf8'))
    media = validate_media(args.video)
    generation = validate_generation_record(record)
    metrics = record.get('metrics', record)
    media_identity = metrics.get('media', {}).get('sha256') == media['sha256']
    passed = media['passed'] and generation['passed'] and media_identity
    result = {'basic_validation_passed': passed, 'media': media, 'generation_record': generation,
              'media_sha_matches_generation_record': media_identity,
              'evaluation': evaluation_status(passed), 'gpu_jobs_submitted': 0}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'basic_validation_passed': passed, 'report': str(args.output),
                      'evaluation': result['evaluation']}))
    return 0 if passed else 1


if __name__ == '__main__':
    # The local audit directory optionally contains the offline PyAV/numpy tool bundle.
    local = Path(__file__).resolve().parent / 'local-tools'
    if local.is_dir():
        sys.path.insert(0, str(local))
    raise SystemExit(main())
