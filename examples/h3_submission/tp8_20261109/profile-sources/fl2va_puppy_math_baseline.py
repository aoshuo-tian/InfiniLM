"""Stage 3: full-checkpoint FL2VA video baseline. No communication/TP block tests."""
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import zipfile

PINNED = {
    'h3-video-deps-1-20261031.tar.gz': '75c17466d83e68b1a426afb23c0f9d164611f175a28602df7ae5a4a4c5ea4c02',
    'h3-video-deps-2-20261031.tar.gz': 'eeeef86ad68689370747bba83aa6cd6dcc1ad6344bb68dea244e3a4a49ee70cc',
}

RUNTIME_SHA = '3e79d685a81ff92db2ef24282b8252e9839807f8706250713d9a0cf98db81450'
RUNTIME_DIR = 'h3-tp-block-long-20261031-053410'
PARAMETERS = {'prompt': 'A dog slowly falls asleep. The camera remains still.', 'seed': 42, 'steps': 50,
              'height': 256, 'width': 256, 'num_frames': 124, 'gpus': list(range(8)), 'fps': 24}


def save(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    path.write_bytes(data)
    if path.read_bytes() != data:
        raise IOError('AFS write verification failed: ' + str(path))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def worker(work, output, model_path):
    import torch
    metrics = {'parameters': PARAMETERS, 'memory_note': 'driver-used VRAM sampled per device; includes native InfiniCore allocations. Torch peaks are allocator-specific.',
               'quality_note': 'decoded frames and previews only; no reference similarity score yet', 'timings': {}, 'passed': False}
    stop = threading.Event()
    peaks = [0] * 8
    samples = [0]
    errors = []
    timing_objects = []
    started = time.perf_counter()

    def sample():
        while not stop.is_set():
            try:
                for device in range(8):
                    free, total = torch.cuda.mem_get_info(device)
                    peaks[device] = max(peaks[device], int(total - free))
                samples[0] += 1
            except Exception as error:
                errors.append(str(error))
                break
            stop.wait(.5)

    sampler = None
    try:
        if not torch.version.hip or torch.cuda.device_count() < 8:
            raise RuntimeError('Expected the authorized 8-DCU compute node')
        metrics['devices'] = [torch.cuda.get_device_name(i) for i in range(8)]
        metrics['torch_version'], metrics['hip_version'] = torch.__version__, torch.version.hip
        import importlib.metadata
        metrics['dependency_versions'] = {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'diffusers', 'accelerate', 'huggingface_hub', 'tokenizers', 'safetensors', 'av')}
        for device in range(8):
            torch.cuda.reset_peak_memory_stats(device)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        # Exercise the exact formerly failing processor path before expensive model loading.
        import torchvision
        from transformers import Qwen3VLProcessor
        from PIL import Image
        processor = Qwen3VLProcessor.from_pretrained(str(model_path), subfolder='processor', local_files_only=True)
        assert processor is not None and callable(processor.image_processor)
        assert callable(processor.create_mm_token_type_ids)
        mm_ids = processor.create_mm_token_type_ids([[processor.image_token_id,processor.video_token_id]])
        assert mm_ids == [[1,2]], mm_ids
        print('FL2VA_MM_TOKEN_IDS_PASS',mm_ids,flush=True)
        image = Image.open(os.environ['H3_FIRST_FRAME']).convert('RGB')
        vision = processor.image_processor(images=[image], return_tensors='pt')
        assert torch.isfinite(vision['pixel_values']).all().item()
        metrics['processor_preflight'] = {'torchvision':torchvision.__version__, 'torchvision_path':torchvision.__file__, 'processor':type(processor).__name__, 'image_processor':type(processor.image_processor).__name__, 'pixel_shape':list(vision['pixel_values'].shape),'grid':vision['image_grid_thw'].tolist()}
        save(output / 'processor-preflight.json', metrics['processor_preflight'])
        print('FL2VA_PROCESSOR_PREFLIGHT_PASS',json.dumps(metrics['processor_preflight']),flush=True)
        del vision, processor
        state = json.loads((work / 'state.json').read_text())
        entry = Path(state['lm']) / 'examples/minimax_h3.py'
        metrics['entry'], metrics['entry_sha256'] = str(entry), sha(entry)
        namespace = runpy.run_path(str(entry), run_name='fl2va_baseline_entry')
        original_init = namespace['TimingReport'].__init__

        def capture(self, devices):
            original_init(self, devices)
            timing_objects.append(self)

        namespace['TimingReport'].__init__ = capture
        video = output / 'fl2va-first-baseline.mp4'
        sys.argv = [str(entry), 'fl2va', '--model-path', str(model_path), '--text', PARAMETERS['prompt'],
                    '--gpus', ','.join(map(str, PARAMETERS['gpus'])), '--seed', str(PARAMETERS['seed']),
                    '--steps', str(PARAMETERS['steps']), '--height', str(PARAMETERS['height']),
                    '--width', str(PARAMETERS['width']), '--num-frames', str(PARAMETERS['num_frames']),
                    '--output', str(video)]
        first = Path(os.environ['H3_FIRST_FRAME'])
        assert first.is_file() and sha(first) == 'c120c8b9d27341c872c9864a69b89b324c770089d71bf876d967efe5120c69ea'
        sys.argv += ['--first-frame', str(first)]
        metrics['first_frame'] = {'path':str(first),'sha256':sha(first)}
        metrics['command'] = list(sys.argv)
        print('FL2VA_GENERATION_BEGIN', json.dumps(PARAMETERS), flush=True)
        from torch.nn.attention import sdpa_kernel, SDPBackend
        # Use standard PyTorch math SDPA for the diagnostic framework VAE.
        # No replacement model, custom device kernel or reduction of generation work.
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            q = torch.randn(1, 2, 16, 64, device='cuda', dtype=torch.bfloat16)
            probe = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            assert torch.isfinite(probe).all().item()
            print('FL2VA_MATH_SDPA_PREFLIGHT_PASS', flush=True)
            namespace['main']()
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError('Pipeline returned without a video file')
        import av
        container = av.open(str(video))
        stream = container.streams.video[0]
        decoded = []
        for frame in container.decode(stream):
            decoded.append(frame.to_image())
        fps = float(stream.average_rate) if stream.average_rate else None
        container.close()
        if len(decoded) != PARAMETERS['num_frames'] or not decoded:
            raise RuntimeError('Unexpected video frame count: ' + str(len(decoded)))
        metrics['video'] = {'path': str(video), 'bytes': video.stat().st_size, 'sha256': sha(video),
                            'frames': len(decoded), 'fps': fps, 'width': decoded[0].width, 'height': decoded[0].height}
        for label, index in (('first', 0), ('middle', len(decoded) // 2), ('last', len(decoded) - 1)):
            decoded[index].save(output / ('preview-' + label + '.png'))
        metrics['passed'] = True
    except BaseException:
        metrics['error'] = traceback.format_exc()
        print(metrics['error'], flush=True)
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=10)
        metrics['worker_wall_seconds'] = time.perf_counter() - started
        metrics['driver_peak_used_bytes_per_device'] = peaks
        metrics['driver_memory_samples'] = samples[0]
        metrics['memory_sampling_errors'] = errors
        if timing_objects:
            metrics['timings'] = {key: list(values) for key, values in timing_objects[-1].values.items()}
        try:
            metrics['torch_peak_allocated_bytes'] = [torch.cuda.max_memory_allocated(i) for i in range(8)]
            metrics['torch_peak_reserved_bytes'] = [torch.cuda.max_memory_reserved(i) for i in range(8)]
        except Exception as error:
            metrics['torch_memory_error'] = str(error)
        save(output / 'generation-metrics.json', metrics)
        print('FL2VA_GENERATION_RESULT', json.dumps(metrics, ensure_ascii=False), flush=True)
    return 0 if metrics['passed'] else 1


def parent():
    home = Path(os.environ['IFAB_AFS_HOME'])
    output = home / ('h3-fl2va-baseline-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S'))
    output.mkdir(exist_ok=False)
    report = {'passed': False, 'stage': 'asset-discovery', 'output': str(output), 'parameters': PARAMETERS,
              'scope': '256x256 diagnostic full-weight FL2VA; user-provided puppy image; not formal evaluation or NineToothed compliance validation'}
    save(output / 'result.json', report)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('FL2VA_BASELINE_OUTPUT', output, flush=True)
    try:
        model = Path('/public/home/aclsylqidf/shared/MiniMax-H3/diffusers')
        if not model.is_dir():
            raise FileNotFoundError('Previously located model directory unavailable: ' + str(model))
        report['model_path'] = str(model)
        source = home / RUNTIME_DIR
        meta = json.loads((source / 'tp-runtime.json').read_text())
        archive = source / 'tp-qualified-runtime.tar.gz'
        # Routine integrity check when loading the package, not a new qualification test.
        if meta['sha256'] != RUNTIME_SHA or sha(archive) != RUNTIME_SHA:
            raise RuntimeError('Runtime archive integrity mismatch')
        report['runtime_sha256'] = RUNTIME_SHA
        work = (Path.cwd() / 'fl2va-runtime').resolve()
        work.mkdir(exist_ok=False)
        with tarfile.open(archive) as bundle:
            for original in bundle:
                member = copy.copy(original)
                relative = Path(member.name).relative_to('runtime')
                if str(relative) == '.':
                    continue
                if not (work / relative).resolve().is_relative_to(work):
                    raise RuntimeError('Archive escapes work directory')
                member.name = str(relative)
                bundle.extract(member, work, filter='data')
        for name in ('setup.sh', 'state.json', 'build-status.json'):
            path = work / name
            path.write_text(path.read_text().replace(meta['original_work'], str(work)))
        report['stage'] = 'full-model-video-generation'
        save(output / 'result.json', report)
        deps = (Path.cwd()/'video-deps').resolve()
        assert not deps.exists()
        for filename, digest in PINNED.items():
            bundle = home/filename
            assert sha(bundle)==digest,filename
            with tarfile.open(bundle) as tar:
                for m in tar:
                    assert m.isfile() or m.isdir()
                    assert Path(m.name).parts[0]=='video-deps'
                    assert (Path.cwd()/m.name).resolve().is_relative_to(deps)
                    tar.extract(m,Path.cwd(),filter='data')
        manifest = json.loads((deps/'manifest.json').read_text())
        site = deps/'site-packages';site.mkdir(exist_ok=False)
        for record in manifest['wheels']:
            name = record['name'].lower().replace('_','-')
            if name == 'transformers': continue  # pinned 5.10.0 replacement below
            assert name not in {'torch','numpy','torchvision','torchaudio','triton'}
            wheel = deps/'wheels'/record['file']
            assert sha(wheel)==record['sha256'],wheel.name
            with zipfile.ZipFile(wheel) as bundle:
                for m in bundle.infolist():
                    if m.is_dir():continue
                    parts=Path(m.filename).parts
                    assert parts and not Path(m.filename).is_absolute() and '..' not in parts
                    if parts[0].endswith('.data'):
                        if len(parts)<3 or parts[1] not in ('purelib','platlib'):continue
                        parts=parts[2:]
                    target=site.joinpath(*parts).resolve()
                    assert target.is_relative_to(site)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(bundle.read(m))
        # Only torchvision auxiliary preprocessing; never replace the Hygon Torch runtime.
        vision_wheel = home / 'torchvision-0.24.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl'
        assert sha(vision_wheel) == '51fe8d72ddcc88a50ff27c50dd4987d82fba8b0dfd591898ee1f60e2902d18d1'
        with zipfile.ZipFile(vision_wheel) as bundle:
            for info in bundle.infolist():
                if info.is_dir(): continue
                target = (site / info.filename).resolve()
                assert target.is_relative_to(site)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(info))
        report['supplemental_torchvision'] = {'file':vision_wheel.name,'sha256':sha(vision_wheel),'source':'official download.pytorch.org CPU wheel, 0.24.0 matching torch 2.9.0; target compatibility verified below'}
        tf_wheel = home / 'transformers-5.10.0-py3-none-any.whl'
        assert sha(tf_wheel) == 'eae143e46d8e521da604bc4fdc45af7dd90e96eacbbc27f68d2754150aa5175a'
        with zipfile.ZipFile(tf_wheel) as bundle:
            for info in bundle.infolist():
                if info.is_dir(): continue
                target = (site / info.filename).resolve()
                assert target.is_relative_to(site)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(info))
        report['transformers_override'] = {'version':'5.10.0','sha256':sha(tf_wheel),'reason':'Diffusers requires ProcessorMixin.create_mm_token_type_ids, absent from 5.0.0'}
        manifest['wheels'] = [r for r in manifest['wheels'] if r['name'].lower() != 'transformers']
        manifest['supplemental'] = [report['supplemental_torchvision'], report['transformers_override']]

        save(output / 'offline-dependency-manifest.json', manifest)
        env = os.environ.copy()
        env['H3_FIRST_FRAME'] = str(home / 'user-puppy-first-frame.jpg')
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   HF_HOME=str(output / 'cache'), OMP_NUM_THREADS='1')
        command = ['bash', '-c', 'source "$1"; export PYTHONPATH="$2:${PYTHONPATH:-}"; shift 2; exec "$@"', 'h3-fl2va', str(work / 'setup.sh'), str(site),
                   str(work / 'venv/bin/python'), '-u', str(Path(__file__).resolve()), '--worker', str(work), str(output), str(model)]
        print('FL2VA_PIPELINE_START', flush=True)
        start = time.monotonic()
        with (output / 'generation.log').open('w') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                proc.wait(timeout=1450)
            except subprocess.TimeoutExpired:
                report['timed_out'] = True
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
        report['generation_exit_code'] = proc.returncode
        report['generation_wall_seconds'] = time.monotonic() - start
        metrics_file = output / 'generation-metrics.json'
        if metrics_file.is_file():
            report['metrics'] = json.loads(metrics_file.read_text())
        report['passed'] = proc.returncode == 0 and report.get('metrics', {}).get('passed') is True
        print((output / 'generation.log').read_text(errors='replace')[-18000:], flush=True)
        if not report['passed']:
            raise RuntimeError('Full-model video run failed; see original generation.log and metrics')
        report['stage'] = 'video-saved'
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    finally:
        save(output / 'result.json', report)
        print('FL2VA_BASELINE_FINAL', json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(worker(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
                     if len(sys.argv) > 1 and sys.argv[1] == '--worker' else parent())
