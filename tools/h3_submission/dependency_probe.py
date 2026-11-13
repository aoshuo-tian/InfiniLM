"""Read-only Python dependency/API probe; never installs or executes DCU kernels."""
import argparse
import contextlib
import importlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import subprocess
import sys

try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement

IMPORTS = {'janus':'janus', 'xxhash':'xxhash', 'transformers':'transformers',
           'diffusers':'diffusers', 'accelerate':'accelerate', 'av':'av',
           'pillow':'PIL', 'scikit-image':'skimage', 'scipy':'scipy',
           'torchvision':'torchvision'}


def inspect(requirements, vision=False):
    requested=[]
    for path in requirements:
        requested.extend(line.strip() for line in Path(path).read_text().splitlines()
                         if line.strip() and not line.lstrip().startswith('#'))
    if vision:
        requested.append('torchvision==0.24.0+cpu')
    missing, versions, errors, origins = [], {}, [], {}
    for text in requested:
        requirement=Requirement(text)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        name=requirement.name.lower().replace('_','-')
        try:
            version=metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            missing.append(text)
            continue
        versions[name]=version
        if not requirement.specifier.contains(version, prereleases=True):
            missing.append(text)
            continue
        module=IMPORTS.get(name)
        if module:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    imported=importlib.import_module(module)
                origins[name]=getattr(imported,'__file__',None)
            except Exception as error:
                errors.append(dict(distribution=name, stage='import', error=repr(error)))
    # A matching version number alone does not prove the H3 entry's lazy APIs exist.
    inference=any(Requirement(x).name.lower()=='diffusers' for x in requested)
    if inference and not missing and not errors:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
                from accelerate.utils import set_module_tensor_to_device
                from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
                from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference
                from diffusers.utils.export_utils import encode_video
                from safetensors.torch import load_file
                from transformers import Qwen3VLForConditionalGeneration
        except Exception as error:
            errors.append(dict(stage='H3 Python API',error=repr(error)))
    return dict(passed=not missing and not errors, missing_or_version_mismatch=missing,
        import_errors=errors, versions=versions, module_paths=origins,
        python=sys.executable, python_version=sys.version, platform=sys.platform,
        prefix=sys.prefix, model_loaded=False, gpu_kernel_executed=False)


def probe(python, requirements, vision=False):
    command=[str(python),str(Path(__file__).resolve()),'--requirements',*[str(p) for p in requirements]]
    if vision:
        command.append('--vision')
    run=subprocess.run(command,capture_output=True,text=True)
    rows=[line[len('H3_DEPENDENCY_PROBE '):] for line in run.stdout.splitlines()
          if line.startswith('H3_DEPENDENCY_PROBE ')]
    if run.returncode or len(rows)!=1:
        raise RuntimeError('依赖检查进程失败；未执行安装：'+run.stderr[-4000:])
    result=json.loads(rows[0])
    result['diagnostic_stderr']=run.stderr[-4000:]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements',nargs='+',type=Path,required=True)
    parser.add_argument('--vision',action='store_true')
    args=parser.parse_args()
    print('H3_DEPENDENCY_PROBE '+json.dumps(inspect(args.requirements,args.vision),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
