"""Direct CCL correctness probe against the just-built library; no build/restorer."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
from runtime_links import loaded_objects

def save(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def worker(libpath, output):
    import torch
    lib = C.CDLL(str(libpath), mode=C.RTLD_GLOBAL)
    ptr, integer, size = (C.c_void_p, C.c_int, C.c_size_t)
    signatures = {'infinicclCommInitAll': [integer, C.POINTER(ptr), integer, C.POINTER(integer)], 'infinicclCommDestroy': [ptr], 'infinicclGroupStart': [ptr], 'infinicclGroupEnd': [ptr], 'infinicclAllReduce': [ptr, ptr, size, integer, integer, ptr, ptr], 'infinicclAllGather': [ptr, ptr, size, integer, ptr, ptr]}
    for name, signature in signatures.items():
        getattr(lib, name).argtypes = signature
        getattr(lib, name).restype = integer

    def check(code):
        if code != 0:
            raise RuntimeError('InfiniCCL status ' + str(code))
    result = {'scope': 'Direct InfiniCCL C API; single process multi-device; not H3 TP', 'library': str(libpath), 'checks': [], 'passed': False}
    for world in (2, 4, 8):
        comms = (ptr * world)()
        ids = (integer * world)(*range(world))
        check(lib.infinicclCommInitAll(8, comms, world, ids))
        print('NATIVE_CCL_INIT', world, flush=True)
        try:
            for dtype, enum in ((torch.float32, 13), (torch.float16, 12), (torch.bfloat16, 19)):
                for n in (1, 257, 65536, 1048576):
                    for epoch in (0, 1):
                        sources, reduced, gathered, bases = ([], [], [], [])
                        for rank in range(world):
                            with torch.cuda.device(rank):
                                b = (torch.arange(n, device='cuda', dtype=torch.int64) % 13 - 6).to(dtype) / 16
                                bases.append(b)
                                sources.append(b + (rank + epoch) / 8)
                                reduced.append(torch.empty_like(b))
                                gathered.append(torch.empty((world, n), device='cuda', dtype=dtype))
                        for rank in range(world):
                            torch.cuda.synchronize(rank)
                        for op in ('allreduce', 'allgather'):
                            check(lib.infinicclGroupStart(comms[0]))
                            for rank in range(world):
                                torch.cuda.set_device(rank)
                                stream = torch.cuda.current_stream(rank).cuda_stream
                                if op == 'allreduce':
                                    check(lib.infinicclAllReduce(sources[rank].data_ptr(), reduced[rank].data_ptr(), n, enum, 0, comms[rank], stream))
                                else:
                                    check(lib.infinicclAllGather(sources[rank].data_ptr(), gathered[rank].data_ptr(), n, enum, comms[rank], stream))
                            check(lib.infinicclGroupEnd(comms[0]))
                            for rank in range(world):
                                torch.cuda.synchronize(rank)
                                if op == 'allreduce':
                                    expected = bases[rank] * world + (world * (world - 1) / 2 + world * epoch) / 8
                                    ok = torch.equal(reduced[rank], expected)
                                else:
                                    ok = all((torch.equal(gathered[rank][j], bases[rank] + (j + epoch) / 8) for j in range(world)))
                                if not ok:
                                    raise RuntimeError(f'{op} mismatch world={world} rank={rank} dtype={dtype} n={n}')
                            result['checks'].append(dict(world=world, dtype=str(dtype), n=n, epoch=epoch, op=op, all_ranks_exact=True))
                        save(output / 'native-result.json', result)
            print('NATIVE_CCL_WORLD_PASS', world, flush=True)
        finally:
            for comm in comms:
                check(lib.infinicclCommDestroy(comm))
    result['passed'] = True
    result['loaded_libraries'] = loaded_objects()
    save(output / 'native-result.json', result)
    print('INFINICCL_NATIVE_2_4_8_PASS', len(result['checks']), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    state = json.loads((work / 'state.json').read_text())
    status = json.loads((work / 'ccl-build-status.json').read_text())
    if not status.get('passed') or not status.get('complete'):
        parser.error('A completed CCL source build is required')
    for path, digest in status['libraries'].items():
        if (not Path(path).resolve().is_relative_to(work)
                or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest):
            parser.error('New CCL library path/SHA mismatch')
    if args.output.exists():
        parser.error('Use a new output directory; existing evidence is not overwritten')
    import torch
    if not torch.version.hip or torch.cuda.device_count() != 8:
        parser.error('The specified eight-device Hygon node is required')
    args.output.mkdir(parents=True)
    worker(Path(state['lib']) / 'libinfiniccl.so', args.output)


if __name__ == '__main__':
    main()
