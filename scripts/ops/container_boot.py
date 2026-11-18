"""Container-local protection only; never load CANN or initialize an NPU."""
import json, pathlib, subprocess, shutil, ctypes, os

def mounts():
    return {p[4]:p[5].split(',') for p in (line.split() for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines())}

for target in ['/sys','/proc/sys','/proc/sysrq-trigger']:
    if 'ro' not in mounts().get(target,[]):
        if shutil.which('mount'):
            subprocess.run(['mount','--bind',target,target],check=True)
            subprocess.run(['mount','-o','remount,bind,ro',target],check=True)
        else:
            # Some existing minimal images have no mount executable. This is the
            # same bind/remount operation in our container's mount namespace.
            libc=ctypes.CDLL(None,use_errno=True)
            libc.mount.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong,ctypes.c_void_p]
            for flags in [4096,4096|32|1]:
                if libc.mount(target.encode(),target.encode(),None,flags,None):
                    raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),target)
assert all('ro' in mounts().get(p,[]) for p in ['/sys','/proc/sys','/proc/sysrq-trigger'])
assert 'private-memory-view' not in pathlib.Path('/proc/self/mountinfo').read_text()
print(json.dumps({'boot':'PUBLIC_PATHS_READ_ONLY','npu_initialized':False}),flush=True)
