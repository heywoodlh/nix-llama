"""Hardware probe.

The probe answers one question: *what does this machine look like right now?*
It deliberately knows nothing about llama-server flags -- turning facts into an
argument vector is :mod:`nix_llama.defaults`' job. Keeping the two apart means
the probe can be unit-tested against synthetic ``/sys`` and ``/proc`` trees.

Every filesystem lookup is rooted at :attr:`Probe.root` so tests can point the
whole thing at a fixture directory.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# PCI vendor IDs, as they appear in /sys/.../vendor.
VENDOR_NVIDIA = "0x10de"
VENDOR_AMD = "0x1002"
VENDOR_INTEL = "0x8086"

# Backend preference, best first. A backend only wins if the *binary* reports a
# matching device, so this is a ranking rather than a guess.
BACKEND_PRIORITY = ["cuda", "metal", "rocm", "vulkan", "sycl", "opencl", "blas", "cpu"]

# Where the OS keeps the helpers the probe shells out to (sysctl on macOS),
# searched only when PATH does not already find them.
SYSTEM_BIN_PATH = os.pathsep.join(("/usr/sbin", "/sbin", "/usr/bin", "/bin"))

_LIST_DEVICES_RE = re.compile(
    r"^\s+(?P<backend>[A-Za-z0-9_]+):\s+(?P<name>.*?)\s+\((?P<total>\d+)\s*MiB"
    r"(?:,\s*(?P<free>\d+)\s*MiB free)?\)\s*$"
)


def _run(argv: list[str], timeout: float = 10.0) -> str | None:
    """Run a helper binary, returning stdout or ``None`` if it is unusable."""
    # launchd daemons and Nix builders on macOS run with a PATH that omits
    # /usr/sbin, where sysctl lives; without the fallback the probe reports
    # 0 bytes of memory and the planner sizes everything for nothing.
    exe = shutil.which(argv[0]) or shutil.which(argv[0], path=SYSTEM_BIN_PATH)
    if exe is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - argv is built from literals
            [exe, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


@dataclass
class Accelerator:
    vendor: str
    backend: str
    name: str
    memory_bytes: int | None = None
    count: int = 1
    source: str = "sysfs"


@dataclass
class CPUInfo:
    logical: int = 1
    physical: int = 1
    # Cores we are actually allowed to run on, after cgroup quotas and affinity
    # masks. This -- not `logical` -- is what sizes --threads.
    available: int = 1
    performance: int | None = None
    isa: list[str] = field(default_factory=list)
    model: str = ""


@dataclass
class MemoryInfo:
    total_bytes: int = 0
    available_bytes: int = 0
    # cgroup ceiling, when one applies (containers, systemd MemoryMax).
    limit_bytes: int | None = None

    @property
    def budget_bytes(self) -> int:
        """Memory we may plan against."""
        values = [v for v in (self.total_bytes, self.limit_bytes) if v]
        return min(values) if values else 0


@dataclass
class StorageInfo:
    path: str = ""
    rotational: bool | None = None
    free_bytes: int = 0
    total_bytes: int = 0

    @property
    def is_ssd(self) -> bool:
        # ``None`` means "could not tell" (network/overlay/virtio with no queue
        # attributes). Treating unknown as SSD would hand slow disks a KV
        # offload path, so unknown stays false.
        return self.rotational is False


@dataclass
class Probe:
    root: str = "/"
    os: str = ""
    arch: str = ""
    container: bool = False
    container_runtime: str | None = None
    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    accelerators: list[Accelerator] = field(default_factory=list)
    devices: list[dict[str, Any]] = field(default_factory=list)
    acceleration: str = "cpu"
    vram_bytes: int | None = None
    unified_memory: bool = False
    storage: StorageInfo = field(default_factory=StorageInfo)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("root", None)
        data["memory"]["budget_bytes"] = self.memory.budget_bytes
        data["storage"]["is_ssd"] = self.storage.is_ssd
        return data

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


class _Fs:
    """Filesystem accessor rooted at an arbitrary prefix."""

    def __init__(self, root: str = "/") -> None:
        self.root = Path(root)

    def path(self, p: str) -> Path:
        return self.root / p.lstrip("/")

    def read(self, p: str) -> str | None:
        try:
            return self.path(p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def read_int(self, p: str) -> int | None:
        raw = self.read(p)
        if raw is None:
            return None
        raw = raw.strip()
        try:
            return int(raw, 0)
        except ValueError:
            return None

    def exists(self, p: str) -> bool:
        return self.path(p).exists()

    def glob(self, pattern: str) -> list[Path]:
        return sorted(self.root.glob(pattern.lstrip("/")))


# --------------------------------------------------------------------------- #
# container
# --------------------------------------------------------------------------- #


def detect_container(fs: _Fs) -> tuple[bool, str | None]:
    if fs.exists("/.dockerenv"):
        return True, "docker"
    if fs.exists("/run/.containerenv"):
        return True, "podman"
    # Lowercase on purpose: "container" is the variable systemd-nspawn and the
    # OCI runtimes actually set.
    env_marker = os.environ.get("container")  # noqa: SIM112
    if env_marker:
        return True, env_marker
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True, "kubernetes"
    cgroup = fs.read("/proc/1/cgroup") or ""
    for needle, runtime in (
        ("docker", "docker"),
        ("kubepods", "kubernetes"),
        ("libpod", "podman"),
        ("containerd", "containerd"),
        ("lxc", "lxc"),
    ):
        if needle in cgroup:
            return True, runtime
    # A PID-1 that is not init/systemd is a strong hint inside minimal images.
    sched = fs.read("/proc/1/sched") or ""
    if sched:
        first = sched.splitlines()[0].split(" ", 1)[0]
        if first not in ("systemd", "init", "launchd", "bash", "sh"):
            return True, "unknown"
    return False, None


# --------------------------------------------------------------------------- #
# cpu
# --------------------------------------------------------------------------- #


def _cgroup_cpu_quota(fs: _Fs) -> float | None:
    # cgroup v2: "<quota|max> <period>"
    v2 = fs.read("/sys/fs/cgroup/cpu.max")
    if v2:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                return int(parts[0]) / int(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    quota = fs.read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = fs.read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and quota > 0:
        return quota / period
    return None


def detect_cpu_linux(fs: _Fs) -> CPUInfo:
    info = CPUInfo()
    cpuinfo = fs.read("/proc/cpuinfo") or ""

    # /proc/cpuinfo is a sequence of blocks separated by blank lines, and the
    # order of keys within a block is not guaranteed, so each block is
    # collected whole before being interpreted.
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in cpuinfo.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        key, sep, value = line.partition(":")
        if sep:
            current[key.strip()] = value.strip()
    if current:
        blocks.append(current)

    cores: set[tuple[str, str]] = set()
    for block in blocks:
        if not info.model:
            info.model = block.get("model name") or block.get("Model") or ""
        if not info.isa:
            flags = block.get("flags")
            if flags:
                info.isa = sorted(
                    f
                    for f in flags.split()
                    if f in {"avx", "avx2", "avx512f", "avx512_vnni", "amx_int8", "f16c", "fma"}
                )
            features = block.get("Features")  # aarch64
            if features and not info.isa:
                info.isa = sorted(
                    f
                    for f in features.split()
                    if f in {"asimd", "asimddp", "sve", "sve2", "i8mm", "bf16"}
                )
        core_id = block.get("core id")
        if core_id is not None:
            cores.add((block.get("physical id", "0"), core_id))

    info.logical = len(blocks) or os.cpu_count() or 1
    info.physical = len(cores) or _physical_from_topology(fs) or info.logical

    allowed = os.cpu_count() or info.logical
    try:
        allowed = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
    quota = _cgroup_cpu_quota(fs)
    if quota:
        allowed = min(allowed, max(1, int(quota)))
    # Hyper-threads rarely help llama.cpp, so cap at the physical core count.
    info.available = max(1, min(allowed, info.physical))
    return info


def _physical_from_topology(fs: _Fs) -> int | None:
    cores: set[str] = set()
    for path in fs.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list"):
        try:
            cores.add(path.read_text(encoding="utf-8").strip())
        except OSError:
            continue
    return len(cores) or None


def detect_cpu_darwin() -> CPUInfo:
    info = CPUInfo()

    def sysctl_int(name: str) -> int | None:
        out = _run(["sysctl", "-n", name])
        if out is None:
            return None
        try:
            return int(out.strip())
        except ValueError:
            return None

    info.logical = sysctl_int("hw.logicalcpu") or os.cpu_count() or 1
    info.physical = sysctl_int("hw.physicalcpu") or info.logical
    # On Apple silicon, perflevel0 is the performance cluster. Scheduling onto
    # the efficiency cores costs more than it adds, so they are left out.
    info.performance = sysctl_int("hw.perflevel0.physicalcpu")
    info.model = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()
    info.available = info.performance or info.physical
    return info


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #


def detect_memory_linux(fs: _Fs) -> MemoryInfo:
    mem = MemoryInfo()
    for line in (fs.read("/proc/meminfo") or "").splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if not parts:
            continue
        try:
            kib = int(parts[0])
        except ValueError:
            continue
        if key == "MemTotal":
            mem.total_bytes = kib * 1024
        elif key == "MemAvailable":
            mem.available_bytes = kib * 1024

    limit = fs.read("/sys/fs/cgroup/memory.max")
    if limit and limit.strip() != "max":
        try:
            mem.limit_bytes = int(limit.strip())
        except ValueError:
            pass
    if mem.limit_bytes is None:
        v1 = fs.read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        # cgroup v1 spells "unlimited" as a huge sentinel value.
        if v1 and v1 < (1 << 62):
            mem.limit_bytes = v1
    if not mem.available_bytes:
        mem.available_bytes = mem.total_bytes
    return mem


def detect_memory_darwin() -> MemoryInfo:
    mem = MemoryInfo()
    out = _run(["sysctl", "-n", "hw.memsize"])
    if out:
        try:
            mem.total_bytes = int(out.strip())
        except ValueError:
            pass
    mem.available_bytes = mem.total_bytes
    return mem


# --------------------------------------------------------------------------- #
# accelerators
# --------------------------------------------------------------------------- #


def detect_nvidia(fs: _Fs) -> list[Accelerator]:
    found: list[Accelerator] = []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if out:
        for line in out.strip().splitlines():
            name, _, mem = line.partition(",")
            try:
                mem_bytes = int(float(mem.strip()) * 1024 * 1024)
            except ValueError:
                mem_bytes = None
            found.append(
                Accelerator(
                    vendor="nvidia",
                    backend="cuda",
                    name=name.strip(),
                    memory_bytes=mem_bytes,
                    source="nvidia-smi",
                )
            )
        if found:
            return found

    # nvidia-smi is frequently absent inside containers even when the devices
    # are passed through, so fall back to device nodes and sysfs.
    device_nodes = list(fs.glob("/dev/nvidia[0-9]*"))
    if device_nodes or fs.exists("/proc/driver/nvidia/version"):
        count = len(device_nodes) or 1
        found.append(
            Accelerator(
                vendor="nvidia",
                backend="cuda",
                name="NVIDIA GPU",
                count=count,
                source="devnode",
            )
        )
        return found
    for vendor_file in fs.glob("/sys/bus/pci/devices/*/vendor"):
        try:
            if vendor_file.read_text(encoding="utf-8").strip() == VENDOR_NVIDIA:
                found.append(
                    Accelerator(vendor="nvidia", backend="cuda", name="NVIDIA GPU", source="pci")
                )
                break
        except OSError:
            continue
    return found


def detect_amd(fs: _Fs) -> list[Accelerator]:
    found: list[Accelerator] = []
    for card in fs.glob("/sys/class/drm/card[0-9]*"):
        vendor = None
        try:
            vendor = (card / "device" / "vendor").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if vendor != VENDOR_AMD:
            continue
        mem = None
        try:
            mem = int((card / "device" / "mem_info_vram_total").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            mem = None
        found.append(
            Accelerator(
                vendor="amd",
                backend="rocm" if fs.exists("/dev/kfd") else "vulkan",
                name=card.name,
                memory_bytes=mem,
                source="sysfs",
            )
        )
    if not found and fs.exists("/dev/kfd"):
        found.append(Accelerator(vendor="amd", backend="rocm", name="AMD GPU", source="devnode"))
    return found


def detect_intel(fs: _Fs) -> list[Accelerator]:
    found: list[Accelerator] = []
    for card in fs.glob("/sys/class/drm/card[0-9]*"):
        try:
            vendor = (card / "device" / "vendor").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if vendor != VENDOR_INTEL:
            continue
        mem = None
        for attr in ("lmem_total_bytes", "mem_info_vram_total"):
            try:
                mem = int((card / "device" / attr).read_text(encoding="utf-8").strip())
                break
            except (OSError, ValueError):
                continue
        found.append(
            Accelerator(
                vendor="intel",
                backend="vulkan",
                name=card.name,
                memory_bytes=mem,
                source="sysfs",
            )
        )
    return found


def detect_apple(mem_total: int) -> list[Accelerator]:
    # Apple silicon has no separate VRAM: Metal may address roughly 75% of
    # system memory, which is also what recommendedMaxWorkingSetSize reports.
    return [
        Accelerator(
            vendor="apple",
            backend="metal",
            name=(_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon").strip(),
            memory_bytes=int(mem_total * 0.75) if mem_total else None,
            source="platform",
        )
    ]


def list_llama_devices(llama_server: str | None) -> list[dict[str, Any]]:
    """Ask the llama-server binary which backends it was actually built with.

    Hardware we can see is not the same as hardware the binary can use: a
    CPU-only build on a CUDA box must still run on the CPU. This is the
    authoritative signal, and the probe treats it as such.
    """
    if not llama_server:
        return []
    out = _run([llama_server, "--list-devices"], timeout=60.0)
    if out is None:
        return []
    devices: list[dict[str, Any]] = []
    for line in out.splitlines():
        match = _LIST_DEVICES_RE.match(line)
        if not match:
            continue
        total = int(match.group("total")) * 1024 * 1024
        free = match.group("free")
        devices.append(
            {
                "backend": match.group("backend").lower(),
                "name": match.group("name"),
                "memory_bytes": total or None,
                "free_bytes": int(free) * 1024 * 1024 if free else None,
            }
        )
    return devices


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #


def _block_device_for(fs: _Fs, path: Path) -> str | None:
    """Map a path to the kernel block device name backing it."""
    try:
        st = path.stat()
    except OSError:
        return None
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    if major == 0:
        # Virtual filesystem (tmpfs, overlay, btrfs subvol, ZFS): no block dev.
        return None
    link = fs.path(f"/sys/dev/block/{major}:{minor}")
    try:
        resolved = link.resolve()
    except OSError:
        return None
    name = resolved.name
    # Walk up from a partition (sda3) to its disk (sda), where `rotational`
    # actually lives.
    if (resolved.parent / "queue" / "rotational").exists():
        return resolved.parent.name
    return name


def detect_storage(fs: _Fs, target: str) -> StorageInfo:
    info = StorageInfo(path=target)
    probe_path = Path(target)
    # The state directory may not exist yet; walk up to something that does.
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    try:
        usage = shutil.disk_usage(probe_path)
        info.free_bytes, info.total_bytes = usage.free, usage.total
    except OSError:
        pass

    if platform.system() == "Darwin":
        # Every Apple silicon Mac, and every Mac still receiving updates,
        # ships flash storage.
        info.rotational = False
        return info

    device = _block_device_for(fs, probe_path)
    if device:
        rotational = fs.read_int(f"/sys/block/{device}/queue/rotational")
        if rotational is not None:
            info.rotational = bool(rotational)
    if info.rotational is None:
        # Device-mapper, virtio and overlayfs hide the rotational flag. If any
        # physical disk reports non-rotational and none report rotational, call
        # it flash.
        flags = []
        for rot in fs.glob("/sys/block/*/queue/rotational"):
            try:
                flags.append(int(rot.read_text(encoding="utf-8").strip()))
            except (OSError, ValueError):
                continue
        if flags and all(f == 0 for f in flags):
            info.rotational = False
    return info


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def probe(
    root: str = "/",
    state_dir: str | None = None,
    llama_server: str | None = None,
    force_acceleration: str | None = None,
) -> Probe:
    """Collect every fact nix-llama needs about the host."""
    fs = _Fs(root)
    result = Probe(root=root)
    result.os = platform.system().lower()
    result.arch = platform.machine()
    result.container, result.container_runtime = detect_container(fs)

    if result.os == "darwin":
        result.cpu = detect_cpu_darwin()
        result.memory = detect_memory_darwin()
    else:
        result.cpu = detect_cpu_linux(fs)
        result.memory = detect_memory_linux(fs)

    if result.os == "darwin" and result.arch in ("arm64", "aarch64"):
        result.accelerators = detect_apple(result.memory.total_bytes)
        result.unified_memory = True
    else:
        result.accelerators = detect_nvidia(fs) + detect_amd(fs) + detect_intel(fs)

    result.devices = list_llama_devices(llama_server)
    result.acceleration = choose_backend(result, force_acceleration)
    result.vram_bytes = vram_for(result, result.acceleration)
    result.storage = detect_storage(fs, state_dir or os.getcwd())
    return result


def choose_backend(result: Probe, forced: str | None = None) -> str:
    """Pick the backend to serve on."""
    if forced and forced not in ("auto", ""):
        if forced == "none":
            return "cpu"
        return forced

    usable = {d["backend"] for d in result.devices}
    # Backends llama.cpp reports that are not real offload targets.
    usable.discard("cpu")
    usable.discard("blas")
    usable.discard("rpc")

    detected = [a.backend for a in result.accelerators]
    for backend in BACKEND_PRIORITY:
        if backend == "cpu":
            break
        if backend in usable:
            return backend
    if detected and not usable and result.devices:
        # Hardware is present but this build cannot address it. Say so loudly:
        # silently running 30x slower is the single most confusing failure mode.
        result.warnings.append(
            f"detected {', '.join(sorted(set(detected)))} hardware but the llama-server build "
            f"exposes no matching device; falling back to CPU. "
            f"Set services.nix-llama.acceleration to rebuild with the right backend."
        )
    return "cpu"


def vram_for(result: Probe, backend: str) -> int | None:
    """Device memory available to the chosen backend, in bytes."""
    if backend == "cpu":
        return None
    reported = [
        d["memory_bytes"]
        for d in result.devices
        if d["backend"] == backend and d.get("memory_bytes")
    ]
    if reported:
        return sum(reported)
    detected = [
        a.memory_bytes for a in result.accelerators if a.backend == backend and a.memory_bytes
    ]
    if detected:
        return sum(detected)
    return None
