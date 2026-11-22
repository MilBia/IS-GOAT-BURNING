"""Child process of ``test_no_outbound.py`` (#210): runs local inference and reports network/process side effects.

Usage: ``python no_outbound_child.py <mode> <model.onnx> <wait_seconds>``. ``mode`` is an ``EdgeInferenceDetector``
backend (``cv2dnn``, ``openvino``) or a negative control that imports a library *without* the app's telemetry guard
(``raw_openvino``) or opens a plain Python connection (``raw_socket``). A
Python-level socket guard is installed before anything else is imported: it records and refuses every DNS lookup /
connect / sendto / sendmsg that is not to an ``AF_UNIX`` socket or loopback. Native code and ``multiprocessing``
children bypass it, which is why the parent also checks the temporary directories for files and leftover processes
and may run this script in a network namespace.

The last stdout line is a JSON report: ``attempts``, ``children`` (PIDs of child processes still alive after the wait)
and ``results`` (inference verdicts, as a sanity check that inference really ran).
"""

from collections.abc import Callable
import contextlib
import ipaddress
import json
from pathlib import Path
import socket
import sys
import time

ATTEMPTS: list[str] = []


def _is_loopback(host: object) -> bool:
    """Whether ``host`` is ``localhost`` or a loopback IP address (127.0.0.0/8, ::1)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


def _refuse(what: str) -> OSError:
    """Records a blocked attempt and returns the error to raise."""
    ATTEMPTS.append(what)
    return ConnectionRefusedError(f"blocked by the no-outbound test guard: {what}")


def _guard_method(name: str, address_of: Callable[[tuple], object]) -> None:
    """Replaces ``socket.socket.<name>`` with a wrapper that refuses non-local destinations.

    Args:
        name: Method name (``connect``, ``connect_ex``, ``sendto``, ``sendmsg``).
        address_of: Extracts the destination from the call's positional arguments (``None``: no destination given,
            i.e. an already connected and therefore already checked socket).
    """
    real = getattr(socket.socket, name)

    def wrapper(self: socket.socket, *args: object) -> object:
        address = address_of(args)
        if address is None or self.family == socket.AF_UNIX or (isinstance(address, tuple) and _is_loopback(address[0])):
            return real(self, *args)
        raise _refuse(f"{name}({address!r})")

    setattr(socket.socket, name, wrapper)


def _install_socket_guard() -> None:
    """Wraps the ``socket`` entry points so non-local network use is recorded and refused."""
    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        if host is None or _is_loopback(host) or str(host).startswith("/"):
            return real_getaddrinfo(host, port, *args, **kwargs)
        raise _refuse(f"getaddrinfo({host!r}, {port!r})")

    socket.getaddrinfo = getaddrinfo
    _guard_method("connect", lambda args: args[0])
    _guard_method("connect_ex", lambda args: args[0])
    _guard_method("sendto", lambda args: args[-1])  # sendto(data[, flags], address)
    _guard_method("sendmsg", lambda args: args[3] if len(args) > 3 else None)  # sendmsg(buffers, anc, flags, address)
    # create_connection() resolves via socket.getaddrinfo and connects via socket.socket.connect: both guarded.


def _child_pids() -> list[int]:
    """Returns the PIDs of this process's live child processes (all threads), read from ``/proc``."""
    pids: list[int] = []
    for children in Path("/proc/self/task").glob("*/children"):
        try:
            pids += [int(pid) for pid in children.read_text().split()]
        except OSError:
            continue
    return sorted(set(pids))


def _run_detector(mode: str, model_path: str) -> list[bool]:
    """Builds ``EdgeInferenceDetector`` for ``mode`` and classifies a red (fire) and a blue (no fire) frame."""
    import asyncio

    import numpy as np

    from is_goat_burning.config import Accelerator
    from is_goat_burning.config import EdgeSettings
    from is_goat_burning.fire_detection.edge_detector import EdgeInferenceDetector

    accelerator = {"cv2dnn": Accelerator.CPU, "openvino": Accelerator.OPENVINO}[mode]
    settings = EdgeSettings(model_path=model_path, accelerator=accelerator, input_size=8, confidence_threshold=0.5)
    detector = EdgeInferenceDetector(settings)
    red = np.zeros((20, 30, 3), np.uint8)
    red[..., 2] = 255
    blue = np.zeros((20, 30, 3), np.uint8)
    blue[..., 0] = 255

    async def classify() -> list[bool]:
        return [await detector.verify(frame) for frame in (red, blue, red, blue)]

    return asyncio.run(classify())


def _run_raw(mode: str, model_path: str) -> list[bool]:
    """Negative control: uses the library directly, bypassing the app's telemetry guards."""
    if mode == "raw_socket":  # the guard refuses this before any DNS query or packet (reserved .invalid TLD anyway)
        with contextlib.suppress(OSError):
            socket.create_connection(("telemetry.invalid", 443), timeout=1)
        return [True]
    import numpy as np

    blob = np.zeros((1, 3, 8, 8), np.float32)
    blob[:, 0] = 1.0
    import openvino

    logits = openvino.Core().compile_model(model_path, "CPU")({0: blob})[0]
    return [bool(logits[0][1] > logits[0][0])]


def main() -> None:
    """Runs the requested mode, waits for delayed uploaders and prints the JSON report."""
    _install_socket_guard()
    mode, model_path, wait_seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    results = _run_raw(mode, model_path) if mode.startswith("raw_") else _run_detector(mode, model_path)
    time.sleep(wait_seconds)
    import multiprocessing

    children = sorted(set(_child_pids()) | {p.pid for p in multiprocessing.active_children()})
    print(json.dumps({"attempts": ATTEMPTS, "children": children, "results": results}), flush=True)


if __name__ == "__main__":
    main()
