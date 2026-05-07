from __future__ import annotations

import os
import platform
import re
import subprocess
from datetime import timedelta
from typing import Any


def configure_gloo_socket_ifname(master_addr: str) -> str:
    """Prefer the Tailscale interface for multi-machine Gloo process groups."""

    existing = os.environ.get("GLOO_SOCKET_IFNAME", "").strip()
    if existing:
        return existing
    if master_addr in {"127.0.0.1", "localhost", "::1"}:
        return ""

    ifname = _tailscale_interface_name()
    if ifname:
        os.environ["GLOO_SOCKET_IFNAME"] = ifname
    return ifname


def create_gloo_pg_options(master_addr: str, dist_module: Any, timeout: timedelta) -> tuple[Any | None, str]:
    """Force Gloo to advertise the local Tailscale IPv4 address when possible.

    On macOS Tailscale uses utun interfaces that can also expose link-local IPv6
    addresses. Gloo may select one of those unusable addresses even when
    GLOO_SOCKET_IFNAME is correct, so explicit device creation is more reliable.
    """

    if master_addr in {"127.0.0.1", "localhost", "::1"}:
        return None, ""

    tailscale_ip = _tailscale_ipv4()
    if not tailscale_ip:
        return None, ""

    process_group_gloo = getattr(dist_module, "ProcessGroupGloo", None)
    if process_group_gloo is None:
        _set_gloo_socket_ifname_fallback(tailscale_ip)
        return None, tailscale_ip

    create_device = getattr(process_group_gloo, "create_device", None)
    options_cls = getattr(process_group_gloo, "_Options", None)
    if create_device is None or options_cls is None:
        _set_gloo_socket_ifname_fallback(tailscale_ip)
        return None, tailscale_ip

    try:
        options = options_cls()
        options._devices = [create_device(hostname=tailscale_ip)]
        if hasattr(options, "_timeout"):
            options._timeout = timeout
    except Exception:
        # Explicit device creation failed; fall back to GLOO_SOCKET_IFNAME so
        # Gloo at least binds to the right interface instead of picking any one.
        _set_gloo_socket_ifname_fallback(tailscale_ip)
        return None, tailscale_ip

    os.environ.pop("GLOO_SOCKET_IFNAME", None)
    return options, tailscale_ip


def _set_gloo_socket_ifname_fallback(tailscale_ip: str) -> None:
    """Set GLOO_SOCKET_IFNAME to the Tailscale interface when pg_options are unavailable."""
    if os.environ.get("GLOO_SOCKET_IFNAME", "").strip():
        return
    ifname = _tailscale_interface_name()
    if ifname:
        os.environ["GLOO_SOCKET_IFNAME"] = ifname


def _tailscale_interface_name() -> str:
    tailscale_ip = _tailscale_ipv4()
    if not tailscale_ip:
        return ""
    system = platform.system().lower()
    if system == "darwin":
        return _darwin_interface_for_ip(tailscale_ip)
    if system == "linux":
        return _linux_interface_for_ip(tailscale_ip)
    return ""


def _tailscale_ipv4() -> str:
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""


def _darwin_interface_for_ip(ip: str) -> str:
    try:
        result = subprocess.run(
            ["ifconfig"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""

    current = ""
    for line in result.stdout.splitlines():
        if line and not line.startswith(("\t", " ")):
            current = line.split(":", 1)[0]
            continue
        if current and re.search(rf"\binet\s+{re.escape(ip)}\b", line):
            return current
    return ""


def _linux_interface_for_ip(ip: str) -> str:
    try:
        result = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[2] == "inet" and fields[3].split("/", 1)[0] == ip:
            return fields[1]
    return ""
