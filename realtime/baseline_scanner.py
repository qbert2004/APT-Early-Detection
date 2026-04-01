"""
Network Baseline Scanner
========================
Discovers active hosts on a subnet using ARP (Layer-2, very fast).
Stores the "known hosts" list in models/baseline_hosts.json.

The detector uses this list to flag flows from unknown/new hosts
with a bonus risk score (+0.15), since new devices appearing mid-session
are a classic APT lateral-movement indicator.

Usage:
    # Scan local /24 subnet and save baseline
    python -X utf8 -m realtime.baseline_scanner --subnet 192.168.1.0/24

    # Check single IP against saved baseline
    python -X utf8 -m realtime.baseline_scanner --check 10.0.0.42

Requirements:
    - Scapy (already in requirements.txt)
    - Admin / root privileges for ARP scan (or use --demo for testing)
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

ROOT      = Path(__file__).parent.parent
HOSTS_FILE = ROOT / "models" / "baseline_hosts.json"

# Risk bonus for unknown hosts (added on top of ml_score)
UNKNOWN_HOST_BONUS = 0.15


# ── Host discovery ─────────────────────────────────────────────────────────────

def scan_subnet(subnet: str, timeout: float = 2.0) -> list[dict]:
    """
    ARP-scan a subnet and return list of active hosts.
    Requires Scapy and admin/root privileges.

    Returns list of dicts: {ip, mac, hostname, first_seen}
    """
    try:
        from scapy.all import ARP, Ether, srp
    except ImportError:
        log.error("Scapy not available for ARP scan")
        return []

    print(f"[scanner] Scanning {subnet} (timeout={timeout}s) …")
    try:
        arp_req   = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=subnet)
        answered, _ = srp(arp_req, timeout=timeout, verbose=False)
    except PermissionError:
        print("[scanner] ⚠ Permission denied — ARP scan requires admin/root.")
        print("[scanner] Try: sudo python -m realtime.baseline_scanner --subnet ...")
        return []
    except Exception as exc:
        log.error("ARP scan failed", error=str(exc))
        return []

    hosts = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for _, rcv in answered:
        ip  = rcv.psrc
        mac = rcv.hwsrc
        try:
            hostname = socket.gethostbyaddr(ip)[0]
        except socket.herror:
            hostname = ""
        hosts.append({"ip": ip, "mac": mac, "hostname": hostname, "first_seen": ts})
        print(f"  ✓ {ip:16s}  {mac}  {hostname}")

    print(f"[scanner] Found {len(hosts)} active hosts")
    return hosts


def scan_demo(n: int = 10) -> list[dict]:
    """
    Generate a synthetic baseline without needing admin privileges.
    Useful for testing and demo.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    hosts = []
    for i in range(1, n + 1):
        hosts.append({
            "ip":         f"192.168.1.{i}",
            "mac":        f"aa:bb:cc:dd:ee:{i:02x}",
            "hostname":   f"host-{i}.local",
            "first_seen": ts,
        })
    print(f"[scanner] Demo baseline: {len(hosts)} synthetic hosts")
    return hosts


# ── Baseline management ────────────────────────────────────────────────────────

def save_baseline(hosts: list[dict], merge: bool = True) -> Path:
    """Save (or merge with existing) baseline to JSON."""
    existing: dict[str, dict] = {}

    if merge and HOSTS_FILE.exists():
        try:
            existing = {h["ip"]: h for h in json.loads(HOSTS_FILE.read_text())}
        except Exception:
            pass

    for h in hosts:
        existing[h["ip"]] = h

    HOSTS_FILE.parent.mkdir(exist_ok=True)
    data = list(existing.values())
    HOSTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info("baseline saved", hosts=len(data), path=str(HOSTS_FILE))
    print(f"[scanner] Baseline saved → {HOSTS_FILE}  ({len(data)} hosts)")
    return HOSTS_FILE


def load_baseline() -> set[str]:
    """Return set of known IP addresses from baseline file."""
    if not HOSTS_FILE.exists():
        return set()
    try:
        data = json.loads(HOSTS_FILE.read_text(encoding="utf-8"))
        return {h["ip"] for h in data}
    except Exception:
        return set()


# ── Risk enrichment ────────────────────────────────────────────────────────────

class BaselineChecker:
    """
    Checks if a source IP is in the known baseline.
    Used by the detector to add risk bonus for unknown hosts.
    """

    def __init__(self):
        self._known: set[str] = load_baseline()
        self._unknown_seen: set[str] = set()
        count = len(self._known)
        if count:
            log.info("baseline loaded", known_hosts=count)
        else:
            log.warning(
                "No baseline file found — all hosts treated as known. "
                "Run: python -m realtime.baseline_scanner --subnet <your-subnet>"
            )

    def reload(self):
        """Reload baseline from disk (useful for hot-reload in long-running sessions)."""
        self._known = load_baseline()

    def risk_bonus(self, ip: str) -> float:
        """
        Return extra risk score for unknown hosts.
        0.0 if host is known or baseline is empty.
        UNKNOWN_HOST_BONUS if host is NOT in baseline.
        """
        if not self._known:
            return 0.0  # no baseline → neutral

        # strip port if present
        clean_ip = ip.split(":")[0]

        if clean_ip not in self._known:
            if clean_ip not in self._unknown_seen:
                self._unknown_seen.add(clean_ip)
                log.warning("unknown host detected", ip=clean_ip,
                            bonus=UNKNOWN_HOST_BONUS)
            return UNKNOWN_HOST_BONUS
        return 0.0

    def is_known(self, ip: str) -> bool:
        clean_ip = ip.split(":")[0]
        return not self._known or clean_ip in self._known

    def stats(self) -> dict:
        return {
            "known_hosts":   len(self._known),
            "unknown_seen":  len(self._unknown_seen),
            "unknown_ips":   list(self._unknown_seen),
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network Baseline Scanner")
    parser.add_argument("--subnet",  default=None, help="Subnet to scan, e.g. 192.168.1.0/24")
    parser.add_argument("--demo",    action="store_true", help="Generate synthetic demo baseline")
    parser.add_argument("--check",   default=None, metavar="IP",
                        help="Check if a specific IP is in baseline")
    parser.add_argument("--show",    action="store_true", help="Show current baseline")
    parser.add_argument("--timeout", default=2.0, type=float, help="ARP timeout seconds")
    args = parser.parse_args()

    if args.show:
        if HOSTS_FILE.exists():
            hosts = json.loads(HOSTS_FILE.read_text())
            print(f"Baseline: {len(hosts)} hosts")
            for h in hosts:
                print(f"  {h['ip']:16s}  {h.get('mac','')}  {h.get('hostname','')}")
        else:
            print("No baseline file found.")

    elif args.check:
        checker = BaselineChecker()
        known = checker.is_known(args.check)
        bonus = checker.risk_bonus(args.check)
        status = "✅ KNOWN" if known else f"⚠ UNKNOWN (risk +{bonus:.2f})"
        print(f"{args.check} → {status}")

    elif args.demo:
        hosts = scan_demo()
        save_baseline(hosts)

    elif args.subnet:
        hosts = scan_subnet(args.subnet, timeout=args.timeout)
        if hosts:
            save_baseline(hosts)
        else:
            print("[scanner] No hosts found or scan failed.")

    else:
        parser.print_help()
