"""Prints the outbound endpoints the current configuration needs (#212).

Derives the hosts and ports from the settings (stream source, Discord webhooks, SMTP server, cloud verifier) without
printing secrets: only host names and ports, never webhook paths or tokens, passwords or API keys.

Usage (inside the app container, so it reads the deployment's environment):

    python3 egress_allowlist.py                 # human-readable list
    python3 egress_allowlist.py --squid         # allow-list file for deploy/egress/squid.conf

On a development machine pass the env file explicitly (otherwise ``.env.tests`` from the repository root is read):

    PYTHONPATH=. python tools/egress_allowlist.py --env-file /path/outside/repo/goat.env --squid
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from urllib.parse import urlparse

from is_goat_burning.config import Settings

#: Host names of YouTube watch pages; such a source needs the domains below (page, player API, HLS manifests and
#: segments). Measured with the pinned yt-dlp: www.youtube.com and manifest/rr*.googlevideo.com.
YOUTUBE_HOSTS = ("youtube.com", "youtu.be")
YOUTUBE_DOMAINS = (".youtube.com", ".googlevideo.com")
#: Endpoint of the Gemini API (google-genai), used by the cloud verifier strategies only.
GEMINI_HOST = "generativelanguage.googleapis.com"
CLOUD_STRATEGIES = frozenset({"gemini", "hybrid_cloud", "gemini_decision", "hybrid_decision"})
DEFAULT_PORTS = {"http": 80, "https": 443, "rtsp": 554, "rtmp": 1935}


@dataclass(frozen=True)
class Endpoint:
    """One outbound destination.

    Attributes:
        host: Host name, or a squid-style domain with a leading dot (the domain and all its subdomains).
        port: TCP port.
        purpose: What needs it.
    """

    host: str
    port: int
    purpose: str


def _host_port(url: str) -> tuple[str, int] | None:
    """Returns the host and port of a network URL, or None for a local file, a device index or a malformed URL."""
    parsed = urlparse(url)
    try:
        port = parsed.port or DEFAULT_PORTS.get(parsed.scheme.lower())
    except ValueError:
        return None
    if not parsed.hostname or port is None:
        return None
    return parsed.hostname.lower(), port


def _is_youtube(host: str) -> bool:
    """Tells whether a host belongs to YouTube."""
    return any(host == name or host.endswith("." + name) for name in YOUTUBE_HOSTS)


def required_endpoints(settings: Settings) -> list[Endpoint]:
    """Lists the outbound endpoints the given configuration needs, without duplicates.

    Args:
        settings: The application settings.

    Returns:
        The endpoints in a stable order: stream source, Discord, SMTP, cloud verifier.
    """
    endpoints: list[Endpoint] = []
    source = _host_port(settings.source)
    if source is not None:
        host, port = source
        if _is_youtube(host):
            endpoints += [Endpoint(domain, 443, "stream (YouTube via yt-dlp, HLS)") for domain in YOUTUBE_DOMAINS]
            if host == "youtu.be":
                endpoints.append(Endpoint(host, port, "stream (short link)"))
        else:
            endpoints.append(Endpoint(host, port, "stream source"))
    webhooks = [(url, "Discord alerts") for url in settings.discord.hooks if settings.discord.use_discord]
    webhooks += [(url, "Discord service channel") for url in settings.discord.service_hooks]
    for url, purpose in webhooks:
        hook = _host_port(url)
        if hook is not None:
            endpoints.append(Endpoint(*hook, purpose))
    if settings.email.use_emails and settings.email.email_host and settings.email.email_port:
        endpoints.append(Endpoint(settings.email.email_host.lower(), settings.email.email_port, "SMTP (email alerts)"))
    if settings.detection_strategy in CLOUD_STRATEGIES:
        endpoints.append(Endpoint(GEMINI_HOST, 443, f"cloud verifier ({settings.detection_strategy})"))
    unique: dict[tuple[str, int], Endpoint] = {}
    for endpoint in endpoints:
        unique.setdefault((endpoint.host, endpoint.port), endpoint)
    return list(unique.values())


def _covered(host: str, domains: set[str]) -> bool:
    """Tells whether a host is matched by one of the leading-dot domains."""
    return any(host == domain[1:] or host.endswith(domain) for domain in domains if host != domain)


def squid_allowlist(endpoints: list[Endpoint]) -> str:
    """Renders a squid ``dstdomain`` file for the HTTPS endpoints.

    The proxy tunnels HTTPS (port 443) only. Other endpoints are listed as comments so they are not lost: SMTP goes
    through the ``email`` profile of the compose file, anything else needs its own rule.

    Args:
        endpoints: The endpoints from :func:`required_endpoints`.

    Returns:
        The file content, one domain per line (squid rejects a name that a listed domain already covers).
    """
    https = [e for e in endpoints if e.port == 443]
    domains = {e.host for e in https if e.host.startswith(".")}
    lines = ["# Generated by tools/egress_allowlist.py from the current settings (#212)."]
    lines += [e.host for e in https if not _covered(e.host, domains)]
    lines += [f"# not through this proxy: {e.host}:{e.port} ({e.purpose})" for e in endpoints if e.port != 443]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Prints the endpoints, or the squid allow-list with ``--squid``.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--squid", action="store_true", help="print the squid dstdomain allow-list instead")
    parser.add_argument("--env-file", help="read this env file instead of the default .env.tests/.env")
    args = parser.parse_args(argv)
    if args.env_file:
        settings = Settings(_env_file=args.env_file)
    else:
        from is_goat_burning.config import settings
    endpoints = required_endpoints(settings)
    if args.squid:
        sys.stdout.write(squid_allowlist(endpoints))
        return 0
    for endpoint in endpoints:
        print(f"{endpoint.host}:{endpoint.port}\t{endpoint.purpose}")
    print("DNS: resolved by the proxy in the Docker variant; with a host firewall, allow the configured resolver.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
