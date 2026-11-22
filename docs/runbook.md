# Operations runbook

How to run, check and recover the production deployment (Docker, `hybrid_decision`, segments recording, Discord alerts plus service channel). Settings and the full preset are in the README section [Production run](../README.md#production-run); setting names are in `.env.example`.

Examples use the container name `burning_goat_detection_container` and the env file `/path/outside/repo/goat.env`. Docker env files do **not** strip quotes: write values unquoted. Quoted values in `.env.example` are fine for the app's own `.env` loader, but must be unquoted in a Docker `--env-file`.

## 1. How to check it is running

```bash
docker ps --filter name=burning_goat_detection_container
docker logs --tail 100 burning_goat_detection_container
```

After a start, `docker logs` shows in this order:

```text
Successfully resolved stream URL for <stream URL> with preferred format.
Segments ring: N slots of >= 2s in /dev/shm/goat-ring.
Initialized hybrid detector with verifier DecisionFireDetector
Segment archiver (ffmpeg -c copy) started.
```

With `ESCALATION_LOG__ENABLED=true` there is also `Escalation log enabled: directory=/app/recordings/escalations limit=2000 MB` (logged when the log is first created; a missing line right after start is not a failure).

Service channel (`DISCORD__SERVICE_HOOKS`):

-   At every start: `[Goat monitor] Monitoring started. Strategy: ... Source: ... Heartbeat every ...; stream-down warning after ...`.
-   Heartbeat every `DISCORD__SERVICE_HEARTBEAT_INTERVAL_SECONDS` (default 24 h; the production preset uses 6 h): `[Goat monitor] Heartbeat: running for ... Stream: up for .... Fire events since last heartbeat: N (unverified: M). Verifier: available.`
-   **No heartbeat for more than a day means the process or the host's network is down**, even if `docker ps` looks fine. Check `docker logs` first.

Escalation log (every frame sent to the verifier): on the recordings volume, `<recordings dir>/escalations/` with one JPEG per verifier call and an `escalations.jsonl` index (`verdict`, probability, policy). The directory is capped by `ESCALATION_LOG__MAX_MB`; the oldest JPEGs are pruned.

```bash
tail -n 20 /path/to/recordings/escalations/escalations.jsonl
```

## 2. Day zero (the camera goes live)

1.  **Find the stream URL** (the 2026 live stream on the Visit Gävle YouTube channel) and open it in a browser.
2.  **Check it resolves and note the parameters** (pinned `yt-dlp` and format as in the app):

    ```bash
    yt-dlp --print "%(title)s | live=%(is_live)s | %(resolution)s %(fps)sfps | %(vcodec)s | tbr=%(tbr)s" -f "bestvideo/best" "<stream URL>"
    ```

    Expect `live=True`. Write down resolution, fps and bitrate; 60 fps at 1080p needs a host with spare CPU (a Raspberry Pi 4 is too small).
3.  **Control frame.** Grab one frame and look at it: framing, whether the goat stands, privacy mosaic, text overlays, lamps.

    ```bash
    ffmpeg -i "$(yt-dlp -g -f 'bestvideo/best' '<stream URL>')" -frames:v 1 control.jpg
    ```

4.  **Shadow mode.** Start with `SOURCE=<stream URL>` and **test** webhooks in both `DISCORD__HOOKS` and `DISCORD__SERVICE_HOOKS` (they must be different URLs). After 10 minutes check the startup lines (section 1) and CPU:

    ```bash
    docker stats --no-stream burning_goat_detection_container
    ```

5.  **Review the escalation log after the first dusk and the first night.** Count escalations per hour and look at the verdicts in `escalations.jsonl`. If the verifier confirmed anything that was not fire, do not switch to the alert channel yet; tune `DECISION__THRESHOLD` or the gatekeeper threshold first.
6.  **Switch to the alert webhook before the inauguration** (see section 4): put the real webhook in `DISCORD__HOOKS`, recreate the container, confirm the `Monitoring started` message on the service channel.

The README production preset sets `RECONNECT_MAX_DELAY_SECONDS=120` (see section 6).

## 3. After an alert

Alert on `DISCORD__HOOKS`, with the triggering frame attached (`DISCORD__ATTACH_FRAME`).

-   **Verified alert:** the Gemini verifier confirmed the CV detection. Look at the attached frame, then at the stream.
-   **`[UNVERIFIED]` alert:** the verifier was unavailable and `VERIFIER_FAILURE_POLICY=alert_unverified` let the local detector alert anyway. Message note: `[UNVERIFIED] The fire verifier is unavailable; this alert comes from the local camera detector only.` (the text after `[UNVERIFIED]` is configurable via `UNVERIFIED_ALERT_NOTE`). Treat it as a real possibility, check the stream yourself, and see section 6 for the cause.

Recordings are in the recordings volume, in a directory `event_<YYYY-MM-DD_HH-MM-SS>`:

-   `goat-cam_pre-fire-buffer_<timestamp>.mp4`: the ring buffer from before the fire (up to `VIDEO__MEMORY_BUFFER_SECONDS`).
-   `goat-cam_post-fire_<timestamp>.mp4`: consecutive clips of `VIDEO__VIDEO_CHUNK_LENGTH_SECONDS` while the fire lasts (with `VIDEO__RECORD_DURING_FIRE=true`) and after it.
-   With `DISCORD__SEND_VIDEO_CHUNKS=true` clips are uploaded to the alert channel. A clip larger than `DISCORD__FILE_LIMIT_MB` (default 18 MiB) is **not** uploaded; instead the channel gets `Goat on fire! A video chunk (<file>, <size> MiB) is too large to upload to Discord. It is saved on disk.` The file is in the `event_` directory.

Do **not** restart the container during a real fire: recording is in progress and the final clips are written at the end of the event.

**False alarm:** no restart is needed. Alerts are re-armed automatically. `ALERT_COOLDOWN_SECONDS` defaults to 600 s: a new fire within 10 minutes of the last alert is not dropped but deferred to the end of the cooldown (log: `Fire alert deferred: <N>s of the alert cooldown remain.`), and cancelled if the fire goes out first. Keep the frame from the alert, the matching JPEG and line from the escalation log, and the log excerpt (`docker logs --since <time> ...`); they are the material for tuning thresholds. Delete the `event_` directory only after saving what you need.

## 4. Changing the stream URL or other configuration

The env file is read only when the container is **created**; `docker restart` keeps the old values. Edit the file, then recreate:

```bash
docker stop burning_goat_detection_container
docker rm burning_goat_detection_container
docker run -d --name burning_goat_detection_container \
  --restart unless-stopped \
  --stop-timeout 30 \
  --shm-size 1g \
  --env-file /path/outside/repo/goat.env \
  -v /path/to/recordings:/app/recordings \
  burning_goat_detection
```

Recordings and the escalation log live on the volume and survive. Confirm the `Monitoring started` message on the service channel and the startup lines from section 1.

## 5. Updating yt-dlp when YouTube breaks resolution

`yt-dlp` is pinned in `pyproject.toml`. When resolution fails and the stream is up in a browser:

```bash
# on a development machine
pip install -U yt-dlp
yt-dlp --print title "<stream URL>"      # confirm the new version resolves
# bump the pin in pyproject.toml (yt-dlp==<version>)
```

Then regenerate **all** `requirements*.txt` (each file's header has its `pip-compile` command; the list is in [CONTRIBUTING](../CONTRIBUTING.md#dependency-updates-dependabot)). `requirements-openvino.txt` pins yt-dlp too, and the `requirements-in-sync` CI job fails if any file is left behind.

On the host: pull the change, rebuild and recreate the container (section 4):

```bash
docker build --no-cache --target cpu -t burning_goat_detection .
```

`--no-cache` makes sure the pip layer is not reused.

## 6. Failures

| Symptom (log or service message) | Cause | Action |
|---|---|---|
| `Stream unavailable (<error>). Reconnecting in <N>s...` repeating, N growing from `RECONNECT_DELAY_SECONDS` up to `RECONNECT_MAX_DELAY_SECONDS` (default 300) | Stream offline, network down, or YouTube changed format | Open the stream in a browser. If it is up, see section 5. After the network returns, recovery can lag up to `RECONNECT_MAX_DELAY_SECONDS` (rehearsal: 4 min 13 s); the production preset uses `RECONNECT_MAX_DELAY_SECONDS=120`. Service message `WARNING: the stream has been unavailable for ...` after `DISCORD__SERVICE_STREAM_DOWN_WARNING_SECONDS`; `Stream recovered` when it is back. No container restart needed. |
| `Verifier DecisionFireDetector unavailable (<reason>); policy alert_unverified: fires seen by the local detector are reported as UNVERIFIED until it recovers.` (CRITICAL; also `Decisions API failed 5 times in a row; ...`) and service message `WARNING: verifier ... unavailable` | Bad or expired Gemini key, quota, API outage | Alerts keep coming, marked `[UNVERIFIED]`. Fix the key or quota in the env file and recreate the container (section 4) if the cause is configuration. While the cause persists, the verifier is re-probed only when the CV gate fires, at most once per `VERIFIER_COOLDOWN_SECONDS`; once it answers again the service channel reports `recovered; fires are verified again`, which can lag until the next CV positive. With `VERIFIER_FAILURE_POLICY=suppress` alerts are silenced instead: watch the CRITICAL line. |
| `Segment archiver exited with code <N>.` followed by `Restarting segment archiver in <S>s.` | `ffmpeg` died or was killed (rehearsal: `kill -9` restarted after 1 s) | Normal if rare; the supervisor backs off 1 to 30 s. If it repeats, read the masked `Last stderr lines` in the log. |
| Archiver exits with code 228 repeatedly (#156); `Only <N> MB free in the segment ring directory ... in Docker raise --shm-size.` | `/dev/shm` full (default 64 MB) | Recreate with a larger `--shm-size` (1g is the documented minimum for 1080p60) or shorten `VIDEO__MEMORY_BUFFER_SECONDS`. |
| `Video chunk <file> (<N> MiB) exceeds the <L> MiB limit; not uploading, sending a substitute message.` or HTTP 413 from Discord | Clip over the Discord file limit | Not a failure: the clip is on disk in `event_<timestamp>`. Shorten `VIDEO__VIDEO_CHUNK_LENGTH_SECONDS` or lower `DISCORD__FILE_LIMIT_MB` below the real limit if clips should be uploaded. |
| `Escalation log directory <dir> is not usable (<error>); escalation records will be dropped. Set ESCALATION_LOG__DIRECTORY to a writable path (in Docker: /app/recordings/escalations).` (ERROR) | Directory not writable by the container user | Set `ESCALATION_LOG__DIRECTORY=/app/recordings/escalations` (on the volume) and recreate. Detection and alerts are unaffected; only the tuning data is lost. |
| `No alert action succeeded; the alert stays armed and is retried in <S>s.` | Discord (or email) unreachable when the fire started | The alert is retried every `ALERT_RETRY_SECONDS`. Check the webhook and the host's network. |
| No service messages (start, heartbeat, warnings) while the network is down | Service messages go out over the same network as the stream, so they cannot be delivered during an outage (rehearsal: the stream-down warning did not arrive) | Expected. A missing heartbeat is the signal. After the network returns, the `Stream recovered` message is sent (rehearsal: delivered 1/1). |
| `Service notification was not accepted by any service webhook.` | Service webhook deleted or wrong | Fix `DISCORD__SERVICE_HOOKS` and recreate (section 4). |
| `ffmpeg not found; segments archiving is inactive.` or `Segments mode needs the stream URL; archiving is inactive.` | Image without ffmpeg, or `SOURCE` unset | Rebuild from the repository Dockerfile; check `SOURCE`. Detection and alerts still work, no clips. |

## 7. Stop, start, upgrade

```bash
docker stop burning_goat_detection_container     # graceful, ~10 s, sends "Monitoring stopped after ..."
docker start burning_goat_detection_container    # sends "Monitoring started ..."
docker restart burning_goat_detection_container  # same values as before (env is not re-read)
```

`--stop-timeout 30` at `docker run` gives the graceful shutdown time (final clip, pending alert, service message); the worst case is about 15 s (measured stop in rehearsal: about 10 s), which can exceed Docker's default 10 s grace period and be killed before the shutdown message is sent.

Upgrade: `git pull`, `docker build --target cpu -t burning_goat_detection .`, then stop, remove and run as in section 4. Do it outside a fire and check the startup lines afterwards.

## 8. Egress-restricted deployment

Optional hardening (#212): the app container has no route to the internet; its only way out is an allow-list proxy (squid) that tunnels HTTPS to the configured services by host name and logs every other attempt. Filtering is by the host name in the CONNECT request only: anything served under an allowed name (e.g. other services on the same Google front ends) is reachable too, so this narrows egress, it does not inspect it. Files: `deploy/egress/` (`docker-compose.yml`, `squid.conf`, `allowed-domains.txt`). The proxy logs host and port only, never paths or query strings, so webhook tokens do not reach the log.

What goes through it: yt-dlp, ffmpeg (OpenCV capture and the segments archiver), Discord (alerts, clips, service channel) and the Gemini client honour the lowercase `http_proxy` / `https_proxy` / `no_proxy` variables set by the compose file (ffmpeg reads only `http_proxy`, and uses it for `https://` too). The app has no external DNS; the proxy resolves names. SMTP cannot use an HTTP proxy: see *Email* below.

**1. Generate the allow-list from your configuration** (prints host names and ports only, no secrets):

Every `docker compose` command below needs `GOAT_ENV_FILE` and `GOAT_RECORDINGS_DIR` (and `EGRESS_SMTP_*` with the email profile): put them in a `.env` next to the compose file (`deploy/egress/.env`, ignored by git through the repository's `.env` rule; check with `git status`) rather than exporting them in each shell.

```bash
cd deploy/egress
printf 'GOAT_ENV_FILE=/path/outside/repo/goat.env\nGOAT_RECORDINGS_DIR=/path/to/recordings\n' > .env
docker compose build app
docker compose run --rm --no-deps -T --entrypoint python3 \
  -v ../../tools/egress_allowlist.py:/app/egress_allowlist.py:ro \
  app /app/egress_allowlist.py --squid > allowed-domains.txt
```

`-T` and `--entrypoint python3` keep a terminal's CR characters and the entrypoint's messages out of the file. Step 1 overwrites the committed `allowed-domains.txt`: before a `git pull`, run `git stash` (or `git checkout allowed-domains.txt`), then regenerate.

Without `--squid` it prints a readable list (`host:port  purpose`). The committed `allowed-domains.txt` fits the README production preset (YouTube, Discord, Gemini). For a YouTube source the list holds `.youtube.com` and `.googlevideo.com` (watch page, player API, HLS manifests and segments; a leading dot covers all subdomains). For another source only its host is listed; if its HLS segments come from a different CDN host, add that host after reading the proxy log (step 3).

The compose file builds the `cpu` image; for `opencl` or `gpu` change `target` and add the device options from the README. Unlike `docker run --env-file`, compose expands `$` in the env file: write a literal `$` (e.g. in a password or key) as `$$`. If `docker compose up` fails with `Pool overlaps`, change the internal subnet `172.31.212.0/24` (once) and the relay address `172.31.212.25` (twice) in `docker-compose.yml`.

**2. Start** (instead of `docker run`; stop and remove the old container first, the compose file reuses the name `burning_goat_detection_container`):

```bash
docker compose up -d
```

**3. Verify** (all from `deploy/egress`):

-   Stream, alerts and heartbeat: the startup lines from section 1 (`docker logs burning_goat_detection_container`) and the `Monitoring started` message on the service channel.
-   Proxy log: `docker compose logs proxy | grep TCP_`. Allowed connections show `TCP_TUNNEL/200 CONNECT <host>:443`; a tunnel is logged when it closes, so long-lived stream connections appear later.
-   A disallowed host is blocked and logged:

    ```bash
    docker exec burning_goat_detection_container python3 -c "import urllib.request; urllib.request.urlopen('https://example.com', timeout=10)"
    # -> urllib.error.URLError: Tunnel connection failed: 403 Forbidden
    docker compose logs proxy | grep TCP_DENIED
    # -> ... TCP_DENIED/403 CONNECT example.com:443
    ```

-   Direct egress is impossible:

    ```bash
    docker exec burning_goat_detection_container python3 -c "import socket; socket.create_connection(('1.1.1.1', 443), 5)"
    # -> OSError: [Errno 101] Network is unreachable
    ```

**4. When something stops working** (YouTube or Discord change domains, a new setting needs a new service): `docker compose logs proxy | grep TCP_DENIED` shows the refused `host:port`. Add the host (or its domain with a leading dot) to `allowed-domains.txt`, or regenerate it (step 1), then reload without a restart: `docker compose exec proxy squid -k reconfigure`. If every connection is denied, check that `allowed-domains.txt` is not empty: the redirect in step 1 truncates it even when the tool fails. Only HTTPS on port 443 is tunnelled; plain HTTP and other ports are refused by `squid.conf`. A too-narrow list shows up as the usual stream reconnect loop or failed alerts (section 6) together with `TCP_DENIED` lines.

**Email.** SMTP goes through a separate relay that forwards one host and port only. Set `EGRESS_SMTP_HOST` / `EGRESS_SMTP_PORT` to the same values as `EMAIL__EMAIL_HOST` / `EMAIL__EMAIL_PORT` (STARTTLS port, e.g. 587) and start with the profile:

```bash
EGRESS_SMTP_HOST=<smtp host> EGRESS_SMTP_PORT=587 docker compose --profile email up -d
```

Inside the app the SMTP host name resolves to the relay, so the app still connects to the real name and STARTTLS checks the server certificate against it. Keep `EGRESS_SMTP_*` in `deploy/egress/.env` so every later `docker compose` command sees them.

**Without Docker** (host firewall, e.g. `nftables` for the account running the app) is not scripted here. Use the readable output of `PYTHONPATH=. python tools/egress_allowlist.py --env-file <env file>` (from the repository root) as the list of destinations; note that a firewall filters by IP address, and the YouTube and Discord addresses change, which is why the Docker variant filters by name.
