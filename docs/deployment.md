# Deployment: plant server with SQL Server

Target: one always-on machine on the plant network that can reach the camera
(`10.110.4.8`, RTSP 554) and the database server (SQL Server, TCP 1433). Windows Server /
Windows 10+ or Ubuntu 22.04+. A GPU is optional: the heavy model runs about 30 times per
40-minute visit, so a modern CPU keeps up with one camera.

## 1. Install

```powershell
# Windows (PowerShell). Linux: same steps with python3 / .venv/bin/
winget install Gyan.FFmpeg            # ffmpeg on PATH (recorder)
# Python 3.11 or 3.12, 64-bit, from python.org
cd C:\watchin-u                       # the repository folder
py -3.12 -m venv .venv
.venv\Scripts\activate
# GPU only: first install the CUDA build of torch from pytorch.org, then:
pip install -e ".[ml,mssql]"
yolo settings sync=False              # no usage analytics from the plant server
```

SQL Server also needs **"ODBC Driver 18 for SQL Server"** installed on this machine.

## 2. Database (SQL Server)

1. The DBA creates a database `watchinu`, plus a login with `db_datareader`, `db_datawriter`
   and `db_ddladmin` on it. `db_ddladmin` is for migrations only; use a separate login without it
   for the service if your policy requires.
2. Put the connection URL in `.env` (step 3). URL-encode special characters in the password
   (`@` becomes `%40`, `:` becomes `%3A`):
   ```
   # SQL login
   WATCHINU_DATABASE_URL=mssql+pyodbc://watchinu:PASSWORD@SQLHOST:1433/watchinu?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes
   # Windows authentication (service runs as a domain account)
   WATCHINU_DATABASE_URL=mssql+pyodbc://@SQLHOST/watchinu?driver=ODBC+Driver+18+for+SQL+Server&Trusted_Connection=yes&Encrypt=yes
   ```
   Add `&TrustServerCertificate=yes` only if the server uses a self-signed certificate.
3. Create the schema and smoke-test it:
   ```
   watchinu db-upgrade
   watchinu db-check --config configs/camera01.yaml
   ```
   `db-check` connects, checks the schema version, and writes and reads a visit and a decision
   (timezone timestamps, JSON, Unicode) inside a transaction it rolls back. All lines must say `OK`.

PostgreSQL works the same way (`pip install -e ".[postgres]"`,
`postgresql+psycopg://...`). SQLite is for development only.

## 3. Configuration and secrets

`.env` in the repository folder is loaded automatically. Restrict it to the service account
(Windows: file Properties, Security; Linux: `chmod 600`).

```
WATCHINU_CAM01_USER=analytics          # dedicated read-only camera user, NOT admin/IT
WATCHINU_CAM01_PASSWORD=...
WATCHINU_DATABASE_URL=mssql+pyodbc://...
TZ=Asia/Kolkata
```

In `configs/camera01.yaml`:

* `models.*`: paths to the trained `best.pt` files (copied to `models/`), and `device`
  (`cpu`, or `"0"` for the first GPU).
* geometry: re-fit `bays[].polygon` / `bed_quad` on a real frame and check it with
  `watchinu render-config ... --layout linear` and `--layout criss_cross`.
* `camera.reference_image`: a clean main-stream frame (for example from
  `watchinu capture --mode timelapse` running for a minute) with `static_region`. This
  enables the view-drift guard.

## 4. Verify before going live

```
watchinu replay --config configs/camera01.yaml --video <recent recording or time-lapse> --out runs/replay/check
watchinu eval --predictions runs/replay/check/events.jsonl --visit-log <its visit log>
```

Then run live in the foreground once and watch the log:
`watchinu run --config configs/camera01.yaml --log data/events.jsonl`

## 5. Run as a service

* **Windows**: `deploy\windows\install-service.ps1` (uses [NSSM](https://nssm.cc)). It sets auto-start,
  restart on failure, and rotated logs in `data\logs\`.
* **Linux**: `deploy/linux/watchinu.service` (systemd).

Open visits survive restarts: the state machine is persisted in the database, and a resumed
visit is flagged `restarted_mid_visit`.

## 6. Operations

| Task | How |
|---|---|
| Health | Service log; `alerts` table (`view_drift`, `view_restored`); `data/events.jsonl` |
| Results | `decisions` (append-only; latest row per visit is current), with `evidence` image paths |
| Evidence retention | Scheduled task deleting `data/evidence/*` older than the agreed retention |
| Backups | SQL Server maintenance plan (full plus log); evidence folder on the NAS backup |
| Upgrade | stop service, update code, `pip install -e ".[ml,mssql]"`, `db-upgrade`, `db-check`, start |
| Rollback | previous code folder; `alembic downgrade <rev>` only if the new release migrated |
| New model | copy weights, update `models.*`, restart. Every decision records its model versions |

Not in this release (phase 2): the outbox worker that pushes decisions to ERP or alerts.
Rows accumulate in `outbox` until it is enabled.
