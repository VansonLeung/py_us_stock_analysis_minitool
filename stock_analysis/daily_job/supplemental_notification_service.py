import datetime as dt
import html
import json
import os
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

from .daily_job_constants import FEAR_GREED_RUNTIME_DIR, FINANCIAL_FORECAST_RUNTIME_DIR, TELEGRAM_MSG_MAX_CHARS


def _format_half_up_1dp(value: object) -> str:
    try:
        return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    except Exception:
        return "n/a"


def _parse_utc_datetime(value: object) -> dt.datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_issue_lines(issues: object) -> list[str]:
    if not isinstance(issues, list):
        return []
    lines: list[str] = []
    for item in issues:
        if isinstance(item, dict):
            message = str(item.get("message") or item.get("code") or "").strip()
            source = str(item.get("source") or "").strip()
            if message and source:
                lines.append(f"{source}: {message}")
            elif message:
                lines.append(message)
            continue
        text = str(item).strip()
        if text:
            lines.append(text)
    return lines


def truncate_html_message_block(title: str, body: str, max_chars: int = TELEGRAM_MSG_MAX_CHARS) -> str:
    prefix = f"<b>{html.escape(title)}</b>\n<pre>"
    suffix = "</pre>"
    normalized = body.strip()
    candidate = f"{prefix}{html.escape(normalized)}{suffix}"
    if len(candidate) <= max_chars:
        return candidate

    truncated_marker = "\n...\n[truncated]"
    low = 0
    high = len(normalized)
    best = f"{prefix}{html.escape('[truncated]')}{suffix}"
    while low <= high:
        mid = (low + high) // 2
        trial_text = normalized[:mid].rstrip() + truncated_marker
        trial = f"{prefix}{html.escape(trial_text)}{suffix}"
        if len(trial) <= max_chars:
            best = trial
            low = mid + 1
        else:
            high = mid - 1
    return best


def extract_forecast_report_error_text(report: dict) -> str:
    errors = report.get("errors") if isinstance(report, dict) else []
    if isinstance(errors, list):
        for item in errors:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            message = str(item.get("message") or "").strip()
            if code and message:
                return f"{code}: {message}"
            if message:
                return message

    searches = report.get("searches") if isinstance(report, dict) else []
    if isinstance(searches, list):
        for entry in searches:
            if not isinstance(entry, dict):
                continue
            error = entry.get("error")
            if not isinstance(error, dict):
                continue
            code = str(error.get("code") or "").strip()
            message = str(error.get("message") or "").strip()
            if code and message:
                return f"{code}: {message}"
            if message:
                return message

    return "no AI overview available"


def build_fear_greed_html_message(report: dict) -> str:
    report_date = str(report.get("report_date") or dt.date.today().isoformat())
    observation = report.get("observation") if isinstance(report, dict) else None
    if not isinstance(observation, dict) or not observation:
        lines = [
            f"report date  {report_date}",
            "status       index unavailable",
            f"freshness    {report.get('freshness', 'unknown')}",
            f"fetch        {report.get('fetch_status', 'unknown')}",
            f"cached       {'yes' if report.get('is_cached') else 'no'}",
        ]
    else:
        observed_at_utc = _parse_utc_datetime(observation.get("observed_at"))
        observed_at_hkt = observed_at_utc.astimezone(ZoneInfo("Asia/Hong_Kong")) if observed_at_utc is not None else None
        category = str(observation.get("category") or "unknown").replace("_", " ").title()
        lines = [
            f"report date  {report_date}",
            f"score        {_format_half_up_1dp(observation.get('score'))}/100",
            f"category     {category}",
            f"market date  {observation.get('market_date') or 'n/a'}",
            f"observed HKT {observed_at_hkt.strftime('%Y-%m-%d %H:%M:%S') if observed_at_hkt is not None else 'n/a'}",
            f"change pts   {_format_half_up_1dp(report.get('change_points'))}",
            f"freshness    {report.get('freshness', 'unknown')}",
            f"fetch        {report.get('fetch_status', 'unknown')}",
            f"cached       {'yes' if report.get('is_cached') else 'no'}",
            f"new obs      {'yes' if report.get('is_new_observation') else 'no'}",
        ]
        if report.get("entry_event") and report.get("freshness") == "current" and report.get("is_new_observation"):
            lines.append("extreme event new entry")

    data_issue_lines = _format_issue_lines(report.get("quality_issues"))
    chart_issue_lines = _format_issue_lines(report.get("chart_issues"))
    artifacts = report.get("artifacts") if isinstance(report, dict) else []
    if not isinstance(artifacts, list):
        artifacts = []
    chart_kinds = [str(item.get("kind")).strip() for item in artifacts if isinstance(item, dict) and item.get("kind")]
    if chart_kinds:
        lines.append(f"charts       {', '.join(chart_kinds)}")
    if data_issue_lines:
        lines.append("")
        lines.append("data issues")
        lines.extend(data_issue_lines)
    if chart_issue_lines:
        lines.append("")
        lines.append("chart issues")
        lines.extend(chart_issue_lines)

    return f"<b>Fear &amp; Greed</b>\n<pre>{html.escape(chr(10).join(lines))}</pre>"


def prepare_fear_greed_notification_message(
    base_dir: Path,
    charts_mode: str,
    timeout_seconds: int,
    attempts: int,
) -> tuple[str, list[Path]]:
    try:
        from fear_greed import Config, prepare_report
    except ImportError as exc:
        raise RuntimeError("Fear & Greed support requires the cnn-fear-greed-daily package in this venv.") from exc

    runtime_dir = (base_dir / FEAR_GREED_RUNTIME_DIR).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config = Config(
        database=runtime_dir / "state.sqlite3",
        timeout_seconds=timeout_seconds,
        attempts=attempts,
    )
    report = prepare_report(config)

    artifacts: list[Path] = []
    if charts_mode != "off":
        from fear_greed.charts.service import attach_charts

        report = attach_charts(report, config, output_dir=runtime_dir / "charts", mode=charts_mode)
        artifacts_data = report.get("artifacts", [])
        if not isinstance(artifacts_data, list):
            artifacts_data = []
        for item in artifacts_data:
            if not isinstance(item, dict):
                continue
            path_value = item.get("path")
            if not path_value:
                continue
            artifact_path = Path(str(path_value))
            if artifact_path.exists():
                artifacts.append(artifact_path)

    return build_fear_greed_html_message(report), artifacts


def _run_financial_forecast_cli(base_dir: Path, timeout_seconds: int, cli_timeout_ms: int) -> dict:
    runtime_dir = (base_dir / FINANCIAL_FORECAST_RUNTIME_DIR).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    report_path = runtime_dir / "latest-report.json"
    stderr_path = runtime_dir / "latest-stderr.log"

    cli_override = os.environ.get("FINANCIAL_FORECAST_CLI", "").strip()
    if cli_override:
        command = [cli_override]
    else:
        command = [
            "npx",
            "--prefix",
            str((base_dir / "vendor/playwright-altered").resolve()),
            "financial_sector_forecast_fetch",
        ]
    command.extend(["--format", "json", "--headed", "--timeout", str(cli_timeout_ms)])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            check=False,
            cwd=str(base_dir),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        stderr_path.write_text(str(error), encoding="utf-8")
        return {
            "schemaVersion": 1,
            "status": "failed",
            "searches": [],
            "errors": [{"code": "CLI_UNAVAILABLE", "message": str(error)}],
        }

    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "schemaVersion": 1,
            "status": "failed",
            "searches": [],
            "errors": [{"code": "INVALID_OUTPUT", "message": completed.stderr.strip() or "CLI returned no valid JSON"}],
        }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def prepare_financial_forecast_notification_messages(
    today: dt.date,
    base_dir: Path,
    timeout_seconds: int,
    cli_timeout_ms: int,
) -> list[str]:
    report = _run_financial_forecast_cli(
        base_dir=base_dir,
        timeout_seconds=timeout_seconds,
        cli_timeout_ms=cli_timeout_ms,
    )

    overview_messages: list[str] = []
    searches = report.get("searches") if isinstance(report, dict) else []
    if isinstance(searches, list):
        for entry in searches:
            if not isinstance(entry, dict):
                continue
            if entry.get("status") != "success" or entry.get("aiOverviewStatus") != "available":
                continue
            overview = entry.get("aiOverview")
            if not isinstance(overview, dict):
                continue
            text = str(overview.get("text") or "").strip()
            if not text:
                continue
            category = str(entry.get("category") or "Financial forecast").strip()
            overview_messages.append(truncate_html_message_block(f"Forecast AI Overview: {category}", text))

    if overview_messages:
        return overview_messages

    status = str(report.get("status") or "failed").strip()
    error_text = extract_forecast_report_error_text(report)
    message = html.escape(f"Financial forecast AI Overview unavailable on {today.isoformat()}: {status} - {error_text}")
    return [f"<b>{message}</b>"]


build_fear_greed_message = build_fear_greed_html_message
extract_report_error = extract_forecast_report_error_text
prepare_fear_greed_notification = prepare_fear_greed_notification_message
prepare_financial_forecast_notifications = prepare_financial_forecast_notification_messages
truncate_html_block = truncate_html_message_block
