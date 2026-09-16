import mimetypes
from pathlib import Path

import requests

from .daily_job_constants import TELEGRAM_MSG_MAX_CHARS, WEBHOOK_URL


def split_html_message_for_telegram(message: str, max_chars: int = TELEGRAM_MSG_MAX_CHARS) -> list[str]:
    if len(message) <= max_chars:
        return [message]

    parts = message.split("\n\n")
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else (current + "\n\n" + part)
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(part) > max_chars:
            chunks.append(part[:max_chars])
            part = part[max_chars:]
        current = part
    if current:
        chunks.append(current)
    return chunks


def send_webhook_html_message(message: str, timeout: int = 20) -> None:
    response = requests.post(WEBHOOK_URL, json={"msg": message}, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:500] if response.text else ""
        raise requests.HTTPError(f"Webhook message failed: {exc}. Response: {body}") from exc


def send_webhook_html_message_batch(messages: list[str], timeout: int = 20, split_messages: bool = True) -> None:
    for message in messages:
        if not message:
            continue
        chunks = split_html_message_for_telegram(message) if split_messages else [message]
        for chunk in chunks:
            send_webhook_html_message(chunk, timeout=timeout)


def send_webhook_file_attachment(file_path: Path, timeout: int = 40) -> None:
    with file_path.open("rb") as handle:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        files = {"files": (file_path.name, handle, mime_type)}
        response = requests.post(WEBHOOK_URL, files=files, timeout=timeout)
        response.raise_for_status()


post_file = send_webhook_file_attachment
post_webhook_message = send_webhook_html_message
post_webhook_messages = send_webhook_html_message_batch
split_message_chunks = split_html_message_for_telegram
