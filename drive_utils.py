# drive_utils.py
from __future__ import annotations

import json
from typing import Any, Dict, List
from collections.abc import Mapping

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def _normalize_gsa(raw: Any) -> Dict[str, Any]:
    """Chuẩn hoá GOOGLE_SERVICE_ACCOUNT_JSON về dict (hỗ trợ TOML object hoặc JSON string)."""
    if isinstance(raw, Mapping):
        return dict(raw)             # TOML object -> dict
    if isinstance(raw, str):
        return json.loads(raw)       # JSON string -> dict
    raise TypeError(f"GOOGLE_SERVICE_ACCOUNT_JSON kiểu không hỗ trợ: {type(raw).__name__}")


def get_drive_from_secrets(raw_gsa: Any) -> GoogleDrive:
    """
    Tạo GoogleDrive client từ secrets.
    Dùng ServiceAuth(keyfile_dict=...) + scopes để tránh mọi nhánh parse JSON nội bộ.
    """
    client_json = _normalize_gsa(raw_gsa)

    # Scopes chỉ cần quyền đọc
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]

    # Không cấu hình service_config nữa; truyền dict trực tiếp cho ServiceAuth
    gauth = GoogleAuth(settings={"save_credentials": False})
    gauth.ServiceAuth(keyfile_dict=client_json, scopes=scopes)

    return GoogleDrive(gauth)


def list_files_in_folder(
    drive: GoogleDrive,
    folder_id: str,
    mime_filters: List[str] | None = None,
    include_shared: bool = True,
) -> List[Dict[str, Any]]:
    """Liệt kê file trong folder Drive (mặc định PDF + PPTX)."""
    if mime_filters is None:
        mime_filters = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ]
    mime_q = " or ".join([f"mimeType='{m}'" for m in mime_filters])
    q = f"'{folder_id}' in parents and trashed=false and ({mime_q})"

    params = {"q": q}
    if include_shared:
        params["supportsAllDrives"] = True
        params["includeItemsFromAllDrives"] = True

    items = drive.ListFile(params).GetList()
    return [
        {
            "id": f["id"],
            "title": f["title"],
            "mimeType": f.get("mimeType"),
            "modifiedDate": f.get("modifiedDate"),
            "md5Checksum": f.get("md5Checksum"),
        }
        for f in items
    ]


def download_file(drive: GoogleDrive, file_id: str, local_path: str) -> str:
    """Tải file từ Drive về local_path."""
    f = drive.CreateFile({"id": file_id})
    f.GetContentFile(local_path)
    return local_path
