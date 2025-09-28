# drive_utils.py
# Tiện ích Google Drive cho VNA Tech – dùng PyDrive2 + Service Account.

from __future__ import annotations

import json
from typing import Any, Dict, List
from collections.abc import Mapping

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# =========================
# Chuẩn hoá secrets
# =========================
def _normalize_gsa(raw: Any) -> Dict[str, Any]:
    """
    Chuẩn hoá giá trị GOOGLE_SERVICE_ACCOUNT_JSON thành dict:
      - Nếu raw là Mapping (TOML object) -> trả về dict(raw)
      - Nếu raw là string JSON (triple quotes '''...''' hoặc """...""") -> json.loads
    """
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        # raw phải là JSON hợp lệ; lưu ý private_key cần \\n chứ không phải newline thật
        return json.loads(raw)
    raise TypeError(f"GOOGLE_SERVICE_ACCOUNT_JSON có kiểu không hỗ trợ: {type(raw).__name__}")


# =========================
# Khởi tạo Drive client
# =========================
def get_drive(creds_dict_or_raw: Any) -> GoogleDrive:
    """
    Tạo GoogleDrive client từ:
      - dict 'client_json' đã chuẩn hoá, hoặc
      - raw secrets (string JSON hoặc TOML object)
    """
    creds = _normalize_gsa(creds_dict_or_raw)

    gauth = GoogleAuth(settings={
        "client_config_backend": "service",
        "service_config": {"client_json": creds},
        "save_credentials": False,  # không lưu token ra file
    })
    # Không truyền tham số vào ServiceAuth -> PyDrive2 sẽ dùng settings['service_config']['client_json']
    gauth.ServiceAuth()
    return GoogleDrive(gauth)


# =========================
# Tác vụ với file/folder
# =========================
def list_files_in_folder(
    drive: GoogleDrive,
    folder_id: str,
    mime_filters: List[str] | None = None,
    include_shared: bool = True,
) -> List[Dict[str, Any]]:
    """
    Liệt kê file trong một folder.
    - mime_filters: danh sách MIME types cần lọc; mặc định PDF + PPTX.
    - include_shared: True để hỗ trợ Shared Drives.
    """
    if mime_filters is None:
        mime_filters = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ]

    # Xây dựng query MIME
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
    """
    Tải nội dung một file từ Drive về đường dẫn local_path.
    Trả về local_path khi thành công.
    """
    f = drive.CreateFile({"id": file_id})
    f.GetContentFile(local_path)
    return local_path
