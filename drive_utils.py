# drive_utils.py
# Tiện ích kết nối Google Drive bằng PyDrive2 + Service Account
# Tương thích secrets dạng TOML object hoặc JSON string.

from __future__ import annotations

import json
from typing import Any, Dict, List
from collections.abc import Mapping

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# -------------------------
# Chuẩn hoá secrets
# -------------------------
def _normalize_gsa(raw: Any) -> Dict[str, Any]:
    """
    Trả về dict 'client_json' từ giá trị GOOGLE_SERVICE_ACCOUNT_JSON:
    - raw là Mapping (TOML object) -> dict(raw)
    - raw là string JSON ('''...''' hoặc \"\"\"...\"\"\") -> json.loads(raw)
    """
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        # Lưu ý: trong secrets, private_key phải có \\n (không phải newline thật)
        return json.loads(raw)
    raise TypeError(f"GOOGLE_SERVICE_ACCOUNT_JSON có kiểu không hỗ trợ: {type(raw).__name__}")


# -------------------------
# Khởi tạo GoogleDrive
# -------------------------
def get_drive_from_secrets(raw_gsa: Any) -> GoogleDrive:
    """
    Tạo GoogleDrive client từ giá trị GOOGLE_SERVICE_ACCOUNT_JSON lấy trực tiếp
    từ st.secrets (string JSON hoặc TOML object).
    Không truyền đối số vào ServiceAuth để tránh double-parse.
    """
    client_json = _normalize_gsa(raw_gsa)

    gauth = GoogleAuth(settings={
        "client_config_backend": "service",
        "service_config": {"client_json": client_json},
        "save_credentials": False,
    })
    # Quan trọng: KHÔNG truyền tham số ở đây
    gauth.ServiceAuth()
    return GoogleDrive(gauth)


# -------------------------
# Tác vụ với file/folder
# -------------------------
def list_files_in_folder(
    drive: GoogleDrive,
    folder_id: str,
    mime_filters: List[str] | None = None,
    include_shared: bool = True,
) -> List[Dict[str, Any]]:
    """
    Liệt kê file trong một folder Drive.
    - mime_filters: danh sách MIME type cần lọc (mặc định PDF + PPTX)
    - include_shared: bật hỗ trợ Shared Drives
    """
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
    """
    Tải một file từ Drive về local_path. Trả về local_path khi xong.
    """
    f = drive.CreateFile({"id": file_id})
    f.GetContentFile(local_path)
    return local_path
