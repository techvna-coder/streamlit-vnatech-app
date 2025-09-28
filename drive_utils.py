# drive_utils.py
import io, json, os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def get_drive(creds_dict):
    gauth = GoogleAuth()
    gauth.ServiceAuth(creds_dict)  # truyền dict trực tiếp
    drive = GoogleDrive(gauth)
    return drive

def list_files_in_folder(drive: GoogleDrive, folder_id: str):
    # Chỉ lấy PDF, PPTX
    q = f"'{folder_id}' in parents and (mimeType='application/pdf' or mimeType='application/vnd.openxmlformats-officedocument.presentationml.presentation') and trashed=false"
    files = []
    page_token = None
    while True:
        file_list = drive.ListFile({'q': q, 'maxResults': 1000, 'pageToken': page_token}).GetList()
        for f in file_list:
            files.append({
                "id": f["id"],
                "title": f["title"],
                "md5Checksum": f.get("md5Checksum"),  # có với upload qua API
                "modifiedDate": f["modifiedDate"],
                "mimeType": f["mimeType"],
            })
        page_token = None
        if len(file_list) < 1000:
            break
    return files

def download_file(drive: GoogleDrive, file_id: str, local_path: str):
    f = drive.CreateFile({'id': file_id})
    f.GetContentFile(local_path)
    return local_path
