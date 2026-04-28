"""
save_to_drive.py — Zip run results and upload to Google Drive.

Setup (one-time):
  1. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
  2. Go to https://console.cloud.google.com → New project → Enable "Google Drive API"
  3. Credentials → Create → OAuth 2.0 Client IDs → Desktop app → Download JSON
  4. Save it as credentials.json next to this script

Usage:
  # Upload a specific run (creates lsec_final_run.zip → uploads to Google Drive)
  python save_to_drive.py --run_dir runs/final_run

  # Give the zip a custom name
  python save_to_drive.py --run_dir runs/final_run --name my_experiment

  # Upload to a specific Google Drive folder (use folder ID from the URL)
  python save_to_drive.py --run_dir runs/final_run --folder_id 1AbCdEfGhIjKlMnOpQr

  # Upload everything under runs/
  python save_to_drive.py --run_dir runs
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path


# ─────────────────────────────────────────────────
# Zip
# ─────────────────────────────────────────────────

def zip_directory(src: str, zip_path: str) -> int:
    """Zip src recursively into zip_path. Returns total file count."""
    src = Path(src)
    count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for file in sorted(src.rglob('*')):
            if file.is_file():
                zf.write(file, file.relative_to(src.parent))
                count += 1
    return count


# ─────────────────────────────────────────────────
# Google Drive upload
# ─────────────────────────────────────────────────

SCOPES = ['https://www.googleapis.com/auth/drive.file']
TOKEN_FILE = 'token.json'
CREDENTIALS_FILE = 'credentials.json'


def get_drive_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(
                    f"[ERROR] {CREDENTIALS_FILE} not found.\n"
                    "  1. Go to https://console.cloud.google.com\n"
                    "  2. Enable 'Google Drive API'\n"
                    "  3. Create OAuth 2.0 Desktop credentials\n"
                    f"  4. Download and save as {CREDENTIALS_FILE}"
                )
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            # On RunPod (headless): open the printed URL on your local browser
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())
        print(f"  Auth token saved → {TOKEN_FILE}  (reused next time)")

    return build('drive', 'v3', credentials=creds)


def upload_file(service, local_path: str, remote_name: str, folder_id: str | None) -> str:
    from googleapiclient.http import MediaFileUpload

    file_meta = {'name': remote_name}
    if folder_id:
        file_meta['parents'] = [folder_id]

    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  Uploading {remote_name}  ({size_mb:.1f} MB) ...")

    media = MediaFileUpload(local_path, mimetype='application/zip', resumable=True)
    uploaded = service.files().create(
        body=file_meta,
        media_body=media,
        fields='id,name,size',
    ).execute()

    file_id = uploaded['id']
    print(f"  Done. File ID: {file_id}")
    print(f"  Link: https://drive.google.com/file/d/{file_id}/view")
    return file_id


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.exists(args.run_dir):
        print(f"[ERROR] run_dir not found: {args.run_dir}")
        sys.exit(1)

    # ── determine zip name ───────────────────────
    run_name = args.name or Path(args.run_dir).name
    zip_path = f"{run_name}.zip"

    # ── zip ─────────────────────────────────────
    print(f"\nZipping  {args.run_dir}  →  {zip_path}")
    count = zip_directory(args.run_dir, zip_path)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  {count} files  |  {size_mb:.1f} MB")

    if args.zip_only:
        print("  --zip_only set, skipping upload.")
        return

    # ── upload ───────────────────────────────────
    try:
        import googleapiclient  # noqa: F401
    except ImportError:
        print(
            "\n[ERROR] Google API client not installed.\n"
            "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
        sys.exit(1)

    print("\nConnecting to Google Drive ...")
    service = get_drive_service()
    upload_file(service, zip_path, os.path.basename(zip_path), args.folder_id)

    if not args.keep_zip:
        os.remove(zip_path)
        print(f"  Local zip removed: {zip_path}")


def parse_args():
    p = argparse.ArgumentParser(description='Zip run results and upload to Google Drive')
    p.add_argument('--run_dir', required=True,
                   help='Run directory to zip and upload (e.g. runs/final_run)')
    p.add_argument('--name', default=None,
                   help='Base name for the zip file (default: directory name)')
    p.add_argument('--folder_id', default=None,
                   help='Google Drive folder ID to upload into (from folder URL)')
    p.add_argument('--zip_only', action='store_true',
                   help='Only create zip, skip upload')
    p.add_argument('--keep_zip', action='store_true',
                   help='Keep local zip after upload (deleted by default)')
    return p.parse_args()


if __name__ == '__main__':
    main()
