from apiclient import discovery
from httplib2 import Http
from oauth2client import file
import io
from googleapiclient.http import MediaIoBaseDownload


def download(new_file_name, file_id):
    """
    A function that accesses and downloads files from Google Drive.

    Parameters
    ----------
    new_file_name: string
        The file name you want a google doc to have after download.

    file_if: string
        The file id of your desired file on google drive.

    Returns
    -------
    N/A:
        The function downloads a file onto the actual computer
        under new_file_name

    """
    SCOPES = 'https://www.googleapis.com/auth/drive.readonly'  # noqa: F841
    store = file.Storage('token.json')
    creds = store.get()
    DRIVE = discovery.build('drive', 'v3', http=creds.authorize(Http()))
    # if you get the shareable link, the link contains this id,
    # replace the file_id below
    request = DRIVE.files().get_media(fileId=file_id)
    # replace the filename and extension in the first field below
    fh = io.FileIO(new_file_name, mode='w')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    return
