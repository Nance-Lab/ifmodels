from apiclient import discovery
from httplib2 import Http
import oauth2client
from oauth2client import file, client, tools
import pandas as pd
import nrrd
import nibabel as nib
import io
from googleapiclient.http import MediaIoBaseDownload

def download(new_file_name, file_id):
    #Downloads files from google drive
    SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
    store = file.Storage('token.json')
    creds = store.get()
    DRIVE = discovery.build('drive', 'v3', http=creds.authorize(Http()))
    # if you get the shareable link, the link contains this id, replace the file_id below
    request = DRIVE.files().get_media(fileId=file_id)
    # replace the filename and extension in the first field below
    fh = io.FileIO(new_file_name, mode='w')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    return
