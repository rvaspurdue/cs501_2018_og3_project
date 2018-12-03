from __future__ import print_function
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import io,os
from apiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/drive.readonly'

def sync():
    
    path = os.getcwd()
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    store = file.Storage('token.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('drive', 'v3', http=creds.authorize(Http()))

    # Call the Drive v3 API
    results = service.files().list(
        pageSize=20, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

#    if not items:
#        print('No files found.')
#    else:
#        print('Files:')
#        for item in items:
#            print(u'{0} ({1})'.format(item['name'], item['id']))
    
    for item in items:            
        request = service.files().get_media(fileId=item['id'])
        fh = io.FileIO(os.path.join(path,'GoogleDriveContents',item['name']),'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print ("Download %d%%." % int(status.progress() * 100))