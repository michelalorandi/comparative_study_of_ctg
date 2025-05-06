import os
import time

from google.oauth2 import service_account
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/documents'
]


def init_service(access_with_iam=False,
                 service_account_file=os.path.join('', 'google_key.json'),
                 token_path=os.path.join('', 'token.json'),
                 creds_path=os.path.join('', 'client_secret.json')):
    """

    :param access_with_iam:
    :param service_account_file:
    :param token_path:
    :param creds_path:
    :return:
    """
    creds = None
    if access_with_iam:
        creds = service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    else:
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            if creds.expired:
                if creds.refresh_token:
                    creds.refresh(Request())
                else:
                    creds = None

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

    global service_sheet
    service_sheet = build('sheets', 'v4', credentials=creds)
    global service_drive
    service_drive = build('drive', 'v3', credentials=creds)
    global service_docs
    service_docs = build('docs', 'v1', credentials=creds)


def modify_sheet(cell_range_insert, values, spreadsheet_id, worksheet_name):
    """

    :param cell_range_insert: range of cell in format 'A1:B1'
    :param values:
    :param spreadsheet_id:
    :param worksheet_name:
    :return:
    """
    value_range_body = {
        'majorDimension': 'ROWS',
        'values': values
    }
    while True:
        try:
            service_sheet.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                valueInputOption='USER_ENTERED',
                range=worksheet_name + '!' + cell_range_insert,
                body=value_range_body).execute()
            return
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def add_conditional_formatting(sheet_id: int, spreadsheet_id: str, color: dict, start_row: int = 1, end_row: int = 31,
                               start_col: int = 1, end_col: int = 2, index_rule: int = 0, conditions: list = None,
                               condition_type: str = 'CUSTOM_FORMULA', foreground_color:dict=None):
    """

    :param sheet_id:
    :param spreadsheet_id:
    :param start_row:
    :param end_row:
    :param start_col:
    :param end_col:
    :param index_rule:
    :param condition:
    :param color:
    :param condition_type:
    :return:
    """
    requests = [{
        'addConditionalFormatRule': {
            'rule': {
                'ranges': [
                    {
                        'sheetId': sheet_id,
                        'startRowIndex': start_row,
                        "endRowIndex": end_row,
                        "startColumnIndex": start_col,
                        "endColumnIndex": end_col
                    }
                ],
                'booleanRule': {
                    'condition': {
                        'type': condition_type
                    },
                    'format': {
                        'backgroundColor': {
                            'red': color['red']/255,
                            'green': color['green']/255,
                            'blue': color['blue']/255,
                            'alpha': color['alpha']
                        }
                    }
                }
            },
            'index': index_rule
        }
    }]
    if foreground_color is not None:
        requests[0]['addConditionalFormatRule']['rule']['booleanRule']['format']["textFormat"] = {
            "foregroundColor": {
                            'red': foreground_color['red']/255,
                            'green': foreground_color['green']/255,
                            'blue': foreground_color['blue']/255,
                            'alpha': foreground_color['alpha']
                        }
        }
    if conditions is not None:
        requests[0]['addConditionalFormatRule']['rule']['booleanRule']['condition']['values'] = []
        for cond in conditions:
            requests[0]['addConditionalFormatRule']['rule']['booleanRule']['condition']['values'].append({"userEnteredValue": cond})

    body = {
        'requests': requests
    }
    while True:
        try:
            response = service_sheet.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=body).execute()
            return response
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def add_borders_in_sheet(sheet_id, spreadsheet_id, color, start_row=1, end_row=31, start_col=1, end_col=2,
                         directions=['bottom'], style='SOLID'):
    """

    :param sheet_id:
    :param spreadsheet_id:
    :param start_row:
    :param end_row:
    :param start_col:
    :param end_col:
    :param directions:
    :param style:
    :param color:
    :return:
    """
    requests = [{
        'updateBorders': {
            'range': {
                'sheetId': sheet_id,
                'startRowIndex': start_row,
                "endRowIndex": end_row,
                "startColumnIndex": start_col,
                "endColumnIndex": end_col
            },
        }
    }]
    for dir in directions:
        requests[0]['updateBorders'][dir] = {
            'style': style,
            'color': {
                'red': color['red'],
                'green': color['green'],
                'blue': color['blue'],
                'alpha': color['alpha']
            }
        }
    body = {
        'requests': requests
    }
    while True:
        try:
            response = service_sheet.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=body).execute()
            return response
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def add_sheet(title: str, spreadsheet_id: str):
    """
    Add new sheet
    :param title: name of the sheet
    :param spreadsheet_id: id of the spreadsheet
    :return: sheet id
    """
    add_sheet_to_spreadsheet_request_body = {
        'properties': {
            'title': title
        }
    }
    body = {
        'requests': {
            'addSheet': add_sheet_to_spreadsheet_request_body
        }
    }
    while True:
        try:
            request = service_sheet.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body)
            response = request.execute()
            return response['replies'][0]['addSheet']['properties']['sheetId']
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)



def copy_sheet(destination_spreadsheet_id: int, origin_spreadsheet_id: str, origin_sheet_id: int = 0):
    """

    :param destination_spreadsheet_id:
    :param origin_spreadsheet_id:
    :param origin_sheet_id:
    :return:
    """
    copy_sheet_to_another_spreadsheet_request_body = {
        # The ID of the spreadsheet to copy the sheet to.
        'destination_spreadsheet_id': destination_spreadsheet_id
    }

    request = service_sheet.spreadsheets().sheets().copyTo(spreadsheetId=origin_spreadsheet_id, sheetId=origin_sheet_id,
                                                           body=copy_sheet_to_another_spreadsheet_request_body)
    while True:
        try:
            response = request.execute()
            return response['sheetId']
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def create_spreadsheet(title):
    """
    Create a new spreadsheet with the specified title.
    :param title: Name of the file.
    :return: id of the created spreadsheet.
    """
    spreadsheet = {
        'properties': {
            'title': title
        }
    }
    while True:
        try:
            spreadsheet = service_sheet.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
            return spreadsheet.get('spreadsheetId')
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def create_spreadsheet_in_folder(name, parents):
    """

    :param name:
    :param parents:
    :return:
    """
    if parents is None:
        print("ERROR - Parent directory is None")
        return None
    file_metadata = {
        'name': name,
        'parents': parents,
        'mimeType': 'application/vnd.google-apps.spreadsheet',
    }
    while True:
        try:
            res = service_drive.files().create(body=file_metadata).execute()
            return res['id']
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def search_file_by_name(filename, parent=None):
    """

    :param filename:
    :param parent:
    :return:
    """
    page_token = None
    while True:
        try:
            query = f"name = '{filename}' and mimeType != 'application/vnd.google-apps.folder'"
            if parent is not None:
                query += f" and '{parent}' in parents"
            response = service_drive.files().list(q=query,
                                                  spaces='drive',
                                                  fields='nextPageToken, files(id, name)',
                                                  pageToken=page_token).execute()
            for file in response.get('files', []):
                # Process change
                print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
                return file.get('name'), file.get('id')
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                return None, None
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def rename_sheet(spreadsheet_id, sheet_id, name):
    """

    :param spreadsheet_id:
    :param sheet_id:
    :param name:
    :return:
    """
    # Change the spreadsheet's title.
    requests = [{
        'updateSheetProperties': {
            'properties': {
                'title': name,
                'sheetId': sheet_id
            },
            'fields': 'title'
        }
    }]
    body = {
        'requests': requests
    }
    while True:
        try:
            response = service_sheet.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=body).execute()
            return response
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def create_folder(folder_name, parents:list=None):
    """

    :param folder_name:
    :param parents:
    :return:
    """
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
    }
    if parents is not None:
        file_metadata['parents'] = parents
    while True:
        try:
            file = service_drive.files().create(body=file_metadata,
                                                fields='id').execute()
            print('Folder ID: %s' % file.get('id'))
            return file.get('id')
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def search_folder_by_name(foldername: str, parent: str=None):
    """
    Search a folder by name in Drive
    :param foldername: name of the folder
    :param parent: optional parent id to restrict search
    :return: (name, id) name and id of the found folder otherwise (None, None) if not found
    """
    page_token = None
    while True:
        query = f"name = '{foldername}' and mimeType = 'application/vnd.google-apps.folder'"
        if parent is not None:
            query += f" and '{parent}' in parents"
        response = service_drive.files().list(q=query,
                                              spaces='drive',
                                              fields='nextPageToken, files(id, name)',
                                              pageToken=page_token).execute()
        for file in response.get('files', []):
            # Process change
            print('Found folder: %s (%s)' % (file.get('name'), file.get('id')))
            return file.get('name'), file.get('id')
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            return None, None


def get_spreadsheet(spreadsheet_id, ranges=[], include_grid_data=False):
    """

    :param spreadsheet_id:
    :param ranges: The ranges to retrieve from the spreadsheet.
    :param include_grid_data: True if grid data should be returned. This parameter is ignored if a field mask was set in the request.
    :return: response obtained
    """
    while True:
        try:
            request = service_sheet.spreadsheets().get(spreadsheetId=spreadsheet_id, ranges=ranges,
                                                       includeGridData=include_grid_data)
            response = request.execute()
            return response
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def read_sheet(spreadsheet_id: str, sheet_name: str):
    """

    :param spreadsheet_id:
    :param sheet_name:
    :return:
    """
    if service_sheet is None:
        init_service()
    while True:
        try:
            result = service_sheet.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=sheet_name).execute()
            return result.get('values', [])
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def check_sheet_in_spreadsheet(sheet_name, spreadsheet_id):
    """

    :param sheet_name:
    :param spreadsheet_id:
    :return:
    """
    spreadsheet = get_spreadsheet(spreadsheet_id)
    for sheet in spreadsheet['sheets']:
        if sheet['properties']['title'] == sheet_name:
            return True, sheet['properties']['sheetId']
    return False, None


def get_spreadsheet_id(gdoc_filename: str, folder_id: str):
    """
    Searches for the specified spreadsheet and create a new one if it doesn't exist
    :param gdoc_filename: name of the file
    :param folder_id: id of the folder containing the spreadsheet
    :return: id of the spreadsheet
    """
    _, gdoc_id = search_file_by_name(gdoc_filename, folder_id)
    if gdoc_id is None:
        gdoc_id = create_spreadsheet_in_folder(gdoc_filename, [folder_id])
    return gdoc_id


def delete_sheet_with_id(spreadsheet_id, sheet_id):
    """

    :param spreadsheet_id:
    :param sheet_id:
    :return:
    """
    requests = [{
        'deleteSheet': {
            'sheetId': sheet_id
        }
    }]
    body = {
        'requests': requests
    }
    while True:
        try:
            response = service_sheet.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=body).execute()
            return response
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def set_background_color(spreadsheet_id: str, sheet_id: int, start_row_index: int, end_row_index: int,
                         start_column_index: int, end_column_index: int, background_color: dict):
    """
    Set background color of specified cells.
    :param spreadsheet_id: id of the spreadsheet.
    :param sheet_id: id of the sheet
    :param start_row_index:
    :param end_row_index:
    :param start_column_index:
    :param end_column_index:
    :param background_color:
    :return:
    """
    body = {
        "requests": [
            {
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": start_row_index,
                        "endRowIndex": end_row_index,
                        "startColumnIndex": start_column_index,
                        "endColumnIndex": end_column_index
                    },
                    "rows": [
                        {
                            "values": [
                                {
                                    "userEnteredFormat": {
                                        "backgroundColor": {
                                            "red": background_color['red']/255.0,
                                            "green": background_color['green']/255.0,
                                            "blue": background_color['blue']/255.0
                                        }
                                    }
                                }
                            ]
                        }
                    ],
                    "fields": "userEnteredFormat.backgroundColor"
                }
            }
        ]
    }
    while True:
        try:
            res = service_sheet.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
            return res
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)


def set_dropdown_cells(spreadsheet_id:int, sheet_id:int, start_row_index:int, end_row_index:int, start_column_index:int,
                       end_column_index:int, values:list):
    formatted_values = []
    for val in values:
        formatted_values.append({
            "userEnteredValue": str(val)
        })
    body = {
        "requests": [
            {
                "setDataValidation": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": start_row_index,
                        "endRowIndex": end_row_index,
                        "startColumnIndex": start_column_index,
                        "endColumnIndex": end_column_index
                    },
                    "rule": {
                        'showCustomUi': True,
                        'strict': True,
                        'condition': {
                            "type": 'ONE_OF_LIST',
                            "values": formatted_values
                        }
                    }
                }
            }
        ]
    }
    while True:
        try:
            res = service_sheet.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
            return res
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)



def copy_file_in_folder(file_id: int, name: str, parents: list):
    file_metadata = {
        'name': name,
        'parents': parents
    }
    while True:
        try:
            res = service_drive.files().copy(fileId=file_id, body=file_metadata).execute()
            return res['id']
        except Exception as e:
            print(e)
            print('Retrying in 30 seconds')
            time.sleep(30)

