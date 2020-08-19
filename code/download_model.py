import os
from google_drive_downloader import GoogleDriveDownloader as gdd

model_url = "1pC9OGFrsV4l_JCUT5Mn6eZgZXsDxFL8a"
def download_model(model_url=model_url, model_dir="../model"):
    model_name = "finetuned_token_cls_model"
    if not os.path.exists(model_dir):
    	os.mkdir(model_dir)
    path_name = "/".join([model_dir, model_name])
    gdd.download_file_from_google_drive(file_id=model_url,
                                    dest_path=path_name)

if __name__ == '__main__':
	print(' == Download model == ')
	download_model()
	print(' == Done == ')
