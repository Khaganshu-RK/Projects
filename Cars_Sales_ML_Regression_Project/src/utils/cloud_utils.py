import boto3

def s3_upload(bucket_name: str, file_path: str, object_name: str):
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, bucket_name, object_name)

def s3_download(bucket_name: str, object_name: str, file_path: str):
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, object_name, file_path)

