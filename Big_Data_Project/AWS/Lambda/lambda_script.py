import boto3
import datetime

s3 = boto3.client('s3')

def lambda_handler(event, context):

    source_bucket = event["source_bucket"]
    source_key = event["source_key"]
    use_event_time = event.get("use_event_time", True)

    if use_event_time:
        dt = datetime.datetime.now(datetime.timezone.utc)
    else:
        timestamp_str = event.get("upload_time")
        if not timestamp_str:
            raise ValueError("upload_time is required when use_event_time is False")
        dt = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

    year = dt.year
    month = str(dt.month).zfill(2)

    full_filename = source_key.split("/")[-1]
    filename = full_filename.split(".")[0]
    if filename.startswith("Your_condination"):
        target_key = f"processed/{filename}/year={year}/month={month}/{full_filename}"
    else:
        print("Filename does not start with 'Your_condination', skipping copy.")
        return {
            'statusCode': 200,
            'message': "Filename does not start with 'Your_condination', skipping copy."
        }
    
    target_bucket = source_bucket

    print(f"Copying {source_key} → {target_key}")
    copy_source = {'Bucket': source_bucket, 'Key': source_key}
    s3.copy_object(CopySource=copy_source, Bucket=target_bucket, Key=target_key)

    return {
        'statusCode': 200,
        'message': f"Copied to {target_key}",
        'partition_path': f"year={year}/month={month}/"
    }