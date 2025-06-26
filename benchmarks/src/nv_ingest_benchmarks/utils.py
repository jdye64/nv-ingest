import subprocess
import requests
import json
import copy
import socket
import os
import zipfile
from datetime import datetime
import inspect
import docker
import inspect

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import pypdfium2 as pdfium
import glob, time
from pymilvus import MilvusClient, Collection


def segment_results(results):
    text_results = [[element for element in results if element["document_type"] == "text"] for results in results]
    table_results = [
        [element for element in results if element["metadata"]["content_metadata"]["subtype"] == "table"]
        for results in results
    ]
    chart_results = [
        [element for element in results if element["metadata"]["content_metadata"]["subtype"] == "chart"]
        for results in results
    ]
    return text_results, table_results, chart_results


def unload_collection(collection_name):
    client = MilvusClient(uri="http://localhost:19530")
    client.release_collection(collection_name=collection_name)


def load_collection(collection_name):
    client = MilvusClient(uri="http://localhost:19530")
    client.load_collection(collection_name)


def milvus_chunks(collection_name):
    client = MilvusClient(
        uri="http://localhost:19530",
    )
    stats = client.get_collection_stats(collection_name)
    log(f"{collection_name}_chunks", f"{stats}")


def save_extracts(results):
    parent_fn = get_parent_script_filename()
    dt_hour = date_hour()
    t0 = time.time()
    with open(f"/datasets/nv-ingest/extracts/{parent_fn}_{dt_hour}.json", "w") as fp:
        fp.write(json.dumps(results))
    t1 = time.time()
    log("disk_write_time", t1 - t0)


def get_parent_script_filename():
    stack = inspect.stack()
    caller_frame = stack[-1]
    caller_filename = caller_frame.filename
    return os.path.basename(caller_filename)


def pdf_page_count_glob(pattern):
    total_pages = 0
    for filepath in glob.glob(pattern, recursive=True):
        if filepath.lower().endswith(".pdf"):
            pdf = pdfium.PdfDocument(filepath)
            total_pages += len(pdf)
    return total_pages


def pdf_page_count(directory):
    total_pages = 0
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            try:
                pdf = pdfium.PdfDocument(filepath)
                total_pages += len(pdf)
            except Exception as e:
                print(f"{filepath} failed: {e}")
                continue
    return total_pages


def date_hour():
    now = datetime.now()
    return now.strftime("%m-%d-%y_%I")


def check_container_running(container_name):
    client = docker.from_env()
    containers = client.containers.list()

    for container in containers:
        if container_name in container.image.tags[0]:
            return True

    return False


def embed_info():
    if check_container_running("llama-3.2-nemoretriever-1b-vlm-embed-v1"):
        return "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1", 2048
    elif check_container_running("llama-3.2-nv-embedqa-1b-v2"):
        return "nvidia/llama-3.2-nv-embedqa-1b-v2", 2048
    else:
        return "nvidia/nv-embedqa-e5-v5", 1024


logs = []


def log(key, val):
    print(f"{date_hour()}: {key}: {val}")
    caller_fn = get_parent_script_filename()

    log_fn = "test_results/" + caller_fn.split(".")[0] + ".json"
    # print(os.getcwd())
    # print(log_fn)
    logs.append({key: val})
    with open(log_fn, "w") as fp:
        fp.write(json.dumps(logs))


# Runs a system command and returns the response code
# stderr and stdout are written the the configured location
def run_system_command(command: list, print_output=False):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)

    return_code = -1
    all_output = ""
    while True:
        output = process.stdout.readline()
        if print_output:
            print(output.strip())
        all_output = all_output + output.strip()
        return_code = process.poll()
        if return_code is not None:
            for output in process.stdout.readlines():
                print(output.strip())
                all_output = all_output + output.strip()
            break

    return return_code, all_output


slack_webhook_url = "https://hooks.slack.com/services/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"
post_template = {
    "blocks": [
        {"type": "header", "text": {"type": "plain_text", "text": "REPORT_NAME"}},
        {"type": "section", "text": {"type": "plain_text", "text": "SUMMARY"}},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "Google Drive Link"},
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "emoji": True, "text": "Open"},
                "url": "DRIVE_LINK",
            },
        },
        {"type": "context", "elements": [{"type": "mrkdwn", "text": "PIC_ID"}]},
    ]
}

pic_dict = {
    "randy": "U9TBF6WT0",
    "dante": "UAXKYQR6V",
    "brad": "U7K466FGC",
    "rick": "UFACDNN5U",
    "greg": "U02MDG8B15H",
    "vibhu": "UGFRT88RE",
    "ayush": "UBDSVEQPL",
    "chris": "U01EUUZKG1Y",
}


def slack_post(report_name, summary, pic, link):
    message = copy.deepcopy(post_template)
    message["blocks"][0]["text"]["text"] = report_name
    message["blocks"][1]["text"]["text"] = summary
    message["blocks"][2]["accessory"]["url"] = link
    if pic in pic_dict:
        pic_id = pic_dict[pic]
        message["blocks"][3]["elements"][0]["text"] = f"cc <@{pic_id}>"
    else:
        message["blocks"][3]["elements"][0]["text"] = ""
    print(message)
    print(json.dumps(message))
    resp = requests.post(slack_webhook_url, data=json.dumps(message), headers={"Content-Type": "application/json"})
    print(resp)
    print(resp.text)


def ntfy(topic, message):
    requests.post(f"https://ntfy.sh/{topic}", data=message.encode(encoding="utf-8"))


def time_delta(fn):
    file_ctime = os.path.getctime(fn)
    file_creation_time = datetime.fromtimestamp(file_ctime)
    current_time = datetime.now()
    time_difference = current_time - file_creation_time
    minutes = int(time_difference.total_seconds() / 60)
    return minutes


def timestr_fn(test_type):
    return datetime.now().strftime("%Y-%m-%d_%H") + "_" + test_type + "_" + socket.gethostname()


def filter_versions(fn, libs):
    versions = open(fn).read().split("\n")[:-1]
    versions = set([x for x in versions if any(lib in x.split()[0] for lib in libs)])
    return "\n".join(versions)


def gdrive(service_account_json_file, drive_folder_id, fn):
    gauth = GoogleAuth()
    scope = ["https://www.googleapis.com/auth/drive"]
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(service_account_json_file, scope)
    drive = GoogleDrive(gauth)

    drive_file = drive.CreateFile({"parents": [{"id": drive_folder_id}], "title": fn})
    drive_file.SetContentFile(fn)
    drive_file.Upload()
    return drive_file["alternateLink"]


def zip(path, fn):
    # create a new ZipFile object in write mode
    zip_obj = zipfile.ZipFile(fn, "w")

    # loop through all the files in the directory and add them to the ZIP file
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            zip_obj.write(file_path)

    # close the ZIP file
    zip_obj.close()


def failed_passed(raw_data_dir, fn):
    pytest_results = open(raw_data_dir + "/" + fn).read().split("PASSED\n")
    failed_passed = [x for x in pytest_results if "Compute Failed" in x or "distributed.worker - ERROR" in x]
    with open(raw_data_dir + "/failed_passed.txt", "w") as fp:
        fp.write("\n".join(failed_passed))
    return failed_passed


def failure_summary(raw_data_dir, fn):
    test_results = open(raw_data_dir + "/" + fn).read().split("PASSED\n")
    failure_summary = test_results[-1].split(
        "=========================== short test summary info ============================\n"
    )[1]
    with open(raw_data_dir + "/summary.txt", "w") as fp:
        fp.write(failure_summary)
    return failure_summary


def get_gpu_name():
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # return pynvml.nvmlDeviceGetName(handle).decode("utf-8")
    return pynvml.nvmlDeviceGetName(handle)


def get_num_workers(client):
    return len(client.scheduler_info()["workers"])


def summary(text, packages):
    versions = filter_versions("/rapids/raw_data/versions.txt", packages)
    report_dict = {
        "Summary": text,
        "Versions": versions,
        "Host": socket.gethostname(),
        "Hardware": get_gpu_name(),
    }

    summary = [f"{key}:\n{value}\n\n" for key, value in report_dict.items()]
    return "".join(summary)


def get_associated_pr(commit_sha, OWNER, REPO):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/commits/{commit_sha}/pulls"
    TOKEN = open("github_token.txt").read().strip()

    headers = {
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        prs = response.json()
        if prs:
            return str(prs[0]["number"])  # Return the first associated PR number
    return "N/A"
