import os
import subprocess
import requests

# GitHub repository URL
repo_url = "https://github.com/dreamento/dreamento"

# Local directory to clone/update the repository
local_repo_path = "."

def get_latest_remote_commit_hash(repo_url):
    try:
        api_url = f"https://api.github.com/repos/{repo_url.split('/')[-2]}/{repo_url.split('/')[-1]}/commits/main"
        response = requests.get(api_url)
        response_json = response.json()
        return response_json["sha"]
    except Exception as e:
        print("Error fetching remote commit hash:", e)
        return None

def check_and_update_repository(repo_url, local_path):
    try:
        if os.path.exists(local_path):
            os.chdir(local_path)
            
            subprocess.call(["git", "fetch"])
            
            local_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            remote_commit_hash = get_latest_remote_commit_hash(repo_url)
            
            if local_commit_hash == remote_commit_hash:
                print("Repository is up to date.")
            else:
                print("Updating the repository...")
                subprocess.call(["git", "pull"])
                print("Repository updated.")
        else:
            print("Cloning the repository...")
            subprocess.call(["git", "clone", repo_url, local_path])
            print("Repository cloned.")
            
    except Exception as e:
        print("Error:", e)

def download_missing_files(repo_url, local_path):
    try:
        os.chdir(local_path)
        
        remote_files = get_all_files_in_repo(repo_url)
        local_files = get_all_local_files(local_path)
        
        missing_files = [file for file in remote_files if file not in local_files]
        
        if missing_files:
            print("Downloading missing files...")
            for file in missing_files:
                file_path = os.path.join(local_path, file)
                if not os.path.exists(file_path):  # Check if the file doesn't exist locally
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file_url = f"{repo_url}/raw/main/{file}"
                    response = requests.get(file_url)
                    if response.status_code == 200:
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        print(f"Downloaded: {file}")
        else:
            print("No missing files found.")
            
    except Exception as e:
        print("Error:", e)

def get_all_files_in_repo(repo_url):
    try:
        api_url = f"https://api.github.com/repos/{repo_url.split('/')[-2]}/{repo_url.split('/')[-1]}/git/trees/main?recursive=1"
        response = requests.get(api_url)
        response_json = response.json()
        
        file_paths = [item["path"] for item in response_json["tree"] if item["type"] == "blob"]
        return file_paths
    except Exception as e:
        print("Error fetching remote repository files:", e)
        return []

def get_all_local_files(path):
    local_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), path)
            local_files.append(file_path)
    return local_files

if __name__ == "__main__":
    check_and_update_repository(repo_url, local_repo_path)
    download_missing_files(repo_url, local_repo_path)

    # Keep the command prompt window open
    print("Updating Dreamento is ended. You can follow the results above.")
    input("Press Enter to exit...")
