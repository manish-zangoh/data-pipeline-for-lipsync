import os
import requests
from tqdm import tqdm

API_KEY = "bNN4IDQiaFftE7DwxyBk5zreydQoVkvFrSJm6vBi6DiwIoFAsCJEwYdj"
SEARCH_QUERY = "interview person talking face"
PER_PAGE = 10
TOTAL_VIDEOS = 5

OUTPUT_DIR = "dataset/input_videos"

headers = {
    "Authorization": API_KEY
}


def search_videos(page=1):
    url = "https://api.pexels.com/videos/search"
    params = {
        "query": SEARCH_QUERY,
        "per_page": PER_PAGE,
        "page": page
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def download_file(url, path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    downloaded = 0
    page = 1

    while downloaded < TOTAL_VIDEOS:
        data = search_videos(page)

        for video in data["videos"]:
            if downloaded >= TOTAL_VIDEOS:
                break

            video_files = video["video_files"]

            # choose best quality (mp4)
            best_file = max(video_files, key=lambda x: x["width"] or 0)

            video_url = best_file["link"]
            video_id = video["id"]

            file_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

            if os.path.exists(file_path):
                print(f"Skipping existing {video_id}")
                continue

            print(f"Downloading {video_id}...")

            try:
                download_file(video_url, file_path)
                downloaded += 1
            except Exception as e:
                print(f"Failed {video_id}: {e}")

        page += 1


if __name__ == "__main__":
    main()