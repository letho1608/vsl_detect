import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re
from bs4 import BeautifulSoup

class VideoDownloader:
    def __init__(self):
        self.base_url = "https://qipedc.moet.gov.vn"
        self.dictionary_url = f"{self.base_url}/dictionary"
        self.output_dir = os.path.join("Dataset", "Video")
        self.max_workers = 4
        
        # Tạo thư mục output
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Thư mục lưu video: {os.path.abspath(self.output_dir)}")

    def get_video_list(self):
        """Lấy danh sách video trực tiếp từ trang web"""
        try:
            # Lấy nội dung trang web
            response = requests.get(self.dictionary_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            videos = []
            # Tìm tất cả các thẻ video
            for video_elem in soup.select("section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1) a"):
                try:
                    title = video_elem.find('p').text.strip()
                    thumb_url = video_elem.find('img')['src']
                    # Lấy ID video từ URL thumbnail
                    video_id = re.search(r'/thumbs/(.+?)\.jpg', thumb_url).group(1)
                    video_url = f"{self.base_url}/videos/{video_id}.mp4"
                    
                    videos.append({
                        "title": title,
                        "url": video_url
                    })
                    print(f"Tìm thấy video: {title}")
                except Exception as e:
                    print(f"Lỗi khi xử lý video: {str(e)}")
                    
            return videos
            
        except Exception as e:
            print(f"Lỗi khi lấy danh sách video: {str(e)}")
            return []

    def download_video(self, video_info):
        """Tải một video"""
        try:
            title = video_info["title"]
            url = video_info["url"]
            
            # Tạo tên file an toàn
            safe_filename = "".join(x for x in title if x.isalnum() or x in [' ', '-', '_']).strip()
            output_path = os.path.join(self.output_dir, f"{safe_filename}.mp4")
            
            if os.path.exists(output_path):
                print(f"Đã tồn tại: {safe_filename}")
                return
            
            print(f"\nBắt đầu tải: {title}")
            
            # Tải video với progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=safe_filename) as pbar:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
            print(f"Hoàn thành: {safe_filename}")
            
        except Exception as e:
            print(f"Lỗi khi tải {title}: {str(e)}")

    def download_all(self):
        """Tải tất cả video"""
        videos = self.get_video_list()
        if not videos:
            print("Không tìm thấy video nào")
            return
            
        print(f"\nBắt đầu tải {len(videos)} video...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(self.download_video, videos)
        print("\nHoàn thành tải video!")

if __name__ == "__main__":
    downloader = VideoDownloader()
    downloader.download_all()