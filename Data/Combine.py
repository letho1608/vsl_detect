from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import os
import csv
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Các biến global
BASE_URL = "https://qipedc.moet.gov.vn"
VIDEOS_API = "https://qipedc.moet.gov.vn/videos"
output_dir = "Dataset/Video"
text_dir = "Dataset/Text"
os.makedirs(output_dir, exist_ok=True)  
os.makedirs(text_dir, exist_ok=True)
csv_path = os.path.join(text_dir, "Label.csv")
csv_lock = threading.Lock()

def handle_recursive_scrapping(dict: dict, driver):
    vids = WebDriverWait(driver=driver, timeout=3).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1)"))
    )
    vids = driver.find_elements(By.CSS_SELECTOR, "section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1) a")
    for vid in vids:
        gross = vid.find_element(By.CSS_SELECTOR, "p").text
        raw_thumb_url = vid.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
        base_thumbs_url = "https://qipedc.moet.gov.vn/thumbs/"
        videosID = raw_thumb_url[len(base_thumbs_url):len(raw_thumb_url) - 4]
        video_url = BASE_URL + "/videos/" + videosID + ".mp4"

        item = {
            "gross": gross,
            "url": video_url
        }
        print(f"✓ Tìm thấy: {gross}")
        dict.append(item)
    return

def init_csv():
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["STT", "VIDEO", "TEXT"])

def add_to_csv(stt, video_name, text):
    with csv_lock:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([stt, video_name, text])

def download_video(video_data):
    url = video_data.get("url")
    gross = video_data.get("gross")  # Lấy gross từ data crawl được
    if not url:
        return

    filename = os.path.basename(urlparse(url).path)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"⏭️  Bỏ qua: {filename} - đã tồn tại")
        return
        
    try:
        print(f"\n⬇️  Đang tải: {filename}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=f"Tiến trình tải {filename}",
            total=total_size,
            unit='B', 
            unit_scale=True,
            unit_divisor=1024,
            ncols=100
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        stt = sum(1 for _ in open(csv_path, encoding='utf-8'))
        add_to_csv(stt, filename, gross)  # Sử dụng gross thay vì f"Video {filename}"
                
        print(f"✅ Hoàn thành: {filename}")
        print(f"📝 Đã cập nhật Label.csv với nhãn: {gross}")
        
    except Exception as e:
        print(f"❌ Lỗi khi tải {filename}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)

def crawl_videos():
    print("\n=== 🔍 BẮT ĐẦU CRAWL DỮ LIỆU ===")
    options = Options()
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    videos = []
    
    try:
        driver.get("https://qipedc.moet.gov.vn/dictionary")
        print("🌐 Đã kết nối tới trang từ điển")
        
        handle_recursive_scrapping(videos, driver)

        for i in range(2, 5):
            id = i
            if i != 2: id = i + 1
            button = driver.find_element(By.CSS_SELECTOR, f"button:nth-of-type({id})")
            button.click()
            handle_recursive_scrapping(videos, driver)
            
        for i in range(5, 218):
            id = 6
            button = driver.find_element(By.CSS_SELECTOR, f"button:nth-of-type({id})")
            button.click()
            handle_recursive_scrapping(videos, driver)

        for i in range(218, 220):
            id = 6
            if i != 218: id = 7
            button = driver.find_element(By.CSS_SELECTOR, f"button:nth-of-type({id})")
            button.click()
            handle_recursive_scrapping(videos, driver)

    except Exception as e:
        print(f"❌ Lỗi khi crawl: {e}")
    finally:
        driver.close()
        return videos

def main():
    print("\n====================================")
    print("🤖 CHƯƠNG TRÌNH CRAWL VÀ TẢI VIDEO")
    print("====================================\n")
    
    videos = crawl_videos()
    if videos:
        print(f"✅ Đã tìm thấy {len(videos)} video\n")
    
    print("=== ⬇️  BẮT ĐẦU TẢI VIDEO ===")
    init_csv()
    
    if not videos:
        print("❌ Không có video nào để tải")
        return
        
    with ThreadPoolExecutor(max_workers=3) as executor:
        print(f"⚡ Tải song song tối đa 3 video cùng lúc")
        executor.map(download_video, videos)
        
    print("\n====================================")
    print(f"✨ HOÀN THÀNH - Đã tải về {output_dir}")
    print("====================================")

if __name__ == "__main__":
    main()
