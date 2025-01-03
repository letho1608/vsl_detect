#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
from tabulate import tabulate
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

def load_and_print_data(data_path):
    """Tải và in dữ liệu từ các tệp .npy trong thư mục đã cho"""
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                print(f"Dữ liệu từ {file_path}:")
                print(data)
                print("\n")

def print_directory_info(data_path):
    """In thông tin về các thư mục trong đường dẫn đã cho"""
    subdirs = next(os.walk(data_path))[1]
    print(f"Số lượng thư mục con trong {data_path}: {len(subdirs)}")
    print(f"Các thư mục con: {subdirs}")

def count_npy_files(folder_path):
    """Đếm số lượng file .npy trong thư mục"""
    count = 0
    for _, _, files in os.walk(folder_path):
        count += sum(1 for file in files if file.endswith('.npy'))
    return count

def analyze_directory(data_path):
    """Phân tích cấu trúc thư mục và file"""
    if not os.path.exists(data_path):
        print(f"Không tìm thấy thư mục {data_path}")
        return

    # Thu thập thông tin
    data = []
    total_folders = 0
    total_files = 0
    
    # Lấy danh sách thư mục gốc (các hành động)
    action_folders = next(os.walk(data_path))[1]
    
    for action in sorted(action_folders):
        action_path = os.path.join(data_path, action)
        sequences = next(os.walk(action_path))[1]  # Các thư mục sequence
        npy_count = count_npy_files(action_path)
        
        total_folders += len(sequences)
        total_files += npy_count
        
        # Thêm vào bảng
        data.append([
            action,
            len(sequences),
            npy_count,
            f"{npy_count/(len(sequences) or 1):.1f}",
            "✓" if npy_count > 0 else "✗"
        ])

    # In bảng thống kê
    headers = ["Hành động", "Số sequence", "Số file .npy", "TB file/seq", "Trạng thái"]
    print("\nThống kê chi tiết:")
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # In tổng quan
    print("\nTổng quan:")
    print(f"- Tổng số hành động: {len(action_folders)}")
    print(f"- Tổng số thư mục sequence: {total_folders}")
    print(f"- Tổng số file .npy: {total_files}")
    print(f"- Trung bình file/hành động: {total_files/len(action_folders):.1f}")

def validate_npy_file(file_path):
    """Kiểm tra tính hợp lệ của file .npy"""
    try:
        data = np.load(file_path)
        # Kiểm tra kích thước (phải là 126 features cho mỗi frame)
        if data.shape != (126,):
            return False, f"Kích thước không đúng: {data.shape} (cần 126,)"
        # Kiểm tra giá trị
        if np.isnan(data).any():
            return False, "Chứa giá trị NaN"
        if np.isinf(data).any():
            return False, "Chứa giá trị Inf"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def check_dataset_integrity(data_path):
    """Kiểm tra toàn bộ dataset"""
    if not os.path.exists(data_path):
        print(f"Không tìm thấy thư mục {data_path}")
        return

    problems = []
    stats = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0
    }

    # Lấy danh sách các hành động
    actions = sorted(next(os.walk(data_path))[1])
    
    for action in tqdm(actions, desc="Kiểm tra dữ liệu"):
        action_path = os.path.join(data_path, action)
        sequences = sorted(next(os.walk(action_path))[1])
        
        for seq in sequences:
            seq_path = os.path.join(action_path, seq)
            files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
            
            for file in files:
                stats['total_files'] += 1
                file_path = os.path.join(seq_path, file)
                is_valid, message = validate_npy_file(file_path)
                
                if is_valid:
                    stats['valid_files'] += 1
                else:
                    stats['invalid_files'] += 1
                    problems.append([
                        action,
                        seq,
                        file,
                        message
                    ])

    # In kết quả
    print("\nKết quả kiểm tra:")
    print(f"Tổng số files: {stats['total_files']}")
    print(f"Files hợp lệ: {stats['valid_files']}")
    print(f"Files lỗi: {stats['invalid_files']}")
    
    if problems:
        print("\nDanh sách files có vấn đề:")
        headers = ["Hành động", "Sequence", "File", "Lỗi"]
        print(tabulate(problems, headers=headers, tablefmt="grid"))
        
        # Thống kê lỗi theo hành động
        action_stats = {}
        for p in problems:
            action = p[0]
            if action not in action_stats:
                action_stats[action] = 0
            action_stats[action] += 1
        
        print("\nThống kê lỗi theo hành động:")
        action_problems = [[action, count] for action, count in action_stats.items()]
        print(tabulate(action_problems, ["Hành động", "Số files lỗi"], tablefmt="grid"))

def find_and_fix_sequence_issues(data_path, required_sequences=60):
    """Tìm và sửa các vấn đề về số lượng sequence"""
    print("\nKiểm tra số lượng sequence cho mỗi hành động...")
    
    issues = []
    for action in os.listdir(data_path):
        action_path = os.path.join(data_path, action)
        if not os.path.isdir(action_path):
            continue
            
        sequences = sorted([int(seq) for seq in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, seq))])
        if not sequences:
            continue
            
        # Kiểm tra số lượng sequence
        if len(sequences) != required_sequences:
            issues.append([
                action,
                len(sequences),
                sequences[-1] if sequences else 'N/A',
                "Số lượng sequence không đúng"
            ])
            
        # Kiểm tra thứ tự sequence
        for i, seq in enumerate(sequences):
            if seq >= required_sequences:
                issues.append([
                    action,
                    seq,
                    i,
                    "Sequence index vượt quá giới hạn"
                ])
    
    if issues:
        print("\nDanh sách vấn đề về sequence:")
        headers = ["Hành động", "Số sequence/Index", "Sequence cuối/Vị trí", "Vấn đề"]
        print(tabulate(issues, headers=headers, tablefmt="grid"))
        
        # Hỏi người dùng có muốn sửa không
        fix = input("\nBạn có muốn sửa các vấn đề này không? (y/n): ")
        if fix.lower() == 'y':
            for action in os.listdir(data_path):
                action_path = os.path.join(data_path, action)
                if not os.path.isdir(action_path):
                    continue
                    
                sequences = sorted([int(seq) for seq in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, seq))])
                if not sequences:
                    continue
                    
                # Xóa các sequence thừa
                for seq in sequences:
                    if seq >= required_sequences:
                        seq_path = os.path.join(action_path, str(seq))
                        try:
                            import shutil
                            shutil.rmtree(seq_path)
                            print(f"Đã xóa sequence thừa: {seq_path}")
                        except Exception as e:
                            print(f"Lỗi khi xóa {seq_path}: {str(e)}")
            
            print("\nĐã sửa xong các vấn đề!")
    else:
        print("Không tìm thấy vấn đề về số lượng sequence.")

if __name__ == "__main__":
    data_path = 'Data_test'
    
    # Kiểm tra cấu trúc thư mục
    analyze_directory(data_path)
    
    # Kiểm tra và sửa số lượng sequence
    find_and_fix_sequence_issues(data_path)
    
    # Kiểm tra tính toàn vẹn của dữ liệu
    print("\nBắt đầu kiểm tra tính toàn vẹn của dataset...")
    check_dataset_integrity(data_path)
