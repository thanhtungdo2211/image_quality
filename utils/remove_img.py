import os
import pandas as pd

def delete_images_with_issues(df, folder_path, issue_column):
    """
    Xóa các hình ảnh có vấn đề cụ thể.

    Args:
    df (pd.DataFrame): DataFrame chứa dữ liệu phân tích ảnh.
    folder_path (str): Đường dẫn tới thư mục chứa ảnh.
    issue_column (str): Tên cột trong DataFrame chỉ ra vấn đề cần xét (e.g., 'is_dark_issue').

    Returns:
    int: Số lượng hình ảnh đã bị xóa.
    """
    problematic_images = df[df[issue_column] == True]
    count_deleted = 0

    for image_name in problematic_images['image_name']:
        image_path = os.path.join(folder_path, image_name)
        if os.path.exists(image_path):
            os.remove(image_path)
            count_deleted += 1
            print(f"Deleted {image_name}")
        else:
            print(f"File not found: {image_name}")

    return count_deleted