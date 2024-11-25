from src.issue_analysis import IssueAnalyst
from src.visualize import sumary_issue

if __name__ == "__main__":
    folder_path = '/media/tung/New Volume/Ubuntu/Programing/MQSolutions/TaskData/dataset/fire_mq_data'  # Cập nhật đường dẫn tới thư mục chứa ảnh
    analyst = IssueAnalyst(folder_path)
    results_df = analyst.analyze_images(issue_types=['dark', 'light', 'blurry', 'duplicate', 'near_duplicate'])
    results_df.to_csv('output.csv', index=False)  # Xuất kết quả ra file CSV
    print(sumary_issue(results_df))  # In kết quả phân tích
