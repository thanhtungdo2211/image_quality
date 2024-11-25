import os
import pandas as pd

from PIL import Image
from src.issue_score import Brightness, AspectRatio, Entropy, Blurriness, ColorSpace, Size, DuplicateImage,NearDuplicateImage
from multiprocessing import Pool
from tqdm import tqdm

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, FloatType, BooleanType

class IssueAnalyst:
    def __init__(self, folder_path: str, thresholds=None):
        """Khởi tạo IssueAnalyst với đường dẫn thư mục và ngưỡng cho mỗi vấn đề.

        Args:
            folder_path (str): Đường dẫn đến thư mục chứa ảnh.
            thresholds (dict, optional): Ngưỡng cho mỗi vấn đề. Defaults to None.
        """
        self.folder_path = folder_path
        # Định nghĩa ngưỡng mặc định cho mỗi vấn đề
        self.thresholds = thresholds or {
            'dark': 0.32,
            'light': 0.05,
            'odd_aspect_ratio': 0.35,
            'low_information': 0.3,
            'blurry': 0.29,
            'color_space': 0.5,
            'odd_size': 0.5
        }
        self.size_evaluator = Size(self.folder_path)
        #self.spark = SparkSession.builder.appName("ImageAnalyzer").getOrCreate()

    def _calculate_dark_score(self, img: Image, threshold: float) -> (float, bool):
        """Tính điểm cho vấn đề ảnh tối.

        Args:
            img (Image): Đối tượng ảnh từ thư viện PIL.
            threshold (float): Ngưỡng cho vấn đề ảnh tối.

        Returns:
            float, bool: Điểm và có phải là vấn đề hay không.
        """
        score = Brightness.calculate_brightness_score(img)['brightness_perc_95']
        is_issue = bool(score < threshold)
        return float(score), is_issue

    def _calculate_light_score(self, img: Image, threshold: float) -> (float, bool):
        """Tính điểm cho vấn đề ảnh sáng.

        Args:
            img (Image): Đối tượng ảnh từ thư viện PIL.
            threshold (float): Ngưỡng cho vấn đề ảnh sáng.

        Returns:
            float, bool: Điểm và có phải là vấn đề hay không.
        """
        score = 1 - Brightness.calculate_brightness_score(img)['brightness_perc_5']
        is_issue = bool(score < threshold)
        return float(score), is_issue

    def _calculate_odd_aspect_ratio_score(self, img: Image, threshold: float) -> (float, bool):
        """Tính điểm cho vấn đề tỷ lệ khung hình lạ.

        Args:
            img (Image): Đối tượng ảnh từ thư viện PIL.
            threshold (float): Ngưỡng cho vấn đề tỷ lệ khung hình lạ.

        Returns:
            float, bool: Điểm và có phải là vấn đề hay không.
        """
        score = AspectRatio.calc_aspect_ratio_score(img)
        is_issue = bool(score < threshold)
        return float(score), is_issue

    def _calculate_low_information_score(self, img: Image, threshold: float) -> (float, bool):
        """Tính điểm cho vấn đề thông tin thấp.

        Args:
            img (Image): Đối tượng ảnh từ thư viện PIL.
            threshold (float): Ngưỡng cho vấn đề thông tin thấp.

        Returns:
            float, bool: Điểm và có phải là vấn đề hay không.
        """
        score = Entropy.calc_entropy_score(img)
        is_issue = bool(score < threshold)
        return float(score), is_issue

    def _calculate_blurry_score(self, img: Image, threshold: float) -> (float, bool):
        """Tính điểm cho vấn đề ảnh mờ.

        Args:
            img (Image): Đối tượng ảnh từ thư viện PIL.
            threshold (float): Ngưỡng cho vấn đề ảnh mờ.

        Returns:
            float, bool: Điểm và có phải là vấn đề hay không.
        """
        score = Blurriness.calculate_blurriness_score(img)['score']
        is_issue = bool(score < threshold)
        return float(score), is_issue

    def _calculate_color_space_score(self, img: Image, threshold: float) -> (float, bool):
        """Tính điểm cho vấn đề không gian màu.

        Args:
            img (Image): Đối tượng ảnh từ thư viện PIL.
            threshold (float): Ngưỡng cho vấn đề không gian màu.

        Returns:
            float, bool: Điểm và có phải là vấn đề hay không.
        """
        score = ColorSpace.calculate_space_color(img)['color_space']
        is_issue = bool(score < threshold)
        return float(score), is_issue

    def _calculate_odd_size_score(self, img: Image, threshold: float) -> (float, bool):
        """Tính điểm cho vấn đề kích thước lạ.

        Args:
            img (Image): Đối tượng ảnh từ thư viện PIL.
            threshold (float): Ngưỡng cho vấn đề kích thước lạ.

        Returns:
            float, bool: Điểm và có phải là vấn đề hay không.
        """
        score = self.size_evaluator.calculate_image_size_score(img)
        is_issue = bool(score < threshold)
        return float(score), is_issue

    def _process_image(self, file_name, issue_types, duplicate_files=None, near_duplicate_files=None):
        """Xử lý một ảnh và trả về điểm cho mỗi vấn đề.

        Args:
            file_name (str): Tên file ảnh.

        Returns:
            dict: Điểm cho mỗi vấn đề và có phải là vấn đề hay không.
        """
        image_path = os.path.join(self.folder_path, file_name)

        results = {"image_name": file_name}

        with Image.open(image_path) as img:
            # Lấy ngưỡng cho mỗi vấn đề
            dark_threshold = self.thresholds.get('dark', 0.05)
            light_threshold = self.thresholds.get('light', 0.32)
            odd_aspect_ratio_threshold = self.thresholds.get('odd_aspect_ratio', 0.35)
            low_information_threshold = self.thresholds.get('low_information', 0.3)
            blurry_threshold = self.thresholds.get('blurry', 0.29)
            color_space_threshold = self.thresholds.get('color_space', 0.5)
            odd_size_threshold = self.thresholds.get('odd_size', 0.5)

            # Tính toán điểm cho mỗi vấn đề sử dụng ngưỡng tương ứng    
            if 'dark' in issue_types:
                dark_score, is_dark_issue = self._calculate_dark_score(img, dark_threshold)
                results.update({"dark_score": dark_score, "is_dark_issue": is_dark_issue})

            if 'light' in issue_types:
                light_score, is_light_issue = self._calculate_light_score(img, light_threshold)
                results.update({"light_score": light_score, "is_light_issue": is_light_issue})
            
            if  'odd_aspect_ratio' in issue_types:
                odd_aspect_ratio_score, is_odd_aspect_ratio_issue = self._calculate_odd_aspect_ratio_score(img, odd_aspect_ratio_threshold)
                results.update({"odd_aspect_ratio_score": odd_aspect_ratio_score, "is_odd_aspect_ratio_issue": is_odd_aspect_ratio_issue})

            if 'low_information' in issue_types:
                low_information_score, is_low_information_issue = self._calculate_low_information_score(img, low_information_threshold)
                results.update({"low_information_score": low_information_score, "is_low_information_issue": is_low_information_issue})

            if 'blurry' in issue_types:
                blurry_score, is_blurry_issue = self._calculate_blurry_score(img, blurry_threshold)
                results.update({"blurry_score": blurry_score, "is_blurry_issue": is_blurry_issue})

            if 'color_space' in issue_types:
                color_space_score, is_color_space_issue = self._calculate_color_space_score(img, color_space_threshold)
                results.update({"color_space_score": color_space_score, "is_color_space_issue": is_color_space_issue})

            if 'odd_size' in issue_types:
                odd_size_score, is_odd_size_issue = self._calculate_odd_size_score(img, odd_size_threshold)
                results.update({"odd_size_score": odd_size_score, "is_odd_size_issue": is_odd_size_issue})
            
        if 'duplicate' in issue_types:
            results["is_duplicate"] = file_name in (duplicate_files or set())
        if 'near_duplicate' in issue_types:
            results["is_near_duplicate"] = file_name in (near_duplicate_files or set())

        return results

    def analyze_images(self, issue_types) -> pd.DataFrame:
        """Phân tích tất cả các ảnh trong thư mục và trả về kết quả dưới dạng DataFrame.

        Returns:
            pd.DataFrame: DataFrame chứa kết quả phân tích.
        """
        results = []
        file_names = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

        duplicate_files, near_duplicate_files = set(), set()
        if 'duplicate' in issue_types:
            duplicates = DuplicateImage(self.folder_path).find_duplicate_image()
            if duplicates:
                duplicate_files = {item for sublist in duplicates.values() for item in sublist}
        if 'near_duplicate' in issue_types:
            near_duplicates = NearDuplicateImage(self.folder_path).find_near_duplicate_image()
            if near_duplicates:
                near_duplicate_files = {item for sublist in near_duplicates.values() for item in sublist}

        for file_name in tqdm(file_names):
            image_result = self._process_image(file_name, issue_types, duplicate_files, near_duplicate_files)
            results.append(image_result)

        return pd.DataFrame(results)
    
    def analyze_images_pypark(self, issue_types):
        """Phân tích tất cả các ảnh trong thư mục và trả về kết quả dưới dạng DataFrame.

        Returns:
            pyspark.sql.DataFrame: DataFrame chứa kết quả phân tích.
        """
        file_names = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

        duplicate_files, near_duplicate_files = set(), set()
        if 'duplicate' in issue_types:
            duplicates = DuplicateImage(self.folder_path).find_duplicate_image()
            if duplicates:
                duplicate_files = {item for sublist in duplicates.values() for item in sublist}
        if 'near_duplicate' in issue_types:
            near_duplicates = NearDuplicateImage(self.folder_path).find_near_duplicate_image()
            if near_duplicates:
                near_duplicate_files = {item for sublist in near_duplicates.values() for item in sublist}

        results = []
        for file_name in tqdm(file_names):
            image_result = self._process_image(file_name, issue_types, duplicate_files, near_duplicate_files)
            results.append(image_result)

        # Xây dựng schema động dựa trên issue_types
        schema_fields = [StructField("image_name", StringType(), True)]
        if 'dark' in issue_types:
            schema_fields.extend([
                StructField("dark_score", FloatType(), True),
                StructField("is_dark_issue", BooleanType(), True)
            ])
        if 'light' in issue_types:
            schema_fields.extend([
                StructField("light_score", FloatType(), True),
                StructField("is_light_issue", BooleanType(), True)
            ])
        if 'odd_aspect_ratio' in issue_types:
            schema_fields.extend([
                StructField("odd_aspect_ratio_score", FloatType(), True),
                StructField("is_odd_aspect_ratio_issue", BooleanType(), True)
            ])
        if 'low_information' in issue_types:
            schema_fields.extend([
                StructField("low_information_score", FloatType(), True),
                StructField("is_low_information_issue", BooleanType(), True)
            ])
        if 'blurry' in issue_types:
            schema_fields.extend([
                StructField("blurry_score", FloatType(), True),
                StructField("is_blurry_issue", BooleanType(), True)
            ])
        if 'color_space' in issue_types:
            schema_fields.extend([
                StructField("color_space_score", FloatType(), True),
                StructField("is_color_space_issue", BooleanType(), True)
            ])
        if 'odd_size' in issue_types:
            schema_fields.extend([
                StructField("odd_size_score", FloatType(), True),
                StructField("is_odd_size_issue", BooleanType(), True)
            ])
        if 'duplicate' in issue_types:
            schema_fields.append(StructField("is_duplicate", BooleanType(), True))
        if 'near_duplicate' in issue_types:
            schema_fields.append(StructField("is_near_duplicate", BooleanType(), True))

        schema = StructType(schema_fields)

        df = self.spark.createDataFrame(results, schema)
        return df
