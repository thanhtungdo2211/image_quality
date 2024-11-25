import numpy as np
import math
import os
import imagehash
import cv2
import hashlib

from tqdm import tqdm
from PIL import Image
from PIL import ImageStat, ImageFilter
from typing import Union, List, Dict, Any
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict


class Brightness:
    """Phân tích độ sáng của một hình ảnh.

    Cung cấp các phương thức để tính toán chỉ số độ sáng dựa trên đọ sáng trung bình và độ sáng theo phân vị.

    Attributes:
        image (Image): hình ảnh được mở từ thư viện Pillow.
    """
    percentiles = [1, 5, 10, 15, 90, 95, 99]
    
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_brightness(
        red: Union[float, "np.ndarray[Any, Any]"],
        green: Union[float, "np.ndarray[Any, Any]"],
        blue: Union[float, "np.ndarray[Any, Any]"],
    ) -> Union[float, "np.ndarray[Any, Any]"]:
        """Tính độ sáng của hình ảnh hoặc pixel.

        Độ sáng được tính toán dựa trên công thức dựa vào tổng trọng số của bình phương các thành phần
        màu đỏ, xanh lá, và xanh dương.

        Args:
            red (Union[float, np.ndarray]): Mảng chứa thông tin về mức màu đỏ.
            green (Union[float, np.ndarray]): Mảng chứa thông tin về mức màu xanh lá.
            blue (Union[float, np.ndarray]): Mảng chứa thông tin về mức màu xanh dương.

        Returns:
            Union[float, np.ndarray]: Độ sáng tính toán được dưới dạng số thực hoặc mảng numpy.
        """
        cur_bright = (
            np.sqrt(0.241 * (red * red) + 0.691 * (green * green) + 0.068 * (blue * blue))
        ) / 255

        return cur_bright

    @classmethod
    def calc_avg_brightness(cls, image) -> float:
        """Tính độ sáng trung bình của hình ảnh.

        Độ sáng trung bình được tính toán dùng giá trị trung bình của các kênh màu đỏ, xanh lá, và xanh dương.

        Returns:
            float: Độ sáng trung bình của hình ảnh.
        """
        stat = ImageStat.Stat(image)
        try:
            red, green, blue = stat.mean
        except ValueError:
            red, green, blue = (
                stat.mean[0],
                stat.mean[0],
                stat.mean[0],
            )  # For B&W images
        return cls.calculate_brightness(red, green, blue)
    
    @classmethod
    def calc_percentile_brightness(cls, image, percentiles=None) -> "np.ndarray[Any, Any]":
        """Tính độ sáng tại các phân vị xác định của hình ảnh.

        Args:
            percentiles (List[int]): Danh sách các phân vị để tính độ sáng.

        Returns:
            np.ndarray: Các giá trị độ sáng tại các phân vị đã chỉ định.
        """
        imarr = np.asarray(image)
        if len(imarr.shape) == 3:
            r, g, b = (
                imarr[:, :, 0].astype("int"),
                imarr[:, :, 1].astype("int"),
                imarr[:, :, 2].astype("int"),
            )
            pixel_brightness = cls.calculate_brightness(r, g, b)
        else:
            pixel_brightness = imarr / 255.0
        
        percentiles = cls.percentiles if percentiles is None else percentiles
        return np.percentile(pixel_brightness, percentiles)
    
    @classmethod
    def calculate_brightness_score(cls, image, percentiles=None) -> Dict[str, Union[float, str]]:
        """Tính toán và trả về điểm độ sáng tổng thể của hình ảnh.

        Tính toán giá trị độ sáng trung bình và độ sáng tại các phân vị. Độ sáng tại phân vị 5 sẽ được đại diện cho vấn đề Dark issues, 
        độ sáng tại phân vị 99 sẽ đại diện cho vấn đề Light issues.
        
        Returns:
            Dict[str, Union[float, str]]: Một từ điển chứa các giá trị độ sáng tại các phần trăm đã chọn và
            giá trị độ sáng trung bình của hình ảnh. Khóa của dict là 'brightness_perc_X' 
            với 'X' là giá trị phân vị, 'brightness' cho giá trị độ sáng trung bình.
        """
        percentiles = cls.percentiles if percentiles is None else percentiles
        perc_values = cls.calc_percentile_brightness(image, percentiles)
        raw_values = {
            f"brightness_perc_{p}": value for p, value in zip(percentiles, perc_values)
        }
        raw_values["brightness"] = cls.calc_avg_brightness(image)
        return raw_values

class AspectRatio:
    """Đánh tỉ lệ khung hình của một hình ảnh.

    Tính toán và đánh giá tỉ lệ khung hình của hình ảnh dựa trên tỉ lệ giữa chiều rộng và chiều cao của hình ảnh.

    Attributes:
        image (Image): hình ảnh được mở từ thư viện Pillow.
    """
    @staticmethod
    def calc_aspect_ratio_score(image: Image) -> float:
        width, height = image.size
        return min(width / height, height / width)
    
class Entropy:
    """Đánh giá mức độ mang thông tin của một hình ảnh.

    Đánh giá mức độ phức tạp của thông tin trong hình ảnh thông qua entropy

    Attributes:
        image (Image): Đối tượng hình ảnh từ thư viện PIL, mà từ đó entropy sẽ được tính.
    """
    @staticmethod  
    def calc_entropy_score(image : Image) -> float:
        return image.entropy() / 10
class Blurriness:
    """Đánh giá mức độ mờ của hình ảnh.

    Tính toán điểm số mờ, dựa trên phân tích cạnh và độ lệch chuẩn
    của histogram màu xám của hình ảnh. Các phương pháp này giúp xác định mức độ mờ của hình ảnh.

    Attributes:
        MAX_RESOLUTION_FOR_BLURRY_DETECTION (int): Độ phân giải tối đa cho phép khi phát hiện mờ.
    """
    MAX_RESOLUTION_FOR_BLURRY_DETECTION = 64
    @staticmethod
    def get_edges(gray_image: Image) -> Image:
        return gray_image.filter(ImageFilter.FIND_EDGES) #Tìm các cạnh trong hình ảnh màu xám bằng bộ lọc FIND_EDGES

    @classmethod
    def calc_blurriness(cls, gray_image: Image) -> float:
        edges = cls.get_edges(gray_image)
        blurriness = ImageStat.Stat(edges).var[0]
        return np.sqrt(blurriness)

    @staticmethod
    def calc_std_grayscale(gray_image: Image) -> float:
        return np.std(gray_image.histogram()) #Tính độ lệch chuẩn của histogram trên mức màu xám

    @classmethod
    def calculate_blurriness_score(cls, image : Image) -> Dict[str, Union[float, str]]:
        """Tính toán và trả về điểm số mờ của hình ảnh.

        Điểm số mờ được tính toán dựa trên mức độ mờ và độ lệch chuẩn của histogram màu xám,

        Returns:
            Dict[str, Union[float, str]]: Điểm số mờ của hình ảnh, dưới dạng từ điển.
        """
        ratio = max(image.width, image.height) / Blurriness.MAX_RESOLUTION_FOR_BLURRY_DETECTION
        if ratio > 1:
            resized_image = image.resize(
                (max(int(image.width // ratio), 1), max(int(image.height // ratio), 1))
            )
        else:
            resized_image = image.copy()
        gray_image = resized_image.convert("L")
        blur_scores = 1 - np.exp(-1 * cls.calc_blurriness(gray_image) / 100)
        std_scores = 1 - np.exp(-1 * cls.calc_std_grayscale(gray_image) / 100)
        blur_std_score = np.minimum(blur_scores + std_scores, 1)
        return {"score": blur_std_score}
class ColorSpace:
    """Xác định không gian màu của một hình ảnh.

    Cung cấp phương thức để xác định không gian màu của hình ảnh (ví dụ: RGB, L (xám))

    Attributes:
        image (Image): Đối tượng hình ảnh từ thư viện PIL.
    """
    @staticmethod
    def get_image_mode(image : Image) -> str:
        if image.mode:
            return image.mode
        else:
            imarr = np.asarray(image)
            if len(imarr.shape) == 2 or (len(imarr.shape) == 3 and (np.diff(imarr.reshape(-1, 3).T, axis=0) == 0).all()):
                return "L"
            else:
                return "UNK"
    @classmethod
    def calculate_space_color(cls,image :Image) -> Dict[str, Union[float, str]]:
        return {"color_space": 1 if cls.get_image_mode(image) == "RGB" else 0}

class Size:
    """Tính toán và đánh giá kích thước của hình ảnh.

    Class này cung cấp các phương thức để đánh giá kích thước của một hình ảnh và so sánh nó
    với các hình ảnh khác trong một thư mục dựa trên các chỉ số đặc biệt về kích thước.

    Attributes:
        image (Image): Một đối tượng hình ảnh từ thư viện PIL.
    """
    
    def __init__(self, folder_path = None):
        if folder_path is not None:
            self.image_area_sqrt_sizes = Size.get_image_area_sqrt_sizes(folder_path)
        else:
            self.image_area_sqrt_sizes = None

    @staticmethod
    def calc_image_area_sqrt(image : Image) -> float:
        w, h = image.size
        return math.sqrt(w) * math.sqrt(h)

    @classmethod
    def get_image_area_sqrt_sizes(cls, folder_path: str) -> list:
        image_sqrt_sizes = []
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                try:
                    with Image.open(os.path.join(folder_path, file_name)) as img:
                        image_sqrt_size = cls.calc_image_area_sqrt(img)
                        image_sqrt_sizes.append(image_sqrt_size)
                except IOError:
                    print(f"Cannot open {file_name}.")
        return image_sqrt_sizes
    
    def update_image_area_sqrt_sizes(self, folder_path: str):
        self.image_area_sqrt_sizes = Size.get_image_area_sqrt_sizes(folder_path)

    def calculate_image_size_score(self, image: Image, folder_path: str = None, iqr_factor: float = 3.0) -> float:
        """Tính điểm số kích thước cho hình ảnh dựa trên so sánh với các hình ảnh trong một thư mục.

        Phương pháp này so sánh kích thước của hình ảnh với phổ kích thước của các hình ảnh khác trong thư mục,
        sử dụng phương pháp interquartile range (IQR) để xác định ngưỡng và tính toán điểm số.

        Args:
            folder_path (str): Đường dẫn đến thư mục chứa hình ảnh.
            iqr_factor (float): Hệ số nhân của IQR để xác định ngưỡng.

        Returns:
            float: Điểm số kích thước của hình ảnh, dựa trên mức độ khớp với phổ kích thước trong thư mục.
        """
        image_sizes = self.image_area_sqrt_sizes
        if image_sizes is None:
            assert folder_path is not None, "Folder path must be provided to calculate image size score."
            self.update_image_area_sqrt_sizes(folder_path)
            image_sizes = self.image_area_sqrt_sizes

        q1, q3 = np.percentile(image_sizes, [25, 75])
        iqr = q3 - q1
        min_threshold = q1 - iqr_factor * iqr
        max_threshold = q3 + iqr_factor * iqr
        mid_threshold = (min_threshold + max_threshold) / 2

        image_size = self.calc_image_area_sqrt(image)
        distance = abs(image_size - mid_threshold)
        norm_value = max_threshold - min_threshold if max_threshold - min_threshold > 0 else mid_threshold
        norm_dist = distance / norm_value
        score_value = 1 - np.clip(norm_dist, 0, 1)
        return score_value

class NearDuplicateImage:
    def __init__(self, folder_path, hash_size=8):
        self.folder_path = folder_path
        self.hash_size = hash_size

    def find_near_duplicate_image(self):
        """
        Tìm kiếm các hình ảnh trùng lặp trong thư mục chỉ định bằng cách sử dụng perceptual hash.
        Trả về một dictionary với perceptual hash làm khóa và danh sách tên file trùng lặp làm giá trị.
        """
        hash_method = imagehash.phash # Change hash type here
        image_hashes = {}

        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_path = os.path.join(self.folder_path, filename)
                with Image.open(image_path) as img:
                    img_hash = str(hash_method(img, hash_size=self.hash_size))

                    # bytes = img.read() # read entire image as a byte array
                    # img_hash = hashlib.md5(bytes).hexdigest(); # hash the byte array

                    if img_hash in image_hashes:
                        image_hashes[img_hash].append(filename)
                    else:
                        image_hashes[img_hash] = [filename]
                        
        near_duplicates = {}
        for img_hash, filenames in image_hashes.items():
            if len(filenames) > 1:
                longest_name = max(filenames, key=len)
                near_duplicates[longest_name] = [fname for fname in filenames if fname != longest_name]
    
        return near_duplicates

class DuplicateImage:
    """
    Tìm kiếm các hình ảnh trùng lặp trong thư mục chỉ định bằng cách sử dụng d-hash và check lại bằng việc so sánh MSE và SSIM.
    """
    def __init__(self, folder_path, hash_size=8):
        self.folder_path = folder_path
        self.hash_size = hash_size

    def _dhash(self, image):
        resized_img = cv2.resize(image, (self.hash_size + 1, self.hash_size))
        diff = resized_img[:, 1:] > resized_img[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    
    def _phash(self, image):
        resized_img = cv2.resize(image, (self.hash_size, self.hash_size))
        dct = cv2.dct(np.float32(resized_img))
        dct_low_freq = dct[:8, :8]
        median = np.median(dct_low_freq)
        diff = dct_low_freq > median
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    def _mse(self, first_img, second_img):
        err = np.sum((first_img.astype("float") - second_img.astype("float")) ** 2)
        err /= float(first_img.shape[0] * first_img.shape[1])
        return err

    def find_duplicate_image(self):
        image_data = {}
        pic_hashes = {}

        for rel_path in os.listdir(self.folder_path):
            path = os.path.join(self.folder_path, rel_path)
            img = cv2.imread(path, 0)
            if img is None:
                continue
            image_data[rel_path] = img
            image_hash = self._phash(img) # Change hash type in here
            pic_hashes.setdefault(image_hash, []).append(rel_path)

        duplicates = defaultdict(set)
        for files in pic_hashes.values():
            if len(files) > 1:
                for i in range(len(files)):
                    for j in range(i + 1, len(files)):
                        img1 = image_data[files[i]]
                        img2 = image_data[files[j]]
                        if img1.shape == img2.shape:
                            mse_val = self._mse(img1, img2)
                            ssim_val = ssim(img1, img2)
                            if mse_val < 20 and ssim_val > 0.95:
                                duplicates[files[i]].add(files[j])
                                duplicates[files[j]].add(files[i])

        # Consolidate duplicates into a unified list
        seen = set()
        final_duplicates = {}
        for key, values in duplicates.items():
            if key not in seen:
                all_related = set(values)
                all_related.add(key)
                representative = max(all_related, key=lambda x: len(x))
                final_duplicates[representative] = all_related - {representative}
                seen.update(all_related)

        return final_duplicates

