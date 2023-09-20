"""
Kế hoạch 1:

Tạo dữ liệu ảnh barcode độ phân giải cao

Để training REAL-ESRGAN với tập dữ liệu này

Để thấy sự khác biệt

"""

"""
test trên các model khác nhau như: 
Ibicubic x2, srgan, gan, single image super-resolution, realSR,... trên tập dữ liệu barcodes 
REAL-ESRGAN with pure synthetic data 

Để thêm kết quả vào phần related works 
"""

"""
OpenSR_Barcode_Images: contain barcode hight resolution images

để training REAL-ESRGAN trên tập dữ liệu barcode này  

"""

from io import BytesIO

from barcode import EAN13
from barcode.writer import ImageWriter

# Write to a file-like object:
rv = BytesIO()
EAN13(str(100000902922), writer=ImageWriter()).write(rv)

# Or to an actual file:
with open("f111169109.jpg", "wb") as f:
    EAN13("225251155000", writer=ImageWriter()).write(f)
