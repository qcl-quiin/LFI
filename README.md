\-----CHUẨN BỊ CHẠY-----

**Bước 1: Gỡ Python cũ và các thư viện (Windows)**

"pip freeze | xargs pip uninstall -y"

\#Nếu không muốn xoá python cũ, hãy bỏ qua bước này

(khi chạy phải đổi phiên bản python sang 3.11 nếu không sẽ sinh lỗi)



**Bước 2: Cài Python 3.11.9**

"winget install Python.Python.3.11"

\#Sau khi cài xong, mở lại PowerShell/CMD mới trước khi tiếp tục



**Bước 3: Cài các thư viện**

"py -3.11 -m pip install opencv-python numpy pillow pywebview mediapipe==0.10.9"

\#Lưu ý: phải dùng đúng mediapipe==0.10.9, version mới hơn sẽ bị lỗi



Bước 4: Chạy chương trình

"py -3.11 collect\_dataset.py"



**-----ATTENTION-----**

**!GIỮ NGUYÊN CẤU TRÚC THƯ MỤC VÀ THỰC HIỆN THU DỮ LIỆU!**

**//Khuyến nghị chạy bằng Visual Studio Code**

