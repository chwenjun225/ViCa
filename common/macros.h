/*
GIẢI THÍCH TỆP CODE

macros.h is often used to define macros, which are constants 
or simple reusable functions that can make the code cleaner, more 
readable, and easier to maintain. These might include things like:

- Assertions for Debugging: For instance, ASSERT() macros to 
check if conditions are met during runtime, which you've 
already used in your code for checking object states in MemPool.

- Compiler Hints: Macros like LIKELY() or UNLIKELY() that hint to 
the compiler about how often a condition might be true. This can 
help with optimizations, especially in low-latency or 
high-performance applications.

Platform-Specific Code: To ensure compatibility across different 
systems, macros can conditionally include code based on the 
operating system or compiler being used.


Giải thích đoạn code:
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

Đoạn mã này sử dụng các macro `LIKELY` và `UNLIKELY` để thông 
báo cho trình biên dịch biết rằng một biểu thức có khả năng xảy
ra hay không. Đây là kỹ thuật tối ưu hóa thông thường được sử 
dụng trong lập trình C/C++ nhằm giúp cải thiện hiệu suất của 
ứng dụng. 

Giải thích:
1. `__builtin_expect`: Đây là một hàm đặc biệt do GCC (GNU 
Compiler Collection) cung cấp. Nó cho phép lập trình viên thông 
báo cho trình biên dịch về "dự đoán" của họ về một biểu thức có 
thể xảy ra hay không, giúp tối ưu hóa mã máy mà trình biên dịch 
tạo ra.
- `!!(x)`: Biểu thức này đảm bảo rằng giá trị của `x` sẽ được 
chuyển đổi về dạng boolean (0 hoặc 1), bởi vì `x` có thể là bất
kỳ giá trị nào và chúng ta cần một giá trị rõ rằng là `true` (1) 
hoặc `false` (0).
- `1` và `0`: Đây là các tham số chỉ định rằng bạn dự đoán biểu 
thức có khả năng xảy ra (1) hoặc (0). Trình biên dịch sẽ tối ưu 
hóa mã cho tình huống này bằng các tối ưu hóa cấu trúc nhánh 
(branch prediction).

2. `LIKELY(x)`: chỉ ra rằng biểu thức `x` có khả năng xảy ra 
cao (đúng nhiều hơn sai). 
- Ví dụ: `LIKELY(x == 0)` nói rằng việc `x` bằng `0` là khả năng 
cao.

3. `UNLIKELY(x)`: Chỉ ra rằng biểu thức `x` có khả năng xảy ra 
thấp (sai nhiều hơn đúng).
- Ví dụ: `UNLIKELY(x == 10)` nói rằng việc `x` bằng 10 là khả 
năng thấp. 

Ví dụ về cách sử dụng 
```
if (LIKELY(x == 0)) {
    // Nếu x=0, thực hiện một hành động nhanh chóng 
}
else {
    // Nếu x!=0, thực hiện một hành động khác
}
```

Lợi ích: 
- Tối ưu hóa hiệu suất: Trình biên dịch có thể tối ưu hóa mã máy 
và cả thiện hiệu suất nếu biết rằng một nhanh trong mã có thể được 
chọn nhiều hơn (hoặc ít hơn). 
- Branch Prediction: Khi chương trình chạy trên CPU, CPU có thể dự 
đoán nhánh nào sẽ được thực hiện tiếp theo, giúp giảm thiểu thời 
gian dừng lại khi phân nhánh. 

Tổng kết:
Đoạn mã này là một kỹ thuật tối ưu hóa để giúp trình biên dịch hiểu
rõ hơn về cách các điều kiện trong mã của bạn có thể xảy ra, giúp 
tăng hiệu suất, đặc biệt là trong các ứng dụng yêu cầu độ trễ thấp 
như trong lập trình hệ thống hay các ứng dụng thực thi real-time. 




Giải thích đoạn code:
1. `ASSERT` function
```cpp
inline auto ASSERT(bool cond, const std::string &msg) noexcept {
    if (UNLIKELY(!cond)) {
        std::cerr << "ASSERT : " << msg << std::endl; 
        exit(EXIT_FAILURE); 
    }
}
```
Đoạn mã này chứa hai hàm `ASSERT` và `FATAL`, đây là các hàm dùng để 
kiểm tra điều kiện xử lý lỗi trong các ứng dụng C++.

1. `ASSERT`:
- Mục đích: Hàm này dùng để kiểm tra điều kiện `cond`. Nếu điều kiện 
không đúng (tức là `cond` bằng `false`), nó sẽ in ra thông báo và 
dừng chương trình với mã thoát `EXIT_FAILURE`. 

* Chi tiết:
- `UNLIKELY(!cond)`: Sử dụng `UNLIKELY` (như đã giải thích ở trên) để
cho tình biên dịch biết rằng điều kiện này ít khi xảy ra. Điều này 
giúp tối ưu hóa hiệu suất trong trường hợp điều kiện sai. 
- `std::cerr`: In thông báo lỗi ra `stderr` (dòng lỗi tiêu chuẩn). 
- `exit(EXIT_FAILURE)`: Dừng chương trình và trả về mã thoát 
`EXIT_FAILURE`, chỉ ra rằng chương trình kết thúc với lỗi. 

2. `FATAL` function
```cpp
inline auto FATAL(const std::string &msg) noexcept {
    std::cerr << "FATAL : " << msg << std::endl; 
    exit(EXIT_FAILURE); 
}
```
- Mục đích: Hàm này giống như `ASSERT`, nhưng nó không kiểm tra một 
điều kiện cụ thể mà chỉ in thông báo lỗi ra và dừng chương trình. 
- Chi tiết: Hàm này luôn in thông báo với tiền tố `FATAL` và sau đó
dừng chương trình giống như hàm `ASSERT`. 

* Điểm chung:
- `noexcept`: Cả hai hàm đều được khai báo là `noexcept`, có nghĩa 
là chúng không ném ra ngoại lệ. Điều này giúp trình biên dịch tối ưu 
mã và làm rõ ràng rằng các này không gây ra lỗi ngoài kiểm tra điều 
kiện của chúng. 
- `exit(EXIT_FAILURE)`: Cả hai hàm đều kết thúc chương trình nếu điều 
kiện sai hoặc có lỗi, đảm bảo rằng ứng dụng không tiếp tục thực thi 
trong trạng thái không hợp lệ.
* Ứng dụng: 
- `ASSERT`: Thường được sử dụng để kiểm tra các điều kiện mà bạn kỳ 
vọng là đúng trong quá trình phát triển (ví dụ: kiểm tra các giá trị
đầu vào, trạng thái của đối tượng). Nếu điều kiện sai, ứng dụng sẽ 
dừng lại ngay lập tức để tránh các lỗi nghiêm trọng. 
- `FATAL`: Thường dùng trong các tình huống mà bạn không thể tiếp tục 
chương trình vì lỗi nghiêm trọng (chẳng hạn như lỗi cấu hình, dữ liệu
không hợp lệ,...).

* Tổng kết: Cả hai hàm đều có mục đích giúp ứng dụng dễ dàng phát hiện 
lỗi và ngừng hoạt động khi có vấn đề xảy ra, giúp bạn tránh được các 
lỗi khó phát hiện và gây ra sự cố trong suốt vòng đời của ứng dụng. 
*/

#pragma once 
#include <cstring>
#include <iostream>

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

inline auto ASSERT(bool cond, const std::string &msg) noexcept {
    if (UNLIKELY(!cond)) {
        std::cerr << "ASSERT : " << msg << std::endl; 
        exit(EXIT_FAILURE); 
    }
}

inline auto FATAL(const std::string &msg) noexcept {
    std::cerr << "FATAL : " << msg << std::endl; 
    exit(EXIT_FAILURE); 
}
