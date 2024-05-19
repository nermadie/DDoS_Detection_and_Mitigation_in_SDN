# Phát hiện và giảm thiểu tấn công DDoS bằng Machine Learning trong SDN

## Thành viên




| **Tên**              | **Lớp**               | **Mã sinh viên**          |
|-------------------    |-------------------------|--------------------------|
| **Lê Phạm Công**    | 20KTMT1                 | 106200221                |
| **Phan Công Danh**       | 20KTMT1                 | 106200222                |


---
## Link video demo (Youtube): https://youtu.be/hJq-6aTnBpk


## Tổng quan
Dự án này triển khai giải pháp Phát hiện và giảm thiểu DDoS bằng cách sử dụng Bộ điều khiển Mininet và Ryu. Mục đích là mô phỏng môi trường mạng trong các trường hợp thông thường và trong trường hợp xảy ra các cuộc tấn công DDoS để kiểm tra tính hiệu quả của mô hình Cây quyết định trong việc phát hiện DDoS và hiệu suất của mô-đun giảm nhẹ.

## Cấu trúc thư mục

**REPORT Folder**: Mã nguồn Latex, file báo cáo (pdf) và file slide thuyết trình (pdf).

**RESULT Folder**: Các kết quả thu được dạng ảnh

**SOURCE Folder**
- **CONTROLLER**
+ ***collect_ddos_traffic.py**: Mã python controller đê thu thập data DDoS
+ **collect_normal_traffic.py**: Mã python controller để thu thập data bình thường
+ **DT_controller.py**: Controller dựa trên RYU
+ **mitigation_module.py**: Module giảm thiểu (Bao gồm phát hiện và giảm thiểu DDoS).
+ **no_mitigation_module.py**: Module chỉ phát hiện không bao gồm giảm thiểu
- **MININET**:
+ **client.csh**: Mã Script để thực hiện lệnh request đến http server
+ **ddos.csh**: Mã Script thực hiện lệnh tấn công ICMP
+ **iperf.csh**: Mã Script thực hiện kiểm tra băng thông của Iperf server
+ **ping.csh**: Mã Script thực hiện lệnh ping đến server
+ **server.csh**: Mã Script thực hiện khởi tạo Iperf server
: **topology.py**: Khởi tạo mạng trong môi trường mininet
+ **generate_ddos_traffic.py**: Giả lập lưu lượng tấn công DDoS
+ **generate_normal_traffic.py**: Giả lập lưu lượng bình thường
- **MACHINELEARNING**:
+ **DT.py**: source thực hiện train và kiểm tra model
+ **model.pkl**: trọng số của model
 
 
 

## Yêu cầu

- Python 3.10.x
- Mininet 2.3.0
- Ryu Controller 4.34
- Linux Ubuntu 22.04LTS

## Cài đặt

1. Clone repository:
    ```bash
    git clone https://github.com/lephamcong/DDoS_Detection_and_Mitigation_in_SDN.git
    ```
    ```
    cd SDN_LoadBalancing
    ```


2. Thiết lập Mininet và Ryu theo hướng dẫn cài đặt tương ứng:
    - [Mininet Installation Guide](http://mininet.org/download/)
    - [Ryu Controller Installation Guide](https://ryu.readthedocs.io/en/latest/getting_started.html)

3. Dataset
    - [Link Dataset](https://drive.google.com/file/d/1Ll9zG8IB4gWwevBKIiB9lse6BvnSYVQ_/view)
## Cách sử dụng

### Thu thập dữ liệu 
Mở 2 terminal, terminal 1 cd đến thư mục controller, terminal 2 cd đến thư mục mininet


#### Thu thập dữ liệu bình thường

1. Terminal 1:
    ```bash
    ryu-manager collect_normal_traffic.py
    ```

2. Terminal 2:
    Thay đổi địa chỉ IP controller tương ứng với thiết bị chạy controller
    ```bash
    sudo python generate_normal_traffic.py
    ```

#### Thu thập dữ liệu DDoS

1. terminal 1:
    ```bash
    ryu-manager collect_ddos_traffic.py
    ```

2. Terminal 2:
    Thay đổi địa chỉ IP controller tương ứng với thiết bị chạy controller
    ```bash
    sudo python generate_ddos_traffic.py
    ```
    

### Train và đánh giá mô hình Decision Tree

#### Train và đánh giá

1. Mở 1 terminal, chuyển đến thư mục machinelearning:
    ```bash
    python DT.py
    ```

2. Load trọng số model đã train:
    ```bash
    import pickle
    with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
    ```

### Phát hiện và giảm thiểu tấn công DDoS bằng Machine Learning
Mở 2 terminal, terminal 1 cd đến thư mục controller, terminal 2 cd đến thư mục mininet
1. Terminal 1:
    Chạy DT_controller.py với module mitigation bằng cách import mitigation_module.py vào DT_controller.py
    ```bash
    ryu-manager DT_controller.py
    ```

2. Terminal 2:
    Thay đổi địa chỉ IP controller tương ứng với thiết bị chạy controller
    ```bash
    sudo python topology.py
    ```
3. Terminal 2:
    Trong môi trường mininet, xterm vào host tùy chọn và thực thi các file Script (đuôi .csh) như trong thư mục mininet.



## Liên hệ
- Lê Phạm Công
- Email: [lpc051002@gmail.com](mailto:lpc051002@gmail.com).
