# Random Forest and effect of Hyperparameters on result

| Tên thành viên    | MSSV     |
| ----------------- | -------- |
| Văn Viết Hiếu Anh | 19521225 |
| Văn Viết Nhật     | 19521958 |
| Lê Văn Phước      | 19522054 |

# Giới thiệu về Random Forest Classifier
Random Forest Classifier là mô hình sử dụng để giải quyết bài toán classification. Nó sử dụng thuật toán Random forest, đây là thuật toán supervised learning trong máy học.
## Giới thiệu về thuật toán Random Forest
Random là ngẫu nhiên, Forest là rừng, nên ở thuật toán Random Forest sẽ xây dựng nhiều cây quyết định bằng thuật toán Decision Tree, tuy nhiên mỗi cây quyết định sẽ khác nhau (có yếu tố random). Sau đó kết quả dự đoán được tổng hợp từ các cây quyết định.
Đối với bài toán này thì đầu ra của thuật toán Random Forest là loại được chọn bởi hầu hết các cây.

Về mặt kỹ thuật, thuật toán Random Forest là một phương pháp tổng hợp (dựa trên cách tiếp cận chia để trị) của các cây quyết định được tạo trên một tập dữ liệu được phân chia ngẫu nhiên. Tập hợp các bộ phân loại cây quyết định này còn được gọi là rừng. Các cây quyết định riêng lẻ được tạo bằng cách sử dụng một chỉ báo lựa chọn thuộc tính như mức tăng thông tin, tỷ lệ tăng cho mỗi thuộc tính. Mỗi cây phụ thuộc vào một mẫu ngẫu nhiên độc lập. Trong một bài toán phân loại, mỗi cây bình chọn và lớp phổ biến nhất được chọn làm kết quả cuối cùng. Nó đơn giản hơn và mạnh mẽ hơn so với các thuật toán phân loại phi tuyến tính khác.

## Thuật toán

### Decision Tree Learning

Decision Tree là một phương pháp phổ biến cho các tác vụ học máy khác nhau. Đặc biệt, những cây được trồng rất sâu có xu hướng học các kiểu hình bất thường cao, chúng trang bị quá nhiều bộ huấn luyện của mình, tức là có độ lệch thấp, nhưng phương sai rất cao. Random Forest là một cách lấy trung bình nhiều cây quyết định sâu, được huấn luyện trên các phần khác nhau của cùng một tập huấn luyện, với mục tiêu giảm phương sai. Điều này phải trả giá bằng một sự gia tăng nhỏ trong độ chệch và một số mất khả năng diễn giải, nhưng nói chung là tăng đáng kể hiệu suất trong mô hình cuối cùng.

Forest giống như sự kết hợp của các nỗ lực thuật toán Decision Tree. Thực hiện việc làm việc theo nhóm của nhiều cây do đó cải thiện hiệu suất của một cây ngẫu nhiên duy nhất. Mặc dù không hoàn toàn giống nhau, nhưng các khu rừng mang lại hiệu quả của xác thực chéo gấp K lần.

### Bootstrap Aggregating

Thuật toán đào tạo cho các khu rừng ngẫu nhiên áp dụng kỹ thuật tổng hợp bootstrap hay còn gọi là random sampling with replacement. Tức khi mình sample được 1 dữ liệu thì mình không bỏ dữ liệu đấy ra mà vẫn giữ lại trong tập dữ liệu ban đầu, rồi tiếp tục sample cho tới khi sample đủ n dữ liệu. Khi dùng kĩ thuật này thì tập n dữ liệu mới của mình có thể có những dữ liệu bị trùng nhau.

Cho một tập huấn luyện X = <img src="https://render.githubusercontent.com/render/math?math=x_{1}">, ..., <img src="https://render.githubusercontent.com/render/math?math=x_{n}"> với các phản hồi Y = <img src="https://render.githubusercontent.com/render/math?math=y_{1}">, ..., <img src="https://render.githubusercontent.com/render/math?math=y_{n}">, đóng gói lặp đi lặp lại (B lần) chọn một mẫu ngẫu nhiên thay thế tập huấn luyện và lắp các cây vào các mẫu:

Đối với b = 1, ..., B:

1. Ví dụ huấn luyện mẫu, với thay thế n từ X, Y; gọi chúng là <img src="https://render.githubusercontent.com/render/math?math=X_{b}">, <img src="https://render.githubusercontent.com/render/math?math=Y_{b}">.
2. Huấn luyện cây phân loại hoặc hồi quy <img src="https://render.githubusercontent.com/render/math?math=f_{b}"> trên <img src="https://render.githubusercontent.com/render/math?math=X_{b}">, <img src="https://render.githubusercontent.com/render/math?math=Y_{b}">.

Sau khi huấn luyện, có thể thực hiện dự đoán cho các mẫu chưa nhìn thấy x' bằng cách lấy đa số phiếu của các cây:

<img src="https://render.githubusercontent.com/render/math?math=\hat{f} = \frac{1}{B} \sum_{b=1}^{B} f_{b} (x')">

Quy trình khởi động này dẫn đến hiệu suất mô hình tốt hơn vì nó làm giảm phương sai của mô hình mà không làm tăng độ chệch. Chỉ cần huấn luyện nhiều cây trên một tập huấn luyện duy nhất sẽ cho các cây có tương quan chặt chẽ, lấy mẫu bootstrap là một cách khử tương quan giữa các cây bằng cách hiển thị cho chúng các tập huấn luyện khác nhau. 

Ngoài ra, ước tính về độ không chắc chắn của dự đoán có thể được thực hiện dưới dạng độ lệch chuẩn của các dự đoán từ tất cả các cây hồi quy riêng lẻ trên x ':

<img src="https://render.githubusercontent.com/render/math?math=\sigma = \sqrt{\frac{\sum_{b=1}^{B}(f_{b}(x') - \hat{f})^{2}}{B - 1}}">

Trong đó: 
+ B là số cây.
+ <img src="https://render.githubusercontent.com/render/math?math=x_{i}"> là số lượng mẫu huấn luyện

# Hyperparameter of RandomForestClassifer sklearn

+ n_estimators: Số lượng cây trong rừng.
+ criterion: Chức năng đo lường chất lượng của một lần tách
+ max_depth: Chiều sâu tối đa của cây
+ min_samples_split: Số lượng mẫu tối thiểu cần thiết để tách một nút nội bộ
+ min_samples_leaf: Số lượng mẫu tối thiểu cần thiết để có ở một nút lá
+ min_weight_fraction_leaf: Phần có trọng số tối thiểu của tổng trọng số (của tất cả các mẫu đầu vào) cần thiết để ở một nút lá
+ max_features:  Số lượng các tính năng cần xem xét khi tìm kiếm sự phân chia tốt nhất
+ bootstrap: Các mẫu bootstrap có được sử dụng khi xây dựng cây hay không

'''python
    
    from sklearn.model_selection import RandomizedSearchCV
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    
    # The function to measure the quality of a split
    criterion = ['gini', 'entropy']

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'criterion': criterion,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    print(random_grid)

'''
