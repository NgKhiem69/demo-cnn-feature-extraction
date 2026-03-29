import streamlit as st
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


# Tải mô hình VGG16 (include_top=False để bỏ lớp phân loại, chỉ lấy phần trích chọn đặc trưng)
@st.cache_resource
def load_model():
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    return model


model = load_model()

st.title("Demo Trích Chọn Đặc Trưng CNN")
st.write("Ứng dụng sử dụng mô hình VGG16 để chuyển đổi hình ảnh thành vector đặc trưng.")

# Upload ảnh
uploaded_file = st.file_uploader("Tải một bức ảnh lên...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh đầu vào
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đầu vào', use_column_width=True)

    # Tiền xử lý ảnh cho VGG16 (đưa về kích thước 224x224)
    img_resized = image.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    img_processed = preprocess_input(img_array)

    # Trích xuất đặc trưng
    st.write("Đang tính toán vector đặc trưng...")
    features = model.predict(img_processed)

    st.success("Trích xuất thành công!")

    # In ra kết quả
    st.write("**Kích thước của vector đặc trưng:**", features.shape)
    st.write(f"Mô hình đã nén bức ảnh thành một vector có {features.shape[1]} chiều.")

    # Hiển thị 10 giá trị đầu tiên của vector để demo
    st.write("**Một số giá trị đầu tiên trong vector:**")
    st.write(features[0][:10])