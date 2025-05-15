import streamlit as st
from PIL import Image
import pickle

from transformers import AutoTokenizer

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from models.EfficientNetV2_Transformer.E_T_image_caption_model import Imagecaptionmodel, position_encoding
from utils.E_T_utils.image_utils import extract_image_feature
from utils.E_T_utils.model_utils import load_model_and_vocabulary
from utils.E_T_utils.caption_utils import generate_caption


from torchvision.transforms import Compose, Resize, ToTensor

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

session_state = st.session_state
# Streamlit app layout
st.set_page_config(
    page_title="Image Captioning",
)

# E_T
@st.cache_resource
def load_E_T_model():
    E_T_MODEL_PATH = './models/EfficientNetV2_Transformer/Bestmodel.pth'
    E_T_VOCAB_PATH = './models/EfficientNetV2_Transformer/vocabulary_data.pkl'
    return load_model_and_vocabulary(E_T_MODEL_PATH, E_T_VOCAB_PATH)


E_T_model, word_to_idx, idx_to_word, start_token, pad_token = load_E_T_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_size = 1280
max_sequence_len = 35

model_img = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT).to(device)
model_img.eval()
model_img_layer_4 = model_img._modules.get('features')

#S_B
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
encoder_model_name = "microsoft/swin-large-patch4-window12-384-in22k"
decoder_model_name = "vinai/bartpho-syllable"

S_B_model = VisionEncoderDecoderModel.from_pretrained(
    'Skyler215/Swin_Syllable_30_11'
)

image_processor =  ViTImageProcessor.from_pretrained(encoder_model_name)
tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

def S_B_generate(img):
    img = Image.open(img)
    pixel_values = image_processor(img, return_tensors="pt").pixel_values
    output = S_B_model.generate(
        pixel_values,
        max_length=23,
        num_beams=5,
        temperature=0.1,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result


def compute_metrics_with_pycocoevalcap(hypotheses, ground_truths):
    """
    Tính toán các metric BLEU, ROUGE, CIDEr, METEOR, SPICE bằng pycocoevalcap.
    :param hypotheses: list, caption được tạo (mỗi ảnh một caption).
    :param ground_truths: list of list, các ground truth captions (mỗi ảnh có nhiều ground truth captions).
    :return: dict, các giá trị của các metric.
    """
    # Đảm bảo ground_truths là list of list
    if not all(isinstance(g, list) for g in ground_truths):
        ground_truths = [[g] if isinstance(g, str) else g for g in ground_truths]

    # Chuẩn bị dữ liệu cho pycocoevalcap
    gts_dict = {i: ground_truths[i] for i in range(len(ground_truths))}
    res_dict = {i: [hypotheses[i]] for i in range(len(hypotheses))}
    
    # Kiểm tra tính hợp lệ của keys
    assert gts_dict.keys() == res_dict.keys(), "Keys của ground truths và hypotheses không khớp."

    # Khởi tạo các scorer
    bleu_scorer = Bleu(n=4)
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    meteor_scorer = Meteor()
    spice_scorer = Spice()
    
    # Tính BLEU
    bleu, _ = bleu_scorer.compute_score(gts=gts_dict, res=res_dict)
    # Tính ROUGE
    rouge, _ = rouge_scorer.compute_score(gts=gts_dict, res=res_dict)
    # Tính CIDEr
    # cider, _ = cider_scorer.compute_score(gts=gts_dict, res=res_dict)
    # Tính METEOR
    # meteor, _ = meteor_scorer.compute_score(gts=gts_dict, res=res_dict)
    # Tính SPICE
    # spice, _ = spice_scorer.compute_score(gts=gts_dict, res=res_dict)
    
    # Trả về kết quả dưới dạng dictionary
    return {
        "BLEU-1": bleu[0],
        "BLEU-2": bleu[1],
        "BLEU-3": bleu[2],
        "BLEU-4": bleu[3],
        "ROUGE-L": rouge,
        # "CIDEr": cider,
        # "METEOR": meteor,
        # "SPICE": spice
    }

# Sidebar menu
menu = st.sidebar.radio("Chọn tính năng", ["Tạo Caption", "Lịch sử Generate"])

if menu == "Tạo Caption":
    st.title('Image Captioning')

    image = st.file_uploader("Tải ảnh của bạn lên tại đây!", type=["jpg", "jpeg", "png", "webp"])

    if image:
        st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)
        image_mode = st.radio("Chọn chế độ:", ["Ảnh có ground truth", "Ảnh không có ground truth"])

        if image_mode == "Ảnh có ground truth":
            ground_truths = st.text_area(
                "Nhập các ground truth captions, mỗi caption trên một dòng:",
                height=150
            )

    selected_options = st.multiselect('Chọn Model:', ['Efficient v2 + Transformer', 'Swin+Bart'])
    query_button = st.button("Tạo caption")

    if query_button:
        if image is None:
            st.warning("Ảnh chưa được thêm vào")
        elif not selected_options:
            st.warning("Chưa chọn model")
        else:
            captions = {}
            for option in selected_options:
                if option == 'Efficient v2 + Transformer':
                    with st.spinner("Đang tạo caption với EfficientNet v2 + Transformer..."):
                        img_emb = extract_image_feature(model_img_layer_4, image, device)
                        caption1 = generate_caption(E_T_model, word_to_idx, idx_to_word, img_emb, max_sequence_len, start_token, 15, pad_token, device)
                        captions['Efficient v2 + Transformer'] = caption1
                        st.write(f"EfficientNet v2 + Transformer: {caption1} .")
                
                elif option == "Swin+Bart":
                    with st.spinner("Đang tạo caption với Swin+Bart..."):
                        caption2 = S_B_generate(image)
                        captions['Swin+Bart'] = caption2
                        st.write(f"Swin+Bart: {caption2}")

            if image_mode == "Ảnh có ground truth":
                if not ground_truths.strip():
                    st.warning("Vui lòng nhập ground truth captions.")
                else:
                    ground_truth_list = ground_truths.split("\n")
                    with st.spinner("Đang tính score cho EfficientNet v2 + Transformer..."):
                        scores_efficient = compute_metrics_with_pycocoevalcap(
                            [captions['Efficient v2 + Transformer']], [ground_truth_list]
                        )
                    
                    with st.spinner("Đang tính score cho Swin+Bart..."):
                        scores_Swin_Bart = compute_metrics_with_pycocoevalcap(
                            [captions['Swin+Bart']], [ground_truth_list]
                        )
                    
                    st.subheader("Kết quả đánh giá:")
                    st.write("**EfficientNet v2 + Transformer:**")
                    st.json(scores_efficient)
                    st.write("**Swin+Bart:**")
                    st.json(scores_Swin_Bart)


            if 'history' not in session_state:
                session_state.history = []
            session_state.history.append({
                "image": image,
                "captions": captions,
                "ground_truths": ground_truths if image_mode == "Ảnh có ground truth" else None,
                "scores": {
                    "EfficientNet v2 + Transformer": scores_efficient if image_mode == "Ảnh có ground truth" else None,
                    "Swin + Bart": scores_Swin_Bart if image_mode == "Ảnh có ground truth" else None
                }
            })



elif menu == "Lịch sử Generate":
    st.title("Lịch sử Generate")

    def display_history():
        if 'history' in session_state and session_state.history:
            for idx, record in enumerate(session_state.history):
                st.image(record['image'], caption=f"Ảnh {idx + 1}", use_container_width=True)
                st.write("Captions:")
                for model_name, caption in record['captions'].items():
                    st.write(f"- {model_name}: {caption}")
                
                if record['ground_truths']:
                    st.write("Ground Truths:")
                    st.write(record['ground_truths'].replace("\n", "<br>"), unsafe_allow_html=True)

                # Hiển thị scores chỉ khi có scores
                if 'scores' in record and record['scores']:
                    st.write("Scores:")
                    for model_name, score in record['scores'].items():
                        st.write(f"**{model_name}:**")
                        st.json(score)
        else:
            st.write("Chưa có lịch sử nào.")

    display_history()
