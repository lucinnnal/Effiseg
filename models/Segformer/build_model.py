from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import sys
import os

# 현재 파일 위치에서 두 단계 위로 올라가서 Effiseg 루트 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from data.dataset import CityScapes_Testdataset

def build_segformer(variant='b0', pretrained=False):
    """
    Segformer 모델을 생성하는 함수
    
    Args:
        variant (str): 모델 변형 ('b0', 'b1', 'b2', 'b3', 'b4', 'b5')
        pretrained (bool): Cityscapes에 대해 사전학습된 모델을 사용할지 여부
    """
    if pretrained:
        model_name = f"nvidia/segformer-{variant}-finetuned-cityscapes-1024-1024"
    else:
        model_name = f"nvidia/mit-{variant}"
    
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    
    return feature_extractor, model

if __name__ == '__main__':
    # B3는 사전학습된 모델 사용
    feature_extractor_b3, model_b3 = build_segformer('b3', pretrained=True)
    
    # B0는 기본 모델 사용
    feature_extractor_b0, model_b0 = build_segformer('b0', pretrained=False)

    breakpoint()