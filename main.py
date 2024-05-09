import base64
from flask import Flask, request, jsonify, render_template
import cv2
import requests
from ultralytics import YOLO
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json

app = Flask(__name__)

# YOLO 모델 초기화
model = YOLO("model\clothesDetectModel_pretrained_false_30.pt")

cred = credentials.Certificate('firebase_key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
uid = 'q9u1gUypngbpZbKny5vwqDwm6qT2'
doc_id = '20240509'

@app.route('/index')
def index():
  return render_template('index.html')

@app.route('/detect_and_analyze')
def detect_and_analyze_color():
    doc_ref = db.collection("q9u1gUypngbpZbKny5vwqDwm6qT2").document(doc_id)
    doc = doc_ref.get()
    if doc.exists:
       data = doc.to_dict()
       image_url = data.get("imgURL")

    #image_url = storage.child("11283.jpg").get_url(None)
    print(image_url)

    # 이미지를 다운로드하여 바이트 데이터로 변환
    response = requests.get(image_url)
    image = response.content
    
     # 이미지를 OpenCV 형식으로 디코딩
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
    # 이미지 로드에 실패했을 경우 오류 처리
      print("이미지를 로드할 수 없습니다.")

    # YOLO 모델을 사용하여 객체 감지 수행
    results = model.predict(img)

    # 객체 감지 결과 가져오기
    detections = results[0]

    # 가장 확률이 높은 박스 식별
    highest_prob_box = None
    highest_confidence = -1
    for box in detections.boxes:
        if box.conf > highest_confidence:
            highest_prob_box = box
            highest_confidence = box.conf
    idx2class = np.array([value for key,value in results[0].names.items()])
    object_class = idx2class[highest_prob_box.cls.to('cpu').numpy().astype('int')][0]

    # 박스 좌표 얻기
    x1, y1, x2, y2 = highest_prob_box.xyxy[0]

    # 이미지에서 객체 영역 자르기
    cropped_image = img[int(y1):int(y2), int(x1):int(x2)]

    # 객체 영역에서 주요 색상 추출
    main_color = np.average(cropped_image, axis=(0,1))
    main_color_list = main_color.tolist()

    # 주요 색상을 표시하는 이미지 생성
    img_temp = cropped_image.copy()
    img_temp[:, :, 0], img_temp[:, :, 1], img_temp[:, :, 2] = main_color
    _, encoded_image = cv2.imencode('.jpg', img_temp)
    main_color_image_base64 = base64.b64encode(encoded_image).decode('utf-8')

    # 미리 정의된 색상 카테고리와 그에 해당하는 RGB 값
    color_categories = {
        '검정색': [0, 0, 0],
        '하얀색': [255, 255, 255],
        '회색': [128, 128, 128],
        '빨간색': [255, 0, 0],
        '핑크색': [255, 192, 203],
        '주황색': [255, 165, 0],
        '베이지': [245, 245, 220],
        '갈색': [153, 56, 0],
        '노랑색': [255, 255, 0],
        '초록색': [29, 219, 22],
        '카키색': [71, 102, 0],
        '민트색': [189, 252, 201],
        '파란색': [0, 0, 255],
        '남색': [0, 0, 128],
        '하늘색': [135, 206, 235],
        '보라색': [128, 0, 128],
        '라벤더': [230, 230, 250],
        '와인색': [114, 47, 55],
        '네온': [183, 255, 0],
        '금색': [191, 155, 48],
    }

    # 주어진 RGB 값과 각 색상 카테고리의 거리 계산
    distances = {color: np.linalg.norm(np.array(main_color_list) - np.array(color_value)) for color, color_value in color_categories.items()}

    # 거리가 가장 작은 색상 카테고리 선택
    closest_color = min(distances, key=distances.get)

    data = {
       'clothes' : object_class,
       'closet_color_category': closest_color,
       'closet_color_RGB': main_color_list,
    }

    doc_ref.update(data)

    # 결과 반환
    return jsonify({
        'object_class': object_class,
        'closest_color_category': closest_color,
        'main_color_image': main_color_image_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
