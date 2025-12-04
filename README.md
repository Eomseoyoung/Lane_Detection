> # Lane_Detection with ROI Selection (OpenCV, Python)
>
> 
>
> 본 프로젝트는 마우스로 선택한 ROI 영역에서 차선을 검출하고, 좌·우 대표 차선을 계산한 뒤 교차점 기준 아래 부분만 출력하는 OpenCV 기반 차선 검출 코드입니다.


> ### 주요 기능
>
> * 마우스 클릭으로 4점 ROI 폴리곤 선택
>
> * ROI 내부만 대상으로 Edge + HoughLines 기반 차선 검출
>
> * 좌/우 차선 대표 선 추출 및 직선 확장
>
> * 두 직선의 교차점(vanishing point) 계산
>
> * 교차점 위 부분은 자동 제거 (도로 표면만 유지)
>
> * 최종 차선을 이미지로 출력


> ### 폴더 구조
>     Lane_detection/
>     │
>     ├── Lane_Detection_image.py   # 이미지 데이터용 실행파일
>     ├── Lane_Detection_video.py   # 영상 데이터용 실행파일
>     └── Lane_Detection_Test_Image.png    # 예시 이미지


