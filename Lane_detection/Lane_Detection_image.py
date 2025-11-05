import cv2
import numpy as np

# 관심 영역 좌표 저장 리스트
points = []

# 마우스 클릭 이벤트 콜백 함수
def select_points(event, x, y, flags, param):
    global points, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭 시
        points.append((x, y))  # 좌표 저장
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)  # 선택한 점 표시
        cv2.imshow("Select ROI", image_copy)

        if len(points) == 4:  # 4개 점이 선택되면 관심 영역 적용
            apply_roi()


def get_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    b1 = y1 - m1 * x1
    m2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float('inf')
    b2 = y3 - m2 * x3
    
    if m1 == m2:
        return None  # 평행한 경우
    elif m1 == float('inf'):
        x = x1
        y = m2 * x + b2
    elif m2 == float('inf'):
        x = x3
        y = m1 * x + b1
    else:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    
    return int(x), int(y)

def draw_filtered_line(image, line, intersection):
    """
    교차점(intersection) 위로는 선을 그리지 않도록 조정하는 함수
    """
    x1, y1 = line[0]
    x2, y2 = line[1]

    # 만약 두 점이 모두 교차점 아래(y 값이 큼)라면 그대로 그림
    if y1 >= intersection[1] and y2 >= intersection[1]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 만약 한 점만 교차점 위(y 값이 작음)라면, 교차점에서 끊어서 그림
    elif y1 < intersection[1] and y2 >= intersection[1]:
        new_x1 = int(x1 + (x2 - x1) * ((intersection[1] - y1) / (y2 - y1)))
        new_y1 = intersection[1]
        cv2.line(image, (new_x1, new_y1), (x2, y2), (0, 255, 0), 3)

    elif y2 < intersection[1] and y1 >= intersection[1]:
        new_x2 = int(x2 + (x1 - x2) * ((intersection[1] - y2) / (y1 - y2)))
        new_y2 = intersection[1]
        cv2.line(image, (x1, y1), (new_x2, new_y2), (0, 255, 0), 3)

    
def apply_roi():
    global points, image

    cv2.polylines(image, [np.array(points, np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow("masked_edges", masked_edges)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=100)
    
    min_angle = np.pi / 6
    left_lines, right_lines = [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            if abs(angle) > min_angle:
                if (x1 + x2) / 2 < image.shape[1] / 2:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])
    
    global left_line, right_line, intersection
    
    left_line=None
    right_line=None
    
    
    if left_lines:
        left_lines.sort(key=lambda line: min(line[0], line[2]))
        x1, y1, x2, y2 = left_lines[0]
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        y1_new = int(y1 - slope * x1)
        y2_new = int(y2 + slope * (image.shape[1] - x2))
        left_line = (0, y1_new), (image.shape[1], y2_new)
        #cv2.line(image, left_line[0], left_line[1], (0, 255, 0), 3)
    
    if right_lines:
        right_lines.sort(key=lambda line: max(line[0], line[2]))
        x1, y1, x2, y2 = right_lines[0]
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        y1_new = int(y1 - slope * x1)
        y2_new = int(y2 + slope * (image.shape[1] - x2))
        right_line = (0, y1_new), (image.shape[1], y2_new)
        #cv2.line(image, right_line[0], right_line[1], (0, 255, 0), 3)
    
    if left_line and right_line:
        
        intersection = get_intersection(left_line[0] + left_line[1], right_line[0] + right_line[1])
        
        if intersection:
            cv2.circle(image, intersection, 5, (255, 0, 0), -1)
            if left_line and intersection:
                draw_filtered_line(image, left_line, intersection)

            if right_line and intersection:
                draw_filtered_line(image, right_line, intersection)
            
        else:
            print("교차점 없음 (직선이 평행함)")
    else:
        print("검출된 선이 부족하여 교차점을 찾을 수 없음")
    
    
    cv2.imshow("Lane Detection", image)


image = cv2.imread('img.png')
image_copy = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

cv2.imshow("Select ROI", image)
cv2.setMouseCallback("Select ROI", select_points)
cv2.waitKey(0)
cv2.destroyAllWindows()
