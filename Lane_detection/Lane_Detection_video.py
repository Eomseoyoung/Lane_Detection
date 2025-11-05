import cv2
import numpy as np

# 관심 영역 좌표 저장 리스트
points = []
roi_selected = False
prev_left_line = None
prev_right_line = None

def select_points(event, x, y, flags, param):
    global points, frame_copy, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select ROI", frame_copy)

        if len(points) == 4:
            roi_selected = True

def get_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    b1 = y1 - m1 * x1
    m2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float('inf')
    b2 = y3 - m2 * x3
    
    if m1 == m2:
        return None
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
    x1, y1 = line[0]
    x2, y2 = line[1]

    if y1 >= intersection[1] and y2 >= intersection[1]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    elif y1 < intersection[1] and y2 >= intersection[1]:
        new_x1 = int(x1 + (x2 - x1) * ((intersection[1] - y1) / (y2 - y1)))
        new_y1 = intersection[1]
        cv2.line(image, (new_x1, new_y1), (x2, y2), (0, 255, 0), 3)
    elif y2 < intersection[1] and y1 >= intersection[1]:
        new_x2 = int(x2 + (x1 - x2) * ((intersection[1] - y2) / (y1 - y2)))
        new_y2 = intersection[1]
        cv2.line(image, (x1, y1), (new_x2, new_y2), (0, 255, 0), 3)

def process_frame(frame):
    global points, roi_selected, prev_left_line, prev_right_line

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    if roi_selected:
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=100)
        min_angle = np.pi / 6
        left_lines, right_lines = [], []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)

                if abs(angle) > min_angle:
                    if (x1 + x2) / 2 < frame.shape[1] / 2:
                        left_lines.append(line[0])
                    else:
                        right_lines.append(line[0])

        left_line, right_line = prev_left_line, prev_right_line

        if left_lines:
            left_lines.sort(key=lambda line: min(line[0], line[2]))
            x1, y1, x2, y2 = left_lines[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            y1_new = int(y1 - slope * x1)
            y2_new = int(y2 + slope * (frame.shape[1] - x2))
            left_line = (0, y1_new), (frame.shape[1], y2_new)

        if right_lines:
            right_lines.sort(key=lambda line: max(line[0], line[2]))
            x1, y1, x2, y2 = right_lines[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            y1_new = int(y1 - slope * x1)
            y2_new = int(y2 + slope * (frame.shape[1] - x2))
            right_line = (0, y1_new), (frame.shape[1], y2_new)

        if left_line and right_line:
            prev_left_line, prev_right_line = left_line, right_line
            intersection = get_intersection(left_line[0] + left_line[1], right_line[0] + right_line[1])
            if intersection:
                cv2.circle(frame, intersection, 5, (255, 0, 0), -1)
                draw_filtered_line(frame, left_line, intersection)
                draw_filtered_line(frame, right_line, intersection)

    return frame

cap = cv2.VideoCapture('clip.mp4')

ret, frame = cap.read()
if not ret:
    print("비디오를 불러올 수 없습니다.")
    cap.release()
    exit()

frame_copy = frame.copy()
cv2.imshow("Select ROI", frame)
cv2.setMouseCallback("Select ROI", select_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)

    cv2.imshow("Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
