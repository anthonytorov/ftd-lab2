import cv2
import numpy as np
import time

DO_SLEEP = False
ALL_VARIANTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
STATISTICS_DICT = dict()


def inside_bounding_square(frame: np.ndarray, x, y, w, h):
    frame_h, frame_w, channels = frame.shape

    return  x > (frame_w * 0.5 - 100) and \
            y > (frame_h * 0.5 - 100) and \
            (x + w) < (frame_w * 0.5 + 100) and \
            (y + h) < (frame_h * 0.5 + 100)


def rect_right_of_frame_centre(frame: np.ndarray, x, y, w, h):
    frame_h, frame_w, channels = frame.shape
    return x > frame_w * 0.5


def main(variants, do_sleep):

    cap = cv2.VideoCapture('cam_video.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            c = max(contours, key= cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            rect_color = (0, 255, 0)        
        
            # Console printing variants:
            if 3 in variants:
                print(f'Bounding rect centre: ({x + w * 0.5}, {y + h * 0.5})')
            
            if 8 in variants:
                if 'average_coord' in STATISTICS_DICT:
                    avg_prev = STATISTICS_DICT['average_coord']
                    frame_n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # accumulating average
                    STATISTICS_DICT['average_coord'] = (avg_prev[0] + (x - avg_prev[0]) / frame_n, avg_prev[1] + (y - avg_prev[1]) / frame_n)
                else:
                    STATISTICS_DICT['average_coord'] = (x + w * 0.5, y + h * 0.5)
                
                print(f'Average coordinate: {STATISTICS_DICT["average_coord"]}')
            
            if 10 in variants:
                area_object = w * h
                area_frame = frame.shape[1] * frame.shape[0]
                print(f'Object occupies approximately {round(area_object / area_frame * 100, 2)}% of the frame')

            # Shape drawing variants:
            if 2 in variants:
                rect_color = (0, 255, 0) if inside_bounding_square(frame, x, y, w, h) else (0, 0, 255)
            
            if 4 in variants:
                rect_color = (255, 0, 0) if rect_right_of_frame_centre(frame, x, y, w, h) else (0, 0, 255)
            
            if 7 in variants:
                line_px_y = round(y + h * 0.5)
                cv2.line(frame, pt1=(0, line_px_y), pt2=(frame.shape[1], line_px_y), color=(0,0,255), thickness=1)
                line_px_x = round(x + w * 0.5)
                cv2.line(frame, pt1=(line_px_x, 0), pt2=(line_px_x, frame.shape[0]), color=(0,255,0), thickness=1)
            
            if 9 in variants:
                centre = (round(x + w * 0.5), round(y + h * 0.5))
                radius = round(np.sqrt(w * w + h * h) / 2)
                cv2.circle(frame, center=centre, radius=radius, color=(0,0,255), thickness=2)
            
            # Text drawing variants:
            if 1 in variants:
                cv2.putText(frame, text=f'Rect centre: ({x + w * 0.5}, {y + h * 0.5})', org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 255))
            
            if 6 in variants:
                distance = np.sqrt(np.power(x + w * 0.5 - frame.shape[1] * 0.5, 2) + np.power(y + h * 0.5 - frame.shape[0] * 0.5, 2))
                cv2.putText(frame, f'Distance to image center: {round(distance, 1)} px', (0, 75), cv2.FONT_HERSHEY_SIMPLEX, .5, color=(255, 255, 255))
            
            if 5 in variants:
                if 'rect_before' in STATISTICS_DICT:
                    right_of_frame_centre_before =  STATISTICS_DICT['rect_before'][0] > frame.shape[1] * 0.5
                    right_of_frame_centre_now = x > frame.shape[1] * 0.5

                    if right_of_frame_centre_before ^ right_of_frame_centre_now:
                        if 'image_side_change' not in STATISTICS_DICT:
                            STATISTICS_DICT['image_side_change'] = 0
                        STATISTICS_DICT['image_side_change'] += 1
                # store rect for the next frame
                STATISTICS_DICT['rect_before'] = (x, y, w, h)

                side_changes = 0
                if 'image_side_change' in STATISTICS_DICT:
                    side_changes = STATISTICS_DICT['image_side_change']
                cv2.putText(frame, f'Number of side changes: {side_changes}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, color=(255, 255, 255))

            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if do_sleep:
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(ALL_VARIANTS, DO_SLEEP)
