import cv2

def draw_bottom_indicator(image, text, capturer: cv2.VideoCapture, left_offset = 0, color = (255, 0, 0)):
    vid_width = capturer.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_height = capturer.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cv2.putText(img=image, text=text, org=(int(vid_width/2) - left_offset, int(vid_height - 40)), color=color, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=4)
