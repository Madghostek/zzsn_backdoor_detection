import cv2
# export DISPLAY=192.168.20.127:0.0 # Or any other port
# IP camera
vid_cam = cv2.VideoCapture('http://192.168.20.127:56000/mjpeg')

while(vid_cam.isOpened()):
    ret, image_frame = vid_cam.read()
    cv2.imshow('frame', image_frame)

    

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

vid_cam.release()
cv2.destroyAllWindows()

# while(1):
#     img = cv2.imread('/home/sense/cleanse_pytorch/data-2/0_adv.jpg')
#     cv2.imshow('frame', img)
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()    