import cv2


cap = cv2.VideoCapture(0)
i = 0
limit = 1000
flag = 0
val = 1
while (cap.isOpened()):
    ret, frame = cap.read()
    if i == limit or ret == False:
        break
    frame = cv2.flip(frame, 1)
    ext = frame[10:310, 320:620]
    cv2.rectangle(frame, (320, 10), (620, 310), (0, 0, 0), 2)
    cv2.putText(frame, "Go For " + str(val)+" :)", (30, 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        flag = 1
    if flag == 1:
        cv2.imwrite('Tanishka'+str(i)+'.jpg', ext)
        i += 1

cap.release()
cv2.destroyAllWindows()
