import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from twilio.rest import Client

account_sid = '************'
auth_token = '*****************'
twilio_phone_number = '**********'
recipient_phone_number = '***********'

client = Client(account_sid, auth_token)

def send_alert():
    message = client.messages.create(
        body='ALERT!!! Accident Detected! Send response team immediately!',
        from_=twilio_phone_number,
        to=recipient_phone_number
    )


model = YOLO('best.pt')
accident_detected = False

cv2.namedWindow('RGB')
cap = cv2.VideoCapture('cr.mp4')

my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
while True:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'accident' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            if not accident_detected:
                send_alert()
                accident_detected = True
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()



