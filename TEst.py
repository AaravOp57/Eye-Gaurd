import cv2

print("OpenCV version:", cv2.__version__)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ CAMERA COULD NOT BE OPENED")
    exit()

print("✅ CAMERA OPENED!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Could not read frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()