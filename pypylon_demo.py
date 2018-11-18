import time
from pydarknet import Detector, Image
from pypylon import pylon
import cv2

if __name__ == "__main__":
    #pydarknet.set_cuda_device(0)

    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))

try:
    #Creating a camera object with the first found camera.
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    #Print out the name of the used camera
    print("Using cam: ", camera.GetDeviceInfo().GetModelName())
    #Start Grabbing
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 

    #Converter from Basler grab to OpenVC bgr
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    #The Main Loop
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            
            dark_frame = Image(img)
            results = net.detect(dark_frame)
            print(results)
            print("----\n\r")
            for cat, score, bounds in results:
                    x, y, w, h = bounds
                    cv2.rectangle(img, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                    cv2.putText(img, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

                    cv2.imshow("preview", img)

            k = cv2.waitKey(1)

        grabResult.Release()

except KeyboardInterrupt:
    print("Interrupt received, stopping…")
finally:
    #Clean up
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()



