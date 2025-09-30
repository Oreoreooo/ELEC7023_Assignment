from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

net = detectNet("ssd-mobilenet-v2", threshold=0.5)
#image_file = ["cow17.jpeg", "horse5.jpeg"]
camera = videoSource("cow17.jpeg")

display = videoOutput("result2.jpg") # 'my_video.mp4' for file

while display.IsStreaming(): # main loop will go here
	img = camera.Capture()
	if img is None: # capture timeout
		continue

	detections = net.Detect(img)

	for i, detection in enumerate(detections[:2]): 
		print(f"etection results {i+1}")
		print(f"ClassID: {detection.ClassID}")
		print(f"Class Name: {net.GetClassDesc(detection.ClassID)}")
		print(f"Confidence: {detection.Confidence:.3f}")
		print(f"Left: {detection.Left:.1f}")
		print(f"Top: {detection.Top:.1f}")
		print(f"Right: {detection.Right:.1f}")
		print(f"Bottom: {detection.Bottom:.1f}")
		print(f"Width: {detection.Right - detection.Left:.1f}")
		print(f"Height: {detection.Bottom - detection.Top:.1f}")
		print(f"Area: {(detection.Right - detection.Left) * (detection.Bottom - detection.Top):.1f}")
		print(f"Center: ({(detection.Left + detection.Right) / 2:.1f}, {(detection.Top + detection.Bottom) / 2:.1f})")


	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
