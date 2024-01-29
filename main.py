# import cv2
# # Face_cap = cv2.CascadeClassifier("")
# video_cap = cv2.VideoCapture(0)
# while True :
#        ret, video_data = video_cap.read()
#        cv2.imshow("video_live",video_data)
#        if cv2.waitKey(10) == ord("a"):
#         break
# video_cap.release()
import os

# import cv2
# path = r"""C:\Users\Diksha\Pictures\Screenshots"""
# img_show ='Screenshot (29).png'
# img = cv2.imread(path + '\\' + img_show)             // helps to show the image of a particular directory
# cv2.resize(img,(500,300))
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# How to access the camera and open video of a particular directory.
# import numpy as np
# import cv2
# video_path = r'C:\Users\Diksha\Pictures\Camera Roll\ShreeRAMJi.mp4'
# cap = cv2.VideoCapture(video_path)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#        image =  cv2.resize(frame,(400,200))
#        frame_2 = np.hstack((frame,frame))
#        frame_4 = np.vstack((frame_2, frame_2))
#        cv2.imshow("Video_player",frame_4)
#
#        if cv2.waitKey(25) & 0xff == ord('q'):
#           break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# path = r'C:\Users\Diksha\Pictures\Camera Roll\sush.jpg'
# img = cv2.imread(path,0)     # flag 0 krne sei grey mei ho jayega
# print(img)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# mat = np.zeros((200,200))
# print(mat)
# cv2.imshow("Matx",mat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# How to same image or numpy array
# import cv2
# import numpy as np
# import os
# rand_array = np.random.randint(255, size = (300, 600, 3))
# cv2.imwrite("rand_np_array.png",rand_array)
# img = cv2.imread("rand_np_array.png")
# cv2.imshow('Image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#  Save Images from video
# import cv2
# import numpy as np
# import os
# rand_array = np.random.randint(255, size =(300,600,3))
# print(rand_array)
# cv2.imwrite("rand_np_array.png",rand_array)
# img= cv2.imread("rand_np_array.png")
# cv2.imshow("image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img_path = 'sush.jpg'
# img = cv2.imread(img_path)
# cv2.imshow("image",img)
# cv2.imwrite("sush_write.png",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save images from video:-
# video_path = r"C:\Users\Diksha\PycharmProjects\Face_detection_Project\ShreeRAMJi.mp4"
# os.mkdir("video_to_image")
# cap = cv2.VideoCapture(video_path)
# img_count = 1
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("unable to read frame")
#         break
#     is_img_write = cv2.imwrite(f"video_to_image\image{img_count}.jpeg", frame)
#     if is_img_write:
#         print(f'image save at video_to_image\image{img_count}.jpeg')
#     cv2.imshow("video",frame)
#     cv2.waitKey(25) & 0xff == ord('q')
#     img_count += 1
# cap.release()
# cv2.destroyAllWindows()



# create video from numpy array and images
# import numpy as np
# import cv2
# import os
# height = 1280
# width = 720
# channel = 3
# fps = 30
# sec = 5
# fourcc = cv2.VideoWriter_fourcc(*'MP42')
# video = cv2.VideoWriter('test.mp4',fourcc,float(fps),(width, height))
# for frame_count in range(fps*sec):
#     img = np.random.randint(0,255, (height,width,channel),dtype = np.uint8)
#     video.write(img)
# video.release()
# width = 1280
# height = 720
# channel = 3
# fps = 10
# sec = 5
# fourcc = cv2.VideoWriter_fourcc(*'MP42')
# video = cv2.VideoWriter('image_to_video.avi',fourcc,float(fps),(width, height))
# directory = r'C:\Users\Diksha\PycharmProjects\Face_detection_Project'
# img_name_list = os.listdir(directory)
# for frame_count in range(fps*sec):
#     img_name = np.random.choice(img_name_list)
#     img_path = os.path.join(directory, img_name)
#     img = cv2.imread(img_path)
#     img_resize = cv2.resize(img,(width, height))
#
#     video.write(img_resize)
# video.release()



#  How to put text over a picture
# import cv2
# import numpy as np
# img_path = r"C:\Users\Diksha\PycharmProjects\Face_detection_Project\bhglpur.jpg.png"
# image = cv2.imread(img_path)
# image = cv2.resize(image, (1280,720))
# text = "monkey"
# org = (100, 200)
# font = cv2.FONT_HERSHEY_COMPLEX
# font_scale = 6
# color = (0,0,255)#(B, G, R)
# thickness = 3
# lineType = cv2.LINE_AA
# bottomLeftOrigin = False
# img_text = cv2.putText(image, text, org,font, font_scale, color, thickness, lineType, bottomLeftOrigin)
# # img_text = cv2.putText(image, text, org,font, font_scale, color, thickness, lineType, bottomLeftOrigin=True)
# cv2.imshow("Text image", img_text)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# img_path = r"C:\Users\Diksha\PycharmProjects\Face_detection_Project\img2.jpg"
# image = cv2.imread(img_path)
# image = cv2.resize(image,(1280, 720))
# pt_1 = (400, 40)
# pt_2 = (800, 300)
# color = (0, 255, 0)
# thickness = 4
# lineType = cv2.LINE_4
# img_rect = cv2.rectangle(image, pt_1, pt_2, color, thickness, lineType)
# cv2.imshow("image_rectangle", img_rect)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
face_cap = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_cap = cv2.VideoCapture(0)
while True:
     ret, video_data = video_cap.read()
     col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
     faces = face_cap.detectMultiScale(col,scaleFactor=1.1, minNeighbors= 5, minSize= (30,30), flags= cv2.CASCADE_SCALE_IMAGE)
     for (x, y, w, h) in faces:
         cv2.rectangle(video_data, (x,y), (x+w,y+ h), (0, 255, 0), 2 )
     cv2.imshow("Live Video", video_data)
     if cv2.waitKey(10) == ord('a'):
         break
video_cap.release()


