# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from os.path import abspath
# Finshed importing

cnn = tf.keras.models.load_model(abspath("HandGest3.h5"))
win = Tk()
win.title("Model")
win.state("zoomed")
my_canvas = Canvas(win, width=1280, height=720)
my_canvas.pack(fill="both", expand=True)


# Working on the background
fp = abspath('Images')
# fp = "Images"


def background():
    global background_image1
    print(fp + "\\bk.jpg")
    background_image1 = Image.open(fp + "\\bk.jpg")
    background_image1 = background_image1.resize((1278, 690))
    background_image1 = ImageTk.PhotoImage(background_image1)
    my_canvas.create_image(0, 0, image=background_image1, anchor="nw")
    my_canvas.create_text(600, 50, text="HAND GESTURE IDENTIFICATION",
                          fill='#000000', font=('PT Sans', -20, 'bold'))
# Finished working on background


# Adding different gesture photos
ls = ["one", "two", "three", "four", "five",
      "six", "seven", "eight", "nine", "ten"]


def gesture():
    global x
    x = []
    for i in range(0, len(ls)):
        file_path = fp + '\\' + ls[i] + ".jpg"
        x.append(Image.open(file_path))
        x[len(x)-1] = x[len(x)-1].resize((120, 120))
        x[len(x)-1] = ImageTk.PhotoImage(x[len(x)-1])
        if i <= 4:
            Label(win, image=x[len(x)-1]).place(x=5, y=10+i*138)
        else:
            Label(win, image=x[len(x)-1]).place(x=1150, y=10+(i-5)*138)
# Finished adding different gesture photos


# Adding names of different hand gestures
ls = ["one", "two", "three", "four", "five",
      "six", "seven", "eight", "nine", "ten"]


def names():
    global x1
    x1 = []
    for i in range(0, len(ls)):
        if i <= 4:
            my_canvas.create_text(
                220, 60+i*138, text="<- "+ls[i].upper(), fill='#000000', font=('Consolas', -40, 'bold'))
        else:
            my_canvas.create_text(
                1060, 60+(i-5)*138, text=ls[i].upper()+" ->", fill='#000000', font=('Consolas', -40, 'bold'))
# Finished adding names of different hand gestures


# Displaying the Real-Time Stuff
label = Label(win)
label.pack()
cap = cv2.VideoCapture(0)
user_score = 0
comp_score = 0
user_value = 0
comp_value = 0
b = 0
flag = 1
val = 0


def show_frames():
    global cap, label, b, btn, btn1
    ret, frame = cap.read()
    frame = working(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    kpress = cv2.waitKey(1) & 0xFF
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.place(x=315, y=120)
    if ret != False and b != 2:
        label.after(1, show_frames)
    else:
        cap.release()
        label.after(1, label.destroy())
        btn.after(1, btn.destroy())
        btn1.after(1, btn1.destroy())
        os.remove(fp + "\\test0.jpg")
        os.remove(fp + "\\test1.jpg")

# Finieshed Showing Real-Time stuff


# Working on the frame that is to be displayed
def working(frame):
    global user_score, comp_score, user_value, comp_value, b, val, cap
    ls1 = [1, 10, 2, 3, 4, 5, 6, 7, 8, 9, "nothing"]
    frame = cv2.flip(frame, 1)
    test_image1 = frame[10:310, 320:620]
    cv2.rectangle(frame, (320, 10), (620, 310), (255, 253, 8), 2)
    cv2.rectangle(frame, (0, 0), (318, 610), (77, 23, 2), -1)
    cv2.rectangle(frame, (318, 312), (700, 700), (77, 23, 2), -1)
    cv2.rectangle(frame, (318, 0), (700, 8), (77, 23, 2), -1)
    cv2.rectangle(frame, (622, 0), (700, 700), (77, 23, 2), -1)
    cv2.imwrite(
        fp + "\\test0.jpg", test_image1)
    cv2.imwrite(
        fp + "\\test1.jpg", test_image1)
    try:
        test_image1 = image.load_img(
            fp + "\\test0.jpg", target_size=(300, 300))
    except:
        test_image1 = image.load_img(
            fp + "\\test1.jpg", target_size=(300, 300))
    test_image1 = image.img_to_array(test_image1)
    test_image1 = np.expand_dims(test_image1, axis=0)
    result = cnn.predict(test_image1/255.0)
    val = result[0].argmax()
    if comp_value != 0:
        ext = "Comp Val " + str(comp_value)
        cv2.putText(frame, ext, (30, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 253, 8))
    cv2.putText(frame, str(
        "VALUE IS :" + str(ls1[val])), (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 253, 8))
    return frame
# Finished working on the frame that is to be displayed


# Working on the click of Quit Button
def quitgame():
    global cap, label, btn1, flag
    flag = 0
    cap.release()
    label.after(1, label.destroy())
    btn.after(1, btn.destroy())
    btn1.after(1, btn1.destroy())
    os.remove(fp + "\\test0.jpg")
    os.remove(fp + "\\test1.jpg")


# Finished working on the click of Quit Button

background()
gesture()
names()


def startgame():
    global btn1, strt
    show_frames()
    strt.after(1, strt.destroy())
    btn1 = Button(win, text="Quit", bg="#08fdff", fg="#000000",
                  font=('Consolas', -30, 'bold'), command=quitgame)
    btn1.place(x=700, y=520)


strt = Button(win, text="START", bg="#08fdff", fg="#000000",
              font=('Consolas', -60, 'bold'), command=startgame)
strt.place(x=450, y=250)


win.mainloop()
