import gtts
import speech_recognition as sr
from pywebio.output import *
import os

def speech_to_text():
    r= sr.Recognizer()
    with sr.Microphone() as source:
        print("الرجاء التحدث الان.....")
        audio = r.listen(source)
        try:
            text = r.recognize_google_cloud(audio,language="ar-en")
            print(" تم التعرف على النص: "+ text)
            return text
        except sr.RequestError as e:
            print("لم يتم التعرف على النص, خطأ في الخدمة:{0}".format(e))

#===================================================================
def text_to_speech(text):
    tts= gtts(text=text,lang="ar")
    filename= "output.mp3"
    tts.save(filename)
    os.system("start "+ filename)
text = speech_to_text()
text_to_speech()

put_text("Learn python : ")

put_button("التحويل من الصوت الى النص. ",onclick=speech_to_text )
input('press any key......')


put_button("التحويل من النص الى الصوت . ",onclick=text_to_speech )
input('press any key......')