import pyaudio
import cv2
import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import matplotlib as mpl
from  audiolazy import lazy_lpc
import webrtcvad
import time
import pickle


class VSE():
    def __init__(self):
        self.CHUNK = 1024
        self.Fs = 16000
        self.mic = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1,rate=self.Fs, input=True, frames_per_buffer=self.CHUNK)
        self.audio_data = []
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)
        self.formant_word = [r"bEEt",r"bEd",r"cAt",r"bOOt",
                            r"bOAt",r"bOt"]
        self.formant_vals = {r"bEEt":None,r"bEd":None,r"cAt":None,r"bOOt":None,
                            r"bOAt":None,r"bOt":None}
        self.formant_avg = []
        self.initial = True

    def start(self):
        self.mic.start_stream()
        fig = plt.gcf()
        ax = fig.add_subplot(111)
        plt.ylim(1000, 100)
        plt.xlim(4000, 500)
        fig.show()
        fig.canvas.draw()
        key = None
        start = time.time()
        text = plt.text(3500, 50, "Please voice the capitalized vowels:%s"%(self.formant_word[0]), fontsize=12)
        charx = 500
        chary = 500
        while True:
            data_chunk = np.fromstring(self.mic.read(self.CHUNK), dtype=np.int16)
            time_lapse = time.time() - start
            sec10_index = int(time_lapse/5)
            count_down = 5 - time_lapse%5
            if sec10_index < 6:
                text.set_text("Please voice the capitalized vowels:%s %i"%(self.formant_word[sec10_index], count_down))
                b, formants = self.getF1F2(data_chunk)
                if b:
                    if self.formant_vals[self.formant_word[sec10_index]]:
                        self.formant_vals[self.formant_word[sec10_index]].append(formants[0:3])
                    else:
                        self.formant_vals[self.formant_word[sec10_index]] = [formants[0:3]]
            else:
                if self.initial:
                    sct = ax.scatter(500, 500)
                    arrow = mpl.patches.Arrow(-100, -100, -10, 0, color="#aa0088")
                    a = ax.add_patch(arrow)
                    pickle.dump(self.formant_vals, open('formant_vals.pickle','wb'))
                    t_ee = plt.text(3500, 150, "ee")
                    t_e = plt.text(3500, 550, "e")
                    t_a = plt.text(3500, 950, "a")
                    t_oo = plt.text(1000, 150, "oo")
                    t_oa = plt.text(1000, 550, "oa")
                    t_o = plt.text(1000, 950, "o")
                    self.initial = False
                    for i, word in enumerate(self.formant_word):
                        self.formant_avg.append(np.mean(np.array(self.formant_vals[word]), axis=0))
                        # print(self.formant_avg[i])
                else:
                    b, f = self.getF1F2(data_chunk)
                    if b:
                        f_norms = [np.linalg.norm([f2[0] - f[0], f2[1] - f[1], f2[2] - f[2]]) for f2 in self.formant_avg]
                        i = np.argmin(f_norms)
                        print("%f, %s"%(f_norms[i], self.formant_word[i]))
                        if f_norms[i] < 500:
                            if i == 0:
                                charx -= 10
                                a.remove()
                                arrow = mpl.patches.Arrow(charx,chary, -10, 0, color="#aa0088")
                                a = ax.add_patch(arrow)
                            elif i == 1:
                                charx += 10
                                a.remove()
                                arrow = mpl.patches.Arrow(charx, chary, 10, 0, color="#aa0088")
                                a = ax.add_patch(arrow)
                            elif i == 2:
                                chary -= 10
                                a.remove()
                                arrow = mpl.patches.Arrow(charx, chary, 0, -10, color="#aa0088")
                                a = ax.add_patch(arrow)
                            elif i == 3:
                                chary += 10
                                a.remove()
                                arrow = mpl.patches.Arrow(charx, chary, 0, 10, color="#aa0088")
                                a = ax.add_patch(arrow)
                            elif i == 4:
                                continue
                                # charx = 500
                                # chary = 500
                            elif i == 5:
                                continue
                                # charx = 1000
                                # chary = 1000
                            sct.set_offsets([charx, chary])
            fig.canvas.draw()
            plt.pause(0.001)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        self.mic.close()
    #Formula and code was modeled after https://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html
    # and https://stackoverflow.com/questions/25107806/estimate-formants-using-lpc-in-python
    def getFormant(self, x):
        N = len(x)
        w = np.hamming(N)

        # Apply window and high pass filter.
        x1 = x * w
        x1 = lfilter([1], [1., 0.63], x1)

        # Get LPC.

        rts = lazy_lpc.lpc(x1, int((self.Fs/1000)+2)).zeros

        # Get roots.
        # rts = np.roots(A)
        rts = [r for r in rts if np.imag(r) >= 0]

        # Get angles.
        angz = np.arctan2(np.imag(rts), np.real(rts))

        # Get frequencies.
        indices = np.argsort(angz * (self.Fs / (2 * np.pi)))
        frqs = sorted(angz * (self.Fs / (2 * np.pi)))
        bw = -1/2*(self.Fs/(2*np.pi))*np.log(abs(np.array(rts)[indices]))

        formants = []
        for f, b in zip(frqs, bw):
            if f > 90 and b < 400:
                formants.append(f)
        return formants

    def getF1F2(self, data_chunk):
        speech_detected = self.vad.is_speech(data_chunk[0:480].tobytes(), self.Fs)
        # print(speech_detected)
        if (speech_detected):
            formants = self.getFormant(data_chunk)
            if len(formants) >= 3:
                print(formants)
                return 1, formants
        return 0, []
if __name__ == '__main__':
    # pickle.load(open('formant_vals.pickle','rb'))
    al = VSE()
    al.start()
