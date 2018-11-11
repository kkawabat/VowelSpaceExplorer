import numpy as np
from  audiolazy import lazy_lpc
from scipy.signal import lfilter
import webrtcvad
from scipy.spatial import distance
import math
import pickle

class acoustic_features:
    def __init__(self, formant_space={'A':None, 'EE':None, 'E':None, 'O':None, 'OO':None, 'OA':None}, Fs=16000, numPitchSeg=2, tolerance=0, minPitch=np.nan, maxPitch=np.nan):
        self.formant_space = formant_space
        self.Fs = Fs
        self.numPitchSeg = numPitchSeg
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)
        self.tolerance = tolerance
        self.minPitch = minPitch
        self.maxPitch = maxPitch
        self.vowels = ['A', 'EE', 'E', 'O', 'OO', 'OA']

    def loadConfigFromFile(self, filePath):
        config = pickle.load(open(filePath, 'rb'))
        self.loadFormantSpace('A', config['A'])
        self.loadFormantSpace('EE', config['EE'])
        self.loadFormantSpace('E', config['E'])
        self.loadFormantSpace('O', config['O'])
        self.loadFormantSpace('OO', config['OO'])
        self.loadFormantSpace('OA', config['OA'])
        self.minPitch = config['minPitch']
        self.maxPitch = config['maxPitch']

    def loadFormantSpace(self, vowel, formants):
        if vowel in self.formant_space:
            self.formant_space[vowel] = formants
        else:
            print("ERROR %s is not a recognized vowel"%vowel)

    def has_speech(self, x):
        has_speech1 = self.vad.is_speech(x[0:480].tobytes(), self.Fs)
        has_speech2 = self.vad.is_speech(x[272:752].tobytes(), self.Fs)
        has_speech3 = self.vad.is_speech(x[544:1024].tobytes(), self.Fs)
        return has_speech1 or has_speech2 or has_speech3

    def getFormants(self, x):
        N = len(x)
        w = np.hamming(N)

        # Apply window and high pass filter.
        x1 = x * w
        x1 = lfilter([1], [1., 0.63], x1)

        # Get LPC.
        rts = lazy_lpc.lpc(x1, int((self.Fs / 1000) + 2)).zeros

        # Get roots.
        # rts = np.roots(A)
        rts = [r for r in rts if np.imag(r) >= 0]

        # Get angles.
        angz = np.arctan2(np.imag(rts), np.real(rts))

        # Get frequencies.
        indices = np.argsort(angz * (self.Fs / (2 * np.pi)))
        frqs = sorted(angz * (self.Fs / (2 * np.pi)))
        bw = -1 / 2 * (self.Fs / (2 * np.pi)) * np.log(abs(np.array(rts)[indices]))

        formants = []
        for f, b in zip(frqs, bw):
            if f > 90 and b < 400:
                formants.append(f)
        return formants

    def getPitch(self, x):
        return 0

    def getRawInput(self, x):
        has_input = False
        pitch = -1
        formants = -1

        if self.has_speech(x):
            has_input = True
            pitch = self.getPitch(x)
            formants = self.getFormants(x)

        return [has_input, pitch, formants]

    def getProcessedInputs(self, x):
        [has_input, pitch, formants] = self.getRawInput(x)
        if has_input:
            if math.isnan(self.minPitch) and math.isnan(self.maxPitch):
                pitch_percent = (pitch - self.minPitch)/(self.maxPitch- self.minPitch)
            else:
                pitch_percent = -1

            best_formant = 'N'
            if len(formants) >= 3:
                dist = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
                if self.formant_space['A']:
                    dist[0] = distance.cdist([formants[0:3]], self.formant_space['A']).min()
                if self.formant_space['EE']:
                    dist[1] = distance.cdist([formants[0:3]], self.formant_space['EE']).min()
                if self.formant_space['E']:
                    dist[2] = distance.cdist([formants[0:3]], self.formant_space['E']).min()
                if self.formant_space['O']:
                    dist[3] = distance.cdist([formants[0:3]], self.formant_space['O']).min()
                if self.formant_space['OO']:
                    dist[4] = distance.cdist([formants[0:3]], self.formant_space['OO']).min()
                if self.formant_space['OA']:
                    dist[5] = distance.cdist([formants[0:3]], self.formant_space['OA']).min()
                closest_phoneme_index = np.argmin(dist)

                dist_inv = np.power(dist, -1.0)
                conf = max(np.divide(dist_inv, sum(dist_inv)))
                #print(np.divide(dist_inv, sum(dist_inv)))
                if math.isnan(conf) or conf >= self.tolerance:
                    best_formant = self.vowels[closest_phoneme_index]
        else:
            pitch_percent = -1
            best_formant = 'N'

        return [has_input, pitch_percent, best_formant]

if __name__ == "__main__":
    pass