import numpy as np
from  audiolazy import lazy_lpc
from scipy.signal import lfilter
import webrtcvad
from sklearn.cluster import KMeans
from sklearn import datasets
from scipy.spatial import distance

class acoustic_features:
    def __init__(self, config):
        self.config = config
        self.Fs = config['Fs']
        self.numPitchSeg = config['numPitchSeg']
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)
        self.kmean = KMeans(n_clusters=6)
        self.vowels = ['A', 'EE', 'E', 'O', 'OO', 'OA']

    def has_speech(self, x):
        has_speech1 = self.vad.is_speech(x[0:480].tobytes(), self.Fs)
        has_speech2 = self.vad.is_speech(x[272:752].tobytes(), self.Fs)
        has_speech3 = self.vad.is_speech(x[544:1024].tobytes(), self.Fs)
        return has_speech1 or has_speech2 or has_speech3

    def getFormant(self, x):
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
        #MAZHER ADD CODE HERE
        pass

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
        [has_input, pitch, formants] = self.getRawInput(self, x)
        pitch_percent = (pitch - self.config['minPitch'])/(self.config['maxPitch']- self.config['minPitch'])

        best_formant = 'N'
        if len(formants) >= 3:
            dist = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
            dist[0] = min(distance.cdist(formants[0:3], self.config['A_matrix']))
            dist[1] = min(distance.cdist(formants[0:3], self.config['EE_matrix']))
            dist[2] = min(distance.cdist(formants[0:3], self.config['E_matrix']))
            dist[3] = min(distance.cdist(formants[0:3], self.config['O_matrix']))
            dist[4] = min(distance.cdist(formants[0:3], self.config['OO_matrix']))
            dist[5] = min(distance.cdist(formants[0:3], self.config['OA_matrix']))
            closest_phoneme_index = np.argmin(dist)
            if dist[closest_phoneme_index] <= self.config['tolerance']:
                best_formant = self.vowels[closest_phoneme_index]
        return [has_input, pitch_percent, best_formant]

if __name__ == "__main__":
    iris = datasets.load_iris()
    print(iris)