import pyaudio
import numpy as np
import acoustic_features
import pickle

if __name__ == "__main__":
    config = pickle.load(open('formant_vals.pickle','rb'))
    formant_space = {'A':config['cAt'], 'E':config['bEd'], 'EE':config['bEEt'], 'OO':config['bOOt'], 'OA':config['bOAt'], 'O':config['bOt']}
    af = acoustic_features(formant_space=formant_space)

    mic = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                                 frames_per_buffer=1024)
    while True:
        data_chunk = np.fromstring(mic.read(1024), dtype=np.int16)
        [has_input, pitch_percent, best_formant] = af.getProcessedInputs(data_chunk)