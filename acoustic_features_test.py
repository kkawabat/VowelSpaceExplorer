import pyaudio
import numpy as np
from acoustic_features import acoustic_features
import pickle

def formant_test():
    af = acoustic_features()
    af.loadConfigFromFile('dummy_config.pickle')

    mic = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                                 frames_per_buffer=1024)
    while True:
        data_chunk = np.fromstring(mic.read(1024), dtype=np.int16)
        [has_input, pitch_percent, best_formant] = af.getProcessedInputs(data_chunk)
        print(best_formant)

def pitch_test():
    af = acoustic_features()
    af.loadConfigFromFile('dummy_config.pickle')

    mic = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                                 frames_per_buffer=1024)
    while True:
        data_chunk = np.fromstring(mic.read(1024), dtype=np.int16)
        [has_input, pitch, best_formant] = af.getRawInput(data_chunk)
        print(pitch)

def pickle_demo():
    #to load from a pickle file to a variable
    myConfig = pickle.load(open('dummy_config.pickle', 'rb'))

    #config file should be in this format
    print(myConfig)

    #to write to a pickle file
    pickle.dump(myConfig, open('dummy_config2.pickle', 'wb'))


if __name__ == "__main__":
    # formant_test()
    # pickle_demo()
    pitch_test()