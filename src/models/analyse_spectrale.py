# Ce package contient un ensemble d'outil d'analyse spectrale
# Une classe basée sur la STFT pour calculer le spectrogramme et detecter les fréquences dominantes dans la série.
# Une classe basée sur la fonction d'autocorrélation pour calculer le spectrogramme et detecter les fréquences dominantes dans la série.
# Ces fréquence servirons pour la décomposition.

# Packages 
import pandas as pd
import numpy as np
from scipy.signal import spectrogram, find_peaks
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import find_peaks

class SpectrogramAnalysis(BaseEstimator, TransformerMixin):
    """
    La classe SpectrogramAnalysis, calculer le spectrogramme d'une série temporelle et detecte les fréquences dominantes de celle-ci
    Parameters
    ----------
    in 
    window : str
        fenêtre de lissage par défaut hann, autre 
    nperseg : int, default=256
        longueur de la fênetre 
    noverlap : int, default=128
        longueur du recouvrement 
    threshold : float
        seuil de detection  = max * threshold
    fs : float
        fréquence d'échantillonnage
    out 
    
    frequencies : vecteur de float
        vecteur fréquentielle
    times : vecteur de float
    spectrogram : vecteur de float
        un élément : energie à une fréquence dans une fenêtre temporelle
    dominant_frequencies : vecteur d'int 
        fréquence dominantes 

    dominant_periodes : vecteur d'int 
        périodes dominantes : normalisées par rapport à Ts = 1/fs 

    Examples
    --------
    """

    def __init__(self, window='hann', nperseg=256, noverlap=128,  fs= 1,threshold=0.01):
        self.window = window # fenêtre de lissage par défau hann : autre possibilité  
        self.nperseg = nperseg # longeur de la fen^tre
        self.noverlap = noverlap # recouvrement
        self.threshold = threshold # seuil de détection
        self.frequencies = None 
        self.times = None
        self.spectrogram = None
        self.dominant_frequencies = None
        self.dominant_periodes = None
        self.fs = fs

    def fit(self, X, y=None):
        # Calculer le spectrogramme
        f, t, Sxx = spectrogram(X.values, window=self.window, nperseg=self.nperseg, noverlap=self.noverlap, fs = self.fs)
        Sxx = np.abs(Sxx)
        self.frequencies = f
        self.times = t
        self.spectrogram = np.round(Sxx, decimals=10)
        return self

    def transform(self, X):
        # Identifier les fréquences dominantes

        Sxx_sorted = np.sort(self.spectrogram, axis=1)
        spectrum_mean = np.mean(Sxx_sorted, axis=1)
        spectrum_mean = np.round(spectrum_mean, decimals=10)


        #spectrum_mean = np.mean(self.spectrogram, axis=1)
        #spectrum_mean = np.round(spectrum_mean, decimals=10)
        peaks, _ = find_peaks(spectrum_mean, height=np.max(spectrum_mean)*self.threshold) 
        self.dominant_frequencies = self.frequencies[peaks]
        self.dominant_periodes = np.sort(np.round(self.fs/self.dominant_frequencies)) 
        return pd.DataFrame({'Périodes': self.dominant_periodes}).drop_duplicates()
        

    def plot_spectrogramme(self):
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(self.times, self.frequencies, self.spectrogram, shading='gouraud')
        plt.ylabel('Fréquence [Hz]')
        plt.xlabel('Temps [sec]')
        plt.title('Spectrogramme')
        plt.colorbar(label='Intensité')

        if self.dominant_frequencies is not None:
            for freq in self.dominant_frequencies:
                plt.axhline(freq, color='red', linestyle='--', label=f'Periode : {self.fs/freq:.2f} pas')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def marquage_raie_spectre(self, target_freq):
        freq_idx = np.argmin(np.abs(self.frequencies - target_freq))
        intensity_at_freq = self.spectrogram[freq_idx, :]

        plt.figure(figsize=(12, 5))
        plt.subplot(2, 1, 1)
        plt.pcolormesh(self.times, self.frequencies, self.spectrogram, shading='gouraud')
        plt.axhline(self.frequencies[freq_idx], color='r', linestyle='--', label=f'{target_freq} Hz')
        plt.colorbar(label='Intensité')
        plt.title('Spectrogramme avec fréquence ciblée')

        plt.subplot(2, 1, 2)
        plt.plot(self.times, intensity_at_freq, color='r', label=f"Intensité à {target_freq:.2f} Hz")
        plt.xlabel('Temps [s]')
        plt.ylabel('Intensité')
        plt.title('Évolution temporelle de la fréquence ciblée')
        plt.legend()

        plt.tight_layout()
        plt.show()


#--------------------------------------------------------------------------------------------------------------#
#  Cette classe calcule la fonction d'autocorrélation et détecte les fréquences des pics 
#  Cette permet d'évaluer la cohérence des fréquences identifiées par la calsse précédente
#
##--------------------------------------------------------------------------------------------------------------#

class ACFDominantFrequenciesOptimized(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=10, max_lag=500):
        self.top_n = top_n
        self.max_lag = max_lag
        self.height = None
        self.distance = None
        self.width = None
        self.prominence = None
        self.dominant_periods = None
        self.coherent_periods = None

    def fit(self, X, y=None):
        # Ensure the input is a pandas Series
        if not isinstance(X, pd.Series):
            raise ValueError("Input must be a pandas Series.")

        # Compute Autocorrelation
        acf_values = pd.Series([X.autocorr(lag) for lag in range(1, self.max_lag)])
        acf_values.index += 1  # To match the lag values

        # Adaptive height and prominence
        acf_mean = acf_values.mean()
        acf_std = acf_values.std()
        self.height = acf_mean + 0.1 * acf_std
        self.prominence = 0.05 * (acf_values.max() - acf_values.min())

        # Initial peak detection (large peaks)
        peaks, _ = find_peaks(acf_values, height=self.height, prominence=self.prominence)

        # Secondary peak detection (smaller peaks between primary)
        secondary_peaks, _ = find_peaks(acf_values, height=acf_mean, prominence=0.01 * acf_values.max(), distance=5)

        # Combine and sort all detected peaks
        all_peaks = np.unique(np.concatenate((peaks, secondary_peaks)))
        peak_values = acf_values.iloc[all_peaks]
        self.dominant_periods = peak_values.nlargest(self.top_n).index

        # Analyzing distances between peaks to find coherent periods
        periods = np.diff(self.dominant_periods)
        self.coherent_periods = pd.Series(periods).value_counts().nlargest(3).index.values

        return self

    def transform(self, X):
        if self.dominant_periods is None:
            raise ValueError("Model has not been fitted yet.")

        return pd.DataFrame(
            {'Dominant Periods (Lags)': self.dominant_periods,
            'Coherent Periods': [self.coherent_periods]
            }
            )

    def plot_acf(self, X):
        acf_values = pd.Series([X.autocorr(lag) for lag in range(1, self.max_lag)])
        acf_values.index += 1

        plt.figure(figsize=(12, 6))
        plt.plot(acf_values, label="ACF")

        # Highlighting the detected peaks
        for peak in self.dominant_periods:
            plt.axvline(x=peak, color='red', linestyle='--', label=f'Peak at {peak} lag')

        plt.title("Fonction d'Autocorrélation avec détection des pics (primaires et secondaires)")
        plt.xlabel("Lags")
        plt.ylabel("Autocorrélation")
        plt.legend()
        plt.show()

    def get_coherent_periods(self):
        if self.coherent_periods is None:
            raise ValueError("Model has not been fitted yet.")
        return self.coherent_periods