"""
    Segment Signals
"""
import wfdb
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

FS = 500
W_LEN = 256
W_LEN_1_4 = 256 // 4
W_LEN_3_4 = 3 * (256 // 4)

record = wfdb.rdrecord('BBDD/ecg-id-database-1.0.0/Person_01/rec_1')  
annotation = wfdb.rdann('BBDD/ecg-id-database-1.0.0/Person_01/rec_1', 'atr')  



signal = record.p_signal[:, 0]  
sampling_rate = record.fs  


signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)

signal = signals["ECG_Clean"]
r_peaks_annot = info["ECG_R_Peaks"]


def segmentSignals(signal, r_peaks_annot, normalization=True, person_id= None, file_id=None):
    """
    Segments signals based on the detected R-Peak

    Args:
        signal (numpy array): input signal
        r_peaks_annot (int []): r-peak locations.
        normalization (bool, optional): apply z-normalization or not? . Defaults to True.
        person_id ([type], optional): [description]. Defaults to None.
        file_id ([type], optional): [description]. Defaults to None.

    Returns:
            [tuple(numpy array,numpy array)]: segmented signals and refined r-peaks
    """

    def refine_rpeaks(signal, r_peaks):
        """
        Refines the detected R-peaks. If the R-peak is slightly shifted, this assigns the 
        highest point R-peak.

        Args:
            signal (numpy array): input signal
            r_peaks (int []): list of detected r-peaks

        Returns:
            [numpy array]: refined r-peaks
        """

        r_peaks2 = np.array(r_peaks)            # make a copy

        for i in range(len(r_peaks)):

            r = r_peaks[i]          # current R-peak

            small_segment = signal[max(0,r-100):min(len(signal),r+100)]         # consider the neighboring segment of R-peak
            r_peaks2[i] = np.argmax(small_segment) - 100 + r_peaks[i]           # picking the highest point
            r_peaks2[i] = min(r_peaks2[i],len(signal))                          # the detected R-peak shouldn't be outside the signal
            r_peaks2[i] = max(r_peaks2[i],0)                                    # checking if it goes before zero
        
        return r_peaks2                     # returning the refined r-peak list



    segmented_signals = []                      # array containing the segmented beats
    
    r_peaks = np.array(r_peaks_annot)
    r_peaks = refine_rpeaks(signal, r_peaks)

    for r in r_peaks:

        if ((r-W_LEN_1_4)<0) or ((r+W_LEN_3_4)>=len(signal)):           # not enough signal to segment
            continue
        
        segmented_signal = np.array(signal[r-W_LEN_1_4:r+W_LEN_3_4])        # segmenting a heartbeat


        if (normalization):             # Z-score normalization

            if abs(np.std(segmented_signal))<1e-6:          # flat line ECG, will cause zero division error
                continue

            segmented_signal = (segmented_signal - np.mean(segmented_signal)) / np.std(segmented_signal)            

        if not np.isnan(segmented_signal).any():                    # checking for nan, this will never happen
            segmented_signals.append(segmented_signal)



    return segmented_signals, r_peaks           # returning the segmented signals and the refined r-peaks

segmented_signals, refined_r_peaks = segmentSignals(signal, r_peaks_annot)

#print("Segmentos obtenidos:", segmented_signals)
print("Picos R refinados:", refined_r_peaks)


segment = segmented_signals[3]  
sampling_rate = 500  
time_axis = np.arange(len(segment)) / sampling_rate  

r_peak_index = np.argmax(segment)  
r_peak_value = segment[r_peak_index]  

# Grafica del latido y su pico R
plt.figure(figsize=(10, 5))
plt.plot(time_axis, segment, label="Beat Segmentado")
plt.scatter(time_axis[r_peak_index], r_peak_value, color='red', zorder=5)  # Pico R como punto rojo
plt.text(time_axis[r_peak_index], r_peak_value, '  R', color='red', fontsize=12) 
plt.title("Beat Segmentado de ECG")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Línea base
plt.grid(True)
plt.legend()
plt.savefig('latido.png')
plt.show()

# Grafica de la señal completa con sus picos R
plt.plot(signal, label='ECG Signal')
plt.scatter(r_peaks_annot, signal[r_peaks_annot], color='red', label='R-Peaks')
plt.legend()
plt.title('ECG Signal with R-Peaks')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.savefig('señalcompleta.png')
plt.show()