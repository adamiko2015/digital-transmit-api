from flask import Flask, request, jsonify
import numpy as np
from scipy.signal import correlate, butter, lfilter
import logging

app = Flask(__name__)

def calibration_header(sample_rate, low_amp, high_amp):
  total_duration = 4

  time = np.linspace(0, total_duration, sample_rate*total_duration)
  header = np.zeros_like(time)

  header[0:2*sample_rate] = np.sin(2*np.pi*440*time[0:2*sample_rate])
  header[2*sample_rate:3*sample_rate] = low_amp * np.sin(2*np.pi*880*time[2*sample_rate:3*sample_rate])
  header[3*sample_rate:4*sample_rate] = high_amp * np.sin(2*np.pi*880*time[3*sample_rate:4*sample_rate])

  return header

def calibrate(signal, sample_rate):
  # Find start of message
  header_time = np.linspace(0, 2, sample_rate*2)
  reference_header = np.sin(2*np.pi*440*header_time)
  reference_footer = np.sin(2*np.pi*1200*header_time)

  header_cor = correlate(signal, reference_header, mode='valid')
  header_start_idx = np.argmax(header_cor)

  footer_cor = correlate(signal, reference_footer, mode='valid')
  footer_start_idx = np.argmax(footer_cor)

  trimmed = signal[header_start_idx:footer_start_idx] # message from the start (including header) to the start of footer
  message = signal[header_start_idx + 4*sample_rate:]

  # Find amplitudes corresponding to low and high bits
  low_amp_sig = trimmed[2*sample_rate:3*sample_rate]
  high_amp_sig = trimmed[3*sample_rate:4*sample_rate]

  nyquist = sample_rate / 2
  low_cut, high_cut = 800 / nyquist, 960 / nyquist # Filter cutoff frequencies
  b, a = butter(5, [low_cut, high_cut], btype='band')

  low_filtered = lfilter(b, a, low_amp_sig)
  high_filtered = lfilter(b, a, high_amp_sig)

  low_freqs = np.absolute(np.fft.rfft(low_filtered)) / len(low_filtered)
  high_freqs = np.absolute(np.fft.rfft(high_filtered)) / len(high_filtered)

  amps = (max(low_freqs), max(high_freqs))

  return message, amps

@app.route('/encode', methods=['POST'])
def encode_signals():
  # Retrieve parameters
  data = request.get_json()

  app.logger.info('Hi')

  signals = [ np.asarray(s) for s in data['signals'] ]
  bitrate = data['bitrate']
  sample_rate = data['sample_rate']

  header = calibration_header(sample_rate, 1, 2)

  footer_time = np.linspace(0, 2, 2*sample_rate)
  footer = np.sin(2*np.pi*1200*footer_time)

  duration = len(signals[0]) / bitrate
  time = np.linspace(0, duration, int(duration*sample_rate))

  rep_signals = [np.repeat(signal+1, 1/bitrate * sample_rate) for signal in signals]
  sum_signals = np.zeros_like(time)

  for i, signal in enumerate(rep_signals):
    carrier_freq = bitrate * 100 * (i+1)
    carrier = np.sin(2*np.pi*carrier_freq*time)
    sum_signals += carrier * signal

  signal = np.concatenate((header, sum_signals, footer))

  return jsonify(signal.tolist())

def decode_sample(sample, n_signals, amps):
  freqs = np.absolute(np.fft.rfft(sample)) / len(sample) # normalize amplitudes to 1 representing 1, 0.5 representing 0

  top_freq_idx = np.argpartition(freqs, -n_signals)[-n_signals:]
  top_amplitudes = freqs[np.sort(top_freq_idx)]

  decoded = np.zeros(n_signals)

  # convert amplitudes to bits
  for i, amp in enumerate(top_amplitudes):
    if abs(amp - amps[1]) > abs(amp - amps[0]): # if amplitude is closer to 0.5 (representing a 0 bit)
      decoded[i] = 0
    else:
      decoded[i] = 1

  return decoded

@app.route('/decode', methods=['POST'])
def decode_signals():
  # Retrieve parameters
  data = request.get_json()
  raw = np.asarray(data['signal'])
  n_signals = data['n_signals']
  bitrate = data['bitrate']
  sample_rate = data['sample_rate']

  encoded, amps = calibrate(raw, sample_rate)

  decoded = [np.zeros(int(len(encoded)/sample_rate/bitrate)) for _ in range(n_signals)]
  spb = sample_rate / bitrate # samples per bit - how many samples represent a single bit

  n_bit = 0

  for i in range(0, len(encoded), int(spb)):
    sample = encoded[i:i+int(spb)]
    decoded_k = decode_sample(sample, n_signals, amps)

    for k in range(n_signals):
      decoded[k][n_bit] = decoded_k[k]

    n_bit += 1

  signal = jsonify([ d.tolist() for d in decoded ])

  return signal

if __name__ == '__main__':
  app.run(debug=True)
