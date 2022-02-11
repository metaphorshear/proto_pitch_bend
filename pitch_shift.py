# -*- coding: utf-8 -*-
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy.signal import sawtooth, get_window, square
from scipy.fft import fft,ifft #fftshift,ifftshift
import numpy as np
import matplotlib.pyplot as plt
import dtcwt.tf #note: may need to change this for web release.
#(so that users aren't locked to TensorFlow)


from numba import jit

def phase_vocode(coeffs, pitchFactor):
    #from here, I will make a few references to this paper, which I have been cheating off of:
    #http://users.ece.utexas.edu/~bevans/students/ms/jeff_livingston/ms.pdf
    
    #1. convert complex coefficients to magnitude and phase form.
    mag = np.abs(coeffs)
    phase = np.angle(coeffs)
    
    # phase unwrapping.  I'm not sure it corresponds directly to the paper, but
    # it seems to work.
    phase = np.unwrap(phase)
    
    #here I interpolate the magnitudes from the scaled timepoints to the original.
    #I also skip the instanteous frequencies and phase propagation steps.
    #instead, I interpolate the phases directly.
    #===========================================
    mag_t = np.linspace(0, 1, mag.shape[0])
    phase_t = np.linspace(0, 1, phase.shape[0])
    
    if isinstance(pitchFactor, np.ndarray):
        #may need to adjust this more in case pitchFactor has 2 or more dimensions?
        #(i.e., to work along the correct axis.)
        tp = np.linspace(0, 1, num=pitchFactor.shape[0])
        ip = interp1d(tp, pitchFactor, kind='cubic', fill_value='extrapolate')
        mag_s = mag_t * ip(mag_t)
        phase_s = phase_t * ip(phase_t)
    else:    
        mag_s = mag_t * pitchFactor
        phase_s = phase_t * pitchFactor
        
        
    Ymag = np.transpose(
        np.array(
            [ interp1d(mag_s, mag[:,i], fill_value=mag[-1,i], bounds_error=False)(mag_t)
             for i in range(mag.ndim) if mag.shape[i] >= mag_t.shape[0]
             ]
            )
        )
    
    Yphases = np.transpose(
        np.array(
            [ interp1d(phase_s, phase[:,i], fill_value=phase[-1,i],bounds_error=False)(phase_t) 
             for i in range(phase.ndim) if phase.shape[i] >= phase_t.shape[0]
             ]
            )
        )
    
    amount = Ymag.shape[0] - Yphases.shape[0]
    if amount < 0:
        Ymag = np.pad(Ymag, ( (0, abs(amount)), (0,0) ) )
    else:
        Yphases = np.pad(Yphases, ( (0, amount ), (0,0) ) )
            
    #convert scaled magnitude and phase back to complex     
    result = Ymag * np.exp(-1j * Yphases)
    return result

def taper_end(t):
    #put this here so it would be easier to change.
    #I think this should be good, though.
    y = 1-t/(np.sqrt(2+t))
    return y


def bwl_sawtooth(t, frequency, shape=None, rate=44100):
    #t is a numpy array as from linspace or arange.
    #e.g., t=np.linspace(0, 5, num=5*rate)
    #frequency in Hz.
    limit = rate//2
    y = sawtooth(t*frequency*2*np.pi)
    #I think this should work?
    Y = fft(y)
    Y[limit:] = 0
    Y[:frequency] = 0
    y = ifft(Y)
    return y

def padding_wave(t, freq):
    return bwl_sawtooth(t, freq) * taper_end(t)

def pv_with_transform(waveform, pitchFactor, adjustHighpasses=False):
    transform = dtcwt.Transform1d()
    
    tfd = transform.forward(waveform, nlevels=1)
    new_lowpass = phase_vocode(tfd.lowpass, pitchFactor)
    
    #make sure length is correct.    
    ld = tfd.lowpass.shape[0] - new_lowpass.shape[0]
    if ld > 0:
        tfd.lowpass = np.pad(new_lowpass, ((0, ld), (0,0)) )
    else:
        tfd.lowpass = new_lowpass[:(tfd.lowpass.shape[0])]
        
    if adjustHighpasses:
        new_highpasses = [phase_vocode(tfd.highpasses[i], pitchFactor) 
                          for i in range(len(tfd.highpasses))]
        hds = [tfd.highpasses[i].shape[0] - new_highpasses[i].shape[0] for i in range(len(new_highpasses)) ]
        for i in range(len(hds)):
            if hds[i] > 0:
                new_highpasses[i] = np.pad(new_highpasses[i], ((0,hds[i]),(0,0)) )
            else:
                new_highpasses[i] = new_highpasses[i][:(tfd.highpasses[i].shape[0])]
        tfd.highpasses = tuple(new_highpasses)
    return transform.inverse(tfd)


def pitch_shift(waveform, semitones, adjustHighpasses=False):
    """increase or decrease the pitch of a sound

    Args:
        waveform (numpy array): input sound
        semitones (number): add this many semitones to the original.  12 for an octave up.
        adjustHighpasses (bool, optional): whether or not to apply the effect
        to the highpass(es) returned by the wavelet transform. Defaults to False.

    Returns:
        (numpy array): the pitch bent sound
    """
    #adjustHighpasses: True will run the phase vocoder on each high pass.
    #this probably isn't necessary, since the low pass is likely to have
    #most audible frequencies anyway.
    #to decrease by an octave, we're going to stretch out the frequencies.
    #to increase by an octave, we squash them.
    #(in both cases, they're remapped to the original time)
    pitchFactor = 2**(-semitones/12)
    print(semitones, pitchFactor)
    return pv_with_transform(waveform, pitchFactor, adjustHighpasses)

def amplify(orig, shifted):
    maxx = np.max(np.abs(orig))
    maxy = np.max(np.abs(shifted))
    return (shifted/maxy)*maxx

def demo_scale(inputW, outputW):
    sr, orig = wavfile.read(inputW)
    after = np.concatenate(
        [
            amplify(orig, pitch_shift(orig, i)) for i in range(-12, 13)
        ]
    )
    wavfile.write(outputW, sr, after.astype(np.int16))

def bend_helper(times, pitches, seconds=15, sr=44100, kind='linear'):
    """helper function for pitch bending with the phase vocoder

    Args:
        times (list): when the pitch will change.  0 is the beginning and 1 is the end.
        pitches (list): semitones to adjust the pitch at each time.  must be as long as times.
        seconds (int, optional): how long the bend curve should be. Defaults to 15.
        sr (int, optional): sampling rate. Defaults to 44100.
        kind (str, optional): the kind of interpolation to use. Defaults to 'linear'.
        should be one of the types recognized by scipy.interp1d.

    Returns:
        [type]: [description]
    """
    pitches = [2**(-semitones/12) for semitones in pitches]
    t = np.linspace(0, 1, num=seconds*sr)
    y = interp1d(times, pitches, kind, fill_value='extrapolate')
    plt.plot(t, y(t))
    return y(t)

def pitch_bend(waveform, bend_curve):
    """apply a pitch bend (changing pitch over time)

    Args:
        waveform (numpy array): the input sound
        bend_curve (numpy array): amount to shift pitch at each timepoint.

    Returns:
        bent wave
    """
    return pv_with_transform(waveform, bend_curve)

def pitch_bend_demo():#input_wave, bend_wave):
    t = np.linspace(0, 2, num=44100*2)
    y = bend_helper([0, 0.3, 0.7, 1], [0, 1, 2, 3], 2)
    fig, axs = plt.subplots(2,2)
    axs[0][0].plot(t, y)
    z = 0.7 * np.sin(t*np.pi*10)
    axs[0][1].plot(t, z)
    axs[0][1].set_yticks([-1, 1])
    
    w = pv_with_transform(z, 0.5)
    axs[1][0].plot(np.linspace(0,1,num=w.shape[0]), w)
    axs[1][0].set_yticks([-1, 1])
    
    k = pv_with_transform(z, y)
    axs[1][1].plot(np.linspace(0,1,num=k.shape[0]), k)
    axs[1][1].set_yticks([-1, 1])

#@jit
def echo(waveform, ms, level=0.5, sr=44100):
    """add a simple echo effect

    Args:
        waveform (numpy array): the input wave
        ms (int): delay time in milliseconds
        level (float, optional): how loud the first echo should be. Defaults to 0.5.
        sr (int, optional): sample rate. Defaults to 44100.

    Returns:
        wet (numpy array): waveform with the effect applied
    """
    delay_samples = int((ms/1000) * sr)
    window = get_window('hann', delay_samples, fftbins=False)
    if waveform.ndim == 2:
        window = window[:, np.newaxis]
    wet = np.zeros_like(waveform)
    for i in range(1, len(waveform)//delay_samples):
        adding = window * waveform[i*delay_samples:delay_samples*(i+1)] * level
        for j in range(i, len(waveform)//delay_samples):
            wet[j*delay_samples:(j+1)*delay_samples] += adding
            adding *= level
    return wet

#@jit(nopython=True)
def vibrato(waveform, frequency, depth, lfo_type=np.sin, sr=44100):
    """
    vibrato using variable delay.
    Args:
        waveform (numpy array): the carrier wave to apply vibrato (or FM) to
        frequency (float): frequency of the LFO.
        depth (float): how strongly to apply the effect.  the amplitude of the lfo.
        lfo_type (function, optional): how to generate the LFO.  Defaults to sine wave.
        Should work like np.sin.
        sr (int, optional): sampling rate. Defaults to 44100.

    Returns:
        wet (numpy array): waveform with the effect applied
    """
    #TODO: figure out how to use numba on this
    #also not sure if this is quite right.  parameters don't seem to change it enough.
    length = waveform.shape[0]/sr
    t = np.linspace(0, length, num=waveform.shape[0])
    lfo = lfo_type(t*2*np.pi*frequency) * depth
    wet = np.zeros_like(waveform)
    for i in range(len(t)):
        if lfo[i] == 0:
            wet[i] = waveform[i]
        else:
            other = lfo[i]
            if other < 0:
                other = np.floor(other)
            else:
                other = np.ceil(other)
            wet[i] = lfo[i] * waveform[int(other)] + waveform[i] * (depth-lfo[i])
    return wet



infile = "test.wav"
outfile = "output2.wav"

sr, orig = wavfile.read(infile)
bent = amplify(orig, pv_with_transform(orig, bend_helper([0,0.25,0.5,0.7,1], [0,2,3,4,4])))
#delayed = echo(bent, 180, 0.8, 44100)
#delayed = delayed[0] + delayed[1]
vibed = (vibrato(bent, 5, 0.35) + bent)*0.75
wavfile.write(outfile, sr, vibed.astype(np.int16))


#t = np.linspace(0, 1, num=44100)
#x = bwl_sawtooth(t, 15)
#y = delay(x, 100, level=0.3)
#fig, axs = plt.subplots(1,2)
#axs[0].plot(t, x)
#axs[1].plot(t, y)

#TODO: maybe find a resource on autocorrelation?
#there's also this: https://github.com/audacity/audacity/blob/5519fb9fefc34b9533374d19309d00776cc00022/src/SpectrumAnalyst.cpp

def test_sawtooth():
    #it should now be obvious that I am making things up as I go along.
    fig, axs = plt.subplots(2, 2)
    t = np.linspace(0, 1, num=44100)
    y = sawtooth(t*2*np.pi*50)
    z = bwl_sawtooth(t, 50)
    axs[0][0].plot(t,y)
    axs[0][1].plot(t,z)
    from scipy.signal import correlate
    transform = dtcwt.Transform1d()
    Y = transform.forward(y).lowpass#.highpasses[0]
    Z = transform.forward(z).lowpass#.highpasses[0]
    u = correlate(Y, Y)
    v = correlate(Z, Z)
    axs[1][0].plot(u[u.size//2:])
    axs[1][1].plot(v[v.size//2:])
#test_sawtooth()

#plt.plot(padding_wave(np.linspace(0, 1, num=44100), 440))