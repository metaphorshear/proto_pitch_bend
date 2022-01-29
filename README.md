# proto pitch bend
This was a little project that started when I wanted to build a whammy pedal.  Before ordering a microcontroller/tiny computer, it seemed wiser to get the proof of concept working in Python.  So that's what I did, and that's what this is.  It's still a work in progress, and it isn't designed to be maintainable so much as quick to experiment with (because I don't quite know what I'm doing.)

##What's it do?
It can pitch shift and pitch bend.  To pitch shift:
```
shifted = pitch_shift(waveform, semitones)
```
To pitch bend:
```
curve = up_bend([0, 0.35, 0.6, 0.9, 1], [0, 1, 3, 5, 7, 7])
shifted = pv_with_transform(waveform, curve)
```

## How does it work?
It mostly implements the phase vocoder algorithm from this paper: http://users.ece.utexas.edu/~bevans/students/ms/jeff_livingston/ms.pdf.

So, at first I looked into using a Fast Fourier Transform, because I knew that those would separate audio waveforms into frequencies.  Unfortunately, that's nearly all of what I knew about them, and using FFTs to change pitch wasn't as simple as I first expected.  Luckily, other people have had the same thoughts, and those people were advised to look into the phase vocoder.  The phase vocoder converts a wave to the frequency domain, remaps the phases and magnitudes to a new timescale, and then converts the wave back to the time domain.  

The phase vocoder traditionally uses a "short-time Fourier transform" (STFT), which I mostly didn't look into due to immediately learning about wavelets.  While Fourier transforms split a waveform into a series of sine waves (and maybe cosine waves), wavelet transforms decompose the original waveform into a series of wavelets.  Wavelets are finite, so they combine windowing and the time-to-frequency conversion in one step, while the STFT would've likely required more steps on my part, and thus more points of failure. Wavelet transforms split a wave into high frequencies and low frequencies, so that for every additional decomposition level, the high frequencies are split up again.  Now, it tends to set the high pass and low pass so high that nearly all audible content will be in the first low pass, but there's code to change the high passes, anyway.  Results sound fine to me without changing the high pass at all.

There are several libraries for Python that implement wavelet transforms (including scipy and pywavelets), and using the paper linked above, I didn't have to think too much about it.  Of course, before that paper, I found others.  A few tried using the vanilla complex wavelet transform (CWT), but it's pretty slow.  The discrete wavelet transform (DWT) is *much* faster, but it only works with real numbers, and you really need complex numbers for the phase vocoder algorithm.  The dual-tree complex wavelet transform solves that issue.  It uses discrete wavelet transforms on two trees: one for reals and one for imaginary numbers.  It's pretty neat.  The package `dtcwt` was pretty straightforward to use, supports OpenCL and TensorFlow backends for more speed, and just made this whole process relatively painless.

## What's wrong with it?
When bending a pitch up, you might run out of data for interpolation, which leaves a bit of silence at the end of the audio.  I'm going to try adding delay to fix that.  The code might be kind of a mess from long hours shifting things around and trying to solve different problems.  It still needs to be adapted for/tested on real-time audio.  `up_bend` is a misleading name for a function that can bend in either direction, and it doesn't interpolate the bend curve well enough.  Linear interpolation has some strange sounds, and quadratic or higher kind of curve back in the wrong direction.  I didn't try to learn the math of Fourier series or wavelet transforms, so there may be subtleties (or obviousities) that I missed.  The comments need to be converted to helpful docstrings in a less annoying IDE than Spyder.

#What's next?
As mentioned, I'm going to work on delay so upbends don't end too soon.  This project has been fun, and it feels like I learned a fair bit.  I would like to try creating distortion/overdrive, maybe some more time-based effects, and autocorrelation to help with debugging.

I haven't ordered the Daisy yet, but when I do, I'm going to port this to it and try using my old BOSS expression pedal with it.  If I could find more expression pedals somewhere, and a decent enclosure, I might try selling some on Etsy.  Whammy pedals usually go for around $200, but it seems possible that one could be built for under $50.
