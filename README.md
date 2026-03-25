# Manatee Call Detector

A web-based audio analysis system for detecting manatee vocalizations from uploaded audio files. This project combines signal preprocessing, feature extraction, and machine learning models to identify potential manatee sounds and summarize the results in a browser-based interface.

## Overview

This project was developed as part of a team research effort focused on manatee vocalization detection and classification from underwater audio recordings.

The system includes:

- audio upload and browser-based result display
- preprocessing pipeline for audio filtering and segmentation
- HMM-based detection
- LSTM-based detection
- ensemble decision-making using both models
- optional classification modules for vocal-type prediction

## My Contribution

In this group project, my main contribution focused on the HMM-based system and the audio processing pipeline, including:

- designing and implementing the HMM-based detection workflow
- building the MFCC extraction and preprocessing pipeline
- contributing to filtering, segmentation, and feature preparation
- integrating model inference into the application workflow
- contributing to the web application for real-time analysis

Other deep learning components were developed as part of the broader team system.

## Features

- Upload `.wav` or `.mp3` audio files through a Flask web app
- Automatic duration validation for uploaded files
- Bandpass filtering and noise reduction
- Energy-based segmentation of candidate sound events
- MFCC feature extraction
- HMM and LSTM detection pipelines
- Ensemble-based final manatee detection
- Result table with segment timestamps and confidence scores
- CSV export of results

## Methodology

1. Audio is uploaded through the web interface
2. The audio is converted to WAV if needed
3. The signal is filtered and denoised
4. The waveform is split into candidate segments using energy thresholds
5. MFCC-based features are extracted from each segment
6. HMM and LSTM detection models evaluate each segment
7. Final detection is produced using ensemble voting
8. Results are displayed and can be downloaded as CSV

## Tech Stack

- Python
- Flask
- Librosa
- NumPy / Pandas
- SciPy
- TensorFlow / Keras
- hmmlearn
- pydub

## Notes

This repository contains the inference pipeline, preprocessing workflow, and web application used for manatee vocalization analysis.

Training code, original datasets, and certain research artifacts are not included due to privacy and project restrictions.
