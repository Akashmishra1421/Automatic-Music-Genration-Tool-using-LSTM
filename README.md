# 🎵 Automatic Music Generation Tool Using LSTM

A deep learning-based tool for automatic music generation using LSTM neural networks. This project processes MIDI files, trains an LSTM model to learn musical patterns, and generates new music in MIDI format.

---

## 📁 Directory Structure

```
akashmishra1421-automatic-music-genration-tool-using-lstm/
├── demo.py
├── maestro-v2.0.0-midi.zip
├── music_generator.py
├── requirements.txt
├── weights.best.music3.keras
├── Jazz/
│   ├── *.midi
└── Music Generation/
    └── notes
```

---

## ✨ Features

* Train an LSTM model on a set of MIDI files to learn musical patterns.
* Generate new music in MIDI format based on learned sequences.
* Easy-to-use scripts for both training and generation.
* Includes sample Jazz MIDI dataset for experimentation.

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare the Jazz MIDI Dataset

> **Important:**
> Extract MIDI files from `maestro-v2.0.0-midi.zip` and place all `.mid` or `.midi` files **directly** inside the `Jazz/` directory.
> The `Jazz/` folder must **only contain MIDI files** — no subfolders or other file types.

### 3. Train the Model

```bash
python music_generator.py
```

This command processes MIDI files, extracts musical notes, prepares sequences, trains the LSTM model, and saves the best weights.

### 4. Generate Music

```bash
python demo.py
```

This script loads the trained model and generates a new MIDI file based on learned musical patterns.

---

## 📄 File Descriptions

| File/Folder                 | Description                                               |
| --------------------------- | --------------------------------------------------------- |
| `demo.py`                   | Script to generate music using the trained model.         |
| `music_generator.py`        | Main script for training the model and generating music.  |
| `requirements.txt`          | List of required Python packages.                         |
| `weights.best.music3.keras` | Saved best model weights after training.                  |
| `maestro-v2.0.0-midi.zip`   | Archive containing MIDI files for training.               |
| `Jazz/`                     | Directory to store MIDI files for training.               |
| `Music Generation/notes`    | Pickled notes extracted from MIDI files (auto-generated). |

---

## 📝 Notes

* Do not place any folders or non-MIDI files in the `Jazz/` directory.
* At least **101 notes** are required for the model to function.
* If not enough notes are found, the tool will prompt to add more MIDI files.
* Generated MIDI files are saved with timestamped filenames in the project root.

---

## 💻 Example Usage

```python
# demo.py
from music_generator import generate

generate()
```

---

## 📜 License

[MIT License](LICENSE) – See the LICENSE file for more details.



---

## 🙏 Acknowledgements

* Uses `music21` for MIDI file processing.
* LSTM model implemented using TensorFlow/Keras.

> **Special Instruction:**
> After downloading or cloning the repository, extract all MIDI files from `maestro-v2.0.0-midi.zip` and place them directly inside the `Jazz/` directory.
> Ensure the `Jazz/` directory contains only `.mid` or `.midi` files for proper functioning.
