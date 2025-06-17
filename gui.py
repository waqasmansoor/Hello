import sys
import os
import time
import sounddevice as sd
import soundfile as sf
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QInputDialog,QListWidget,QHBoxLayout,QLineEdit,
    QDialog
)
from PyQt5.QtCore import QTimer
from embeddings import save_embeddings
from speechbrain.pretrained import EncoderClassifier
from predict import predict_speaker

classifier = EncoderClassifier.from_hparams(source="model/spkrec-ecapa-voxceleb")

class SpeakerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent=parent
        self.setWindowTitle("Save Recording")

        self.name_input = QLineEdit()
        self.label = QLabel("Enter speaker name:")

        # Buttons
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.test_button = QPushButton("Test")

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.test_button.clicked.connect(self.test_action)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.test_button)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.name_input)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.test_clicked = False

    def test_action(self):
        self.test_clicked = True
        self.pred,self.score=predict_speaker(self.parent.audio,self.parent.samplerate,classifier)
        self.accept()

    def get_data(self):
        return self.name_input.text().strip(), self.test_clicked,self.pred,self.score
    
class VoiceRecorder(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Voice Recorder')

        self.status_label = QLabel('Status: Ready')
        
        
        self.record_btn = QPushButton('Record 5 seconds')
        self.record_btn.clicked.connect(self.start_countdown)

        center_layout = QVBoxLayout()
        center_layout.addWidget(self.status_label)
        center_layout.addWidget(self.record_btn)

        self.speaker_list = QListWidget()
        self.speaker_label = QLabel("Saved Speakers:")
        right_layout=QVBoxLayout()
        right_layout.addWidget(self.speaker_label)
        right_layout.addWidget(self.speaker_list)
        self.update_speaker_list()

        main_layout=QHBoxLayout()
        main_layout.addLayout(center_layout,1)
        main_layout.addLayout(right_layout,2)
        self.setLayout(main_layout)


        self.timer = QTimer()
        self.timer.timeout.connect(self.update_countdown)
        self.counter = 5
        self.audio = None
        self.samplerate = 16000

    def update_speaker_list(self):
        dataset_dir = "dataset"
        self.speaker_list.clear()
        if not os.path.exists(dataset_dir):
            return 0
        speakers = sorted([
            name for name in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir,name))
        ])
        self.speaker_list.addItems(speakers)


    def start_countdown(self):
        self.record_btn.setEnabled(False)
        self.counter = 5
        self.record_btn.setText(str(self.counter))
        self.status_label.setText("Recording...")
        self.audio = sd.rec(int(5 * self.samplerate), samplerate=self.samplerate, channels=1, dtype='float32')
        self.timer.start(1000)

    def update_countdown(self):
        self.counter -= 1
        if self.counter >= 0:
            self.record_btn.setText(str(self.counter))
        if self.counter < 0:
            self.timer.stop()
            sd.wait()
            self.ask_and_save()
            self.record_btn.setText("Record 5 seconds")
            self.record_btn.setEnabled(True)

    def ask_and_save(self):
        
        dialog = SpeakerDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name,test_clicked,pred,score = dialog.get_data()
            
            save_embdgs=False
            if not test_clicked and name:
                name = name.strip()
                folder = os.path.join('dataset', name)
                os.makedirs(folder, exist_ok=True)
                timestamp = int(time.time())
                filepath = os.path.join(folder, f'{timestamp}.wav')
                sf.write(filepath, self.audio, self.samplerate)
                
                self.update_speaker_list()
                save_embdgs=True
            elif test_clicked:
                self.status_label.setText(f"Speaker: {pred}:{score:.4f}")
            else:
                self.status_label.setText("Recording discarded (no name entered)")
        if save_embdgs:
            save_embeddings(name,self.status_label,classifier)
            self.status_label.setText(f'Thanks {name} for recording your voice.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    recorder = VoiceRecorder()
    recorder.show()
    sys.exit(app.exec_())
