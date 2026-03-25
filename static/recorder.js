let mediaRecorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            audioChunks = [];

            mediaRecorder.ondataavailable = e => {
                audioChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(audioChunks, { type: 'audio/wav' });
                const file = new File([blob], 'recorded_audio.wav');

                const input = document.querySelector('input[name="audioFile"]');
                const container = new DataTransfer();
                container.items.add(file);
                input.files = container.files;

                const audio = document.getElementById('recordedAudio');
                audio.src = URL.createObjectURL(blob);
                audio.style.display = 'block';
            };
        });
}

function stopRecording() {
    if (mediaRecorder) mediaRecorder.stop();
}
