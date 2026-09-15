import io

import datasets
import numpy as np
import soundfile

from garak import _config
from garak.probes.audio import AudioAchillesHeel


class _FakeDataset:
    def __init__(self, audio):
        self._train = [{"audio": audio}]
        self.cast_column_name = None
        self.cast_column_feature = None

    def cast_column(self, name, feature):
        self.cast_column_name = name
        self.cast_column_feature = feature
        return self

    def __getitem__(self, key):
        if key == "train":
            return self._train
        raise KeyError(key)


def test_audio_probe_decodes_dataset_without_torchcodec(monkeypatch, tmp_path):
    audio_bytes = io.BytesIO()
    source_audio = np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32)
    soundfile.write(audio_bytes, source_audio, 16000, format="WAV")
    fake_dataset = _FakeDataset({"bytes": audio_bytes.getvalue(), "path": "sample.wav"})

    audio_data_dir = tmp_path / "audio_achilles"
    monkeypatch.setattr("garak.data.path", tmp_path)

    def load_dataset(_):
        return fake_dataset

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    probe = AudioAchillesHeel(config_root=_config)

    output_audio, sampling_rate = soundfile.read(audio_data_dir / "sample.wav")
    assert fake_dataset.cast_column_name == "audio", "The audio column must be cast"
    assert (
        fake_dataset.cast_column_feature.decode is False
    ), "The dataset must disable TorchCodec decoding"
    assert sampling_rate == 16000, "The source sampling rate must be preserved"
    np.testing.assert_allclose(output_audio, source_audio, atol=1e-4)
    assert len(probe.audio) == 1, "The probe must expose the converted audio file"
