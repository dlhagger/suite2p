import pytest

from suite2p.io import tiff


def scanimage_metadata(**overrides):
    metadata = {
        "SI.hRoiManager.scanFrameRate": 30.0,
        "SI.hRoiManager.scanVolumeRate": 30.0,
        "SI.hRoiManager.linesPerFrame": 512,
        "SI.hRoiManager.pixelsPerLine": 512,
        "SI.hStackManager.enable": False,
        "SI.hStackManager.actualNumSlices": 1,
        "SI.hStackManager.numSlices": 1,
        "SI.hStackManager.numFramesPerVolumeWithFlyback": 1,
        "SI.hFastZ.discardFlybackFrames": False,
        "SI.hChannels.channelSave": 1,
    }
    metadata.update(overrides)
    return metadata


def test_single_plane_sampling_frequency_uses_frame_rate():
    result = tiff.sampling_frequency_from_scanimage_metadata(
        scanimage_metadata(
            **{
                "SI.hRoiManager.scanFrameRate": 29.9659,
                "SI.hRoiManager.scanVolumeRate": 29.9659,
            }
        ),
        configured_nplanes=1,
    )

    assert result["sampling_frequency"] == pytest.approx(29.9659)
    assert result["source_key"] == "SI.hRoiManager.scanFrameRate"
    assert not result["plane_mismatch"]


def test_multiplane_sampling_frequency_uses_volume_rate():
    result = tiff.sampling_frequency_from_scanimage_metadata(
        scanimage_metadata(
            **{
                "SI.hRoiManager.scanFrameRate": 60.0,
                "SI.hRoiManager.scanVolumeRate": 18.0,
                "SI.hStackManager.enable": True,
                "SI.hStackManager.numSlices": 3,
                "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
            }
        ),
        configured_nplanes=3,
    )

    assert result["sampling_frequency"] == pytest.approx(18.0)
    assert result["source_key"] == "SI.hRoiManager.scanVolumeRate"


def test_scanimage_metadata_prefills_saved_channel_and_frame_shape():
    result = tiff.scanimage_acquisition_metadata(
        scanimage_metadata(
            **{
                "SI.hChannels.channelSave": 2,
                "SI.hRoiManager.imagingFovUm": [
                    [-387.878, -387.878],
                    [387.878, -387.878],
                    [387.878, 387.878],
                    [-387.878, 387.878],
                ],
            }
        )
    )

    assert result["nplanes"] == 1
    assert result["nchannels"] == 1
    assert result["functional_chan"] == 1
    assert result["saved_channels"] == [2]
    assert result["pixels_per_line"] == 512
    assert result["lines_per_frame"] == 512
    assert result["microns_per_pixel"] == pytest.approx([1.5151484375, 1.5151484375])


def test_scanimage_metadata_prefills_flyback_planes_when_saved():
    result = tiff.scanimage_acquisition_metadata(
        scanimage_metadata(
            **{
                "SI.hStackManager.enable": True,
                "SI.hStackManager.actualNumSlices": 3,
                "SI.hStackManager.numSlices": 3,
                "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
                "SI.hFastZ.discardFlybackFrames": False,
            }
        )
    )

    assert result["nplanes"] == 4
    assert result["ignore_flyback"] == [3]


def test_scanimage_metadata_uses_saved_slice_count_when_flyback_discarded():
    result = tiff.scanimage_acquisition_metadata(
        scanimage_metadata(
            **{
                "SI.hStackManager.enable": True,
                "SI.hStackManager.actualNumSlices": 3,
                "SI.hStackManager.numSlices": 3,
                "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
                "SI.hFastZ.discardFlybackFrames": True,
            }
        )
    )

    assert result["nplanes"] == 3
    assert result["ignore_flyback"] == []


def test_scanimage_metadata_prefills_two_saved_channels_without_guessing_functional():
    result = tiff.scanimage_acquisition_metadata(
        scanimage_metadata(**{"SI.hChannels.channelSave": [1, 2]})
    )

    assert result["nchannels"] == 2
    assert result["saved_channels"] == [1, 2]
    assert result["functional_chan"] is None


def test_multiplane_sampling_frequency_falls_back_to_flyback_aware_rate():
    metadata = scanimage_metadata(
        **{
            "SI.hRoiManager.scanFrameRate": 60.0,
            "SI.hRoiManager.scanVolumeRate": None,
            "SI.hStackManager.enable": "true",
            "SI.hStackManager.numSlices": 3,
            "SI.hStackManager.numFramesPerVolumeWithFlyback": 4,
        }
    )
    result = tiff.sampling_frequency_from_scanimage_metadata(
        metadata, configured_nplanes=3
    )

    assert result["sampling_frequency"] == pytest.approx(15.0)
    assert "numFramesPerVolumeWithFlyback" in result["source_key"]


def test_inference_rejects_inconsistent_tiff_rates(monkeypatch):
    rates = {"first.tif": 30.0, "second.tif": 20.0}

    def fake_read(filename, configured_nplanes=None):
        result = tiff.sampling_frequency_from_scanimage_metadata(
            scanimage_metadata(
                **{"SI.hRoiManager.scanFrameRate": rates[filename]}
            ),
            configured_nplanes=configured_nplanes,
        )
        result["filename"] = filename
        return result

    monkeypatch.setattr(tiff, "read_scanimage_sampling_frequency", fake_read)

    with pytest.raises(ValueError, match="inconsistent"):
        tiff.infer_tiff_sampling_frequency(
            ["first.tif", "second.tif"], configured_nplanes=1
        )


def test_inference_rejects_plane_count_mismatch(monkeypatch):
    def fake_read(filename, configured_nplanes=None):
        result = tiff.sampling_frequency_from_scanimage_metadata(
            scanimage_metadata(), configured_nplanes=configured_nplanes
        )
        result["filename"] = filename
        return result

    monkeypatch.setattr(tiff, "read_scanimage_sampling_frequency", fake_read)

    with pytest.raises(ValueError, match="does not match"):
        tiff.infer_tiff_sampling_frequency(
            ["first.tif"], configured_nplanes=2
        )


def test_acquisition_inference_rejects_inconsistent_channel_metadata(monkeypatch):
    channels = {"first.tif": [1], "second.tif": [1, 2]}

    def fake_read(filename, configured_nplanes=None):
        result = tiff.scanimage_acquisition_metadata(
            scanimage_metadata(
                **{"SI.hChannels.channelSave": channels[filename]}
            ),
            configured_nplanes=configured_nplanes,
        )
        result["filename"] = filename
        return result

    monkeypatch.setattr(tiff, "read_scanimage_acquisition_metadata", fake_read)

    with pytest.raises(ValueError, match="nchannels"):
        tiff.infer_tiff_acquisition_metadata(["first.tif", "second.tif"])
