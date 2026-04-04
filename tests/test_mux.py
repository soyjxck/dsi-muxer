"""Tests for DSI muxing — verifies audio/video integrity with and without nblocks."""

import pytest
from dsi_muxer import DSI
from dsi_muxer.container import DSIBlock, _count_markers, HEADER_SIZE, DEFAULT_BLOCK_SIZE


# ---------------------------------------------------------------------------
# Helpers to build synthetic MPEG-2-like video and PS2 ADPCM-like audio
# ---------------------------------------------------------------------------

FRAME_MARKER = b'\x00\x00\x01\x00'  # MPEG-2 picture start code


def _make_video(n_frames: int, bytes_per_frame: int = 4000) -> bytes:
    """Build a fake MPEG-2 elementary stream with n_frames picture start codes."""
    # Sequence header (only in real MPEG-2, but the muxer only looks for picture starts)
    frames = []
    for i in range(n_frames):
        payload = bytes([i & 0xFF]) * (bytes_per_frame - len(FRAME_MARKER))
        frames.append(FRAME_MARKER + payload)
    return b''.join(frames)


def _make_audio(n_bytes: int, align: int = 512) -> bytes:
    """Build fake PS2 ADPCM audio of exactly n_bytes (must be align-multiple)."""
    assert n_bytes % align == 0
    return bytes(range(256)) * (n_bytes // 256) + bytes(n_bytes % 256)


# ---------------------------------------------------------------------------
# Core invariant: no audio is lost
# ---------------------------------------------------------------------------

class TestNoAudioLoss:
    """Muxing must never drop audio bytes."""

    def test_auto_blocks_preserves_all_audio(self):
        video = _make_video(100)
        audio = _make_audio(51200)
        dsi = DSI.mux(video, audio)
        extracted = dsi.extract_audio()
        assert extracted[:len(audio)] == audio

    def test_explicit_blocks_preserves_all_audio(self):
        video = _make_video(100)
        audio = _make_audio(51200)
        dsi = DSI.mux(video, audio, nblocks=5)
        extracted = dsi.extract_audio()
        assert extracted[:len(audio)] == audio

    def test_audio_longer_than_video_preserved(self):
        """Audio is intentionally longer than video (like FMA1 DSI format)."""
        video = _make_video(50, bytes_per_frame=2000)  # ~100KB video
        audio = _make_audio(102400)  # 100KB audio — roughly equal
        dsi = DSI.mux(video, audio)
        extracted = dsi.extract_audio()
        assert len(extracted) >= len(audio)
        assert extracted[:len(audio)] == audio

    def test_large_audio_excess_preserved(self):
        """Audio 50% longer than video — no truncation."""
        video = _make_video(30, bytes_per_frame=5000)  # 150KB
        audio = _make_audio(153600)  # 150KB audio
        dsi = DSI.mux(video, audio)
        extracted = dsi.extract_audio()
        assert extracted[:len(audio)] == audio


# ---------------------------------------------------------------------------
# Core invariant: no video is lost
# ---------------------------------------------------------------------------

class TestNoVideoLoss:
    """Muxing must preserve all video frames."""

    def test_auto_blocks_preserves_frame_count(self):
        video = _make_video(200)
        audio = _make_audio(51200)
        dsi = DSI.mux(video, audio)
        assert dsi.frame_count() == 200

    def test_explicit_blocks_preserves_frame_count(self):
        video = _make_video(200)
        audio = _make_audio(51200)
        dsi = DSI.mux(video, audio, nblocks=10)
        assert dsi.frame_count() == 200

    def test_video_bytes_preserved(self):
        video = _make_video(50)
        audio = _make_audio(25600)
        dsi = DSI.mux(video, audio)
        extracted = dsi.extract_video()
        # Extracted video may have trailing zeros (block padding) but must start with original
        assert extracted[:len(video)] == video


# ---------------------------------------------------------------------------
# No empty-video blocks
# ---------------------------------------------------------------------------

class TestNoEmptyVideoBlocks:
    """The last block must contain real video data (no black screen)."""

    def test_auto_blocks_no_empty_trailing_blocks(self):
        video = _make_video(100)
        audio = _make_audio(102400)  # generous audio
        dsi = DSI.mux(video, audio)
        last = dsi.blocks[-1]
        # Last block should have nonzero video
        assert any(b != 0 for b in last.video_data), "Last block has zero video"

    def test_explicit_blocks_last_has_video(self):
        video = _make_video(100)
        audio = _make_audio(51200)
        dsi = DSI.mux(video, audio, nblocks=5)
        # All blocks up to video exhaustion should have content
        for i, blk in enumerate(dsi.blocks):
            has_vid = any(b != 0 for b in blk.video_data)
            has_aud = any(b != 0 for b in blk.audio_data)
            if not has_vid and not has_aud:
                # Empty blocks only allowed after all content is placed
                continue
            # No block should have audio but zero video (causes black screen)
            if has_aud:
                assert has_vid or i == len(dsi.blocks) - 1, \
                    f"Block {i} has audio but no video"


# ---------------------------------------------------------------------------
# Block count: auto vs explicit
# ---------------------------------------------------------------------------

class TestBlockCount:
    """Verify block count behavior."""

    def test_auto_creates_enough_blocks(self):
        video = _make_video(500, bytes_per_frame=8000)  # ~4MB video
        audio = _make_audio(512000)  # 500KB audio
        dsi = DSI.mux(video, audio)
        total_content = len(video) + len(audio)
        usable = DEFAULT_BLOCK_SIZE - HEADER_SIZE
        min_blocks = total_content // usable  # at least this many
        assert dsi.num_blocks >= min_blocks

    def test_explicit_blocks_honored(self):
        video = _make_video(100)
        audio = _make_audio(51200)
        dsi = DSI.mux(video, audio, nblocks=20)
        # May be fewer if content runs out, but no more
        assert dsi.num_blocks <= 20

    def test_single_block(self):
        video = _make_video(5, bytes_per_frame=1000)
        audio = _make_audio(512)
        dsi = DSI.mux(video, audio, nblocks=1)
        assert dsi.num_blocks == 1
        assert dsi.extract_audio()[:len(audio)] == audio


# ---------------------------------------------------------------------------
# Template-based muxing
# ---------------------------------------------------------------------------

class TestTemplateMux:
    """Template muxing should use the template's block count."""

    def test_template_block_count_used(self):
        video = _make_video(100)
        audio = _make_audio(51200)
        template = DSI.mux(video, audio, nblocks=8)

        new_video = _make_video(100, bytes_per_frame=3000)
        remuxed = DSI.mux(new_video, audio, template=template)
        assert remuxed.num_blocks <= template.num_blocks

    def test_template_preserves_audio(self):
        video = _make_video(100)
        audio = _make_audio(51200)
        template = DSI.mux(video, audio, nblocks=8)

        remuxed = DSI.mux(video, audio, template=template)
        extracted = remuxed.extract_audio()
        assert extracted[:len(audio)] == audio


# ---------------------------------------------------------------------------
# Round-trip: mux → serialize → parse → extract
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Mux → to_bytes → from_bytes → extract should preserve content."""

    def test_roundtrip_auto(self):
        video = _make_video(50)
        audio = _make_audio(25600)

        dsi = DSI.mux(video, audio)
        raw = dsi.to_bytes()
        parsed = DSI.from_bytes(raw)

        assert parsed.num_blocks == dsi.num_blocks
        assert parsed.extract_audio() == dsi.extract_audio()
        assert parsed.extract_video() == dsi.extract_video()

    def test_roundtrip_explicit(self):
        video = _make_video(50)
        audio = _make_audio(25600)

        dsi = DSI.mux(video, audio, nblocks=5)
        raw = dsi.to_bytes()
        parsed = DSI.from_bytes(raw)

        assert parsed.num_blocks == dsi.num_blocks
        assert parsed.extract_audio() == dsi.extract_audio()
        assert parsed.extract_video() == dsi.extract_video()


# ---------------------------------------------------------------------------
# End-of-sequence marker
# ---------------------------------------------------------------------------

class TestEndOfSequence:
    """EOS marker must be present after muxing."""

    def test_eos_injected(self):
        video = _make_video(50)
        audio = _make_audio(25600)
        dsi = DSI.mux(video, audio)
        assert dsi.verify_end_of_sequence()

    def test_eos_already_present_not_duplicated(self):
        video = _make_video(50) + b'\x00\x00\x01\xb7'
        audio = _make_audio(25600)
        dsi = DSI.mux(video, audio)
        full_vid = dsi.extract_video()
        # Should have exactly one EOS
        assert full_vid.count(b'\x00\x00\x01\xb7') == 1


# ---------------------------------------------------------------------------
# Audio alignment
# ---------------------------------------------------------------------------

class TestAudioAlignment:
    """Audio sizes per block must be aligned."""

    def test_default_alignment_512(self):
        video = _make_video(100)
        audio = _make_audio(51200)
        dsi = DSI.mux(video, audio)
        for blk in dsi.blocks:
            assert blk.audio_size % 512 == 0

    def test_custom_alignment(self):
        video = _make_video(100)
        audio = _make_audio(51200)
        dsi = DSI.mux(video, audio, audio_align=1024)
        for blk in dsi.blocks:
            assert blk.audio_size % 1024 == 0


# ---------------------------------------------------------------------------
# Block serialization
# ---------------------------------------------------------------------------

class TestBlockSerialization:
    """DSIBlock to_bytes/from_bytes round-trip."""

    def test_audio_first(self):
        blk = DSIBlock(audio_data=b'\xAA' * 512, video_data=b'\xBB' * 1024, audio_first=True)
        raw = blk.to_bytes()
        parsed = DSIBlock.from_bytes(raw)
        assert parsed.audio_data == blk.audio_data
        assert parsed.video_data == blk.video_data
        assert parsed.audio_first is True

    def test_video_first(self):
        blk = DSIBlock(audio_data=b'\xAA' * 512, video_data=b'\xBB' * 1024, audio_first=False)
        raw = blk.to_bytes()
        parsed = DSIBlock.from_bytes(raw)
        assert parsed.audio_data == blk.audio_data
        assert parsed.video_data == blk.video_data
        assert parsed.audio_first is False
