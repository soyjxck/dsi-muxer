"""
DSI Muxer — Racjin PS2 DSI container multiplexer/demultiplexer.

The DSI format is a proprietary streaming container used by Racjin for
PS2 cutscenes (Fullmetal Alchemist, Busin series). It interleaves MPEG-2
video and PS2 SPU ADPCM audio in fixed 0x40000-byte blocks.

Usage:
    from dsi_muxer import DSI

    # Demux
    dsi = DSI.from_bytes(data)
    video = dsi.extract_video()
    audio = dsi.extract_audio()

    # Mux
    dsi = DSI.mux(video_data, audio_data, nblocks=307)
    output = dsi.to_bytes()
"""

from .container import DSI, DSIBlock

__version__ = "1.0.0"
__all__ = ["DSI", "DSIBlock"]
