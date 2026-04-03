"""
DSI container format implementation.

Block structure (configurable block size, default 0x40000 bytes):
    +0x00  [32 bytes]  Header (8 x uint32 LE)
    +0x20  [32 bytes]  Zero padding
    +0x40  [varies]    Stream 1 data
    +????  [varies]    Stream 2 data

Header fields:
    f0: Stream count (always 2)
    f1: Offset of stream 1 data (always 64)
    f2: Stream 1 type tag (-8192 = audio, -16384 = video)
    f3: Stream 1 data size
    f4: Offset of stream 2 data
    f5: Stream 2 type tag
    f6: Stream 2 data size
    f7: Reserved (always 0)
"""

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Defaults (matching FMA / Racjin games)
DEFAULT_BLOCK_SIZE = 0x40000
HEADER_SIZE = 64
AUDIO_TAG = -8192
VIDEO_TAG = -16384
DEFAULT_AUDIO_ALIGN = 512  # PS2 ADPCM stereo interleave pair


@dataclass
class DSIBlock:
    """A single DSI block."""
    audio_data: bytes
    video_data: bytes
    audio_first: bool = True

    @property
    def audio_size(self) -> int:
        return len(self.audio_data)

    @property
    def video_size(self) -> int:
        return len(self.video_data)

    @classmethod
    def from_bytes(cls, data: bytes, block_size: int = DEFAULT_BLOCK_SIZE) -> 'DSIBlock':
        """Parse a single block."""
        if len(data) < block_size:
            raise ValueError(f"Block too small: {len(data)} < {block_size}")

        hdr = struct.unpack('<IIiIIiII', data[:32])

        if hdr[2] == AUDIO_TAG:
            aud_off, aud_sz = hdr[1], hdr[3]
            vid_off, vid_sz = hdr[4], hdr[6]
            audio_first = True
        else:
            vid_off, vid_sz = hdr[1], hdr[3]
            aud_off, aud_sz = hdr[4], hdr[6]
            audio_first = False

        return cls(
            audio_data=data[aud_off:aud_off + aud_sz],
            video_data=data[vid_off:vid_off + vid_sz],
            audio_first=audio_first,
        )

    def to_bytes(self, block_size: int = DEFAULT_BLOCK_SIZE) -> bytes:
        """Serialize to a block of the given size."""
        aud_sz = len(self.audio_data)
        vid_sz = len(self.video_data)
        out = bytearray(block_size)

        if self.audio_first:
            struct.pack_into('<IIiIIiII', out, 0,
                2, HEADER_SIZE, AUDIO_TAG, aud_sz,
                HEADER_SIZE + aud_sz, VIDEO_TAG, vid_sz, 0)
            out[HEADER_SIZE:HEADER_SIZE + aud_sz] = self.audio_data
            out[HEADER_SIZE + aud_sz:HEADER_SIZE + aud_sz + vid_sz] = self.video_data
        else:
            struct.pack_into('<IIiIIiII', out, 0,
                2, HEADER_SIZE, VIDEO_TAG, vid_sz,
                HEADER_SIZE + vid_sz, AUDIO_TAG, aud_sz, 0)
            out[HEADER_SIZE:HEADER_SIZE + vid_sz] = self.video_data
            out[HEADER_SIZE + vid_sz:HEADER_SIZE + vid_sz + aud_sz] = self.audio_data

        return bytes(out)


class DSI:
    """DSI container — a sequence of blocks containing interleaved audio and video."""

    def __init__(self, blocks: List[DSIBlock], block_size: int = DEFAULT_BLOCK_SIZE):
        self.blocks = blocks
        self.block_size = block_size

    @classmethod
    def from_bytes(cls, data: bytes, block_size: int = DEFAULT_BLOCK_SIZE) -> 'DSI':
        """Parse a DSI file from raw bytes."""
        nblocks = len(data) // block_size
        blocks = []
        for i in range(nblocks):
            block_data = data[i * block_size:(i + 1) * block_size]
            blocks.append(DSIBlock.from_bytes(block_data, block_size))
        return cls(blocks, block_size)

    @classmethod
    def from_file(cls, path: str, block_size: int = DEFAULT_BLOCK_SIZE) -> 'DSI':
        """Parse a DSI file from disk."""
        with open(path, 'rb') as f:
            return cls.from_bytes(f.read(), block_size)

    def to_bytes(self) -> bytes:
        """Serialize all blocks to raw bytes."""
        return b''.join(block.to_bytes(self.block_size) for block in self.blocks)

    def to_file(self, path: str):
        """Write DSI to disk."""
        with open(path, 'wb') as f:
            f.write(self.to_bytes())

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def usable_per_block(self) -> int:
        return self.block_size - HEADER_SIZE

    def extract_audio(self) -> bytes:
        """Extract the complete audio stream."""
        return b''.join(block.audio_data for block in self.blocks)

    def extract_video(self) -> bytes:
        """Extract the complete video stream."""
        return b''.join(block.video_data for block in self.blocks)

    def last_block_template(self) -> Tuple[int, int, bool]:
        """Get the last block's audio size, video size, and stream order."""
        last = self.blocks[-1]
        return last.audio_size, last.video_size, last.audio_first

    @classmethod
    def mux(cls, video: bytes, audio: bytes,
            nblocks: Optional[int] = None,
            template: Optional['DSI'] = None,
            block_size: int = DEFAULT_BLOCK_SIZE,
            audio_align: int = DEFAULT_AUDIO_ALIGN,
            video_frame_marker: bytes = b'\x00\x00\x01\x00',
            ) -> 'DSI':
        """Mux video + audio into DSI with proportional audio distribution.

        The video is byte-sliced across blocks as a continuous stream.
        Audio per block is proportional to the number of video frames
        in that block, ensuring A/V sync on playback.

        Args:
            video: Video elementary stream (e.g. MPEG-2 .m2v)
            audio: Audio stream (e.g. PS2 SPU ADPCM)
            nblocks: Number of blocks (required if no template)
            template: Optional existing DSI to copy structure from
                      (block count, last block sizes, block size)
            block_size: Block size in bytes (default 0x40000)
            audio_align: Audio size alignment in bytes (default 512)
            video_frame_marker: Byte sequence marking frame starts
                                (default: MPEG-2 picture start code)

        Returns:
            New DSI instance
        """
        if template is not None:
            nblocks = template.num_blocks
            block_size = template.block_size
            last_aud, last_vid, last_aud_first = template.last_block_template()
        else:
            if nblocks is None:
                raise ValueError("Must provide either nblocks or template")
            # Auto-calculate last block from total data
            usable = block_size - HEADER_SIZE
            last_vid = min(usable - audio_align, len(video) % usable or usable)
            last_aud = audio_align
            last_aud_first = False  # V→A for last block

        usable = block_size - HEADER_SIZE
        total_frames = _count_markers(video, video_frame_marker)
        if total_frames == 0:
            raise ValueError(f"No video frames found (marker: {video_frame_marker.hex()})")

        audio_per_frame = len(audio) / total_frames

        # Estimate typical frames per block for initial audio calculation
        avg_audio = max(audio_align,
                        (round(len(audio) / nblocks) // audio_align) * audio_align)
        avg_vid_cap = usable - avg_audio
        est_frames_per_block = max(1, avg_vid_cap // (len(video) // total_frames))

        blocks = []
        vid_pos = 0
        aud_pos = 0

        for blk in range(nblocks):
            is_last = (blk == nblocks - 1)

            if is_last:
                aud_sz = last_aud
                vid_cap = last_vid
                aud_first = last_aud_first
            else:
                # Estimate frames to calculate proportional audio
                est_aud = max(audio_align,
                              (round(est_frames_per_block * audio_per_frame)
                               // audio_align) * audio_align)
                est_vid_cap = usable - est_aud
                chunk = video[vid_pos:vid_pos + est_vid_cap] if vid_pos < len(video) else b''
                actual_frames = _count_markers(chunk, video_frame_marker)
                aud_sz = max(audio_align,
                             (round(actual_frames * audio_per_frame)
                              // audio_align) * audio_align)
                vid_cap = usable - aud_sz
                aud_first = True

            # Byte-slice video
            vc = video[vid_pos:vid_pos + vid_cap] if vid_pos < len(video) else b''
            if len(vc) < vid_cap:
                vc += b'\x00' * (vid_cap - len(vc))

            ac = audio[aud_pos:aud_pos + aud_sz]
            if len(ac) < aud_sz:
                ac += b'\x00' * (aud_sz - len(ac))

            blocks.append(DSIBlock(
                audio_data=ac,
                video_data=vc,
                audio_first=aud_first,
            ))

            vid_pos += vid_cap
            aud_pos += aud_sz

        dsi = cls(blocks, block_size)
        dsi.ensure_end_of_sequence()
        return dsi

    def ensure_end_of_sequence(self, marker: bytes = b'\x00\x00\x01\xb7'):
        """Inject end-of-sequence marker into the last block with video data."""
        for block in reversed(self.blocks):
            vid = block.video_data
            if marker in vid:
                return  # already has one
            # Find last nonzero byte
            last_nz = len(vid) - 1
            while last_nz > 0 and vid[last_nz] == 0:
                last_nz -= 1
            if last_nz > 0 and last_nz + len(marker) + 1 < len(vid):
                vid = bytearray(vid)
                vid[last_nz + 1:last_nz + 1 + len(marker)] = marker
                block.video_data = bytes(vid)
                return

    def verify_end_of_sequence(self, marker: bytes = b'\x00\x00\x01\xb7') -> bool:
        """Verify that end-of-sequence marker exists somewhere in the video."""
        for block in reversed(self.blocks):
            if marker in block.video_data:
                return True
        return False

    def frame_count(self, marker: bytes = b'\x00\x00\x01\x00') -> int:
        """Count total video frames across all blocks."""
        return _count_markers(self.extract_video(), marker)

    def block_info(self, marker: bytes = b'\x00\x00\x01\x00') -> List[dict]:
        """Get per-block info."""
        info = []
        for i, block in enumerate(self.blocks):
            info.append({
                'block': i,
                'audio_size': block.audio_size,
                'video_size': block.video_size,
                'frames': _count_markers(block.video_data, marker),
                'audio_first': block.audio_first,
            })
        return info


def _count_markers(data: bytes, marker: bytes = b'\x00\x00\x01\x00') -> int:
    """Count occurrences of a marker pattern in data."""
    count = 0
    mlen = len(marker)
    pos = 0
    while pos <= len(data) - mlen:
        if data[pos:pos + mlen] == marker:
            count += 1
            pos += mlen
        else:
            pos += 1
    return count


def ensure_end_of_sequence(video: bytes,
                           end_marker: bytes = b'\x00\x00\x01\xb7') -> bytes:
    """Ensure video stream ends with the given end marker."""
    data = bytearray(video)
    if data.rfind(end_marker) < 0:
        last = len(data)
        while last > 0 and data[last - 1] == 0:
            last -= 1
        data = data[:last]
        data.extend(end_marker)
    return bytes(data)
