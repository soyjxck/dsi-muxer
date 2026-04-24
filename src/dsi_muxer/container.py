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

        Block layout matches Racjin's DSI convention: audio-first, fixed
        video cap per block (zero-padded if video is short for the cap),
        block count derived from total content size.

        Audio is distributed proportionally to the fraction of the video
        stream consumed by each block, using cumulative-target tracking to
        prevent drift. No trailing audio dump — the full audio stream is
        spread across all blocks so no block ever gets a spike of leftover.

        Args:
            video: Video elementary stream (e.g. MPEG-2 .m2v)
            audio: Audio stream (e.g. PS2 SPU ADPCM)
            nblocks: Explicit block count (auto-sized from content if None)
            template: Optional existing DSI to copy block_size + nblocks from
            block_size: Block size in bytes (default 0x40000)
            audio_align: Audio size alignment in bytes (default 512)
            video_frame_marker: Frame start marker (unused here, kept for API)

        Returns:
            New DSI instance
        """
        if template is not None:
            nblocks = template.num_blocks
            block_size = template.block_size

        usable = block_size - HEADER_SIZE

        if nblocks is None:
            import math
            total = len(video) + len(audio)
            nblocks = max(1, math.ceil(total / usable))

        blocks = []
        vid_pos = 0
        aud_pos = 0

        for blk in range(nblocks):
            vid_remaining = len(video) - vid_pos
            aud_remaining = len(audio) - aud_pos
            if vid_remaining <= 0 and aud_remaining <= 0:
                break

            # If video is done, fold remaining audio into the last video block
            # up to its headroom, then emit audio-only trailing block(s) for
            # any remainder. Avoids the old muxer's single-block audio spike.
            if vid_remaining <= 0 and aud_remaining > 0:
                if blocks:
                    last = blocks[-1]
                    headroom = usable - last.audio_size - last.video_size
                    take = min(aud_remaining,
                               (headroom // audio_align) * audio_align)
                    if take > 0:
                        new_aud = last.audio_data + audio[aud_pos:aud_pos + take]
                        blocks[-1] = DSIBlock(
                            audio_data=new_aud,
                            video_data=last.video_data,
                            audio_first=last.audio_first,
                        )
                        aud_pos += take
                        aud_remaining -= take
                while aud_remaining > 0:
                    aud_sz = min(aud_remaining,
                                 ((usable - audio_align) // audio_align) * audio_align)
                    aud_sz = ((aud_sz + audio_align - 1) // audio_align) * audio_align
                    aud_sz = min(aud_sz, aud_remaining)
                    blocks.append(DSIBlock(
                        audio_data=audio[aud_pos:aud_pos + aud_sz],
                        video_data=b'',
                        audio_first=True,
                    ))
                    aud_pos += aud_sz
                    aud_remaining -= aud_sz
                break
            is_last = (blk == nblocks - 1)

            # Cumulative-target audio: what should aud_pos be at end of this block?
            # Distribute proportional to total block progress so drift stays bounded
            # and no leftover dumps into the final block.
            target_frac = (blk + 1) / nblocks
            target_aud_end = round(target_frac * len(audio))
            want_aud = target_aud_end - aud_pos

            # Round to nearest align
            aud_sz = ((want_aud + audio_align // 2) // audio_align) * audio_align
            aud_sz = max(audio_align, aud_sz)
            aud_sz = min(aud_sz, aud_remaining)
            aud_sz = min(aud_sz, usable - audio_align)

            # On the last block, take all remaining audio (in case of rounding drift)
            if is_last and aud_remaining > 0:
                need = ((aud_remaining + audio_align - 1) // audio_align) * audio_align
                aud_sz = max(aud_sz, min(need, usable - audio_align))

            # Video cap fills the rest of the block
            vid_cap = usable - aud_sz
            vid_cap = min(vid_cap, vid_remaining) if vid_remaining > 0 else 0
            vc = video[vid_pos:vid_pos + vid_cap]

            ac = audio[aud_pos:aud_pos + aud_sz]
            if len(ac) < aud_sz:
                ac += b'\x00' * (aud_sz - len(ac))

            blocks.append(DSIBlock(
                audio_data=ac,
                video_data=vc,
                audio_first=True,
            ))

            vid_pos += vid_cap
            aud_pos += aud_sz

        dsi = cls(blocks, block_size)
        dsi.ensure_end_of_sequence()
        return dsi

    def replace_video(self, new_video: bytes) -> 'DSI':
        """Replace video content while preserving the exact block structure.

        Splices new video bytes into each block's video region. Headers,
        audio data, stream order, and padding are untouched. This is the
        safest way to swap video (e.g. with burned subtitles) without
        disturbing the block layout that hardware decoders expect.

        Args:
            new_video: Replacement video elementary stream.

        Returns:
            New DSI instance with replaced video.
        """
        new_blocks = []
        vid_pos = 0
        for block in self.blocks:
            vid_sz = block.video_size
            chunk = new_video[vid_pos:vid_pos + vid_sz]
            if len(chunk) < vid_sz:
                chunk += b'\x00' * (vid_sz - len(chunk))
            new_blocks.append(DSIBlock(
                audio_data=block.audio_data,
                video_data=chunk,
                audio_first=block.audio_first,
            ))
            vid_pos += vid_sz
        result = DSI(new_blocks, self.block_size)
        result.ensure_end_of_sequence()
        return result

    def ensure_end_of_sequence(self, marker: bytes = b'\x00\x00\x01\xb7',
                               trailing_pad: int = 256):
        """Ensure the video stream ends with the marker + stuffing bytes.

        The PS2 IPU (MPEG-2 decoder) needs stuffing bytes after the
        end-of-sequence marker to flush its pipeline; without them, the
        decoder stalls and the cutscene freezes on its final frame.
        Original Racjin DSI outputs include ~171 such bytes; we default
        to 256 zero bytes of padding to be safe.

        Args:
            marker: MPEG-2 end-of-sequence marker (default 0x000001B7).
            trailing_pad: Number of zero bytes that must follow the marker
                          within the last block's video region. Capped by
                          available block headroom.
        """
        usable = self.block_size - HEADER_SIZE
        for block in reversed(self.blocks):
            vid = block.video_data
            if len(vid) == 0:
                continue

            idx = vid.rfind(marker) if isinstance(vid, (bytes, bytearray)) else -1

            if idx < 0:
                # Marker missing — overwrite trailing zeros with marker, or append
                last_nz = len(vid) - 1
                while last_nz > 0 and vid[last_nz] == 0:
                    last_nz -= 1
                if last_nz > 0 and last_nz + len(marker) + 1 < len(vid):
                    new_vid = bytearray(vid)
                    new_vid[last_nz + 1:last_nz + 1 + len(marker)] = marker
                    idx = last_nz + 1
                    vid = bytes(new_vid)
                elif block.audio_size + block.video_size + len(marker) <= usable:
                    vid = vid + marker
                    idx = len(vid) - len(marker)
                else:
                    continue  # no room — try previous block

            # Marker is at idx; ensure trailing_pad zero bytes follow it within
            # the video region. Extend video_data if needed (within headroom).
            trailing = len(vid) - (idx + len(marker))
            if trailing < trailing_pad:
                need = trailing_pad - trailing
                headroom = usable - block.audio_size - len(vid)
                add = min(need, headroom)
                if add > 0:
                    vid = vid + b'\x00' * add
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
