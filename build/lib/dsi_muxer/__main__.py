"""
CLI interface for the DSI muxer.

Usage:
    # Demux a DSI file
    python -m dsi_muxer demux input.dsi --video output.m2v --audio output.adpcm

    # Mux video + audio into DSI
    python -m dsi_muxer mux --video input.m2v --audio input.adpcm --blocks 307 -o output.dsi

    # Info about a DSI file
    python -m dsi_muxer info input.dsi
"""

import sys
import argparse
from .container import DSI, ensure_end_of_sequence


def cmd_demux(args):
    dsi = DSI.from_file(args.input)
    if args.video:
        with open(args.video, 'wb') as f:
            f.write(dsi.extract_video())
        print(f"Video: {args.video}")
    if args.audio:
        with open(args.audio, 'wb') as f:
            f.write(dsi.extract_audio())
        print(f"Audio: {args.audio}")


def cmd_mux(args):
    with open(args.video, 'rb') as f:
        video = f.read()
    with open(args.audio, 'rb') as f:
        audio = f.read()

    video = ensure_end_of_sequence(video)
    dsi = DSI.mux(video, audio, nblocks=args.blocks)
    dsi.to_file(args.output)

    frames = dsi.frame_count()
    print(f"Muxed: {frames} frames, {args.blocks} blocks -> {args.output}")


def cmd_info(args):
    dsi = DSI.from_file(args.input)
    video = dsi.extract_video()
    audio = dsi.extract_audio()
    frames = dsi.frame_count()

    print(f"DSI: {dsi.num_blocks} blocks")
    print(f"Video: {len(video):,} bytes, {frames} frames")
    print(f"Audio: {len(audio):,} bytes")
    print()

    info = dsi.block_info()
    print(f"{'Block':>5} {'Frames':>7} {'Audio':>8} {'Video':>8} {'Order':>5}")
    print("-" * 40)
    for b in info[:10]:
        order = "A→V" if b['audio_first'] else "V→A"
        print(f"{b['block']:5d} {b['frames']:7d} {b['audio_size']:8,} {b['video_size']:8,} {order:>5}")
    if len(info) > 20:
        print(f"  ... ({len(info) - 20} blocks omitted)")
    for b in info[-10:]:
        order = "A→V" if b['audio_first'] else "V→A"
        print(f"{b['block']:5d} {b['frames']:7d} {b['audio_size']:8,} {b['video_size']:8,} {order:>5}")


def main():
    parser = argparse.ArgumentParser(description="Racjin PS2 DSI container tool")
    sub = parser.add_subparsers(dest='command')

    p_demux = sub.add_parser('demux', help='Extract video/audio from DSI')
    p_demux.add_argument('input', help='Input DSI file')
    p_demux.add_argument('--video', '-v', help='Output video file (.m2v)')
    p_demux.add_argument('--audio', '-a', help='Output audio file (.adpcm)')

    p_mux = sub.add_parser('mux', help='Create DSI from video + audio')
    p_mux.add_argument('--video', '-v', required=True, help='Input MPEG-2 video (.m2v)')
    p_mux.add_argument('--audio', '-a', required=True, help='Input PS2 ADPCM audio')
    p_mux.add_argument('--blocks', '-b', type=int, required=True, help='Number of DSI blocks')
    p_mux.add_argument('--output', '-o', required=True, help='Output DSI file')

    p_info = sub.add_parser('info', help='Show DSI file info')
    p_info.add_argument('input', help='Input DSI file')

    args = parser.parse_args()
    if args.command == 'demux':
        cmd_demux(args)
    elif args.command == 'mux':
        cmd_mux(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
