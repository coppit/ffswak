#!/usr/bin/env python3

import os

DEFAULT_OUTPUT_DIR = os.path.expanduser('~/Pictures/Import')
FFMPEG = 'ffmpeg'
# Implies the container type. Examples: mov, mkv, mp4
OUTPUT_FILENAME_EXTENSION = 'mp4'
# Warn if the output file is this percent smaller or less. (Always warn if it's larger.)
WARNING_THRESHOLD = 10
# Copy the file if we can (user didn't specify any tranformations), and the size difference is less than this percentage
COPY_THRESHOLD = 2
# I'm not sure what the right settings are here. I had a video of a snow blower that required 200 frames and a
# threshold of .5 to not falsely claim it's interlaced.
INTERLACED_FRAME_SAMPLE = 200
INTERLACED_THRESHOLD = .6
# For 10-bit depth
#REENCODE_THRESHOLD = .15

#-----------------------------------------------------------------------------------------------------------------------

import argparse, atexit, datetime, ffmpeg, humanize, math, os, psutil, random
import re, select, shlex, shutil, signal, stat, subprocess, sys, tempfile, threading, time

from collections import namedtuple
from enum import Enum
from fractions import Fraction as FractionBase
from functools import lru_cache
from pprint import pformat
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, ProgressColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
from rich.traceback import install as install_pretty_exceptions

#-----------------------------------------------------------------------------------------------------------------------

# Need python 3.7 for ordered dicts
MIN_PYTHON_MAJOR = 3
MIN_PYTHON_MINOR = 7

# Check the running Python version
if sys.version_info < (MIN_PYTHON_MAJOR, MIN_PYTHON_MINOR):
    # Construct the error message
    version_string = f""
    sys.exit(f"Error: This script requires Python {MIN_PYTHON_MAJOR}.{MIN_PYTHON_MINOR} or newer. "
             f"You are running Python {sys.version.split(' ')[0]}.")

#-----------------------------------------------------------------------------------------------------------------------

class Dimensions( namedtuple('Dimensions', ['width', 'height']) ):
    def __new__(cls, width, height):
        # Perform validation on the arguments before the object is created.
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("Dimensions must be integers.")
        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive integers.")

        return super().__new__(cls, width, height)

    def __int__(self):
        return self.width * self.height

    def __str__(self):
        return f"{self.width}x{self.height}"

    def __repr__(self):
        return self.__str__()

    def swap(self):
        return Dimensions(self.height, self.width)

    @property
    def aspect_ratio(self):
        return self.width / self.height

#-----------------------------------------------------------------------------------------------------------------------

class Fraction(FractionBase):
    def __repr__(self):
        return self.__str__() + f' ({float(self)})'

#-----------------------------------------------------------------------------------------------------------------------

class CropType(Enum):
    ASPECT = 'aspect'
    FRACTION = 'fraction'
    PIXELS = 'pixels'

#-----------------------------------------------------------------------------------------------------------------------

class Crop:
    def __init__(self, crop_type, width, height, x, y):
        assert(x in ('l', 'c', 'r') or isinstance(x, float) and 0 <= x <= 1)
        assert(y in ('t', 'm', 'b') or isinstance(y, float) and 0 <= y <= 1)

        if crop_type == CropType.FRACTION: assert(0 < width <= 1 and 0 < height <= 1)
        if crop_type == CropType.PIXELS: assert(width == int(width) and height == int(height))

        self.type = crop_type
        self.width = width
        self.height = height
        self.x = x
        self.y = y

    @property
    def aspect_ratio(self):
        assert(self.type in (CropType.ASPECT, CropType.PIXELS))

        return self.width / self.height

    def __str__(self):
        if self.type == CropType.ASPECT:
            sep = ':'
        elif self.type == CropType.FRACTION:
            sep = ','
        elif self.type == CropType.PIXELS:
            sep = 'x'
        else:
            assert(False)

        return f'[{self.type.value} {self.width}{sep}{self.height} {self.x}{self.y}]'

    def __repr__(self):
        return self.__str__()

#-----------------------------------------------------------------------------------------------------------------------

class TimeRange( namedtuple('TimeRange', ['start', 'end']) ):
    def __str__(self):
        return f"{in_hms(self.start)}-{in_hms(self.end)}"

#-----------------------------------------------------------------------------------------------------------------------

# Save originals (Rich replaces them when install_pretty_exceptions() is called)
ORIGINAL_SYS_HOOK = sys.excepthook
ORIGINAL_THREADING_HOOK = getattr(threading, "excepthook", None)

DEBUG=False

def disable_pretty_exceptions():
    sys.excepthook = ORIGINAL_SYS_HOOK
    if hasattr(threading, "excepthook") and threading.excepthook is not None:
        threading.excepthook = ORIGINAL_THREADING_HOOK

def enable_pretty_exceptions():
    install_pretty_exceptions(show_locals=DEBUG)

CONSOLE = Console()
ECONSOLE = Console(stderr=True)

enable_pretty_exceptions()

#-----------------------------------------------------------------------------------------------------------------------

# Monkey patch the highlighter pattern to not highlight "0x1080" in 1920x1080 as a hex number
from rich.highlighter import ReprHighlighter

for i, regex in enumerate(ReprHighlighter.highlights):
    if '0x' in regex:
        parts = regex.split('0x')
        ReprHighlighter.highlights[i] = r'\b0x'.join(parts)

#-----------------------------------------------------------------------------------------------------------------------

# Rich.console pretty-printing to STDOUT and STDERR
def cprint(*args, **kwargs):
    CONSOLE.print(*args, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------

def eprint(*args, **kwargs):
    ECONSOLE.print(*args, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------

# Debug printing
def dprint(*args, **kwargs):
    if not DEBUG:
        return

    if not hasattr(dprint, 'pending'):
        dprint.pending = ''

    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")

    text = sep.join(str(a) for a in args)

    dprint.pending += f'{text}{end}'

    if end != '\n':
        return

    lines = dprint.pending.splitlines(True)  # keep line breaks
    dprint.pending = ''

    prefix = kwargs.pop('prefix', '')

    for line in lines:
        if line.endswith("\n"):
            t = Text()
            t.append("DEBUG: ", style="yellow")

            do_highlight = '<' not in line

            # Let Rich highlight the rest, but disable markup parsing
            line = prefix + line.rstrip('\n')
            highlighted = CONSOLE.render_str(line, markup=False, highlight=do_highlight)
            t += highlighted

            cprint(t, end=end, **kwargs, soft_wrap=True)
        else:
            # incomplete line stays in buffer
            dprint.pending = line

#-----------------------------------------------------------------------------------------------------------------------

class Video(list):
    def __init__(self, output_dir, output_file, dimensions_limit, frame_rate_limit):
        self.output_dir = output_dir
        self._output_file = output_file
        self.dimensions_limit = dimensions_limit
        self.frame_rate_limit = frame_rate_limit

        self.clips_adjusted = False
        self.output_dims = None

    #-------------------------------------------------------------------------------------------------------------------

    def adjust_clip_durations_for_transitions(self):
        old_time_ranges = [ TimeRange(f.start, f.end) for f in self ]

        dprint('Adjusting clip start and end times for transitions')

        for clip in self:
            if clip.index == 0 or clip.transition_duration == 0:
                continue

            prev_clip = self[clip.index-1]
            transition_duration = clip.transition_duration

            dprint(f'- Before: {prev_clip} --> {clip.transition_duration} s transition --> {clip}')

#            dprint(f'  - Adjusting end of clip {clip.index-1} and start of clip {clip.index} for transition {clip.index}')

            # First fix the end of the previous clip if it's too close to the end of the file
            if prev_clip.end + transition_duration * prev_clip.speedup <= prev_clip.input_duration:
                prev_clip_adjusted_end = prev_clip.end + transition_duration * prev_clip.speedup
            else:
                cprint(f'[yellow1]WARNING[/]: Cannot increase the end time of clip {clip.index-1} past the '
                    'end of the input video. '
                    f'{transition_duration * prev_clip.speedup - (prev_clip.input_duration-prev_clip.end):.2f} '
                    'seconds of the desired time range will be in the transition.')

                prev_clip_adjusted_end = prev_clip.input_duration

            # Then fix the start of the current clip if it's too close to the start of the file
            if clip.start - transition_duration * clip.speedup >= 0:
                clip_adjusted_start = clip.start - transition_duration * clip.speedup
            else:
                cprint(f'[yellow1]WARNING[/]: Cannot decrease the start time of clip {clip.index} before the '
                    f'start of the input video. {transition_duration * clip.speedup - clip.start:.2f} '
                    'seconds of the desired time range will be in the transition.')

                clip_adjusted_start = 0

            prev_clip.end = prev_clip_adjusted_end
            clip.start = clip_adjusted_start

            dprint(f'- After: {prev_clip} --> {clip.transition_duration}s transition --> {clip}')

        # Warn about overlaps
        sorted_clips = sorted(self, key=lambda c: c.start)

        for i in range(1, len(sorted_clips)):
            clip = sorted_clips[i]
            prev_clip = sorted_clips[i - 1]

            if clip.start >= prev_clip.end or clip.input_file != prev_clip.input_file:
                continue

            cprint(f'[yellow1]WARNING[/]: {prev_clip} overlaps with {clip}. Consider merging them.')

            dprint(f'- Fixing overlap for a smooth transition')
            dprint(f'  - Before: {prev_clip} --> {clip.transition_duration} s transition --> {clip}')
            adjustment = (prev_clip.end - clip.start - clip.transition_duration) / 2
            prev_clip.end -= adjustment
            clip.start += adjustment
            dprint(f'  - After: {prev_clip} --> {clip.transition_duration} s transition --> {clip}')

        # Report the adjustments
        for clip in self:
            new_time_range = TimeRange(clip.start, clip.end)
            old_time_range = old_time_ranges[clip.index]

            if old_time_range == new_time_range:
                continue

            transition_durations = []
            if clip.index > 0:
                transition_durations += [ f'previous transition with duration {clip.transition_duration}' ]

            if clip.index < len(self)-1:
                transition_durations += [ f'next transition with duration {self[clip.index+1].transition_duration}' ]

            speedup_str = '' if clip.speedup == 1 else f'a speedup of {clip.speedup}x and '
            transition_durations_str = ' and '.join(transition_durations)

            dprint(f'- Adjusted time for clip {clip.index} from {old_time_range} to {new_time_range} based on '
                f'{speedup_str}{transition_durations_str}')

        self.clips_adjusted = True

    #-------------------------------------------------------------------------------------------------------------------

    @property
    def can_copy_video(self):
        if len(self) != 1 or self.max_video_bitrate is None:
            return False

        clip = self[0]

        non_rotation_filters = [f for f in clip.video_filters if f[0] != 'transpose']

        if non_rotation_filters:
            return False

        return True

        # The following gets more into "should encode". I did some testing to see if there was a good heuristic for when
        # we should re-encode an h265 video. Unfortunately, even videos with bits/pixel of .13 sometimes got a lot
        # smaller with re-encoding. Nothing more than .17 failed to benefit from re-encoding. This makes me think that
        # the strategy should be to always re-encode, and if the resulting file is larger, then fast copy to avoid
        # quality loss. This is what I implemented

#        if codec != 'hevc':
#            return False
#
#        input_bit_depth = parse_pixel_format(clip.pixel_format)['depth']
#
#        bits_per_pixel = clip.video_bitrate / clip.input_dims.width / clip.input_dims.height / clip.avg_frame_rate
#
#        # Adjust for bit depth
#        bits_per_pixel *= 10/input_bit_depth
#
#        if bits_per_pixel < REENCODE_THRESHOLD:
#            return False
#
#        return True

    #-------------------------------------------------------------------------------------------------------------------

    @property
    def can_copy_audio(self):
        if len(self) != 1 or video.max_audio_bitrate is None:
            return False

        for clip in self:
            if clip.audio_filters:
                return False

        if video.max_audio_bitrate > 192 * 1024:
            return False

        return True

    #-------------------------------------------------------------------------------------------------------------------

    @property
    def output_file(self):
        if self._output_file is not None:
            return self._output_file

        assert(self.clips_adjusted)

        filenames = [ os.path.splitext(os.path.basename(clip.input_file))[0] for clip in self ]

        merged_filename = '-'.join(filenames)

        output_file = os.path.join(self.output_dir, f'{merged_filename}.{OUTPUT_FILENAME_EXTENSION}')

        while os.path.exists(output_file):
            output_file = os.path.join(self.output_dir,
                f'{merged_filename}-{random.randrange(16**3):03x}.{OUTPUT_FILENAME_EXTENSION}')

        self._output_file = output_file

        return self._output_file

    #-------------------------------------------------------------------------------------------------------------------

    # Null value implies that there should be no video in the output.
    @property
    def max_pixel_format(self):
        parsed = [ parse_pixel_format(clip.pixel_format) for clip in self ]

        if not parsed:
            return None

        # Check for matching families
        families = {p["family"] for p in parsed}
        if len(families) != 1:
            if not hasattr(self, '_warned_about_pixel_format'):
                cprint(f'[yellow1]WARNING[/]: Found mixed pixel families {families}. Converting to the more common '
                    f'yuv family.')
                self._warned_about_pixel_format = True

            family = 'yuv'
        else:
            family = parsed[0]['family']

        # Take the maximum of each
        max_sub = max(p["subsampling"] for p in parsed)
        max_depth = max(p["depth"] for p in parsed)
        endian = next((p["endian"] for p in parsed if p["endian"]), "le")

        # ffmpeg uses a short name like "yuvj420p" instead of "yuvj420p8le"
        if max_depth == 8:
            max_depth = ''
        if endian == 'le':
            endian = ''

        return f"{family}{max_sub}p{max_depth}{endian}"

    #-------------------------------------------------------------------------------------------------------------------

    # Null value implies that there should be no video in the output.
    @property
    def max_video_bitrate(self):
        video_bitrates = [f.video_bitrate for f in self if f.video_bitrate is not None]
        return max(video_bitrates) if video_bitrates else None

    #-------------------------------------------------------------------------------------------------------------------

    # Null value implies that there should be no video in the output.
    @property
    def max_avg_frame_rate(self):
        avg_frame_rates = [f.avg_frame_rate for f in self if f.avg_frame_rate is not None]
        return min(max(avg_frame_rates), self.frame_rate_limit) if avg_frame_rates else None

    #-------------------------------------------------------------------------------------------------------------------

    # Null value implies that there should be no audio in the output.
    @property
    def max_audio_bitrate(self):
        audio_bitrates = [f.audio_bitrate for f in self if f.audio_bitrate is not None]
        return max(audio_bitrates) if audio_bitrates else None

    #-------------------------------------------------------------------------------------------------------------------

    @property
    def output_creation_time(self):
        creation_times = [f.creation_time for f in self if f.creation_time is not None]

        # Fall back to file creation times
        if not creation_times:
            creation_times = [file_creation_time_iso8601(f.input_file) for f in self]

        # Sanity check. Sometimes the file time stamps are garbage.
        creation_times = [t for t in creation_times if t > '1971']

        if not creation_times:
            return None

        min_time = min(creation_times)

        offset = self[0].start

        return adjust_iso8601_time(min_time, offset)

    #-------------------------------------------------------------------------------------------------------------------

    @property
    def output_duration(self):
        assert(self.clips_adjusted)

        if hasattr(self, '_output_duration'):
            return self._output_duration

        dprint('Computing output duration')

        output_duration = 0

        for clip in self:
            dprint(f'- {clip}')

            # Don't double-count the transitions
            next_transition_duration = 0 if clip.index == len(self)-1 else self[clip.index+1].transition_duration

            adjusted_duration = clip.output_duration - next_transition_duration

            dprint(f'  - Adding duration {adjusted_duration:.2f} (adjusted by {next_transition_duration} '
                'to avoid double-counting the transition to the next clip)')

            output_duration += adjusted_duration

        dprint(f'Final output duration = {output_duration:.2f}')

        self._output_duration = output_duration

        return output_duration

    #-------------------------------------------------------------------------------------------------------------------

    @property
    def estimated_file_size(self):
        assert(self.clips_adjusted)

        dprint('Estimating output file size')

        estimated_file_size = 0

        for clip in self:
            dprint(f'- {clip}')

            # Don't double-count the transitions
            next_transition_duration = 0 if clip.index == len(self)-1 else self[clip.index+1].transition_duration

            adjusted_duration = clip.output_duration - next_transition_duration

            dprint(f'  - Adding estimated file size {humanize.naturalsize(clip.overall_bitrate*adjusted_duration)}')

            estimated_clip_size = clip.overall_bitrate * adjusted_duration

            if self.output_dims != None:
                estimated_clip_size *= int(self.output_dims) / int(clip.filtered_dims)
#            else:
#                # TODO: Use crop information to estimate.
#                estimated_clip_size *= int(self.output_dims) / int(clip.filtered_dims)

            estimated_file_size += estimated_clip_size

        dprint(f'Final estimated file size = {humanize.naturalsize(estimated_file_size)}')

        return estimated_file_size

    #-------------------------------------------------------------------------------------------------------------------

    def __str__(self):
        str = ''

        str += f'output_file: {self.output_file}\n'
        str += f'output_duration: {self.output_duration}\n'
        str += f'estimated_file_size: {self.estimated_file_size}\n'
        str += f'max_pixel_format: {self.max_pixel_format}\n'
        str += f'max_video_bitrate: {self.max_video_bitrate}\n'
        str += f'max_avg_frame_rate: {repr(self.max_avg_frame_rate)}\n'
        str += f'max_audio_bitrate: {self.max_audio_bitrate}\n'
        str += f'dimensions_limit: {self.dimensions_limit}\n'
        str += f'frame_rate_limit: {self.frame_rate_limit}\n'
        str += f'output_creation_time: {self.output_creation_time}\n'

        str = str.removesuffix('\n')

        return str

#-----------------------------------------------------------------------------------------------------------------------

class Clip:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self._set_attributes()

        # We have to do this after _set_attributes(), when the input duration becomes known
        self._normalize_end_time()

        self._set_stabilization_parameters()

        self.video_filters = None
        self.audio_filters = None

        self.video_transition_filter = None
        self.audio_transition_filter = None

    #-------------------------------------------------------------------------------------------------------------------

    def _set_attributes(self):
        self.input_dims, self.input_duration, self.creation_time = None, None, None
        self.video_bitrate, self.audio_bitrate, self.avg_frame_rate = None, None, None

        try:
            probe_result = ffprobe(self.input_file)

            self.input_duration = float(probe_result.get('format', {}).get('duration'))
            self.creation_time = probe_result.get('format', {}).get('tags', {}).get('creation_time')

            # Get the streams information
            streams = probe_result.get('streams', [])

        except ffmpeg.Error as e:
            cprint(f"[red]Error probing file[/]: {e.stderr.decode('utf-8')}")
            sys.exit(1)

        for stream in streams:
            if stream.get('codec_type') == 'video':
                # Let's ignore the album art stream from audio files
                disposition = stream.get('disposition', {})
                if disposition.get('attached_pic') or disposition.get('album_art'):
                    continue

                self.interlaced = is_interlaced(self.interlace_test, self.input_file)
                self.input_dims = Dimensions(int(stream['width']), int(stream['height']))
                self.video_bitrate = int(stream['bit_rate'])
                self.pixel_format = stream['pix_fmt']

                # Parse frame rate
                fps_str = stream['avg_frame_rate']

                if '/' in fps_str:
                    numerator, denominator = map(int, fps_str.split('/'))
                    self.avg_frame_rate = Fraction(numerator, denominator)
                else:
                    self.avg_frame_rate = float(fps_str)

                # Check for rotation in side data (e.g., from mobile phones)
                self.presentation_rotation = None
                side_data_list = stream.get('side_data_list', [])
                for data in side_data_list:
                    if data['side_data_type'] == 'Display Matrix' and 'rotation' in data:
                        self.presentation_rotation = int(data['rotation'])
                        break

                # Fallback to tags for rotation
                if self.presentation_rotation is None:
                    tags = stream.get('tags', {})
                    self.presentation_rotation = int(tags.get('rotate', 0))

                self.presentation_rotation %= 360

                # Change the dimensions immediately, because that's how video players and ffmpeg behave, and we don't
                # want to confuse the user
                self._fix_dimensions_for_rotation()

                # XXX: clip.presentation_rotation is informational only, and only used later with the fast-copy
                # optimization. Don't use it to apply any rotation filter!

            elif stream['codec_type'] == 'audio' and self.volume != 0:
                self.audio_bitrate = int(stream['bit_rate'])

            if self.video_bitrate is not None and self.audio_bitrate is not None:
                break

    #-------------------------------------------------------------------------------------------------------------------

    @property
    def transforms_file(self):
        if hasattr(self, '_transforms_file'):
            return self._transforms_file

        self._transforms_file = make_temp_filename(self.input_file, extension='.trf')

        return self._transforms_file

    #-------------------------------------------------------------------------------------------------------------------

    # Include overhead
    @property
    def overall_bitrate(self):
        return os.path.getsize(self.input_file) / self.input_duration

    #-------------------------------------------------------------------------------------------------------------------

    @property
    def output_duration(self):
        if self.end is None or self.start is None or self.speedup is None:
            return None

        return (self.end - self.start) / self.speedup

    #-------------------------------------------------------------------------------------------------------------------

    def _normalize_end_time(self):
        if self.end is None:
            self.end = self.input_duration
        elif self.end > self.input_duration:
            raise argparse.ArgumentTypeError(f'End time {self.end} cannot be after the video length '
                f'{self.input_duration} for input "{self.input_file}".')
        elif self.start >= self.end:
            raise argparse.ArgumentTypeError(f'Start time must be before end time.')

    #-------------------------------------------------------------------------------------------------------------------

    def _set_stabilization_parameters(self):
        if not self.stabilize:
            return

        if self.video_bitrate is None:
            raise argparse.ArgumentTypeError(f'Cannot stabilize a movie with no video stream.')

        # https://github.com/georgmartius/vid.stab

        self.mincontrast = .1

        # 1-10 with 1 meaning little shakiness. Default 5
        self.shakiness = 8

        # == Transform options ==
        # Setting this too high sometimes crops the video more than one would like
        self.smoothing = 20
        # Percentage
        self.zoom = 20
        # 0=disabled, 1=strong movements lead to borders, 2=no borders
        self.optzoom = 2

    #-------------------------------------------------------------------------------------------------------------------

    # Cell phones will report a vertical video as 1920x1080 with a rotation of -90 degrees. Let's correct that.
    def _fix_dimensions_for_rotation(self):
        if self.presentation_rotation % 180 == 0:
            return

        if self.presentation_rotation % 180 == 90:
            self.input_dims = self.input_dims.swap()

            return

        # Phones should not have a rotation that is not a multiple of 90 degrees
        assert(False)

    #-------------------------------------------------------------------------------------------------------------------

    def __str__(self):
        speedup_str = '' if self.speedup == 1 else f' {self.speedup}x'

        return f'Clip {self.index} ({os.path.basename(self.input_file)} {TimeRange(self.start, self.end)}{speedup_str})'

#-----------------------------------------------------------------------------------------------------------------------

def parse_pixel_format(format_string):
    pattern = re.compile(r'^([a-z]+)(\d{3})(?:p)?(\d+)?(le|be)?$')

    m = pattern.match(format_string)

    if not m:
        raise ValueError(f"Unrecognized format: {format_string}")

    family, subs, depth, endian = m.groups()

    return {
        "family": family,
        "subsampling": int(subs),
        "depth": int(depth) if depth else 8,
        "endian": endian or ""
    }

#-----------------------------------------------------------------------------------------------------------------------

@lru_cache
def ffprobe(input_file):
    dprint(f'  - Calling ffprobe to fetch video properties from {input_file}', style='violet')

    return ffmpeg.probe(input_file)

#-----------------------------------------------------------------------------------------------------------------------

# http://www.aktau.be/2013/09/22/detecting-interlaced-video-with-ffmpeg/
@lru_cache
def is_interlaced(interlace_test, input_file):
    if not interlace_test:
        return None

    cprint(f'[violet]Calling ffmpeg to analyze interlacing of {input_file}')

    # I'm assuming here that if the first part of the file is interlaced, the whole thing is.
    command = [FFMPEG, '-filter:v', 'idet', '-frames:v', str(INTERLACED_FRAME_SAMPLE), '-an', '-f', 'rawvideo', '-y',
        '/dev/null', '-i', input_file]

    dprint_command('Test video for interlacing command', command)

    ffmpeg_stdout, ffmpeg_stderr = run_ffmpeg(command, None)

    results = '\n'.join(ffmpeg_stderr)

    try:
        report = re.search(r'(?s).*Multi frame detection: ([^\n]*)', results).group(1)

        tff = re.search(r'TFF:\s+(\d+)', report).group(1)
        prog = re.search(r'Progressive:\s+(\d+)', report).group(1)

        tff = int(tff)
        prog = int(prog)
    except (AttributeError) as e:
        cprint(f"[red]Error during parsing interlacing report[/]: {e}")
        cprint(f"Output below:\n{results}")
        sys.exit(1)

    if tff + prog == 0:
        cprint(f"[red]Error[/]: Couldn't determine if video is interlaced or not. Output below:\n{results}")
        sys.exit(1)

    if tff/(tff+prog) > INTERLACED_THRESHOLD:
        cprint(f'Detected {tff} interlaced frames and {prog} progressive frames. Treating the input as '
            'interlaced! Continuing in 10 seconds. CTRL-c to abort.')

        time.sleep(10)

        return True

    return False

#=======================================================================================================================

def global_options_parser():
    usage = '%(prog)s [global options] -- [per-file options] input_file [time_ranges ...] ' \
        '[ [per-file options] input_file [time_ranges... ] ... ]'

    parser = argparse.ArgumentParser(usage=usage, formatter_class=argparse.RawDescriptionHelpFormatter,
        description='A Python wrapper for ffmpeg that simplies common video editing tasks.\n\n' \
            'TIME FORMAT is [[HH:]MM:]SS[.frac] or NNN[.frac] or .frac. Time ranges do not include\n' \
            'transition times. A warning will be issued if the end of an input file requires the transition to\n' \
            'include part of the specified time range.\n\n' \
            'Global options apply to all files. Video options override global options for a specific\n' \
            'file. "--" can be omitted if there are no per-video options.',
        epilog= 'Generally speaking, bitrate, frame rate, etc. will be chosen to avoid degrading the quality.\n' \
            'Width and height will be automatically adjusted (with a warning) when it\'s obvious that they are\n' \
            'wrong. e.g. Width and height will be swapped when all the inputs are portrait instead of landscape.',
        add_help=False)

    # These options can't be used at the per-file level
    # XXX: All of these should be passed to Video.__init__()
    global_group = parser.add_argument_group('Global-Only Options',
        description='General options, and options for the output video.')

    global_group.add_argument('-F', '--frame-rate-limit', type=float, default=60.0, help='Maximum frame rate.')
    global_group.add_argument('-D', '--dimensions-limit', type=dimensions_type, default='1920x1080',
        help='Maximum dimensions. .5 means 50%% as wide and tall; .5,1 means half as wide, full height; '
          '16:9 means the largest possible video with that aspect ratio; 1280x720 means exactly that size')
    global_group.add_argument('-o', '--output-file',
        help='Output file. (Default is input.mp4, or input-abc.mp4 if needed, for re-encoding. With -c extension '
        'is kept the same)')
    global_group.add_argument('-O', '--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory.')
    global_group.add_argument('-d', '--debug', default=False, action='store_true', help='Enable debugging messages')

    global_group.add_argument('--help', action='help', help='Show this help message and exit.')

    # These options can be specified globally for all files, or separately for each file. Use the global values as the
    # defaults for the per-file arguments. XXX: Be sure to update the other function if you update this one!
    clip_group = parser.add_argument_group('Video Options',
        description='Options for videos. Can be specified at the global or per-video level.')

    clip_group.add_argument('-cl', '--crop-location', type=crop_location_type, default=('c', 'm'),
        help='Cropped portion should be in the top/middle/bottom and left/center/right. ".2,.3" means 20%% over '
            'from the left, and 30%% down from the top. 100%% means the right side of the crop window will be '
            'aligned with the right side of the original video.')
    clip_group.add_argument('-cs', '--crop-size', type=crop_size_type, default=(CropType.FRACTION, 1.0, 1.0),
        help='Cropped portion size. .5 means 50%% as wide and tall; .5,1 means half as wide, full height; '
          '1280x720 means exactly that size')
    clip_group.add_argument('-p', '--speedup', type=float, default=1.0,
        help='Change the speed. 2 means twice as fast. Disables audio.')
    clip_group.add_argument('-r', '--rotate', type=rotation_type, default=0,
        help='Rotate the video, cropping as needed. Positive values are clockwise.')
    clip_group.add_argument('-v', '--volume', type=float, default=1.0,
        help='Modify volume level. 2 means twice as loud. 0 means omit the audio track.')
    clip_group.add_argument('-s', '--stabilize', default=False, action='store_true',
        help='Stabilize the video')
    clip_group.add_argument('-t', '--tripod', type=time_type, default=None,
        help='Enable tripod mode, stabilizing on the time specified in TIME FORMAT.')
    clip_group.add_argument('-R',  '--reverse', action='store_true',
        help='Reverse the video.')
    clip_group.add_argument('-T', '--transition-duration', type=time_type, default=0.5,
        help='Transition duration when concatenating ranges, in TIME FORMAT.')
    clip_group.add_argument('-I', '--interlace-test', action='store_true',
        help='Enable testing the video for interlacing.')

    return parser

#-----------------------------------------------------------------------------------------------------------------------

class ClipArgumentParser(argparse.ArgumentParser):
    def __init__(self, global_parser, global_args, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._global_parser = global_parser

        # These options can be specified globally for all files, or separately for each file. Use the global values as
        # the defaults for the per-file arguments. XXX: Be sure to update the other function if you update this one!
        self.add_argument('-cl', '--crop-location', type=crop_location_type, default=global_args.crop_location,
            help='Cropped portion should be in the top/middle/bottom and left/center/right. ".2,.3" means 20%% over '
                'from the left, and 30%% down from the top. 100%% means the right side of the crop window will be '
                'aligned with the right side of the original video.')
        self.add_argument('-cs', '--crop-size', type=crop_size_type, default=global_args.crop_size,
            help='Cropped portion size. .5 means 50%% as wide and tall; .5,1 means half as wide, full height; '
              '16:9 means the largest possible video with that aspect ratio; 1280x720 means exactly that size')
        self.add_argument('-p', '--speedup', type=float, default=global_args.speedup,
            help='Change the speed. 2 means twice as fast. Disables audio.')
        self.add_argument('-r', '--rotate', type=rotation_type, default=global_args.rotate,
            help='Rotate the video, cropping as needed. Positive values are clockwise.')
        self.add_argument('-v', '--volume', type=float, default=global_args.volume,
            help='Modify volume level. 2 means twice as loud.')
        self.add_argument('-s', '--stabilize', default=global_args.stabilize, action='store_true',
            help='Stabilize the video')
        self.add_argument('-t', '--tripod', type=time_type, default=global_args.tripod,
            help='Enable tripod mode, stabilizing on the time specified in TIME FORMAT')
        self.add_argument('-R', '--reverse', action='store_true', default=global_args.reverse,
            help='Reverse the video.')
        self.add_argument('-T', '--transition-duration', type=time_type, default=global_args.transition_duration,
            help='Transition duration when concatenating ranges, in TIME FORMAT')
        self.add_argument('-I', '--interlace-test', action='store_true', default=global_args.interlace_test,
            help='Enable testing the video for interlacing.')

        # These options are per-file only
        self.add_argument('input_file', type=video_file_type)
        self.add_argument('time_ranges', type=time_range_type, nargs='*',
            help='Format is start-end, where the times are in TIME FORMAT, or empty to indicate the start or end of the '
            'video')

    def format_usage(self):
        return self._global_parser.format_usage()

    def format_help(self):
        return self._global_parser.format_help()

#-----------------------------------------------------------------------------------------------------------------------

def dimensions_type(arg_value):
    try:
        width, height = map(int, arg_value.split('x'))
        return Dimensions(width, height)
    except:
        raise argparse.ArgumentTypeError(f'Invalid dimensions format: "{arg_value}". Expected format is "1280x720"')

#-----------------------------------------------------------------------------------------------------------------------

# Returns (CropType, int, int) or (CropType, float, float)
def crop_size_type(arg_value):
    try:
        if ':' in arg_value:
            width, height = map(float, arg_value.split(':'))
            return (CropType.ASPECT, width, height)
        elif 'x' in arg_value:
            width, height = map(int, arg_value.split('x'))
            return (CropType.PIXELS, width, height)
        else:
            if ',' in arg_value:
                width, height = map(float, arg_value.split(','))
            else:
                width, height = float(arg_value), float(arg_value)

            return (CropType.FRACTION, width, height)
    except:
        raise argparse.ArgumentTypeError(f'Invalid crop format: "{arg_value}". '
            'Expected format is ".5", ".5,.9", "16:9", or "1280x720"')

#-----------------------------------------------------------------------------------------------------------------------

def crop_location_type(arg_value):
    symbolic_positions = ['lt', 'ct', 'rt', 'lm', 'cm', 'rm', 'lb', 'cb', 'rb' ]

    try:
        if ',' in arg_value:
            x, y = map(float, arg_value.split(','))

            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise

            return (x, y)

        if len(arg_value) != 2:
            raise

        (x, y) = arg_value[0:2]

        # User got it backwards. Switch them.
        if x not in ('l', 'c', 'r'):
            x, y = y, x

        if x not in ('l', 'c', 'r') or y not in ('t', 'm', 'b'):
            raise

        return (x, y)
    except:
        raise argparse.ArgumentTypeError(f'Invalid crop location: "{arg_value}". '
            'Expected format is "lt", "ct", "rt", "lm", "cm", "rm", "lb", "cb", "rb", or "x,y"')

#-----------------------------------------------------------------------------------------------------------------------

def time_type(arg_value):
    return in_seconds(arg_value)

#-----------------------------------------------------------------------------------------------------------------------

def is_time_range(arg_value):
    parts = arg_value.split("-")
    if len(parts) != 2:
        raise ValueError(f'Invalid time range {arg_value}')

    start_string, end_string = parts

    if start_string != '' and not is_time_string(start_string): return False
    if end_string != '' and not is_time_string(end_string): return False

    return True

#-----------------------------------------------------------------------------------------------------------------------

def is_time_string(string):
    try:
        in_seconds(string)
        return True
    except:
        pass

    return False

#-----------------------------------------------------------------------------------------------------------------------

def in_seconds(time_string):
    if time_string.count('.') > 1:
        raise ValueError(f'{time_string} has more than one "."')

    rest, _, fraction_digits = time_string.partition('.')

    if rest.count(':') > 2:
        raise ValueError(f'{time_string} has more than two ":"')

    hours, minutes, seconds = (['0', '0'] + rest.split(":"))[-3:]

    seconds = seconds or '0'

    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    fraction = 0 if fraction_digits == '' else float('.'+fraction_digits)

    if hours > 0 and minutes > 59 or minutes > 0 and seconds > 59:
        raise ValueError(f'{time_string} is not a valid [[HH:]MM:]SS value')

    return 3600 * hours + 60 * minutes + seconds + fraction

#-----------------------------------------------------------------------------------------------------------------------

def in_hms(total_seconds, precision=2):
    fraction = total_seconds - int(total_seconds)
    hours = int(total_seconds) // 3600
    minutes = int(total_seconds) % 3600 // 60
    seconds = int(total_seconds) % 60

    hms = f'{hours}:{minutes:02d}:{seconds:02d}'

    # Remove "00:" values from the start. But leave one 0: so that we get 0:05
    hms = hms.removeprefix('0:').removeprefix('0')

    # Append the ".1234" part
    if fraction != 0 and precision > 0:
        frac_str = str(fraction).lstrip('0')

        frac_str = frac_str[0:precision+1]

        hms += frac_str

    hms = hms or '0'

    return hms

#-----------------------------------------------------------------------------------------------------------------------

# Returns None for the end time to indicate the end of the video. We replace that with the video's duration in
# post-processing
def time_range_type(arg_value):
    try:
        if not is_time_range(arg_value): raise

        start_string, end_string = arg_value.split("-")

        start = 0.0 if start_string == '' else in_seconds(start_string)
        end = None if end_string == '' else in_seconds(end_string)

        # We do this once the None has been resolved.
#        if start >= end: raise
    except:
        raise argparse.ArgumentTypeError(f'Invalid time range format: "{arg_value}". Expected format is start-end, '
            'where the times are as described in the help, or empty to indicate the start or end of the video')

    return TimeRange(start, end)

#-----------------------------------------------------------------------------------------------------------------------

def video_file_type(arg_value):
    if not os.path.exists(arg_value):
        raise argparse.ArgumentTypeError(f'File "{arg_value}" does not exist.')

    return arg_value

#-----------------------------------------------------------------------------------------------------------------------

def positive_int_type(arg_value):
    try:
        value = int(arg_value)

        if value <= 0: raise
    except:
        raise argparse.ArgumentTypeError(f'"{arg_value}" must be a positive integer.')

    return value

#-----------------------------------------------------------------------------------------------------------------------

def rotation_type(arg_value):
    angle = float(arg_value)

    angle %= 360

    return angle

#-----------------------------------------------------------------------------------------------------------------------

def split_clip_args(remainder_args, global_parser):
    blocks = []
    current_block = []
    found_file = False

    for arg in remainder_args:
        # Finalize if we saw the file already and (1) it's a file, or (2) it's a flag (not a time string)
        if found_file and (os.path.isfile(arg) or not is_time_range(arg)):
            blocks += [current_block]
            current_block = []
            found_file = False

        if os.path.isfile(arg):
            found_file = True

        current_block += [arg]

    if current_block:
        if found_file:
            blocks += [current_block]
        else:
            global_parser.error(f'Extra arguments found while parsing multi-file input. (Does the input file exist?): '
                f'{current_block}')

    return blocks

#-----------------------------------------------------------------------------------------------------------------------

# Flatten the structure so that the video is a series of clips, each with a start and end
def make_clips(clip_args, video):
    if len(clip_args.time_ranges) == 0:
        time_ranges = [ TimeRange(0, None) ]
    else:
        time_ranges = clip_args.time_ranges
        del clip_args.time_ranges

    clip_args.crop = Crop(*clip_args.crop_size, *clip_args.crop_location)

    del clip_args.crop_size
    del clip_args.crop_location

    for time_range in time_ranges:
        attrs = vars(clip_args)

        attrs['start'] = time_range.start
        attrs['end'] = time_range.end
        attrs['index'] = len(video)

        video += [ Clip(**attrs) ]

#-----------------------------------------------------------------------------------------------------------------------

def parse_arguments():
    global_parser = global_options_parser()

    # There's the single input and multi-input syntaxes. The latter can be identified by "--"
    if '--' in sys.argv:
        global_argv = sys.argv[1:sys.argv.index('--')]
        clips_argv = sys.argv[sys.argv.index('--')+1:]

        global_args = global_parser.parse_args(global_argv)
    else:
        original_args = list(sys.argv)

        global_args, clips_argv = global_parser.parse_known_args()

        if clips_argv < original_args[-len(clips_argv):]:
            cprint('[red]Found options within the input files and time ranges. Please use "--"'
                'to specify per-file options, or place global options before input files.[/]')
            sys.exit(1)

    if global_args.debug:
        global DEBUG
        DEBUG = True

        # Re-enable to add local vars to stack traces
        disable_pretty_exceptions()
        enable_pretty_exceptions()

    video = Video(global_args.output_dir, global_args.output_file, global_args.dimensions_limit,
        global_args.frame_rate_limit)

    dprint(f'Parsing remaining arguments {clips_argv}')

    # Process per-input options and time ranges
    for file_number, arg_block in enumerate(split_clip_args(clips_argv, global_parser)):
        dprint(f'- Parsed argument block {arg_block}')
        file_parser = ClipArgumentParser(global_parser, global_args)
        clip_args, remaining_args = file_parser.parse_known_args(arg_block)

        clip_args.file_number = file_number

        # Parse ones that look like -1:20 (an option)
        for arg_value in remaining_args:
            time_range = time_range_type(arg_value)
            clip_args.time_ranges.append(time_range)

        make_clips(clip_args, video)

    video.adjust_clip_durations_for_transitions()

#    dprint(f'Global args:')
#    dprint(pformat(vars(global_args)))

    # To force debug output to go before the "Video information" line below
    video.output_duration

    dprint(f'Video information:')
    dprint(str(video), prefix='- ')

    dprint(f'Video Clips:')
    dprint(pformat([vars(f) for f in video], width=CONSOLE.width))

    return video

#-----------------------------------------------------------------------------------------------------------------------

def dprint_command(text, command):
    if not DEBUG:
        return

    print_command(text, command)

#-----------------------------------------------------------------------------------------------------------------------

def print_command(text, command):
    if text:
        cprint(f'[violet]{text}[/]:')

    if command is None:
        str_command = str(command)
    else:
        str_command = " ".join([command[0]] + [shlex.quote(j) for j in command[1:]])

    cprint(str_command, style='royal_blue1', highlight=False, markup=False, soft_wrap=True)

#-----------------------------------------------------------------------------------------------------------------------

def encoded_file_not_much_smaller(video):
    if DEBUG:
        return True

    output_size = os.path.getsize(video.output_file)

    if len(video) > 1 or (video[0].start, video[0].end) != (0, video[0].input_duration):
        input_size = video.estimated_file_size
        is_estimate = True
    else:
        input_size = os.path.getsize(video[0].input_file)
        is_estimate = False

    if output_size > input_size:
        smaller_or_larger = 'larger'
        percent_difference = -100 * (input_size - output_size) / input_size
    else:
        smaller_or_larger = 'smaller'
        percent_difference = 100 * (input_size - output_size) / input_size

    percent_difference_text = f' ({percent_difference:.0f}% {smaller_or_larger} than ' \
        f'the{" estimated" if is_estimate else ""} original size of {humanize.naturalsize(input_size)})'

    cprint(f'[violet]Output file size is {humanize.naturalsize(output_size)}{percent_difference_text}[/]')

    if output_size > input_size:
        cprint(f'[yellow1]WARNING[/]: Output file size is LARGER than {"estimated " if is_estimate else ""}'
            f'input size ({humanize.naturalsize(output_size)} > {humanize.naturalsize(input_size)})')
    elif percent_difference < WARNING_THRESHOLD:
        cprint(f'[yellow1]WARNING[/]: Output file size is only {percent_difference:.0f}% smaller than '
            f'{"estimated" if is_estimate else ""} input size '
            f'({humanize.naturalsize(output_size)} > {humanize.naturalsize(input_size)})')

    return output_size > input_size or percent_difference < COPY_THRESHOLD

#-----------------------------------------------------------------------------------------------------------------------

def build_video_stream_filters(video, clip, kind, skip_stabilization):
    # Clear out the filters from the prep run
    clip.video_filters = []

    if video.max_video_bitrate is None:
        return

    if clip.video_bitrate is None:
        clip.video_filters += blank_video(video, clip, kind)
        return

    # First start with the transformative stuff that would affect the stabilization data.
    clip.video_filters += compute_trim(clip, 'video', kind, skip_stabilization)
    clip.video_filters += compute_video_speedup(clip)
    clip.video_filters += compute_deinterlace(clip)
    clip.video_filters += compute_rotation(clip)
    clip.video_filters += compute_crop(clip.crop)
    clip.video_filters += compute_reverse(clip.reverse, 'video')
    # Force every clip to use the max fps, so that xfade will work. Also normalize the timebase, for the same reason
    clip.video_filters += compute_fps(clip, video.max_avg_frame_rate)
    clip.video_filters += compute_timebase(video)

    # Boost red for underwater
#    clip.video_filters += [ ( 'curves', [], { 'red': '0/.75 .25/1 1/1' } ) ]
    clip.video_filters += compute_stabilize(clip, kind, skip_stabilization)

    # Now add in the stuff that shouldn't be used for stabilization detection
    if kind == 'encode':
        clip.video_filters += compute_color(clip, video)
        clip.video_filters += compute_scale(clip, video)
        clip.video_filters += compute_unsharp(clip)

#-----------------------------------------------------------------------------------------------------------------------

def blank_video(video, clip, kind):
    if kind == 'prep':
        return []

    return [ ( 'color', [], {
        'color':'black',
        'size': video.output_dims,
        'duration': clip.end-clip.start,
        'rate': video.max_avg_frame_rate
    } ) ]

#-----------------------------------------------------------------------------------------------------------------------

def compute_trim(clip, video_or_audio, kind, skip_stabilization=False):
    prefix = 'a' if video_or_audio == 'audio' else ''

    trim_params = {}

    if skip_stabilization or kind == 'prep':
        trim_params['start_frame'] = 1
        trim_params['end_frame'] = 2
    else:
        if clip.start > 0:
            trim_params['start'] = clip.start

        if clip.end != clip.input_duration:
            trim_params['end'] = clip.end

    if not trim_params:
        return []

    trim_filters = []

    trim_filters += [ ( f'{prefix}trim', [], trim_params ) ]
    trim_filters += [ ( f'{prefix}setpts', [ 'PTS-STARTPTS' ], {} ) ]

    return trim_filters

#-----------------------------------------------------------------------------------------------------------------------

def compute_video_speedup(clip):
    if clip.speedup == 1.0:
        return []

    return [ ( f'setpts', [ f'PTS/{clip.speedup}' ], {} ) ]

#-----------------------------------------------------------------------------------------------------------------------

def compute_audio_speedup(clip):
    filters = []
    speedup = clip.speedup

    # atempo only works between 0.5 and 2.0
    while speedup < 0.5:
        filters += [ ( 'atempo', [0.5], {} ) ]
        speedup /= 0.5

    while speedup > 2.0:
        filters += [ ( 'atempo', [2], {} ) ]
        speedup /= 2

    if speedup != 1.0:
        filters += [ ( 'atempo', [speedup], {} ) ]
        speedup = 1.0

    return filters

#-----------------------------------------------------------------------------------------------------------------------

def compute_deinterlace(clip):
    if clip.interlaced is None or not clip.interlaced:
        return []

    return [ 'yadif', [], {} ]

#-----------------------------------------------------------------------------------------------------------------------

def compute_rotation(clip):
    # Clockwise 90 degrees
    if clip.rotate == 0:
        return []
    elif clip.rotate == 90:
        return [ ( 'transpose', [], { 'dir': 'clock' } ) ]
    elif clip.rotate == 270:
        return [ ( 'transpose', [], { 'dir': 'cclock' } ) ]
    elif clip.rotate == 180:
        return [ ( 'transpose', [], { 'dir': 'cclock' } ), ( 'transpose', [], { 'dir': 'cclock' } ) ]

    # Here we go. Thanks to ChatGPT for helping me finish the initial derivation.
    in_angle = clip.rotate % 180
    in_width, in_height = clip.input_dims.width, clip.input_dims.height

    angle_1 = in_angle % 180
    normalized_angle = angle_1 if angle_1 <= 90 else 180-angle_1
    angle = math.radians(normalized_angle)

    if normalized_angle < 45:
        s_h = in_height / (in_height * math.cos(angle) + in_width * math.sin(angle))
        s_w = in_width / (in_width * math.cos(angle) + in_height * math.sin(angle))

        s = min(s_h, s_w)

        width, height = int(in_width * s), int(in_height * s)
    else:
        normalized_angle = 90 - normalized_angle
        angle = math.radians(normalized_angle)

        s_h = in_width / (in_width * math.cos(angle) + in_height * math.sin(angle))
        s_w = in_height / (in_height * math.cos(angle) + in_width * math.sin(angle))

        s = min(s_h, s_w)

        width, height = int(in_height * s), int(in_width * s)

    crop = Crop(CropType.PIXELS, width, height, 'c', 'm')
    ogar = math.radians(clip.rotate)

    # Make it even for the encoder, rounding up
    return [ ( 'rotate', [ogar], { 'ow': f'ceil(rotw({ogar})/2)*2', 'oh': f'ceil(roth({ogar})/2)*2' } ),
             ( 'pad', ['iw+2', 'ih+2'], {} ) ] + compute_crop(crop)

#-----------------------------------------------------------------------------------------------------------------------

def compute_crop(crop):
    if crop.width == 1 and crop.height == 1 and crop.type == CropType.FRACTION:
        return []

    if crop.x == 'l':
        x_pos = '0'
    elif crop.x == 'c':
        x_pos = f'(iw-ow)/2'
    elif crop.x == 'r':
        x_pos = f'iw-ow'
    else:
        # it's a fractional offset
        x_pos = f'(iw-ow)*{crop.x}'

    if crop.y == 't':
        y_pos = '0'
    elif crop.y == 'm':
        y_pos = f'(ih-oh)/2'
    elif crop.y == 'b':
        y_pos = f'ih-oh'
    else:
        # it's a fractional offset
        y_pos = f'(ih-oh)*{crop.y}'

    # Make it even for the encoder. Round down
    if crop.type == CropType.PIXELS:
        width = f'trunc({crop.width}/2)*2'
        height = f'trunc({crop.height}/2)*2'
    elif crop.type == CropType.ASPECT:
        ar = crop.aspect_ratio

        width = f'trunc((if(gte(iw/ih,{ar}),ih*{ar},iw))/2)*2'
        height = f'trunc((if(gte(iw/ih,{ar}),ih,iw/{ar}))/2)*2'
    elif crop.type == CropType.FRACTION:
        width = f'trunc({crop.width}*iw/2)*2'
        height = f'trunc({crop.height}*ih/2)*2'
    else:
        assert(False)

    return [ ( 'crop', [], { 'w': width, 'h': height, 'x': x_pos, 'y': y_pos } ) ]

#-----------------------------------------------------------------------------------------------------------------------

def compute_reverse(is_reversed, video_or_audio):
    if not is_reversed:
        return []

    prefix = 'a' if video_or_audio == 'audio' else ''

    reverse_filters = []

    reverse_filters += [ ( f'{prefix}reverse', [], {} ) ]
    reverse_filters += [ ( f'{prefix}setpts', [ 'PTS-STARTPTS' ], {} ) ]

    return reverse_filters

#-----------------------------------------------------------------------------------------------------------------------

# Always return an FPS filter because otherwise we can't cross-fade properly. Even when we don't "need" it because
# fps == max_avg_frame_rate, xfade will still fail, perhaps due to floating point differences
# (Example: 29.978449996009257).
def compute_fps(clip, max_avg_frame_rate):
# I found that ffmpeg would complain if I didn't always put the frame rate in. Probably floating point errors.
#    if clip.avg_frame_rate == max_avg_frame_rate:
#        return []

    if len(video) == 1:
        return []

    return [ ( 'fps', [], { 'fps': max_avg_frame_rate } ) ]

#-----------------------------------------------------------------------------------------------------------------------

def compute_timebase(video):
    if len(video) == 1:
        return []

    return [ ( 'settb', [ 'AVTB' ], {} ) ]

#-----------------------------------------------------------------------------------------------------------------------

def compute_stabilize(clip, kind, skip_stabilization):
    if not clip.stabilize:
        return []

    if skip_stabilization:
        return []

    stabilization_options = {}

    if clip.tripod is not None:
        stabilization_options['tripod'] = int(clip.tripod*clip.avg_frame_rate)

    if kind == 'prep':
        stabilization_options['mincontrast'] = clip.mincontrast
        stabilization_options['shakiness'] = clip.shakiness
        stabilization_options['result'] = clip.transforms_file

        return [ ( 'vidstabdetect', [], stabilization_options ) ]
    else:
        stabilization_options['input'] = clip.transforms_file
        stabilization_options['smoothing'] = clip.smoothing

        return [ ( 'vidstabtransform', [], stabilization_options ) ]

#-----------------------------------------------------------------------------------------------------------------------

def compute_color(clip, video):
    clip_pxl_fmt = parse_pixel_format(clip.pixel_format)
    video_pxl_fmt = parse_pixel_format(video.max_pixel_format)

    if clip_pxl_fmt['family'] == video_pxl_fmt['family']:
        return []

    return [ ( 'scale', [], { 'out_range': 'limited' } ) ]

#-----------------------------------------------------------------------------------------------------------------------

def compute_scale(clip, video):
    if clip.filtered_dims == video.output_dims:
        return []

    filters = [ ( 'scale',  [], { 'w': clip.output_dims.width, 'h': clip.output_dims.height,
            'force_original_aspect_ratio': 'decrease' } ) ]

    if clip.output_dims != video.output_dims:
        x = int( (video.output_dims.width - clip.output_dims.width) / 2 )
        y = int( (video.output_dims.height - clip.output_dims.height) / 2 )

        filters += [ ( 'pad', [], { 'w': video.output_dims.width, 'h': video.output_dims.height, 'x': x, 'y': y } ) ]

    return filters

#-----------------------------------------------------------------------------------------------------------------------

def compute_unsharp(clip):
    if not clip.stabilize:
        return []

    # Recommended by vid.stab documentation
    return [ ( 'unsharp', [5, 5, 0.8, 3, 3, 0.4], {} ) ]

#-----------------------------------------------------------------------------------------------------------------------

def build_audio_stream_filters(video, clip, kind):
    # Clear out the filters from the prep run
    clip.audio_filters = []

    if kind == 'prep' or video.max_audio_bitrate is None:
        return

    if clip.audio_bitrate is None:
        clip.audio_filters += blank_audio(clip)
        return

    clip.audio_filters += compute_trim(clip, 'audio', kind)
    clip.audio_filters += compute_reverse(clip.reverse, 'audio')
    clip.audio_filters += compute_audio_speedup(clip)
    clip.audio_filters += compute_volume(clip)

#-----------------------------------------------------------------------------------------------------------------------

def blank_audio(clip):
    return [ ( 'anullsrc', [], { 'sample_rate': 44100, 'channel_layout': 'stereo',
        'duration': (clip.end-clip.start)*clip.speedup } ) ]

#-----------------------------------------------------------------------------------------------------------------------

def compute_volume(clip):
    if clip.volume == 1:
        return []

    return [ ( 'volume', [ clip.volume ], {} ) ]

#-----------------------------------------------------------------------------------------------------------------------

def build_video_transition(clip, offset):
    assert(offset > 0)

    if clip.transition_duration == 0:
        clip.video_transition_filter = ( 'concat', [], { 'n': 2, 'v': 1, 'a': 0 } )
    else:
        clip.video_transition_filter = \
            ( 'xfade', [], { 'transition': 'fade', 'duration': clip.transition_duration, 'offset': offset } )

#-----------------------------------------------------------------------------------------------------------------------

def build_audio_transition(clip, offset):
    assert(offset > 0)

    if clip.transition_duration == 0:
        clip.audio_transition_filter = ( 'concat', [], { 'n': 2, 'v': 0, 'a': 1 } )
    else:
        clip.audio_transition_filter = ( 'acrossfade', [], { 'duration': clip.transition_duration } )

#-----------------------------------------------------------------------------------------------------------------------

def build_av_filters(video, kind, skip_stabilization):
    # Video clips
    if video.max_video_bitrate is not None:
        for clip in video:
            build_video_stream_filters(video, clip, kind, skip_stabilization)

    # Audio clips
    if video.max_audio_bitrate is not None:
        for clip in video:
            build_audio_stream_filters(video, clip, kind)

    # Video and audio cross-fade transitions
    if kind == 'encode':
        offset = 0

        for clip in video:
            # Transition starts before the end of the previous clip.
            if offset != 0:
                offset -= clip.transition_duration

                build_video_transition(clip, offset)
                build_audio_transition(clip, offset)

            offset += clip.output_duration

    for clip in video:
        dprint(f'Clip {clip.index}:')
        dprint('- Video filters:')
        dprint(pformat(clip.video_filters), prefix='  ')
        dprint('- Audio filters:')
        dprint(pformat(clip.audio_filters), prefix='  ')
        dprint('- Video transition filter:')
        dprint(pformat(clip.video_transition_filter), prefix='  ')
        dprint('- Audio transition filter:')
        dprint(pformat(clip.audio_transition_filter), prefix='  ')

#-----------------------------------------------------------------------------------------------------------------------

TEMPORARY_FILES = []

def cleanup():
    for temp_file in TEMPORARY_FILES:
        if os.path.exists(temp_file):
            os.remove(temp_file)

atexit.register(cleanup)

#-----------------------------------------------------------------------------------------------------------------------

def make_temp_filename(input_file, extension=None):
    basename = os.path.basename(input_file)
    root, ext = os.path.splitext(basename)

    if extension is not None:
        ext = extension

    file = tempfile.NamedTemporaryFile(mode='w+', prefix=f'{root}-', suffix=ext, delete=False)
    file.close()

    temp_file = file.name

    global TEMPORARY_FILES
    TEMPORARY_FILES += [ temp_file ]

    os.remove(temp_file)

    return temp_file

#-----------------------------------------------------------------------------------------------------------------------

# Workaround for https://github.com/kkroening/ffmpeg-python/issues/880
def unique_input_file(input_file):
    temp_file = make_temp_filename(input_file)

    os.symlink(os.path.abspath(input_file), temp_file)

    return temp_file

#-----------------------------------------------------------------------------------------------------------------------

def build_prep_command(video, skip_stabilization=False):
    build_av_filters(video, 'prep', skip_stabilization)

    f_outputs = []

    for clip in video:
        f_input = ffmpeg.input( unique_input_file(clip.input_file) )

        if clip.video_filters != []:
            f_video = f_input.video

            for v_filter in clip.video_filters:
                f_video = ffmpeg.filter(f_video, v_filter[0], *v_filter[1], **v_filter[2])

            # For prep runs, dump output and don't add encoding options. Also add metadata so that I can know which
            # ffmpeg stream is which
            f_outputs += [ ffmpeg.output(f_video, '-', **{'metadata:s:v:0': f'clip={clip.index}'}, f='null') ]

    return f_outputs

#-----------------------------------------------------------------------------------------------------------------------

@lru_cache
def make_blank_1s_video(color=None, size=None, duration=1, rate=None):
    # Force it to be 1s, for speed and disk space
    duration = 1

    blank_video_file = make_temp_filename('blank.mp4', extension='.mp4')

    command = ['ffmpeg', '-f', 'lavfi', '-i', f'color=color={color}:size={size}:duration=1:rate={rate}',
        blank_video_file.name]

    dprint_command('Blank video command', command)

    run_ffmpeg(command, blank_video_file.name)

    return blank_video_file.name

#-----------------------------------------------------------------------------------------------------------------------

@lru_cache
def make_blank_1s_audio(sample_rate=None, channel_layout=None, duration=1):
    # Force it to be 1s, for speed and disk space
    duration = 1

    blank_audio_file = make_temp_filename('blank.m4a', extension='.m4a')

    command = ['ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=sample_rate={sample_rate}:channel_layout={channel_layout}',
        '-t', str(duration), '-c:a', 'aac', '-b:a', '192k', blank_audio_file.name]

    dprint_command('Silent audio command', command)

    run_ffmpeg(command, blank_audio_file.name)

    return blank_audio_file.name

#-----------------------------------------------------------------------------------------------------------------------

def build_video_encode_command(video):
    f_previous_video = None

    for clip in video:
        f_input = ffmpeg.input( unique_input_file(clip.input_file) )

        dprint(f'- Building video pipeline for clip {clip.index}')

        # Video
        if clip.video_bitrate is None:
            continue

        if clip.video_filters == [] or clip.video_filters[0][0] != 'color':
            dprint(f'  - Video: Using input file {clip.input_file}')

            f_video = f_input.video
            video_filters = clip.video_filters
        else:
            dprint(f'  - Video: Creating blank video file')

            # Convert the "color" input source in a real file because ffmpeg-python doesn't know how to do it in the
            # filtergraph
            blank_video_filename = make_blank_1s_video(**(clip.video_filters[0][2]))

            f_video = ffmpeg.input(blank_video_filename).video

            video_filters = [ ( 'setpts', [ f'PTS*{clip.output_duration}' ], {} ) ] + \
                    compute_fps(clip, video.max_avg_frame_rate) + compute_timebase(video) + clip.video_filters[1:]

        dprint('    - Video filters:')
        dprint(pformat(video_filters), prefix='      ')

        for v_filter in video_filters:
            f_video = ffmpeg.filter(f_video, v_filter[0], *v_filter[1], **v_filter[2])

        # Transitions
        if f_previous_video is not None:
            dprint(f'  - Applying video transition filter between clip {clip.index-1} and clip {clip.index}')
            dprint(pformat(clip.video_transition_filter), prefix='    ')

            transition = clip.video_transition_filter
            f_video = ffmpeg.filter([f_previous_video, f_video], transition[0], *transition[1], **transition[2])

        f_previous_video = f_video

    return f_previous_video

#-----------------------------------------------------------------------------------------------------------------------

def build_audio_encode_command(video):
    f_previous_audio = None

    for clip in video:
        f_input = ffmpeg.input( unique_input_file(clip.input_file) )

        dprint(f'- Building audio pipeline for clip {clip.index}')

        # Audio
        if clip.audio_bitrate is None:
            continue

        if clip.audio_filters == [] or clip.audio_filters[0][0] != 'anullsrc':
            dprint(f'  - Audio: Using input file {clip.input_file}')

            f_audio = f_input.audio
            audio_filters = clip.audio_filters
        else:
            dprint(f'  - Audio: Creating blank audio file')

            # Convert the "anullsrc" input source in a real file because ffmpeg-python doesn't know how to do it in
            # the filtergraph
            blank_audio_filename = make_blank_1s_audio(**(clip.audio_filters[0][2]))

            f_audio = ffmpeg.input(blank_audio_filename).audio

            audio_filters = [ ( 'asetpts', [ f'PTS*{clip.output_duration}' ], {} ) ] + clip.audio_filters[1:]

        dprint('    - Audio filters:')
        dprint(pformat(audio_filters), prefix='      ')

        for a_filter in audio_filters:
            f_audio = ffmpeg.filter(f_audio, a_filter[0], *a_filter[1], **a_filter[2])

        # Transitions
        if f_previous_audio is not None:
            dprint(f'  - Applying audio transition filter between clip {clip.index-1} and clip {clip.index}')
            dprint(pformat(clip.audio_transition_filter), prefix='      ')

            transition = clip.audio_transition_filter
            f_audio = ffmpeg.filter([f_previous_audio, f_audio], transition[0], *transition[1], **transition[2])

        f_previous_audio = f_audio

    return f_previous_audio

#-----------------------------------------------------------------------------------------------------------------------

def parse_to_dtu(iso8601_time):
    if isinstance(iso8601_time, datetime.date):
        if iso8601_time.tzinfo is None:
            raise ValueError("Input datetime.datetime object must be timezone-aware (UTC)")

        return iso8601_time

    if isinstance(iso8601_time, str):
        if not iso8601_time.endswith('Z'):
            raise ValueError(f"Input time string must end with 'Z': {iso8601_time}")

        # Remove the 'Z' to allow strptime to parse the date/time components
        dt_str_no_z = iso8601_time.rstrip('Z')

        try:
            dt_naive = datetime.datetime.strptime(dt_str_no_z, "%Y-%m-%dT%H:%M:%S.%f")
            return dt_naive.replace(tzinfo=datetime.timezone.utc)
        except ValueError as e:
            raise ValueError(f"Could not parse time string '{iso8601_time}'. Original error: {e}")

    raise ValueError('unknown time format')

#-----------------------------------------------------------------------------------------------------------------------

def adjust_iso8601_time(iso8601_time, offset):
    dt_new = parse_to_dtu(iso8601_time) + datetime.timedelta(seconds=offset)

    return dt_new.strftime("%Y-%m-%dT%H:%M:%S.%f") + 'Z'

#-----------------------------------------------------------------------------------------------------------------------

def file_creation_time_iso8601(filepath, offset=0):
    timestamp_seconds = os.stat(filepath).st_ctime

    utc_dt = datetime.datetime.fromtimestamp(timestamp_seconds, tz=datetime.timezone.utc)

    return adjust_iso8601_time(utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), offset)

#-----------------------------------------------------------------------------------------------------------------------

def build_encode_command_with_parameters(video, f_previous_video, f_previous_audio, copy_video=False):
    input_streams = [f for f in (f_previous_video, f_previous_audio) if f is not None]

    if not input_streams:
        return None

    named_params = {}

    if f_previous_video is not None:
        if copy_video and video.can_copy_video:
            clip = video[0]

            named_params['vcodec'] = 'copy'
            named_params['strict'] = 'unofficial'
        else:
            # CRF of 20 gave the highest VMAF score on a video directly from my iPhone. Any lower and the size blew
            # up without improving quality.  ffmpeg -i original.mov -i encoded.mov -lavfi libvmaf -f null
            # The best I could do was 83. I'm not sure why.
            named_params['vcodec'] = 'libx265'
            named_params['pix_fmt'] = video.max_pixel_format
            named_params['crf'] = 20

            # Ensure we don't "over-quality" videos with the -crf 20 above, ballooning the file size needlessly.
            named_params['maxrate'] = video.max_video_bitrate
            named_params['bufsize'] = 2*video.max_video_bitrate

            # Needed for thumbnails to work on Mac
            named_params['tag:v'] = 'hvc1'

            # Helps prevent failures with some malformed input videos
            named_params['max_muxing_queue_size'] = 1024

    if f_previous_audio is not None:
        if video.can_copy_audio:
            named_params['acodec'] = 'copy'
        else:
            named_params['acodec'] = 'aac'
            named_params['audio_bitrate'] = '192k'

    if video.output_duration is not None:
        named_params['metadata:g'] = f'creation_time={video.output_creation_time}'

    if f_previous_video is None and f_previous_audio is not None:
        video.output_file = os.path.splitext(video.output_file)[0] + '.m4a'

    return ffmpeg.output(*input_streams, video.output_file, **named_params)

#-----------------------------------------------------------------------------------------------------------------------

def build_encode_command(video):
    build_av_filters(video, 'encode', skip_stabilization=False)

#    sys.exit(0)
    dprint('Using ffmpeg-python to build the ffmpeg encode command line')

    f_previous_video = build_video_encode_command(video)
    f_previous_audio = build_audio_encode_command(video)

    return build_encode_command_with_parameters(video, f_previous_video, f_previous_audio)

#-----------------------------------------------------------------------------------------------------------------------

def build_copy_command(video):
    assert(video.can_copy_video)

    clip = video[0]

    # Ignore the transpose filter. We'll handle it in metadata instead with the -display_rotation argument to the run
    # command
    # This is the reverse of what the user specified. Positive values are *counter-clockwise*
    params = { 'filename': unique_input_file(clip.input_file) }

    if clip.rotate:
        params['display_rotation'] = int(-(-clip.presentation_rotation + clip.rotate))

    f_input = ffmpeg.input( **params )

    dprint(f'- Building video pipeline for clip {clip.index}')

    assert(clip.video_filters == [] or [f[0] for f in clip.video_filters] == ['transpose'])

    # Allow re-encoding of the audio, since it's fast
    f_previous_audio = build_audio_encode_command(video)

    return build_encode_command_with_parameters(video, f_input.video, f_previous_audio, True)

#-----------------------------------------------------------------------------------------------------------------------

# Example with audio only processing:
# size=   19309KiB time=00:13:22.90 bitrate= 197.0kbits/s speed=81.7x elapsed=0:00:09.82

def time_completion(line):
    match = re.search(r"\btime=\s*([0-9:.]+)", line)
    if match:
        return in_seconds(match.group(1))

    return None

#-----------------------------------------------------------------------------------------------------------------------

def run_ffmpeg(command, output_file, progress=None, progress_callback=None):
    # Capture all output in a single buffer
    full_output_buffer = []

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    def sigint_handler(signum, frame):
        if progress:
            progress.stop()

        cprint('[red]Interrupted. Killing ffmpeg, cleaning up files, and exiting...[/]')

        process.send_signal(signal.SIGINT)

        if output_file is not None and os.path.exists(output_file):
            os.remove(output_file)

        exit(1)

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    # Use a dictionary to map file descriptors to streams and a buffer to store output
    stdout_fd, stderr_fd = process.stdout.fileno(), process.stderr.fileno()
    pipes = { stdout_fd: "stdout", stderr_fd: "stderr", }
    partial_lines = {stdout_fd: '', stderr_fd: ''}

    # Poll the pipes for new data
    while pipes:
        ready_to_read, _, _ = select.select(list(pipes.keys()), [], [], 0.1)

        if not ready_to_read:
            if process.poll() is not None:
                # No more data and process is finished, so break
                pipes.clear()
                break
            continue

        for fd in ready_to_read:
            chunk = os.read(fd, 4096).decode('utf-8')

            if not chunk:
                # End of stream
                pipes.pop(fd, None)
                continue

            # So that I can process each line of the progress
            chunk = chunk.replace('\r', '\n')

            partial_lines[fd] += chunk

            lines = partial_lines[fd].split('\n')

            for line in lines[:-1]:
                stream_type = pipes[fd]

                # Store the line and its source
                full_output_buffer.append((line, stream_type))

                # The progress output from ffmpeg is on stderr, so check for that
                if progress_callback is not None:
                    progress_callback(stream_type, line)

            partial_lines[fd] = lines[-1]

    # After the process exits, read any remaining output
    if partial_lines[stdout_fd]:
        full_output_buffer.append((partial_lines[stdout_fd], "stdout"))
    if partial_lines[stderr_fd]:
        full_output_buffer.append((partial_lines[stderr_fd], "stderr"))

    return_code = process.wait()

    # Restore the signal handler after the 'with' block
    signal.signal(signal.SIGINT, original_sigint_handler)

    if return_code != 0:
        cprint(f'[red]ffmpeg failed.[/] Command was:')
        print_command(None, command)

        cprint(f'[red]Full log:[/]')
        for line, stream_type in full_output_buffer:
            if line.endswith('\n'):
                line = line[:-1]

            rline = CONSOLE.render_str(line, markup=False, highlight=False)

            if stream_type == "stdout":
                cprint(rline)
            else:
                eprint(rline, style='yellow')

        sys.exit(return_code)

    return [ f[0] for f in full_output_buffer if f[1] == 'stdout' ], \
        [ f[0] for f in full_output_buffer if f[1] == 'stderr' ]

#-----------------------------------------------------------------------------------------------------------------------

class TimeColumn(ProgressColumn):
    def render(self, task: "Progress") -> Text:
        return Text(in_hms(task.completed, precision=0), style='progress.remaining')

#-----------------------------------------------------------------------------------------------------------------------

class TimeDescription(ProgressColumn):
    def render(self, task: "Progress") -> Text:
        if task.finished:
            return Text('elapsed', style='progress.elapsed')
        else:
            return Text('remaining', style='progress.remaining')

#-----------------------------------------------------------------------------------------------------------------------

def run_ffmpeg_with_progress(command, description, output_file, target_seconds, show_time):
    # Capture all output in a single buffer
    full_output_buffer = []

    columns = [ TextColumn("[progress.description]{task.description}") ] + \
        ( [ TimeColumn() ] if show_time else [] ) + \
        [
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "•",
            TimeRemainingColumn(elapsed_when_finished=True, compact=True),
            TimeDescription()
        ]

    with Progress(*columns) as progress:
        task = progress.add_task(f'[violet]{description}', total=target_seconds)

        def progress_callback(stream_type, line):
            if stream_type != "stderr":
                return

            completed = time_completion(line)

            if completed is None:
                return

            progress.update(task, completed=completed)

        ffmpeg_stdout, ffmpeg_stderr = run_ffmpeg(command, output_file, progress, progress_callback)

        progress.update(task, completed=target_seconds)

        return ffmpeg_stdout, ffmpeg_stderr

#-----------------------------------------------------------------------------------------------------------------------

def prepare(video):
    f_outputs = build_prep_command(video)
    command = ffmpeg.compile(f_outputs)

    dprint_command('Preparation command', command)

    longest_duration_stabilization = max([0] + [clip.output_duration for clip in video if clip.stabilize])

    # In debug mode, run without stabilization, since it's so slow
    if DEBUG and longest_duration_stabilization > 0:
        dprint(f'<Skipping long-running stabilization analysis for debug mode.>')
        f_outputs = build_prep_command(video, skip_stabilization=True)
        command = ffmpeg.compile(f_outputs)

        dprint_command('Running this preparation command', command)

    if not DEBUG and longest_duration_stabilization > 0:
        ffmpeg_stdout, ffmpeg_stderr = run_ffmpeg_with_progress(command, 'Collecting motion information...',
                None, longest_duration_stabilization, False)
    else:
        # Run quickly and quietly
        ffmpeg_stdout, ffmpeg_stderr = run_ffmpeg(command, None)

        for clip in video:
            if clip.stabilize:
                cprint(f"Motion detection data saved to {clip.transforms_file}")

    current_resolution = None

    for line in ffmpeg_stderr:
        # Look for resolution in the Stream line
        m = re.search(r'Stream #\d+:\d+.*?: Video: .*?, (\d+x\d+)', line)
        if m:
            width, height = m.group(1).split('x')
            current_resolution = Dimensions(int(width), int(height))

        # Look for clip metadata line, which comes later
        m = re.search(r'clip\s*:\s*(\d+)', line)
        if m:
            assert(current_resolution is not None)

            clip_index = int(m.group(1))
            video[clip_index].filtered_dims = current_resolution
            current_resolution = None

    dprint('Extracted dimensions from ffmpeg, after applying filters')

    for clip in video:
        if not hasattr(clip, 'filtered_dims'):
            clip.filtered_dims = None

        dprint(f'- Clip {clip.index} has filtered dimensions of {clip.filtered_dims} '
            f'aspect ratio {clip.filtered_dims.aspect_ratio if clip.filtered_dims else None}')

    compute_output_dimensions(video)

#-----------------------------------------------------------------------------------------------------------------------

def encode_video(video):
    f_output = build_encode_command(video)

    dprint_command('Encode command', ffmpeg.compile(f_output))

    if DEBUG:
        dprint('<Skipping ffmpeg encoding command>')
        return

    run_ffmpeg_with_progress(ffmpeg.compile(f_output, overwrite_output=True), 'Running ffmpeg to encode video...',
        video.output_file, video.output_duration, True)

    os.utime(video.output_file,
        (os.path.getatime(video[0].input_file), parse_to_dtu(video.output_creation_time).timestamp()))

#-----------------------------------------------------------------------------------------------------------------------

def copy_video(video):
    if DEBUG:
        return False

    if not video.can_copy_video:
        return True

    clip = video[0]

    # Literally copy the file if there are no changes.
    if clip.video_filters == [] and clip.audio_filters == []:
        cprint(f"[violet]Deleting the encoded file and copying the video (since there were no changes other than "
            "re-encoding) to avoid quality loss")

        os.remove(video.output_file)

        basename = os.path.basename(clip.input_file)
        output_path = os.path.dirname(video.output_file)
        output_file = os.path.join(output_path, basename)

        shutil.copy2(clip.input_file, output_file)
        return False

    f_output = build_copy_command(video)

    dprint_command('Copy command', ffmpeg.compile(f_output))

    if DEBUG:
        dprint('<Skipping ffmpeg copy command>')
        return False

    cprint(f"[violet]Deleting the encoded file and re-running ffmpeg, copying the video stream to avoid quality loss")

    run_ffmpeg(ffmpeg.compile(f_output, overwrite_output=True), video.output_file)

    os.utime(video.output_file,
        (os.path.getatime(video[0].input_file), parse_to_dtu(video.output_creation_time).timestamp()))

    return True

#-----------------------------------------------------------------------------------------------------------------------

# I'm declaring that portrait video pillboxed into a landscape video with black bars on the sides is anathema (and
# vice-versa).  If all the videos are portrait, but the specified width is more than the height, flip them (and
# vice versa).
def adjust_video_orientation(video):
    clip_aspect_ratios = {}

    clip_aspect_ratios = { '>1': 0, '<1': 0, '=1': 0 }

    for clip in video:
        if clip.filtered_dims is None:
            continue

        if clip.filtered_dims.aspect_ratio > 1:
            clip_aspect_ratios['>1'] += 1
        if clip.filtered_dims.aspect_ratio < 1:
            clip_aspect_ratios['<1'] += 1
        if clip.filtered_dims.aspect_ratio == 1:
            clip_aspect_ratios['=1'] += 1

    if video.dimensions_limit.width > video.dimensions_limit.height and clip_aspect_ratios['>1'] == 0 and clip_aspect_ratios['<1'] > 0:
        cprint(f'[yellow1]WARNING[/]: You specified a landscape width and height of {video.dimensions_limit}, '
            'but none of the videos are landscape. Swapping the height and width.')
        video.dimensions_limit = video.dimensions_limit.swap()

    if video.dimensions_limit.width < video.dimensions_limit.height and clip_aspect_ratios['<1'] == 0 and clip_aspect_ratios['>1'] > 0:
        cprint(f'[yellow1]WARNING[/]: You specified a portrait width and height of {video.dimensions_limit}, '
            'but none of the videos are portrait. Swapping the height and width.')
        video.dimensions_limit = video.dimensions_limit.swap()

#-----------------------------------------------------------------------------------------------------------------------

def compute_reduced_clip_dimensions(video):
    clip_dims = []

    # Don't reduce at this point if the max dimensions are relative. Just collect the sizes so that we can compute the
    # max video size, *then* reduce all the clip dimensions
    if isinstance(video.dimensions_limit.width, float):
        for clip in video:
            if clip.filtered_dims is not None:
                clip_dims += [ clip.filtered_dims ]

        return clip_dims

    # First compute the reduced dimensions for each clip
    for clip in video:
        if clip.filtered_dims is None:
            continue

        old_dims = clip.filtered_dims
        (width, height) = old_dims

        if width > video.dimensions_limit.width:
            height *= video.dimensions_limit.width / width
            width = video.dimensions_limit.width

        # Check the other one too in case both dimensions are over, and the previous adjustment wasn't enough
        if height > video.dimensions_limit.height:
            width *= video.dimensions_limit.height / height
            height = video.dimensions_limit.height

        new_dims = Dimensions( int(round(width, 0)), int(round(height, 0)) )
        clip_dims += [ new_dims ]

        if new_dims == old_dims:
            dprint(f'- Clip {clip.index}: {old_dims} (no reduction needed)')
        else:
            dprint(f'- Clip {clip.index}: {old_dims} -> {new_dims} (aspect ratio {new_dims.aspect_ratio})')

    return clip_dims

#-----------------------------------------------------------------------------------------------------------------------

def compute_output_clip_dimensions(video):
    for clip in video:
        if clip.filtered_dims is None:
            continue

        old_dims = clip.filtered_dims

        if clip.filtered_dims.aspect_ratio > video.output_dims.aspect_ratio:
            width = video.output_dims.width
            height = int(width * clip.filtered_dims.height / clip.filtered_dims.width)
        else:
            height = video.output_dims.height
            width = int(height * clip.filtered_dims.width / clip.filtered_dims.height)

        # Ensure the values are even for the encoder
        height -= height % 2
        width -= width % 2

        new_dims = Dimensions(width, height)

        clip.output_dims = new_dims

        if new_dims == old_dims:
            dprint(f'- Clip {clip.index}: {old_dims} (no reduction needed)')
        else:
            dprint(f'- Clip {clip.index}: {old_dims} -> {new_dims} (aspect ratio {new_dims.aspect_ratio})')

#-----------------------------------------------------------------------------------------------------------------------

def compute_final_dimensions(video):
    dprint('Simulating height and width restrictions')

    clip_dims = compute_reduced_clip_dimensions(video)

    max_width = max( [d.width for d in clip_dims] )
    max_height = max( [d.height for d in clip_dims] )

    dprint(f'Computed maximum output dimensions: {max_width}x{max_height}')

    # Make sure they're even for the encoder
    max_width = int( round(max_width/2, 0) * 2 )
    max_height = int( round(max_height/2, 0) * 2 )

    # Set the output dimensions to the maximum of the videos we saw. (Their dimensions are <= the maximum allowed
    # dimensions.)
    if isinstance(video.dimensions_limit.width, float):
        max_width = int( max_width * video.dimensions_limit.width )
        max_height = int( max_height * video.dimensions_limit.height )

        dprint(f'Adjusted maximum output dimensions: {max_width}x{max_height} (based on {video.dimensions_limit})')
    else:
        dprint(f'Maximum output dimensions: {max_width}x{max_height} (based on {video.dimensions_limit})')

    video.output_dims = Dimensions( max_width, max_height )

    # Now set the output dimensions
    compute_output_clip_dimensions(video)

#-----------------------------------------------------------------------------------------------------------------------

# Munge the final dimensions of the clips to all match, so that transitions will work. Honor the size limitations
# specified in the global arguments, while also invert height and width when it's obvious that the user meant the other
# way around.
def compute_output_dimensions(video):
    adjust_video_orientation(video)

    compute_final_dimensions(video)

#-----------------------------------------------------------------------------------------------------------------------

psutil.Process().nice(10)

video = parse_arguments()

prepare(video)

cprint(f'[violet]Video will be {video.output_dims}, {in_hms(video.output_duration)}')

encode_video(video)

cprint(f"[violet]Video encoded to {video.output_file}[/]")

if encoded_file_not_much_smaller(video):
    copy_video(video)
