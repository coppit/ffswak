# ffswak.py

A Python wrapper for `ffmpeg` that simplies common video editing tasks.

Things you can do:

* Scale the video so that it isn't too large
* Extract a short clip from a long video
* Rotate the video, automatically zooming so that there are no black spaces on the sides
* Stabilize a shaky video
* Crop to zoom in on part of the video
* Re-encode the video using the HEVC encoder (h265)

# Getting Started

Make sure you have ffmpeg and ffprobe installed:

```sh
ffmpeg --version
ffprobe --version
```

Make sure you're running python 3.7 or newer:

```sh
python3 --version
```

Install some third-party modules:

```sh
pip install ffmpeg-python humanize psutil rich
```

Install the script:

```sh
mkdir -p ~/.local/bin
```

Save the script to `~/.local/bin/ffswak.py`.

```sh
chmod +x ~/.local/bin/ffswak.py
```

You may need to add `~/.local/bin` to your PATH environment variable (e.g. in `~/.bashrc`), then close and reopen your
shell.

Test it out:

```sh
ffswak.py --help
```

# Usage Examples

## Example 1: Scale and re-encode one video

```sh
ffswak.py input.mov
```

Scale down the video to fit in 1920x1080 (or 1080x1920 if it's a portrait video). Re-encode with the HEVC codec,
using the same video bitrate. Re-encode the audio using the AAC codec, if the bitrate is higher than 192k. Save the
output to `~/Pictures/Import`, generating a new filename if the file already exists. Keep the pixel format, bit depth,
frame rate, etc. the same.

If the resulting file is less than 10% smaller, issue a warning. If it's less than 2% smaller (or even larger!) copy the
input file to the output location and issue a warning.

## Example 2: Join clips from two videos, with a fade transition

```sh
ffswak.py input1.mov -10 input2.mp4 0:15-1:10
```

Same as above, but join portions of two videos with a .5s fade transition. When possible, the time ranges will be
increased by .5s so that the part you care about isn't lost in the transition. If a time is omitted, such as `-10` or
`10-`, it implies the start or end of the input. You can also specify multiple time ranges for a single input, such as
`input.mov 5-10 15-20`.

If one video is 1280x720 and the other is 720x1280, the resulting output will be 1280x1080, with black bars. (1280 wide
because the 1280x720 is less than 1920x1080, but only 1080 high because 720x1280 is taller than the default max size of
1920x1080.) The output frame rate, pixel format, video/audio bitrates, etc. will be the maximum of the values from the
input files.

## Example 3: Stabilize and join multiple videos

```sh
ffswak.py -s input1.mov input2.mov
```

Stabilize all the videos and join them. The output file will be named input1-input2.mp4.

## Example 4: Global versus per-input options

```sh
ffswak.py -O ~/Desktop -v 2 -- input1.mov -s input2.mov
```

Increase the volume of all the videos by 100%, but stabilize only the second video. Write the output file to the
~/Desktop directory.

Use the `--` syntax whenever you need to specify per-input options. Global options go before the `--`. Per-input options
go before the file name, and any time ranges go after the file name.

## Example 5: Cropping and slowing one clip

```sh
ffswak.py -- input.mov 0-10 -cs .5 -cl bc -s .5 input.mov 10-15 input.mov 15-20
```

For a 5-second clip in the middle, slow it down 50% and crop it to be 50% of the original size, centered on the bottom
center of the original image.

# All Options

This help message shows all of the options

```
usage: ffswak.py [global options] -- [per-file options] input_file [time_ranges ...] [ [per-file options] input_file [time_ranges... ] ... ]

A Python wrapper for ffmpeg that simplies common video editing tasks.

TIME FORMAT is [[HH:]MM:]SS[.frac] or NNN[.frac] or .frac. Time ranges do not include
transition times. A warning will be issued if the end of an input file requires the transition to
include part of the specified time range.

Global options apply to all files. Video options override global options for a specific
file. "--" can be omitted if there are no per-video options.

Global-Only Options:
  General options, and options for the output video.

  -F, --frame-rate-limit FRAME_RATE_LIMIT
                        Maximum frame rate.
  -D, --dimensions-limit DIMENSIONS_LIMIT
                        Maximum dimensions. .5 means 50% as wide and tall; .5,1 means half as wide, full height; 16:9 means the largest possible video with that aspect ratio;
                        1280x720 means exactly that size
  -o, --output-file OUTPUT_FILE
                        Output file. (Default is input.mp4, or input-abc.mp4 if needed, for re-encoding. With -c extension is kept the same)
  -O, --output-dir OUTPUT_DIR
                        Output directory.
  -d, --debug           Enable debugging messages
  --help                Show this help message and exit.

Video Options:
  Options for videos. Can be specified at the global or per-video level.

  -cl, --crop-location CROP_LOCATION
                        Cropped portion should be in the top/middle/bottom and left/center/right. ".2,.3" means 20% over from the left, and 30% down from the top. 100% means
                        the right side of the crop window will be aligned with the right side of the original video.
  -cs, --crop-size CROP_SIZE
                        Cropped portion size. .5 means 50% as wide and tall; .5,1 means half as wide, full height; 1280x720 means exactly that size
  -p, --speedup SPEEDUP
                        Change the speed. 2 means twice as fast. Disables audio.
  -r, --rotate ROTATE   Rotate the video, cropping as needed. Positive values are clockwise.
  -v, --volume VOLUME   Modify volume level. 2 means twice as loud. 0 means omit the audio track.
  -s, --stabilize       Stabilize the video
  -t, --tripod TRIPOD   Enable tripod mode, stabilizing on the time specified in TIME FORMAT.
  -R, --reverse         Reverse the video.
  -T, --transition-duration TRANSITION_DURATION
                        Transition duration when concatenating ranges, in TIME FORMAT.
  -I, --interlace-test  Enable testing the video for interlacing.

Generally speaking, bitrate, frame rate, etc. will be chosen to avoid degrading the quality.
Width and height will be automatically adjusted (with a warning) when it's obvious that they are
wrong. e.g. Width and height will be swapped when all the inputs are portrait instead of landscape.
```

# Known Issues and Limitations

iPhones produce extra metadata streams. Those get lost if the file is re-encoded, as does lots of metadata tags. Run
ffprobe on the input and output to compare.

I add features as I need them. Feel free to suggest enhancements at the [https://github.com/coppit/ffswak](github
project page).

# Author

David Coppit `<david@coppit.org>`

# License

See the file LICENSE for licensing information.
