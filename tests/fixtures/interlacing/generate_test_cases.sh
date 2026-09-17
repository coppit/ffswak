#!/usr/bin/env bash
#shellcheck shell=bash
# set -x # Uncomment for debug

#######################################
### generate_testcase.sh
#######################################

export logdir="./logs"
mkdir -p "${logdir}"
printf '%s | %s: %s: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Starting idet test patterns script. Logs will be written to' "${logdir}/log.txt" | tee -a "${logdir}/log.txt"

rm -rf "${logdir}"/*.log "${logdir}"/*.json "${logdir}"/*.txt "${logdir}"/*.csv

export FFREPORT=file="${logdir}/%p-%t.log:level=48"

duration='00:00:10.000'  # Duration limited to 10s to reduce filesize to satisfy Github's filesize limits.
loglevel='level+warning' # FFmpeg loglevel
quality=5                # Documentation states this should be 2-31, but it seems that 0 and 1 are still valid.
maxrate=8000000          # Maximum bitrate for MPEG-2 DVD is 9.8Mbps, but we will use 8Mbps to be safe.
minrate=0                # Older players may require a minimum bitrate of non-zero.
trellis=0                # Trellis quantization.  Note that trellis (1,2) provides more efficient compression, but at the cost of quality.
mbd='rd'                 # Macroblock decision algorithm. simple (0) = mbcmp (default) | bits (1) = fewest bits | rd (2) = best rate distortion (slow)
mbcmp='satd'             # Macroblock comparison function.
precmp='satd'            # Pre-motion estimation comparison function. rd = best rate distortion (slow) | satd (2) sum of absolute Hadamard transformed differences seems popular.  default is dctmax (13)
cmp='satd'               # Motion estimation comparison function. rd = best rate distortion (Very slow).  default is dctmax (13)
subcmp='satd'            # Sub-picture element (sub-PEL) motion estimation (ME) compare function.rd = best rate distortion (Very slow).  default is dctmax (13)
skip_cmp='satd'          # Skip macroblock comparison function. rd = best rate distortion (slow).  default is dctmax (13)
ildctcmp='satd'          # Interlaced discrete cosine transform (DCT) compare function. Only satd (2), zero (7) seem to work.  Other values cause errors.

#######################################
### _check_dependencies
#######################################
function _check_dependencies
{
  printf '%s | %s: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Checking dependecies.' | tee -a "${logdir}/log.txt"
  local -a required_dependencies
  required_dependencies=(
    ffmpeg
    /opt/dgpulldown/dgpulldown # Using dgpulldown 1.0.11, which requires build and install.
    tee
  )
  for dependency in "${required_dependencies[@]}"; do
    command -v "${dependency}" 1> /dev/null || {
      printf '%s | %s: %s %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'ERROR' 'Required Dependency' "${dependency}" 'not found.  Exiting.' | tee -a "${logdir}/log.txt"
      exit 1
    }
  done

  local -a optional_dependencies
  optional_dependencies=(
    foo-ignore-this # Dummy dependency
    ffprobe
    mediainfo
    grep
    jq
    # gnuplot
    # MP4Box
    # mkvtoolnix
    # ffplay
    # mpv
    # vlc
    # avmediainfo # Apple's version of mediainfo
    # dvdauthor # fun-and-games for creating DVDs.
    # ab-av1 for iterative qscale/crf testing using VMAF.
  )
  for dependency in "${optional_dependencies[@]}"; do
    command -v "${dependency}" 1> /dev/null || {
      printf '%s | %s: %s %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'WARNING' 'Optional Dependency' "${dependency}" 'not found.' | tee -a "${logdir}/log.txt"
    }
  done
  return 0
}

#######################################
### _generate_bt601-525_480_interlaced_bff
#######################################
function _generate_bt601-525_480_interlaced_bff()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"

  local basename="$1"
  local gop=15

  ffmpeg -hide_banner -loglevel "${loglevel}" -sws_flags "+accurate_rnd+full_chroma_int" -bitexact \
    -f 'lavfi' -color_range:v 'tv' -colorspace:v 'smpte170m' -color_primaries:v 'smpte170m' -color_trc:v 'smpte170m' -i "color=color='Black':size='hd480':rate='(60000/1001)', format=pix_fmts='yuv422p10le', \
    il=luma_mode='deinterleave':chroma_mode='deinterleave',drawtext=text='TF':fontcolor='White':fontsize='main_h/16':box=0:boxcolor='Gray':font='Monospace':y=0:y_align='text',drawtext=text='BF':fontcolor='White':fontsize='main_h/16':box=0:boxcolor='Gray':font='Monospace':x=0:y='(main_h/2)+(text_h)':y_align='text',il=luma_mode='interleave':chroma_mode='interleave', \
    drawtext=text='%{expr_int_format\\:mod(n\,60)\\:d\\:2}':fontcolor='Orange':fontsize='main_h':font='Monospace':x='((main_w-text_w)/2)':y='((main_h-text_h)/2)':y_align='font', \
    drawtext=text=${basename}_interlaced_bff:fontcolor='Orange':fontsize='main_h/16':font='Monospace':x='((main_w-text_w)/2)':y='(main_h-text_h)':y_align='font', \
    scale=size='ntsc', setdar=ratio='16/9', \
    tinterlace='interleave_bottom', setfield=mode='bff', \
    format=pix_fmts='yuv420p', limiter=planes=1:min=16:max=235, limiter=planes=6:min=16:max=240,setparams=range='tv'[out]; \
  sine=frequency=440:sample_rate=48000, volume=0.2, aresample=in_chlayout='mono':out_chlayout='stereo'[out1]" \
    -map '0:v:0' -codec:v 'mpeg2video' \
    -g:v "${gop}" -bf:v 2 -b_strategy:v 2 -sc_threshold:v 0x7FFFFFFF \
    -qscale:v "${quality}" -non_linear_quant:v true -qmax:v 28 -maxrate:v "${maxrate}" -minrate:v "${minrate}" -bufsize:v 1835008 -dc:v 10 \
    -trellis:v "${trellis}" -precmp:v "${precmp}" -cmp:v "${cmp}" -subcmp:v "${subcmp}" -skip_cmp:v "${skip_cmp}" -mbcmp:v "${mbcmp}" -mbd:v "${mbd}" \
    -mpv_flags:v '+skip_rd+qp_rd+mv0' \
    -flags:v '+ilme+ildct+bitexact' -alternate_scan:v true -ildctcmp:v "${ildctcmp}" \
    -gop_timecode:v '00:00:00;00' -drop_frame_timecode:v true \
    -pix_fmt:v 'yuv420p' -chroma_sample_location:v 'left' \
    -seq_disp_ext:v 'always' \
    -video_format:v 'ntsc' \
    -map '0:a:0' -codec:a 'ac3' -ac:a 2 -ar:a 48000 -ab:a 192000 \
    -flags:a '+bitexact' \
    -timecode '00:00:00;00' \
    -metadata:s:a:0 'language=eng' \
    -t "${duration}" \
    -f 'mpegts' -fflags '+bitexact' -packetsize 2048 "${basename}_interlaced_bff.ts" -y
  return 0
}

#######################################
### _generate_bt601-525_480_interlaced_tff
#######################################
function _generate_bt601-525_480_interlaced_tff()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"
  local basename="$1"
  local gop=15

  ffmpeg -hide_banner -loglevel "${loglevel}" -sws_flags "+accurate_rnd+full_chroma_int" -bitexact \
    -f 'lavfi' -color_range:v 'tv' -colorspace:v 'smpte170m' -color_primaries:v 'smpte170m' -color_trc:v 'smpte170m' -i "color=color='Black':size='hd480':rate='(60000/1001)', format=pix_fmts='yuv422p10le', \
      il=luma_mode='deinterleave':chroma_mode='deinterleave', \
      drawtext=text='TF':fontcolor='White':fontsize='main_h/16':box=0:boxcolor='Gray':font='Monospace':y=0:y_align='text', \
      drawtext=text='BF':fontcolor='White':fontsize='main_h/16':box=0:boxcolor='Gray':font='Monospace':x=0:y='(main_h/2)+(text_h)':y_align='text', \
      il=luma_mode='interleave':chroma_mode='interleave', \
      drawtext=text='%{expr_int_format\\:mod(n\,60)\\:d\\:2}':fontcolor='Yellow':fontsize='main_h':font='Monospace':x='((main_w-text_w)/2)':y='((main_h-text_h)/2)':y_align='font', \
          drawtext=text=${basename}_interlaced_tff:fontcolor='Yellow':fontsize='main_h/16':font='Monospace':x='((main_w-text_w)/2)':y='(main_h-text_h)':y_align='font', \
      scale=size='ntsc', setdar=ratio='16/9', \
      tinterlace='interleave_top', setfield=mode='tff', \
      format=pix_fmts='yuv420p', limiter=planes=1:min=16:max=235, limiter=planes=6:min=16:max=240, setparams=range='tv'[out]; \
    sine=frequency=440:sample_rate=48000, volume=0.2, aresample=in_chlayout='mono':out_chlayout='stereo'[out1]" \
    -map '0:v:0' -codec:v 'mpeg2video' \
    -g:v "${gop}" -bf:v 2 -b_strategy:v 2 -sc_threshold:v 0x7FFFFFFF \
    -qscale:v "${quality}" -non_linear_quant:v true -qmax:v 28 -maxrate:v "${maxrate}" -minrate:v "${minrate}" -bufsize:v 1835008 -dc:v 10 \
    -trellis:v "${trellis}" -precmp:v "${precmp}" -cmp:v "${cmp}" -subcmp:v "${subcmp}" -skip_cmp:v "${skip_cmp}" -mbcmp:v "${mbcmp}" -mbd:v "${mbd}" \
    -mpv_flags:v '+skip_rd+qp_rd+mv0' \
    -flags:v '+ilme+ildct+bitexact' -alternate_scan:v true -ildctcmp:v "${ildctcmp}" \
    -gop_timecode:v '00:00:00;00' -drop_frame_timecode:v true \
    -pix_fmt:v 'yuv420p' -chroma_sample_location:v 'left' \
    -seq_disp_ext:v 'always' \
    -video_format:v 'ntsc' \
    -map '0:a:0' -codec:a 'ac3' -ac:a 2 -ar:a 48000 -ab:a 192000 \
    -flags:a '+bitexact' \
    -timecode '00:00:00;00' \
    -metadata:s:a:0 'language=eng' \
    -t "${duration}" \
    -f 'mpegts' -fflags '+bitexact' -packetsize 2048 "${basename}_interlaced_tff.ts" -y
  return 0
}

#######################################
### _generate_bt601-525_480_telecined_hard
# it is tricky to determine whether alternate scan should be false (for the progressive frames) or true (for the interleaved frames).  Given that there will be a higher proportion of progressive frames, we will choose false.
#######################################
function _generate_bt601-525_480_telecined_hard()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"
  local basename="$1"
  local gop=15
  local pulldownpattern=32 # This was selected to match the output of dgpulldown, although [ 23 | 2332 ] are common.

  ffmpeg -hide_banner -loglevel "${loglevel}" -sws_flags "+accurate_rnd+full_chroma_int" -bitexact \
    -f 'lavfi' -color_range:v 'tv' -colorspace:v 'smpte170m' -color_primaries:v 'smpte170m' -color_trc:v 'smpte170m' -i "color=color='Black':size='hd480':rate='ntsc-film', format=pix_fmts='yuv422p10le', \
      drawtext=text='A':fontcolor='Red':fontsize='(main_h/8)':fontfile='Monospace':x='(main_w-(4*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),0)', \
      drawtext=text='B':fontcolor='Blue':fontsize='(main_h/8)':fontfile='Monospace':x='(main_w-(3*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),1)', \
      drawtext=text='C':fontcolor='Green':fontsize='(main_h/8)':fontfile='Monospace':x='(main_w-(2*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),2)', \
      drawtext=text='D':fontcolor='Purple':fontsize='(main_h)/8':fontfile='Monospace':x='(main_w-(1*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),3)', \
      drawtext=text='%{expr_int_format\\:mod(n\,24)\\:d\\:2}':fontcolor='Blue':fontsize='main_h':font='Monospace':x='((main_w-text_w)/2)':y='((main_h-text_h)/2)':y_align='font', \
                drawtext=text=${basename}_telecined_hard.ts:fontcolor='Blue':fontsize='main_h/16':font='Monospace':x='((main_w-text_w)/2)':y='(main_h-text_h)':y_align='font', \
      scale=size='ntsc', setdar=ratio='16/9', \
      telecine=pattern=${pulldownpattern}, setfield=mode='tff', \
      format=pix_fmts='yuv420p', limiter=planes=1:min=16:max=235, limiter=planes=6:min=16:max=240,setparams=range='tv'[out]; \
    sine=frequency=440:sample_rate=48000, volume=0.2, aresample=in_chlayout='mono':out_chlayout='stereo'[out1]" \
    -map '0:v:0' -codec:v 'mpeg2video' \
    -g:v "${gop}" -bf:v 2 -b_strategy:v 2 -sc_threshold:v 0x7FFFFFFF \
    -qscale:v "${quality}" -non_linear_quant:v true -qmax:v 28 -maxrate:v "${maxrate}" -minrate:v "${minrate}" -bufsize:v 1835008 -dc:v 10 \
    -trellis:v "${trellis}" -precmp:v "${precmp}" -cmp:v "${cmp}" -subcmp:v "${subcmp}" -skip_cmp:v "${skip_cmp}" -mbcmp:v "${mbcmp}" -mbd:v "${mbd}" \
    -mpv_flags:v '+skip_rd+qp_rd+mv0' \
    -flags:v '+ilme+ildct+bitexact' -alternate_scan:v false \
    -gop_timecode:v '00:00:00;00' -drop_frame_timecode:v true \
    -pix_fmt:v 'yuv420p' -chroma_sample_location:v 'left' \
    -seq_disp_ext:v 'always' \
    -video_format:v 'ntsc' \
    -map '0:a:0' -codec:a 'ac3' -ac:a 2 -ar:a 48000 -ab:a 192000 -frame_size:a 1024 \
    -flags:a '+bitexact' \
    -timecode '00:00:00;00' \
    -metadata:s:a:0 'language=eng' \
    -t "${duration}" \
    -f 'mpegts' -fflags '+bitexact' -packetsize 2048 "${basename}_telecined_hard.ts" -y
  return 0
}

#######################################
### _generate_bt601-525_480_telecined_soft
# - Note that timecode format for 23.976fps progressive is 'HH:MM:SS:FF', ie non drop frame (NDF), rather than 'HH:MM:SS;FF' (DF)
#######################################
function _generate_bt601-525_480_telecined_soft()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"
  local basename="$1"
  local gop=12 # Note that for DVD, the maximum GOP pre-soft-telecine is 12, since gop will become 15 after repeatfields expansion.
  ffmpeg -hide_banner -loglevel "${loglevel}" -sws_flags "+accurate_rnd+full_chroma_int" -bitexact \
    -f 'lavfi' -color_range:v 'tv' -colorspace:v 'smpte170m' -color_primaries:v 'smpte170m' -color_trc:v 'smpte170m' -i "color=color='Black':size='hd480':rate='ntsc-film', format=pix_fmts='yuv422p10le',\
      drawtext=text='A':fontcolor='Red':fontsize='(main_h/8)':fontfile='Monospace':x='(main_w-(4*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),0)', \
      drawtext=text='B':fontcolor='Blue':fontsize='(main_h/8)':fontfile='Monospace':x='(main_w-(3*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),1)', \
      drawtext=text='C':fontcolor='Green':fontsize='(main_h/8)':fontfile='Monospace':x='(main_w-(2*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),2)', \
      drawtext=text='D':fontcolor='Purple':fontsize='(main_h)/8':fontfile='Monospace':x='(main_w-(1*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),3)', \
      drawtext=text='%{expr_int_format\\:mod(n\,24)\\:d\\:2}':fontcolor='Green':fontsize='main_h':font='Monospace':x='((main_w-text_w)/2)':y='((main_h-text_h)/2)':y_align='font', \
      drawtext=text=${basename}_telecined_soft.ts:fontcolor='Green':fontsize='main_h/16':font='Monospace':x='((main_w-text_w)/2)':y='(main_h-text_h)':y_align='font', \
      scale=size='ntsc', setdar=ratio='16/9', format=pix_fmts='yuv420p', limiter=planes=1:min=16:max=235, limiter=planes=6:min=16:max=240,setparams=range='tv'[out]; \
    sine=frequency=440:sample_rate=48000, volume=0.2, aresample=in_chlayout='mono':out_chlayout='stereo'[out1]" \
    -map '0:v:0' -codec:v 'mpeg2video' \
    -g:v "${gop}" -bf:v 2 -b_strategy:v 2 -sc_threshold:v 0x7FFFFFFF \
    -qscale:v "${quality}" -non_linear_quant:v true -qmax:v 28 -maxrate:v "${maxrate}" -minrate:v "${minrate}" -bufsize:v 1835008 -dc:v 10 \
    -trellis:v "${trellis}" -precmp:v "${precmp}" -cmp:v "${cmp}" -subcmp:v "${subcmp}" -skip_cmp:v "${skip_cmp}" -mbcmp:v "${mbcmp}" -mbd:v "${mbd}" \
    -mpv_flags:v '+skip_rd+qp_rd+mv0' \
    -alternate_scan:v false \
    -gop_timecode:v '00:00:00:00' -drop_frame_timecode:v false \
    -pix_fmt:v 'yuv420p' -chroma_sample_location:v 'left' \
    -seq_disp_ext:v 'always' \
    -video_format:v 'ntsc' \
    -map '0:a:0' -codec:a 'ac3' -ac:a 2 -ar:a 48000 -ab:a 192000 -frame_size 1024 \
    -flags:a '+bitexact' \
    -metadata:s:a:0 'language=eng' \
    -timecode '00:00:00:00' \
    -t "${duration}" \
    -f tee -fflags '+bitexact' "[select='v':f='mpeg2video':packetsize=2048]./${basename}_progressive.m2v \
    | [select='a':f='ac3']./${basename}_audio.ac3" -y

  # [TODO] Still needs work to avoid using alias in shell script (add to path or symlink to ~/bin/dgpulldown?)
  # [TODO] Add error checking or && to only run if previous command is successful
  [[ $(command -v /opt/dgpulldown/dgpulldown) ]] \
    && /opt/dgpulldown/dgpulldown "./${basename}_progressive.m2v" -o "./${basename}_pulldown.m2v" -srcfps 24000/1001 -destfps 29.970

  # [TODO] Add error checking or && to only run if previous command is successful
  ffmpeg -hide_banner -loglevel "${loglevel}" -sws_flags "+accurate_rnd+full_chroma_int" -bitexact \
    -f 'mpegvideo' -framerate 'ntsc-film' -fflags '+genpts' -i "./${basename}_pulldown.m2v" \
    -f 'ac3' -fflags '+genpts' -i "./${basename}_audio.ac3" \
    -map '0:v:0' -codec:v 'copy' \
    -map '1:a:0' -codec:a 'copy' \
    -metadata:s:a:0 'language=eng' \
    -f 'mpegts' -fflags '+bitexact' -packetsize 2048 "./${basename}_telecined_soft.ts" -y

  rm -f "./${basename}_progressive.m2v" "./${basename}_audio.ac3" "./${basename}_pulldown.m2v" # Housekeeping
  return 0
}

#######################################
### _generate_bt601-525_480_progressive
#######################################
function _generate_bt601-525_480_progressive()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"
  local basename="$1"
  local gop=15

  ffmpeg -hide_banner -loglevel "${loglevel}" -sws_flags "+accurate_rnd+full_chroma_int" -bitexact \
    -f 'lavfi' -color_range:v 'tv' -colorspace:v 'smpte170m' -color_primaries:v 'smpte170m' -color_trc:v 'smpte170m' -i "color=color='Black':size='hd480':rate='ntsc-film', format=pix_fmts='yuv422p10le', \
      drawtext=text='A':fontcolor='Red':fontsize='(main_h/8)':fontfile='Monospace':x='(main_w-(4*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),0)', \
      drawtext=text='B':fontcolor='Blue':fontsize='(main_h/8)':fontfile='Monospace':x='(main_w-(3*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),1)', \
      drawtext=text='C':fontcolor='Green':fontsize='(main_h/8)':fontfile='Monospace':x='(main_w-(2*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),2)', \
      drawtext=text='D':fontcolor='Purple':fontsize='(main_h)/8':fontfile='Monospace':x='(main_w-(1*text_w))':y=0:y_align='text':box=false:boxcolor='Gray':enable='eq((mod(n,4)),3)', \
      drawtext=text='%{expr_int_format\\:mod(n\,24)\\:d\\:2}':fontcolor='Blue':fontsize='main_h':font='Monospace':x='((main_w-text_w)/2)':y='((main_h-text_h)/2)':y_align='font', \
      drawtext=text=${basename}_progressive.ts:fontcolor='Blue':fontsize='main_h/16':font='Monospace':x='((main_w-text_w)/2)':y='(main_h-text_h)':y_align='font', \
      scale=size='ntsc', setdar=ratio='16/9', \
      format=pix_fmts='yuv420p', limiter=planes=1:min=16:max=235, limiter=planes=6:min=16:max=240,setparams=range='tv'[out]; \
    sine=frequency=440:sample_rate=48000, volume=0.2, aresample=in_chlayout='mono':out_chlayout='stereo'[out1]" \
    -map '0:v:0' -codec:v 'mpeg2video' \
    -g:v "${gop}" -bf:v 2 -b_strategy:v 2 -sc_threshold:v 0x7FFFFFFF \
    -qscale:v "${quality}" -non_linear_quant:v true -qmax:v 28 -maxrate:v "${maxrate}" -minrate:v "${minrate}" -bufsize:v 1835008 -dc:v 10 \
    -trellis:v "${trellis}" -precmp:v "${precmp}" -cmp:v "${cmp}" -subcmp:v "${subcmp}" -skip_cmp:v "${skip_cmp}" -mbcmp:v "${mbcmp}" -mbd:v "${mbd}" \
    -mpv_flags:v '+skip_rd+qp_rd+mv0' \
    -flags:v '+bitexact' \
    -pix_fmt:v 'yuv420p' -chroma_sample_location:v 'left' \
    -alternate_scan:v false \
    -seq_disp_ext:v 'always' \
    -video_format:v 'ntsc' \
    -map '0:a:0' -codec:a 'ac3' -ac:a 2 -ar:a 48000 -ab:a 192000 -frame_size:a 1024 \
    -flags:a '+bitexact' \
    -timecode '00:00:00:00' \
    -metadata:s:a:0 'language=eng' \
    -t "${duration}" \
    -f 'mpegts' -fflags '+bitexact' -packetsize 2048 "${basename}_progressive.ts" -y
  return 0
}

#######################################
### _generate_bt601-525_480_progressive_segmented_frame
#######################################
# function _generate_bt601-525_480_progressive_segmented_frame_tff()
# {
# # Poyton claims... "The progressive segmented-frame (PsF) technique is known in consumer SDTV systems as quasi-interlace.
#   # https://forum.videohelp.com/threads/352391-FFMPEG-Ability-to-identify-progressive-segmented-frame-material-in-h-264
#   # https://forum.videohelp.com/threads/400447-Detect-Progressive-Segmented-Frame-PsF-video
# }

#######################################
### _remux function.
# Ideally, this would be done with tee muxer.
# Ideally, would also add MP4, but H.262 in MP4 is a little funky and generates errors with Apple avmediainfo.
# https://developer.apple.com/library/archive/technotes/tn2429/_index.html
#######################################
function _remux()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"
  local infile="$1"
  # Oh matroska, how do I hate thee? Let me count the ways.
  ffmpeg -hide_banner -loglevel "${loglevel}" -sws_flags "+accurate_rnd+full_chroma_int" -bitexact \
    -i "${infile}" \
    -map '0:v:0' -codec:v 'copy' \
    -map '0:a:0' -codec:a 'copy' "./${infile%.*}.mkv" -y
  return 0
}

#######################################
### _analyse function.
#######################################
function _analyse()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"
  local infile="$1"
  if [[ $(command -v mediainfo) ]]; then
    {
      mediainfo --output='JSON' --LogFile="${logdir}/${infile%.*}.mediainfo.json" "${infile}" 1> /dev/null
      if [[ $(command -v jq) ]]; then
        {
          # [TODO] Add the filename
          # [TODO] Write to the logger
          jq --compact-output '
            [.media.track.[] 
            | select(.["@type"] == "Video") 
            | {FrameRate, FrameRate_Num, FrameRate_Den, ScanType, ScanOrder}]' "${logdir}/${infile%.*}.mediainfo.json"
        }
      else
        {
          printf '%s | %s: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'WARNING' 'Optional Dependency is not installed or not on path.  Falling back to mediainfo summary.' | tee -a "${logdir}/log.txt"
          mediainfo "${infile}" | grep -e "Frame\ rate" -e "Original\ frame\ rate" -e "Scan\ type" -e "Scan\ order" > "${logdir}/${infile%.*}.mediainfo.txt"
        }
      fi
    }
  else
    printf '%s | %s: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'WARNING' 'Optional Dependency mediainfo is not installed or not on path.  Skipping.' | tee -a "${logdir}/log.txt"
  fi
  if [[ $(command -v ffprobe) ]]; then
    {
      ffprobe -hide_banner -loglevel 'error' -f 'lavfi' "movie=filename=${infile}" -show_entries 'frame' -print_format 'json' -o "${logdir}/${infile%.*}.ffprobe.json"
      if [[ $(command -v jq) ]]; then
        {
          # [TODO] output needs to be tsv (not csv) for future gnuplot compatibility
          jq --raw-output '[.frames.[]| select(.["media_type"] == "video") | {pict_type,interlaced_frame,top_field_first,repeat_pict}] | (["pict_type", "interlaced_frame", "top_field_first", "repeat_pict"], (.[] | [.pict_type, .interlaced_frame, .top_field_first, .repeat_pict])) | @csv' "${logdir}/${infile%.*}.ffprobe.json" > "${logdir}/${infile%.*}.ffprobe.summary.csv"
        }
      else
        ffprobe -hide_banner -loglevel 'error' -f 'lavfi' "movie=filename=${infile}" -show_entries 'frame=key_frame,pict_type,interlaced_frame,top_field_first,repeat_pict' -print_format 'compact' -o "${logdir}/${infile%.*}.ffprobe.compact.txt"
      fi
    }
  else
    printf '%s | %s: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'WARNING' 'Optional Dependency ffprobe not installed or not on path.  Skipping.' | tee -a "${logdir}/log.txt"
  fi
  return 0
}

#######################################
### _analyse_idet function.
#######################################
function _analyse_idet()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"
  local infile="$1"
  [[ $(command -v ffprobe) ]] \
    && ffprobe -hide_banner -loglevel 'error' \
      -f 'lavfi' "movie=filename=${infile},extractplanes=planes='y',idet=intl_thres=1.04:prog_thres=1.5:rep_thres=3" \
      -show_entries 'frame' \
      -print_format 'json' -o "${logdir}/${infile%.*}.ffprobe.idet.json"
  ##################### Would be better to use jq on the json, if previous command is successful ################
  [[ $(command -v ffprobe) ]] \
    && ffprobe -hide_banner -loglevel 'error' \
      -f 'lavfi' "movie=filename=${infile},extractplanes=planes='y',idet=intl_thres=1.04:prog_thres=1.5:rep_thres=3" \
      -show_entries 'frame=key_frame,pict_type,interlaced_frame,top_field_first,repeat_pict : frame_tags=lavfi.idet.single.current_frame' \
      -print_format 'compact' -o "${logdir}/${infile%.*}.ffprobe.idet.compact.txt"
  return 0
}

#######################################
### _analyse_repeatfields_idet function.
#######################################
function _analyse_repeatfields_idet()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"
  local infile="$1"
  [[ $(command -v ffprobe) ]] \
    && ffprobe -hide_banner -loglevel 'error' \
      -f 'lavfi' "movie=filename=${infile}, repeatfields, extractplanes=planes='y', idet=intl_thres=1.04:prog_thres=1.5:rep_thres=3" \
      -show_entries 'frame' \
      -print_format 'json' -o "${logdir}/${infile%.*}.ffprobe.repeatfields.idet.json"
  ##################### Would be better to use jq on the json################
  [[ $(command -v ffprobe) ]] \
    && ffprobe -hide_banner -loglevel 'error' \
      -f 'lavfi' "movie=filename=${infile}, repeatfields, extractplanes=planes='y', idet=intl_thres=1.04:prog_thres=1.5:rep_thres=3" \
      -show_entries 'frame=key_frame,pict_type,interlaced_frame,top_field_first,repeat_pict : frame_tags=lavfi.idet.single.current_frame' \
      -print_format 'compact' -o "${logdir}/${infile%.*}.ffprobe.repeatfields.idet.compact.txt"
  return 0
}

#######################################
### Main function.
#######################################
function main()
{
  printf '%s | %s: %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" 'INFO' 'Running function' "${FUNCNAME[0]}" | tee -a "${logdir}/log.txt"
  _check_dependencies

  _generate_bt601-525_480_interlaced_bff "bt601-525_480" \
    && {
      _remux "bt601-525_480_interlaced_bff.ts"
      _analyse "bt601-525_480_interlaced_bff.ts"
      _analyse_idet "bt601-525_480_interlaced_bff.ts"
    }
  _generate_bt601-525_480_interlaced_tff "bt601-525_480" \
    && {
      _remux "bt601-525_480_interlaced_tff.ts"
      _analyse "bt601-525_480_interlaced_tff.ts"
      _analyse_idet "bt601-525_480_interlaced_tff.ts"
    }
  _generate_bt601-525_480_progressive "bt601-525_480" \
    && {
      _remux "bt601-525_480_progressive.ts"
      _analyse "bt601-525_480_progressive.ts"
      _analyse_idet "bt601-525_480_progressive.ts"
    }
  _generate_bt601-525_480_telecined_hard "bt601-525_480" \
    && {
      _remux "bt601-525_480_telecined_hard.ts"
      _analyse "bt601-525_480_telecined_hard.ts"
      _analyse_idet "bt601-525_480_telecined_hard.ts"
    }
  _generate_bt601-525_480_telecined_soft "bt601-525_480" \
    && {
      _remux "bt601-525_480_telecined_soft.ts"
      _analyse "bt601-525_480_telecined_soft.ts"
      _analyse_idet "bt601-525_480_telecined_soft.ts"
      _analyse_repeatfields_idet "bt601-525_480_telecined_soft.ts"
    }
}

echo "running main code"
main "${@}"
exit 0
