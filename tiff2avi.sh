# $1 : start index
# $2 : image name pattern
# $3 : output video name
ffmpeg -start_number $1 -i $2 -vcodec copy $3


# Ref: https://video.stackexchange.com/questions/7903/how-to-losslessly-encode-a-jpg-image-sequence-to-a-video-in-ffmpeg
