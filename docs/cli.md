# Command Line Interface

The basic usage is df3d-cli /path/to/your/images/ --order,

- df3d-cli expects videos to be in camera_{cam_id}.mp4 format and images to be in camera_{cam_id}_img_{img_id}.jpg format. 
- In case of videos, df3d-cli will expand videos into images using ffmpeg.

-n flag will limit the number of images you are using. df3d-cli /your/path/to/images/ -n 100 will only process the first 100 images. First n images will be used for 2d prediction, calibration and triangulation.

-vv flag will print more information during processing.

- will save 