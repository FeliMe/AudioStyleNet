import argparse
import os
import youtube_dl

from tqdm import tqdm


def parse_files(path):
    res = []
    with open(path) as file:
        line = file.readline()
        while line:
            url, timings = line.split(' ')
            timings = [t.split('\n')[0] for t in timings.split('|')]
            res.append({
                'url': url,
                'timings': timings
            })
            line = file.readline()

    return res


def download_video(url, output):
    """
    Download a YouTube video with url and save it to output
    """
    ydl_opts = {
        # 'format': 'bestvideo[ext=mp4]+bestaudio[ext=wav]/best[ext=mp4]',
        'format': 'best[ext=mp4]',
        'outtmpl': output
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_audio(url, output):
    """
    Download a YouTube video with url and save it to output
    """
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'outtmpl': f"{output}.%(ext)s",
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info("{}".format(url))
        filename = ydl.prepare_filename(result)
        ydl.download([url])

    return filename


def trim(file, start, stop, output):
    """
    Trim a video
    Args:
        video (str): Path to video
        start (str): Start of trimming in format hh:mm:ss
        stop (str): End of trimming in format hh:mm:ss
        ourput (str): Output path
    """
    assert file != output, "Video path can't be the same as output path"
    os.system(
        f'ffmpeg -hide_banner -loglevel panic -y -i {file} -ss {start} -to {stop} -strict -2 -async 1 {output}')


def resample_video(video, fps, output):
    """
    Resample the frame rate of a video
    Args:
        video (str): Path to video
        fps (int): desired frame rate
    """
    assert video != output, "Video path can't be the same as output path"
    os.system(
        f"ffmpeg -hide_banner -loglevel panic -y -i {video} -filter:v fps=fps={fps} -strict -2 {output}")


def process_video(info, args):
    tmp_dir = args.output_dir + 'full_videos/'
    trim_dir = args.output_dir + 'trimmed_videos/'
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(trim_dir, exist_ok=True)

    for video in tqdm(info):
        url = video['url']
        timings = video['timings']
        video_id = url.split('v=')[-1]
        full_path = tmp_dir + video_id + '.mp4'

        # Download Video
        print(f"Downloading {url}")
        download_video(url, full_path)
        1 / 0

        for i, timing in enumerate(timings):
            # Trim video
            trim_path = f"{trim_dir}{video_id}_{i + 1}.mp4"
            start, stop = timing.split('-')
            print(f"Trimming from {start} to {stop}")
            trim(full_path, start, stop, trim_path)

            # Resample to 25 fps
            print(f"Resampling to {args.fps}fps")
            final_path = f"{args.output_dir}{video_id}_{i + 1}.mp4"
            resample_video(trim_path, args.fps, final_path)


def process_audio(info, args):
    tmp_dir = args.output_dir + 'full_audios/'
    trim_dir = args.output_dir + 'trimmed_audios/'
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(trim_dir, exist_ok=True)

    for video in tqdm(info):
        url = video['url']
        timings = video['timings']
        video_id = url.split('v=')[-1]
        full_dir = tmp_dir + video_id

        # Download Video
        print(f"Downloading {url}")
        filename = download_audio(url, full_dir)

        for i, timing in enumerate(timings):
            # Trim video
            trim_path = f"{trim_dir}{video_id}_{i + 1}.m4a"
            start, stop = timing.split('-')
            print(f"Trimming from {start} to {stop}")
            trim(filename, start, stop, trim_path)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('output_dir', type=str, help='Where to download the videos to')
    # parser.add_argument('download_list', type=str, help='txt file with URLs and timings')
    parser.add_argument('--fps', type=int, default=25)
    args = parser.parse_args()

    args.output_dir = '/mnt/sdb1/meissen/Datasets/YouTubeDataset/'
    args.download_list = '/mnt/sdb1/meissen/Datasets/YouTubeDataset/youtube_download.txt'

    info = parse_files(args.download_list)

    process_video(info, args)
    process_audio(info, args)
