import cv2, yt_dlp, os, time, librosa, tempfile, uuid, glob, time, torch, re
from panns_inference import AudioTagging, SoundEventDetection, labels
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from tqdm import tqdm
import torch.nn.functional as F
import re

# * ------------------ 비디오 프레임 추출 ------------------

"""
def openVideoStream(youtube_url: str) -> cv2.VideoCapture:
    
    # OpenCV를 사용하여 YouTube 비디오의 스트림을 여는 함수
    # @param video_id: YouTube 영상의 ID
    # @return: 열기 실패 시 False, 성공시 cv2.VideoCapture 객체
    
    print(f"'{youtube_url}'에서 비디오 스트림 URL을 가져오는 중...")

    video_stream_url = None
    ydl_options = {
        "format": "bestvideo/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)

        if "formats" in info_dict:
            for f in info_dict["formats"]:
                if f.get("vcodec") != "none" and f.get("url"):
                    if (
                        "bestvideo" in f.get("format_note", "").lower()
                        or "video only" in f.get("format_note", "").lower()
                    ):
                        video_stream_url = f["url"]
                        break

        if not video_stream_url:
            for f in info_dict["formats"]:
                if (
                    f.get("url")
                    and f.get("acodec") != "none"
                    and f.get("vcodec") != "none"
                ):
                    video_stream_url = f["url"]
                    break

        if not video_stream_url:
            print(
                f"오류: '{youtube_url}'에서 유효한 비디오 스트림 URL을 찾을 수 없습니다."
            )
            print(f"yt-dlp 정보: {info_dict}")
            return False

    cap = cv2.VideoCapture(video_stream_url)
    if not cap.isOpened():
        print(
            f"오류: 비디오 스트림을 열 수 없습니다. URL이 유효한지, 코덱 문제가 아닌지 확인하세요: {video_stream_url}"
        )
        return False

    print("스트림이 열렸습니다.")
    return cap
"""


"""
def openVideoStream(youtube_url: str) -> cv2.VideoCapture:
    
    # OpenCV를 사용하여 YouTube 비디오의 스트림(30fps)을 여는 함수, 30fps 실패 시 최대 fps로\n
    # 가져올 수 있는 최고 화질을 가져옴\n
    # 간혹 쿠키 문제가 발생할 수 있음\n
    # getcookies.txt 구글 플러그인 설치 후, 유튜브로 들어가 쿠키 파일 export 후 cookiefile 경로에 저장하면 해결 가능\n
    # 구글 계정이 여러가지 로그인이 되어있다면 바꿔가면서 다 넣어보기\n
    # 보안상 쿠키 파일은 push하지 않았음

    # @param youtube_url: YouTube 비디오의 URL
    # @return: 열기 실패 시 False, 성공시 cv2.VideoCapture 객체
    
    print(f"'{youtube_url}'에서 30fps 비디오 스트림 URL을 가져오는 중...")

    ydl_options = {
        "format": "bestvideo+bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "cookiefile": "./cooks.txt",
    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        formats = info_dict.get("formats", [])

        # 1차 시도: 30fps + video-only + 최대 해상도
        best_format = None
        max_height = -1
        for f in formats:
            if (
                f.get("vcodec") != "none"
                and f.get("acodec") == "none"
                and f.get("url")
                and f.get("fps") == 30
                and f.get("height", 0) > max_height
            ):
                best_format = f
                max_height = f["height"]

        # 2차 시도: fps 무시하고 video-only 중 최대 해상도
        if not best_format:
            print("⚠️ 30fps 스트림 없음 → 최대 화질로 fallback")
            max_height = -1
            for f in formats:
                if (
                    f.get("vcodec") != "none"
                    and f.get("acodec") == "none"
                    and f.get("url")
                    and f.get("height", 0) > max_height
                ):
                    best_format = f
                    max_height = f["height"]

        if not best_format:
            print("❌ 사용 가능한 video-only 스트림을 찾을 수 없습니다.")
            return False

        video_stream_url = best_format["url"]
        print(
            f"🎥 선택된 해상도: {best_format['height']}p @ {best_format.get('fps', 'N/A')}fps\nURL: {video_stream_url}"
        )

        cap = cv2.VideoCapture(video_stream_url)
        if not cap.isOpened():
            print("❌ OpenCV로 스트림을 열 수 없습니다.")
            return False

        return cap
"""


def openVideoStream(youtube_url: str) -> cv2.VideoCapture:
    """
    OpenCV를 사용하여 YouTube progressive 스트림 중 720p + 30fps를 우선 선택하고,
    720p가 없을 경우 progressive 중 최대 해상도 스트림을 fallback으로 선택합니다.
    """
    print(f"'{youtube_url}'에서 progressive 720p 스트림 URL을 가져오는 중...")

    ydl_options = {
        "format": "bestvideo+bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "cookiefile": "./cooks.txt",  # 필요 없다면 생략 가능
    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        formats = info_dict.get("formats", [])

        # ✅ 1차 시도: progressive + 30fps + height == 720
        selected_format = None
        for f in formats:
            if (
                f.get("vcodec") != "none"
                and f.get("acodec") != "none"
                and f.get("url")
                and f.get("height") == 720
                and f.get("fps") == 30
            ):
                selected_format = f
                break

        # 🔁 2차 시도: progressive 중 최대 해상도
        if not selected_format:
            print(
                "⚠️ 720p @30fps progressive 스트림 없음 → fallback으로 최대 해상도 progressive 선택"
            )
            max_height = -1
            for f in formats:
                if (
                    f.get("vcodec") != "none"
                    and f.get("acodec") != "none"
                    and f.get("url")
                    and f.get("height", 0) > max_height
                ):
                    selected_format = f
                    max_height = f["height"]

        if not selected_format:
            print("❌ progressive 스트림을 찾을 수 없습니다.")
            return False

        video_stream_url = selected_format["url"]
        print(
            f"🎬 선택된 스트림: {selected_format['ext']} | "
            f"{selected_format['height']}p @ {selected_format.get('fps', 'N/A')}fps"
        )
        print(f"URL: {video_stream_url}")
        cap = cv2.VideoCapture(video_stream_url)
        if not cap.isOpened():
            print("❌ OpenCV로 스트림을 열 수 없습니다.")
            return False

        return cap


"""
def extractFrames(cap, start_time_sec, duration_sec, fps, output_folder_path):
    
    # 연속적으로 프레임을 읽으며 일정 간격마다 저장
    # duration_sec이 None이면 start_time_sec부터 영상 끝까지 추출

    # @param cap: cv2.VideoCapture 객체
    # @param start_time_sec: 추출을 시작할 영상 지점(초)
    # @param duration_sec: 추출할 길이(초)
    # @param fps: 초당 추출할 프레임 수
    # @param output_folder_path: 이미지 저장 폴더
    

    if not cap.isOpened():
        print("❌ VideoCapture 열기 실패")
        return False

    os.makedirs(output_folder_path, exist_ok=True)

    # 영상 기본 정보
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration_sec = total_frames / video_fps

    # 시작/종료 프레임 계산
    start_frame = int(start_time_sec * video_fps)
    if duration_sec is None:
        end_frame = total_frames
        print(f"📌 duration_sec=None → 끝까지 추출 (총 {total_duration_sec:.2f}s)")
    else:
        end_frame = int((start_time_sec + duration_sec) * video_fps)

    frame_interval = max(1, int(round(video_fps / fps)))

    print(f"\n🎬 시작 프레임: {start_frame}, 종료 프레임: {end_frame}")
    print(f"🎯 프레임 간격: {frame_interval} (video fps: {video_fps:.2f})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extraction_start_time = time.time()
    current_frame = start_frame
    saved_count = 0

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패 또는 영상 끝 도달")
            break

        if (current_frame - start_frame) % frame_interval == 0:
            filename = os.path.join(
                output_folder_path,
                (
                    f"{start_time_sec}s_{fps}fps_{saved_count:03d}.jpg"
                    if duration_sec is None
                    else f"{start_time_sec}s_{duration_sec}s_{fps}fps_{saved_count:03d}.jpg"
                ),
            )
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
            saved_count += 1

        current_frame += 1

    extraction_end_time = time.time()
    print(
        f"✅ 총 {saved_count}개 프레임 저장 완료. 소요 시간: {extraction_end_time - extraction_start_time:.2f}초"
    )
    return saved_count
"""


def extractFrames(
    cap,
    start_time_sec,
    duration_sec,
    fps,
    output_folder_path,
    resize_to=(320, 180),  # 원하는 해상도 (width, height)
):
    """
    duration_sec이 None이면 start_time_sec부터 끝까지 추출\n
    fps는 cap.get(cv2.CAP_PROP_FPS) 권장 (원본 영상의 fps)\n
    grab + retrieve + resize 최적화된 프레임 추출
    """

    if not cap.isOpened():
        print("❌ VideoCapture 열기 실패")
        return False

    os.makedirs(output_folder_path, exist_ok=True)

    # 영상 정보
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration_sec = total_frames / video_fps

    # 시작/종료 프레임 계산
    start_frame = int(start_time_sec * video_fps)
    if duration_sec is None:
        end_frame = total_frames
        print(f"📌 duration_sec=None → 끝까지 추출 (총 {total_duration_sec:.2f}s)")
    else:
        end_frame = int((start_time_sec + duration_sec) * video_fps)

    frame_interval = max(1, int(round(video_fps / fps)))

    print(f"\n🎬 시작 프레임: {start_frame}, 종료 프레임: {end_frame}")
    print(f"🎯 프레임 간격: {frame_interval} (video fps: {video_fps:.2f})")
    print(f"🖼️ 해상도 축소: {resize_to[0]}x{resize_to[1]}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extraction_start_time = time.time()
    current_frame = start_frame
    saved_count = 0

    while current_frame < end_frame:
        grabbed = cap.grab()
        if not grabbed:
            print("❌ grab 실패 또는 영상 끝 도달")
            break

        if (current_frame - start_frame) % frame_interval == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                print("❌ retrieve 실패")
                break

            # 해상도 줄이기
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)

            filename = os.path.join(
                output_folder_path,
                (
                    f"{start_time_sec}s_{fps}fps_{saved_count:03d}.jpg"
                    if duration_sec is None
                    else f"{start_time_sec}s_{duration_sec}s_{fps}fps_{saved_count:03d}.jpg"
                ),
            )
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

        current_frame += 1

    extraction_end_time = time.time()
    print(
        f"✅ 총 {saved_count}개 프레임 저장 완료. 소요 시간: {extraction_end_time - extraction_start_time:.2f}초"
    )
    return saved_count


def extractFrames720p(
    cap: cv2.VideoCapture,
    start_time_sec: int,
    duration_sec: int,
    fps: int,
    output_folder_path: str,
):
    """
    열려 있는 VideoCapture 스트림에서 특정 구간의 프레임을 추출하고 저장\n
    720p로 리사이즈하여 저장함

    @param cap: cv2.VideoCapture 객체
    @param start_time_sec: 추출을 시작할 영상 지점(초)
    @param duration_sec: 추출할 길이(초)
    @param fps: 초당 추출할 프레임 수
    @param output_folder_path: 이미지 저장 폴더
    """
    if not cap.isOpened():
        print("오류: 유효하지 않은 VideoCapture 객체")
        return False

    os.makedirs(output_folder_path, exist_ok=True)
    print(
        f"\n{start_time_sec}초부터 {duration_sec}초 동안 초당 {fps} 프레임 추출 시작..."
    )

    frame_count = 0
    last_saved_msec = start_time_sec * 1000
    target_frame_interval_msec = 1000 / fps

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
    actual_start_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if abs(actual_start_msec - (start_time_sec * 1000)) > 1000:
        print(f"경고: 탐색 정확도 낮음 (시작 위치 {actual_start_msec/1000:.2f}s)")

    extraction_start_time = time.time()

    while True:
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

        if current_msec / 1000.0 >= (start_time_sec + duration_sec):
            print(f"🛑 {start_time_sec}-{start_time_sec + duration_sec}초 추출 완료.")
            break

        ret, frame = cap.read()
        if not ret:
            print("❌ 더 이상 프레임을 읽을 수 없음")
            break

        if (
            frame_count == 0
            or (current_msec - last_saved_msec) >= target_frame_interval_msec
        ):
            # ✅ 720p로 리사이즈
            resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

            filename = os.path.join(
                output_folder_path,
                f"{start_time_sec}s_{duration_sec}s_{fps}fps_{frame_count:03d}.jpg",
            )
            cv2.imwrite(filename, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_count += 1
            last_saved_msec = current_msec

    extraction_end_time = time.time()
    print(
        f"✅ 총 {frame_count}개 프레임 저장 완료. 소요 시간: {extraction_end_time - extraction_start_time:.2f}초"
    )
    return frame_count


# * ------------------- Audio recognition -------------------


def getAudioCropped(autio_path, start_sec=None, end_sec=None):
    """
    현재 오디오 파일에서 오디오를 불러와 (1, size)로 반환

    @param autio_path: 오디오 파일 경로
    @param start_sec: crop할 시작점, None이면 전체
    @param end_sec: crop할 끝점, None이면 전체
    @return: (1, size) shaped numpy 오디오, 샘플레이트
    """
    audio, sr = librosa.load(autio_path, sr=32000, mono=True)

    if start_sec == None and end_sec == None:
        return audio[None, :], sr

    start_sample, end_sample = int(start_sec * sr), int(end_sec * sr)  # type: ignore
    cropped = audio[start_sample:end_sample]
    return cropped[None, :], sr


def getAudioCroppedFromURL(
    youtube_url, start_sec=None, end_sec=None, isMono=True, sr=32000
):
    """
    YouTube URL에서 오디오를 불러와 (1, size)로 반환
    tmp 디렉토리에 파일이 저장되어 함수 실행 후 자동으로 오디오 파일은 삭제됨
    로컬에 ffmpeg 설치 및 환경변수 세팅이 필요

    @param youtube_url: youtube URL
    @param start_sec: crop할 시작점, None이면 전체
    @param end_sec: crop할 끝점, None이면 전체
    @param isMono: mono로 가져올지, stero로 가져올지, default=True
    @param sr: default=32,000 (PANNs 입력을 위해)
    @return: (1, size) shaped numpy 오디오, 샘플레이트
    """
    tmp_dir = tempfile.gettempdir()
    tmp_id = uuid.uuid4().hex
    tmp_basename = os.path.join(tmp_dir, tmp_id)

    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{tmp_basename}.%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "cookiefile": "./cooks.txt",  #
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "128",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            wav_path = f"{tmp_basename}.wav"

        # librosa 로드, PANNs 모델은 32kHz 기준
        audio, sr = librosa.load(wav_path, sr=sr, mono=isMono)

        if start_sec is None and end_sec is None:
            return audio[None, :], sr

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        cropped = audio[start_sample:end_sample]
        return cropped[None, :], sr

    finally:
        # 다운로드한 임시 파일 삭제
        for ext in ["wav", "webm", "m4a"]:
            f = f"{tmp_basename}.{ext}"
            if os.path.exists(f):
                os.remove(f)


def audioTagging(device: str, audio, n=10):
    """
    전체 영상에서 등장하는 오디오 상위 n개 레이블과 확률 출력

    @param device: 'cpu' 또는 'cuda'
    @param audio: 오디오 데이터 (1, N) numpy array
    @param n: 출력할 상위 클래스 수 (최대 527)
    @return: clipwise_output (1, 527) 확률 벡터
    """
    at = AudioTagging(checkpoint_path=None, device=device)
    clipwise_output, _ = at.inference(audio)
    top_indices = np.argsort(clipwise_output[0])[::-1][:n]

    for i in top_indices:
        print(f"{labels[i]:30s} : {clipwise_output[0][i]:.3f}")

    return clipwise_output


def eventDetectionWithOverallTopk(device: str, audio, clipwise_output, n=5):
    """
    전체 영상에서 등장하는 오디오 상위 n개 클래스의 프레임별 확률 시각화

    @param device: 'cpu' 또는 'cuda'
    @param audio: 오디오 데이터 (1, N) numpy array
    @param clipwise_output: audioTagging 결과로 나온 확률 벡터
    @param n: 시각화할 클래스 수
    @return: 없음 (그래프 출력)
    """
    sed = SoundEventDetection(checkpoint_path=None, device=device)
    framewise_output = sed.inference(audio)[0]

    top_indices = np.argsort(clipwise_output[0])[::-1][:n]

    plt.figure(figsize=(12, 6))

    for i in top_indices:
        plt.plot(framewise_output[:, i], label=labels[i])

    plt.title(f"Top-{n} predicted sound classes over frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def eventDetectionWithPerTopk(device: str, audio, min_score=0.2):
    """
    각 프레임에 대해, min_score 이상 오디오 클래스 시각화

    @param device: 'cpu' or 'cuda'
    @param audio: (1, N) shaped numpy audio
    @param min_score: 시각화 대상 클래스의 최소 최대 확률 기준
    """
    sr = 32000
    sed = SoundEventDetection(checkpoint_path=None, device=device)
    framewise_output = sed.inference(audio)[0]  # shape: (T, 527)

    # ⏱️ 시간축 (초 단위)
    num_frames = framewise_output.shape[0]
    duration_sec = audio.shape[1] / sr
    x = np.linspace(0, duration_sec, num_frames)

    # 🎯 클래스별 최대 확률 계산
    max_scores = np.max(framewise_output, axis=0)
    filtered_indices = np.where(max_scores >= min_score)[0]

    # 📋 출력 + 시각화
    if len(filtered_indices) == 0:
        print(f"\n❗ 최대 확률이 {min_score} 이상인 클래스가 없습니다.")
        return

    print(f"\n최대 확률이 {min_score} 이상인 클래스 수: {len(filtered_indices)}개")
    for idx in filtered_indices[np.argsort(max_scores[filtered_indices])[::-1]]:
        label = labels[idx]
        max_val = max_scores[idx]
        print(f"- {label}: 최대 {max_val:.3f}")

    # 📈 시각화
    plt.figure(figsize=(15, 6))
    for idx in filtered_indices:
        plt.plot(x, framewise_output[:, idx], label=labels[idx])
    plt.title(f"Classes with max score ≥ {min_score}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, int(duration_sec) + 1, 1))
    plt.tight_layout()
    plt.show()


# 오디오에 대해 스피치 - 음악 분리
def split_audio_to_speech_music(audio, model):
    """
    오디오 파일을 스피치와 음악으로 분리

    @param audio: 오디오 numpy 배열, getAudioCroppedFromURL 함수에서 isMono = False로 지정하고 가져와야함
    @param model: 모델, get_model(name='htdemucs').to(device)
    @return: sources, sources[0]: drums, sources[1]: bass, sources[2]: other, sources[3]: vocal\n
            Stero이므로 (2, N) 형태임
    """
    import torch
    from demucs.apply import apply_model

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    audio_tensor = torch.tensor(audio, dtype=torch.float32).to(device)

    model.eval()

    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device.type, split=True)

    sources = sources.squeeze(0).cpu()

    return sources


# * ------------------- Audio segmentation & 조음속도 계산 -------------------


# 1. 오디오 자르기
def split_audio(audio_path, start_time, end_time, chunk_size=1, sr=16000):
    """
    오디오 파일을 지정된 시간 구간으로 청크 분할 후 저장

    @param audio_path: 오디오 파일 경로
    @param start_time: 분할 시작 시간 (초 단위)
    @param end_time: 분할 끝 시간 (초 단위)
    @param chunk_size: 각 청크의 크기 (초 단위)
    @param sr: 샘플링 레이트 (기본값: 16000)
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    chunks = []
    print("청크 분할 진행")
    for t in range(start_time, end_time, chunk_size):
        start_sample = int(t * sr)
        end_sample = int(min((t + chunk_size), end_time) * sr)
        chunk = audio[start_sample:end_sample]

        chunk_path = f"./chunk/chunk_{t}_{t+chunk_size}.wav"
        sf.write(chunk_path, chunk, sr)
        chunks.append((t, t + chunk_size, chunk_path))
    return chunks


# 2. 침묵 제외 발화 시간 계산
def get_speech_duration(audio_path, top_db):
    y, sr = librosa.load(audio_path, sr=None)
    intervals = librosa.effects.split(y, top_db=top_db)
    speech_duration = sum((end - start) for start, end in intervals) / sr
    return speech_duration


# 3. fast-whisper로 자막 추출 + 조음속도 계산
def estimate_articulation_rate_fast_whisper(chunks, model, top_db=30):
    results = []
    print("조음 속도 계산 진행중...")

    for start, end, chunk_path in tqdm(chunks, desc="Processing chunks"):
        try:
            segments, _ = model.transcribe(chunk_path, language="ko", beam_size=5)
            text = "".join([seg.text.replace(" ", "") for seg in segments])
            text = re.sub(r"[.,!?\"'“”‘’…\-–—():;]", "", text)
            num_chars = len(text)

            speech_dur = get_speech_duration(chunk_path, top_db)
            articulation_rate = num_chars / speech_dur if speech_dur > 0 else 0

            results.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "chars": num_chars,
                    "speech_duration": round(speech_dur, 3),
                    "articulation_rate": round(articulation_rate, 2),
                }
            )
        except Exception as e:
            results.append(
                {
                    "start": start,
                    "end": end,
                    "text": "",
                    "chars": 0,
                    "speech_duration": 0,
                    "articulation_rate": 0,
                    "error": str(e),
                }
            )
    return results


# * 샷 감지


def compute_hist_diff(frame1, frame2, bins=16):
    # RGB Color 사용
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [bins] * 3, [0, 256] * 3)
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [bins] * 3, [0, 256] * 3)

    # hist 정규화
    hist1 = cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX).flatten()
    hist2 = cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX).flatten()

    # hist 비교
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def shot_boundary_detection(
    video_source,  # 파일 경로, 스트림 URL, 또는 이미 열린 VideoCapture 객체
    scale_factor=2.0,
    bins=16,
    window_size=30,
    resize_dim=None,
):
    """
    비디오에서 샷 경계를 감지하는 함수 (색 히스토그램 차이 기반)

    :param video_source: 비디오 파일 경로(str), 스트림 URL(str), 또는 cv2.VideoCapture 객체
    :param scale_factor: 샷 경계 임계값을 설정하기 위한 평균 차이에 대한 스케일 팩터, 높을수록 확실한 샷 경계만 감지
    :param bins: 히스토그램 빈 개수 (낮을수록 빠르고, 너무 낮으면 정확도 저하)
    :param window_size: 이동 평균 및 표준 편차 계산을 위한 윈도우 크기 (프레임 수)\n
                        이전 N개 프레임의 차이값을 기반으로 임계값을 동적으로 설정
    :param resize_dim: 리사이즈할 (width, height). None이면 원본\n
                        예: (640, 360) 또는 (320, 180). (640, 360)이면 충분한 것으로 보임
    :return: 샷 경계 프레임 인덱스 리스트, 히스토그램 차이 리스트
    """
    # video_source가 문자열이면 새로 VideoCapture 생성
    if isinstance(video_source, str):
        cap = cv2.VideoCapture(video_source)
        video_label = video_source
    elif isinstance(video_source, cv2.VideoCapture):
        cap = video_source
        video_label = "VideoCapture Stream"
    else:
        raise ValueError("video_source는 str 또는 cv2.VideoCapture여야 합니다.")

    if not cap.isOpened():
        print(f"Cannot open video: {video_label}")
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("경고: 스트림에서는 총 프레임 수를 알 수 없습니다.")

    diffs = []
    shot_boundaries = [0]
    recent_diffs_window = []

    ret, prev_frame = cap.read()
    if not ret:
        print("Cannot read video's first frame")
        cap.release()
        return [], []

    if resize_dim:
        prev_frame = cv2.resize(prev_frame, resize_dim)

    print(f"'{video_label}' 비디오 샷 경계 감지 시작...")

    if resize_dim:
        print(f"프레임 크기 조정: {resize_dim[0]}x{resize_dim[1]}")

    # tqdm 사용 여부: 프레임 수 모르면 progress bar 없이
    use_tqdm = total_frames > 0
    progress = tqdm(
        total=(total_frames - 1) if use_tqdm else None, desc="프레임 처리 중"
    )

    frame_idx = 1
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        if resize_dim:
            current_frame = cv2.resize(current_frame, resize_dim)

        diff = compute_hist_diff(prev_frame, current_frame, bins)
        diffs.append(diff)

        recent_diffs_window.append(diff)
        if len(recent_diffs_window) > window_size:
            recent_diffs_window.pop(0)

        threshold_value = 0.0
        is_shot_boundary = False

        if len(recent_diffs_window) >= 5:
            avg = np.mean(recent_diffs_window)
            std = np.std(recent_diffs_window)
            threshold_value = avg + scale_factor * std

            is_shot_boundary = diff > threshold_value
            if is_shot_boundary:
                shot_boundaries.append(frame_idx)

        if use_tqdm:
            progress.set_postfix(
                {
                    "프레임": frame_idx,
                    "차이": f"{diff:.4f}",
                    "임계값": (
                        f"{threshold_value:.4f}"
                        if len(recent_diffs_window) >= 5
                        else "계산 중"
                    ),
                    "샷 경계": "O" if is_shot_boundary else "X",
                }
            )
            progress.update(1)

        prev_frame = current_frame
        frame_idx += 1

    if use_tqdm:
        progress.close()

    cap.release()
    print(f"\n샷 경계 감지 완료. 총 {len(shot_boundaries)}개의 샷 경계 감지됨.")

    return shot_boundaries, diffs


def sum_short_boundaries(boundaries, max_gap=15):
    """
    너무 짧은 샷들을 병합하는 함수

    :param boundaries: 기존에 만들어진 shot boundaries의 프레임 번호
    :param max_gap: max_gap 이하의 shot들은 하나로 합침
    :return: 짧은 샷들이 제거된 shot boundaries
    """

    representatives = []
    cur_group_start = boundaries[0]

    for i in range(1, len(boundaries)):
        if boundaries[i] - cur_group_start > max_gap:
            representatives.append(cur_group_start)
            cur_group_start = boundaries[i]

    representatives.append(cur_group_start)
    return representatives


def visualize_shot_detection(
    diffs,
    shot_boundaries,
    video_fps,
    total_video_frames,
    title="Hist Diff & Shot Boundary per Frame",
):
    """
    프레임 간 히스토그램 차이(diffs)를 그래프로 시각화하고,
    감지된 샷 경계 지점에 샷 번호를 표시합니다. X축은 시간(초)으로 표시됩니다.

    :param diffs: 각 프레임 간의 히스토그램 차이 값 리스트
    :param shot_boundaries: 샷 경계 프레임 인덱스 리스트
    :param video_fps: 비디오의 초당 프레임 수 (Frame Per Second)
    :param total_video_frames: 비디오의 총 프레임 수
    :param title: 그래프 제목
    """
    plt.figure(figsize=(18, 6))  # 그래프 크기 설정

    # 프레임 인덱스를 시간(초)으로 변환
    time_indices_sec = np.arange(1, len(diffs) + 1) / video_fps

    plt.plot(
        time_indices_sec, diffs, label="Hist diff per Frame", color="blue", alpha=0.7
    )

    # 샷 경계 지점에 샷 번호 텍스트 표시
    for i, boundary_frame in enumerate(shot_boundaries):
        if i > 0:  # 샷 2부터
            shot_number = i + 1
            # 텍스트를 표시할 x 좌표를 시간(초)으로 변환
            text_x = boundary_frame / video_fps
            text_y = np.max(diffs) * 0.95  # diffs 최대값의 95% 위치 (조절 가능)

            plt.text(
                text_x,
                text_y,
                str(shot_number),  # 샷 번호 텍스트
                color="red",
                fontsize=10,  # 폰트 크기 10으로 변경 (더 잘 보이도록)
                fontweight="bold",
                ha="center",  # 수평 정렬: 중앙
                va="top",  # 수직 정렬: 위쪽 (y값이 텍스트의 상단에 오도록)
                bbox=dict(
                    facecolor="yellow",
                    alpha=0.5,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),  # 배경 박스
            )

    plt.title(title, fontsize=16)
    plt.xlabel("시간 (초)", fontsize=12)  # X축 레이블 변경
    plt.ylabel("Histogram diffs (Bhattacharyya Distance)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    # x축 범위를 총 비디오 시간(초)까지 설정
    plt.xlim(0, total_video_frames / video_fps)
    plt.tight_layout()  # 레이아웃 자동 조정
    plt.show()


def create_segmented_video(input_video_path, shot_boundaries, output_filename):
    """
    주어진 샷 경계를 기준으로 원본 비디오를 분할하고, 각 샷 경계에 시각적 표시를 추가하여
    하나의 새로운 MP4 비디오 파일로 통합

    :param input_video_path: 원본 비디오 파일 경로
    :param shot_boundaries: 샷 경계 프레임 인덱스 리스트 (shot_boundary_detection 함수에서 반환된 값)
                            첫 프레임(0)은 항상 샷의 시작으로 간주
    :param output_filename: 생성될 출력 비디오 파일 이름
    """

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_video_path}")
        return

    # 비디오 속성
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 코덱 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"'{input_video_path}'에서 비디오 생성 중...")
    print(f"출력 파일: '{output_filename}'")

    shot_boundaries_set = set(shot_boundaries)

    current_shot_idx = 1
    current_shot_text = f"SHOT: {current_shot_idx}"

    # 각 샷마다 랜덤 색상 생성 (BGR 형식)
    current_shot_color = np.random.randint(0, 256, 3).tolist()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3

    with tqdm(total=total_frames, desc="프레임 처리 및 비디오 생성 중") as pbar:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # 현재 프레임이 샷 경계에 해당하는지 확인
            # 첫 프레임(0)은 이미 초기화 시 처리되었으므로, 0이 아닌 샷 경계만 새로운 샷으로 간주
            if frame_idx > 0 and frame_idx in shot_boundaries_set:
                current_shot_idx += 1  # 새로운 샷이 시작되었으므로 샷 번호 증가
                current_shot_text = f"SHOT: {current_shot_idx}"
                current_shot_color = np.random.randint(
                    0, 256, 3
                ).tolist()  # 새로운 랜덤 색상 생성

            # 텍스트 크기 계산 (매 프레임마다 텍스트가 달라지지 않으므로 한 번만 계산해도 되지만,
            # 여기서는 텍스트 위치 계산을 위해 루프 안에 둡니다.)
            (text_width, text_height), baseline = cv2.getTextSize(
                current_shot_text, font, font_scale, font_thickness
            )

            # 텍스트 위치 (프레임 중앙 상단)
            text_x = (width - text_width) // 2
            text_y = text_height + 20  # 상단에서 20픽셀 아래

            # 텍스트 그리기 (현재 샷의 텍스트와 색상 사용)
            cv2.putText(
                frame,
                current_shot_text,
                (text_x, text_y),
                font,
                font_scale,
                current_shot_color,
                font_thickness,
                cv2.LINE_AA,
            )

            out.write(frame)  # 처리된 프레임을 출력 비디오에 쓰기
            pbar.update(1)  # 진행률 바 업데이트

    cap.release()  # 입력 비디오 객체 해제
    out.release()  # 출력 비디오 객체 해제
    print(f"\n비디오 생성 완료: '{output_filename}'")


# * ------------------- SI, TI, Optical Flow -------------------
def extract_frame_number(path):
    """
    파일 이름에서 정수형 프레임 번호 추출 (예: '0s_30.0fps_896.jpg' → 896)
    """
    match = re.search(r"(\d+)\.jpg", path)
    return int(match.group(1)) if match else -1


def calculateSI(path):
    """
    Spatial Information (SI) 계산\n
    SI는 영상의 복잡성을 측정하는 지표로, Sobel 필터를 사용하여 영상의 엣지 강도를 계산함\n
    extractFrames 함수로 프레임을 추출 후 사용 권장

    @param path: 이미지가 저장된 폴더 경로
    """
    si_values = []
    frames = sorted(glob.glob(os.path.join(path, "*.jpg")), key=extract_frame_number)
    for img_path in tqdm(frames, desc="Calculating SI"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.hypot(sobel_x, sobel_y)  # sqrt(sobel_x**2 + sobel_y**2)
        std = np.std(sobel)
        si_values.append(std)

    # return max(si_values) if si_values else 0.0
    return si_values


def calculateTI(path):
    """
    Temporal Information (TI) 계산\n
    TI는 영상의 시간적 변화를 측정하는 지표로, 연속된 프레임 간의 차이를 계산함\n
    extractFrames 함수로 프레임을 추출 후 사용 권장

    @param path: 이미지가 저장된 폴더 경로
    """
    ti_values = []
    prev_img = None
    frames = sorted(glob.glob(os.path.join(path, "*.jpg")), key=extract_frame_number)

    for img_path in tqdm(frames, desc="Calculating TI"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if prev_img is not None:
            diff = cv2.absdiff(img, prev_img)
            std = np.std(diff)
            ti_values.append(std)

        prev_img = img

    # return max(ti_values) if ti_values else 0.0
    return ti_values


def calculateOpticalFlow(path):
    """
    Optical Flow를 사용하여 프레임 간의 움직임을 측정하는 함수\n
    Optical Flow는 연속된 프레임 간의 픽셀 이동을 계산하여 영상의 움직임을 분석함\n
    extractFrames 함수로 프레임을 추출 후 사용 권장

    @param path: 이미지가 저장된 폴더 경로
    """
    frames = sorted(glob.glob(os.path.join(path, "*.jpg")), key=extract_frame_number)
    flow_magnitudes = []

    prev = None
    prev_pts = None
    imputed_indices = []  # 보간된 프레임 인덱스 기록

    for i, img_path in enumerate(tqdm(frames, desc="Calculating Optical Flow")):
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue

        if prev is not None and prev_pts is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, frame, prev_pts, None)

            if next_pts is not None and status is not None:
                good_prev = prev_pts[status.flatten() == 1]
                good_next = next_pts[status.flatten() == 1]

                if len(good_prev) > 0:
                    displacements = np.linalg.norm(good_next - good_prev, axis=1)
                    mean_disp = np.mean(displacements)
                    flow_magnitudes.append(mean_disp)
                else:
                    # 추적 실패 → 이전 값 보간
                    imputed_indices.append(i)
                    flow_magnitudes.append(
                        flow_magnitudes[-1] if flow_magnitudes else 0
                    )
            else:
                # Optical flow 계산 실패 → 보간
                imputed_indices.append(i)
                flow_magnitudes.append(flow_magnitudes[-1] if flow_magnitudes else 0)
        elif prev is not None:
            # feature 추출 실패 → 보간
            imputed_indices.append(i)
            flow_magnitudes.append(flow_magnitudes[-1] if flow_magnitudes else 0)

        prev = frame
        prev_pts = cv2.goodFeaturesToTrack(
            frame, maxCorners=100, qualityLevel=0.3, minDistance=7
        )

    # 결과 요약 출력
    if imputed_indices:
        print(f"[Optical Flow] 보간된 프레임 수: {len(imputed_indices)}개")
        print(f"[Optical Flow] 보간된 프레임 인덱스: {imputed_indices}")
    else:
        print("[Optical Flow] 모든 프레임에서 정상적으로 Optical Flow 계산 완료.")

    return flow_magnitudes


# * ------------------- Audio로 SBD -------------------


def get_topk_indices_from_whole_audio(audio_np, device="cuda", top_k=5):
    """
    전체 오디오에서 상위 K 개의 인덱스를 가져오는 함수

    @param audio_np: 오디오 데이터 (1, N) numpy array
    @param device: 'cpu' 또는 'cuda'
    @param top_k: 상위 K 개의 인덱스를 가져옴
    @return: 상위 K 개의 인덱스 리스트
    """
    at = AudioTagging(checkpoint_path=None, device=device)
    (clipwise_output, _) = at.inference(torch.tensor(audio_np, device=device).float())
    topk_indices = np.argsort(clipwise_output[0])[::-1][:top_k]
    return topk_indices


def extractPANNsVectors(audio_np, sr=32000, segment_sec=1.0, device="cuda", top_k=5):
    """
    [1, N] shaped numpy 오디오 데이터를 받아서 PANNs 벡터 시퀀스로 변환\n
    top_k가 전체 영상에 대한 top이 아닌, 각 segmeent에 대한 top_k임

    @param audio_np: 오디오 데이터 (1, N) numpy array
    @param sr: 샘플링 레이트 (기본값: 32000)
    @param segment_sec: 각 segment의 길이 (초 단위, 기본값: 1.0초)
    @param device: 'cpu' 또는 'cuda'
    @param top_k: 각 segment에서 추출할 상위 K 개 클래스 수 (기본값: 5)
    @return: PANNs 벡터 시퀀스 (shape: [N, top_k]), segment_sec
    """
    audio_tensor = torch.tensor(audio_np, device=device).float()  # shape: [1, N]

    chunk_size = int(sr * segment_sec)
    total_len = audio_tensor.shape[1]
    total_chunks = total_len // chunk_size

    at = AudioTagging(checkpoint_path=None, device=device)
    vector_list = []

    for i in tqdm(range(total_chunks), desc="PANNs 벡터 추출 중"):
        chunk = audio_tensor[:, i * chunk_size : (i + 1) * chunk_size]
        if chunk.shape[1] != chunk_size:
            continue

        (clipwise_output, _) = at.inference(chunk)
        scores = torch.tensor(
            clipwise_output[0], device=device
        ).float()  # shape: (527,)
        vector_list.append(scores)

    if not vector_list:
        return None, segment_sec

    all_vectors = torch.stack(vector_list, dim=0)  # shape: (N, 527)

    # 🎯 평균 기준 top-k 클래스 선택
    mean_scores = torch.mean(all_vectors, dim=0)
    topk_indices = torch.topk(mean_scores, k=top_k).indices  # shape: (top_k,)

    vectors_topk = all_vectors[:, topk_indices]  # shape: (N, top_k)

    return vectors_topk, segment_sec


def extractPANNsVectorsTopK(
    audio_np, topk_indices, sr=32000, segment_sec=1.0, device="cuda"
):
    """
    [1, N] shaped numpy 오디오 데이터를 받아서 PANNs 벡터 시퀀스로 변환\n
    top_k가 전체 영상에 대한 top_k임

    @param audio_np: 오디오 데이터 (1, N) numpy array
    @param topk_indices: PANNs 모델에서 사용할 상위 K 개 클래스 인덱스 리스트
    @param sr: 샘플링 레이트 (기본값: 32000)
    @param segment_sec: 각 segment의 길이 (초 단위, 기본값: 1.0초)
    @param device: 'cpu' 또는 'cuda'
    @return: PANNs 벡터 시퀀스 (shape: [N, top_k]), segment_sec
    """
    audio_tensor = torch.tensor(audio_np, device=device).float()  # shape: [1, N]

    chunk_size = int(sr * segment_sec)
    total_len = audio_tensor.shape[1]
    total_chunks = total_len // chunk_size

    at = AudioTagging(checkpoint_path=None, device=device)
    vector_list = []

    for i in tqdm(range(total_chunks), desc="PANNs 벡터 추출 중"):
        chunk = audio_tensor[:, i * chunk_size : (i + 1) * chunk_size]
        if chunk.shape[1] != chunk_size:
            continue

        (clipwise_output, _) = at.inference(chunk)
        scores = clipwise_output[0][topk_indices]  # shape: (top_k,)
        scores_tensor = torch.tensor(scores, device=device).float()
        vector_list.append(scores_tensor)

    if not vector_list:
        return None, segment_sec

    return torch.stack(vector_list, dim=0), segment_sec


def detectAudioShotBoundaries(
    vectors, threshold=None, scale=1.0, return_distances=False
):
    """
    cosine distance 기반 샷 경계 검출 (GPU 벡터 입력)

    @param vectors: shape (T, top_k) PANNs 벡터 시퀀스
    @param threshold: 샷 경계 임계값, None이면 자동 계산
    @param scale: threshold가 None일 때, 자동 threshold = mean + scale * std
    @param return_distances: 디버깅용, distance 벡터 반환 여부
    @return: 샷 경계 인덱스 리스트, distances (선택적)
    """
    vectors = F.normalize(vectors, dim=1)
    similarities = torch.sum(vectors[1:] * vectors[:-1], dim=1)
    distances = 1 - similarities

    if threshold is None:
        mean = distances.mean().item()
        std = distances.std().item()
        threshold = mean + scale * std
        print(
            f"[AUTO] Threshold = Mean({mean:.4f}) + {scale:.2f} * Std({std:.4f}) = {threshold:.4f}"
        )

    boundary_indices = torch.where(distances > threshold)[0] + 1
    boundaries = [0] + boundary_indices.tolist()

    if return_distances:
        return boundaries, distances.cpu().numpy()

    return boundaries


def visualizeAudioVectors(vectors, boundaries, topk_indices, segment_sec=1.0):
    """
    PANNs 벡터 시퀀스와 샷 경계 시점을 히트맵으로 시각화

    @param vectors: shape (T, top_k) PANNs 벡터 시퀀스
    @param boundaries: 샷 경계 인덱스 리스트
    @param topk_indices: 원래 527차원 중 어떤 index를 top-k로 썼는지
    @param segment_sec: 각 segment의 길이 (초 단위, 기본값: 1.0초)
    @return: 없음 (그래프 출력)
    """

    # torch → numpy 변환
    vectors = vectors.detach().cpu().numpy()

    # label 이름 추출
    selected_labels = [labels[i] for i in topk_indices]

    # 시각화 시작
    plt.figure(figsize=(14, 5))
    sns.heatmap(vectors.T, cmap="magma", xticklabels=False, yticklabels=selected_labels)
    plt.title("Top Audio Event Scores Over Time")
    plt.ylabel("Top Audio Classes")
    plt.xlabel("Time (seconds, per segment)")

    # 샷 경계 시점 선 긋기
    for t in boundaries:
        plt.axvline(x=t / segment_sec, color="cyan", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.show()
