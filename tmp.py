import numpy as np
import cv2, os, time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from tqdm import tqdm
import utils  # utils 모듈 임포트
import matplotlib

matplotlib.use("Agg")


def play_video_with_metrics(cap, si, ti, optic_flow, output_path, window_size=5):
    """
    비디오 프레임을 SI, TI, Optical Flow 메트릭 플롯과 함께 표시하고,
    결합된 시각화를 비디오 파일로 저장합니다.

    이 최적화된 버전은 Matplotlib의 기본 애니메이션 저장 방식보다
    더 빠른 비디오 저장을 위해 OpenCV의 VideoWriter를 사용합니다.

    Args:
        cap (cv2.VideoCapture): 입력 비디오의 OpenCV VideoCapture 객체.
        si (np.array): 공간 정보 (SI) 메트릭 배열.
        ti (np.array): 시간 정보 (TI) 메트릭 배열.
        optic_flow (np.array): 광학 흐름 (Optical Flow) 메트릭 배열.
        output_path (str): 출력 비디오를 저장할 기본 경로 (예: "output_video").
                            함수는 ".mp4"를 추가합니다.
        window_size (int): 플롯의 x축에 대한 시간 창 (초 단위).
    """

    mpl.rcParams["animation.embed_limit"] = 50_000_000

    # --- 1. 비디오 프레임 로드 ---
    # 비디오 속성 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    print("비디오 프레임을 로드 중...")
    # 모든 프레임을 메모리로 읽어들입니다 (매우 긴 비디오의 경우 메모리 사용량 고려).
    for _ in tqdm(range(total_frame), desc="프레임 로드 중"):
        ret, frame = cap.read()
        if not ret:
            break
        # Matplotlib 표시를 위해 BGR을 RGB로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    print(f"{len(frames)}개의 프레임을 로드했습니다.")

    # --- 2. 시각화를 위한 시간 축 준비 ---
    x_time = np.arange(total_frame) / fps
    # TI 및 Optical Flow 배열은 일반적으로 프레임보다 길이가 하나 짧습니다.
    x_time_ti = x_time[:-1]
    duration = x_time[-1]

    # --- 3. Matplotlib Figure 및 Subplots 설정 ---
    # 비디오 렌더링 속도를 향상시키려면 figsize를 줄여보세요.
    # 예: (10, 6) 또는 (12, 7)
    fig = plt.figure(figsize=(14, 8))  # 현재 크기 유지, 필요시 이 값을 줄여보세요.
    # 그리드 레이아웃 정의: 3행 2열, 열 너비 비율 지정
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 3])

    # 비디오 표시를 위한 왼쪽 서브플롯
    ax_video = fig.add_subplot(gs[:, 0])
    img_disp = ax_video.imshow(frames[0])  # 첫 번째 프레임으로 초기화
    ax_video.axis("off")  # 비디오 표시를 위해 축 끄기
    ax_video.set_title("비디오")

    # 메트릭을 위한 오른쪽 서브플롯
    ax_si = fig.add_subplot(gs[0, 1])
    ax_ti = fig.add_subplot(gs[1, 1])
    ax_of = fig.add_subplot(gs[2, 1])

    # SI (공간 정보) 플롯
    ax_si.plot(x_time, si, color="blue", label="SI")
    cursor_si = ax_si.axvline(x=0, color="r", linestyle="--")  # 수직 커서 라인
    ax_si.set_xlim(0, window_size)  # 초기 x축 제한
    ax_si.set_ylim(np.min(si) - 5, np.max(si) + 5)  # 여백을 포함한 Y축 제한
    ax_si.set_ylabel("SI")
    ax_si.grid(True)

    # TI (시간 정보) 플롯
    ax_ti.plot(x_time_ti, ti, color="green", label="TI")
    cursor_ti = ax_ti.axvline(x=0, color="r", linestyle="--")
    ax_ti.set_xlim(0, window_size)
    ax_ti.set_ylim(np.min(ti) - 5, np.max(ti) + 5)
    ax_ti.set_ylabel("TI")
    ax_ti.grid(True)

    # Optical Flow 플롯
    ax_of.plot(x_time_ti, optic_flow, color="orange", label="Optical Flow")
    cursor_of = ax_of.axvline(x=0, color="r", linestyle="--")
    ax_of.set_xlim(0, window_size)
    ax_of.set_ylim(np.min(optic_flow) - 5, np.max(optic_flow) + 5)
    ax_of.set_xlabel("시간 (s)")
    ax_of.set_ylabel("Optical Flow")
    ax_of.grid(True)

    # 제목/레이블 겹침 방지를 위해 레이아웃 조정
    plt.tight_layout()

    # --- 4. 저장을 위한 VideoWriter 준비 ---
    # Matplotlib 그림의 픽셀 단위 크기 가져오기
    # 렌더러 크기를 얻기 위해 캔버스를 한 번 그려야 합니다.
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    # 코덱 정의 및 VideoWriter 객체 생성
    # 'mp4v'는 .mp4 파일에 대한 일반적인 코덱입니다.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{output_path}.mp4", fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"오류: {output_path}.mp4에 대한 비디오 작성기를 열 수 없습니다.")
        return

    # --- 5. 프레임 업데이트 및 저장 루프 ---
    print("비디오 프레임을 렌더링하고 저장 중...")
    for frame_idx in tqdm(range(len(frames)), desc="비디오 저장 중"):
        current_time = frame_idx / fps
        img_disp.set_data(frames[frame_idx])  # 비디오 프레임 업데이트

        # 모든 수직 커서 라인 업데이트
        for cursor in [cursor_si, cursor_ti, cursor_of]:
            cursor.set_xdata([current_time, current_time])

        # 플롯을 위한 슬라이딩 x축 창 구현
        if current_time > window_size / 2:
            left = current_time - window_size / 2
            right = current_time + window_size / 2
            # 창이 비디오 지속 시간을 초과하지 않도록 보장
            if right > duration:
                right = duration
                left = duration - window_size
            # duration < window_size일 때 `left`가 음수가 될 수 있는 경우 처리
            if left < 0:
                left = 0
                right = window_size
        else:
            left, right = 0, window_size

        # 모든 메트릭 플롯에 새 x축 제한 적용
        for ax in [ax_si, ax_ti, ax_of]:
            ax.set_xlim(left, right)

        # 업데이트된 그림을 캔버스에 그립니다.
        fig.canvas.draw()
        # Matplotlib 그림을 NumPy 배열 (ARGB)로 변환
        # `tostring_argb`는 바이트를 제공하며, (높이, 너비, 4)으로 재구성합니다.
        img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(
            height, width, 4
        )
        # OpenCV를 위해 ARGB를 BGR로 변환 (알파 채널은 무시)
        img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        # 출력 비디오 파일에 프레임 쓰기
        out.write(img_array_bgr)

    # --- 6. 정리 ---
    out.release()  # VideoWriter 해제
    plt.close(fig)  # 메모리 확보를 위해 Matplotlib 그림 닫기
    print(f"비디오가 {output_path}.mp4에 저장되었습니다.")


def extractFrames(
    cap,
    start_time_sec,
    duration_sec,
    fps,
    output_folder_path,
    resize_to=(320, 180),  # 원하는 해상도 (width, height)
):
    """
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
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY])
            saved_count += 1

        current_frame += 1

    extraction_end_time = time.time()
    print(
        f"✅ 총 {saved_count}개 프레임 저장 완료. 소요 시간: {extraction_end_time - extraction_start_time:.2f}초"
    )
    return saved_count


if __name__ == "__main__":
    # 이 블록 안의 코드는 스크립트가 직접 실행될 때만 실행됩니다.
    url = "https://www.youtube.com/watch?v=IrRh1rY5SVQ"

    cap = utils.openVideoStream(url)
    # utils.extractFrames 함수가 필요하다면 주석을 해제하세요.
    extractFrames(cap, 0, None, cap.get(cv2.CAP_PROP_FPS), "./sing")

    # 메트릭 계산
    # extractFrames가 실행되지 않았다면 ./sing 경로에 프레임이 없을 수 있습니다.
    # 이 경우, 프레임 추출 로직을 먼저 실행하거나,
    # si, optic_flow, ti 계산에 필요한 데이터를 다른 방식으로 준비해야 합니다.
    print("SI, Optical Flow, TI 메트릭을 계산 중...")
    si = utils.calculateSI("./sing")
    optic_flow = utils.calculateOpticalFlow("./sing")
    ti = utils.calculateTI("./sing")
    print("메트릭 계산 완료.")

    # 비디오 스트림을 다시 열어야 할 수도 있습니다.
    # extractFrames가 cap을 release했기 때문입니다.
    cap = utils.openVideoStream(url)

    # 비디오와 메트릭을 함께 재생하고 저장
    play_video_with_metrics(cap, si, ti, optic_flow, "sing")
