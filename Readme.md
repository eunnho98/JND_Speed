1. 가상환경 세팅 (python: 3.10.16, torch 2.0 이상 & cuda 세팅 권장)
2. https://werw.tistory.com/65 참고하여 ffmpeg 세팅 및 yt-dlp 설치 (ffmpeg 설치는 필수)
3. 해당 가상환경으로 변경 후 pip install -r requirements.txt
4. panns_data 폴더 전체를 윈도우 기준, C:\Users\사용자이름 에 위치시키기

- panns_data 다운로드 링크: https://drive.google.com/file/d/1aynK5yf0IpDxeRsnrWDUHHU_CiEBHTsA/view?usp=sharing

- 파이토치 링크: https://pytorch.org/get-started/locally/

- Mac일 경우 /Users/사용자이름 에 위치시키면 됨 / cuda 사용을 하지 못하여 매우 느리므로 윈도우 권장

- 간혹 getAudioCroppedFromURL 함수와 openVideoStream 함수에서 쿠키 문제 발생 가능 -> 주석 참조
