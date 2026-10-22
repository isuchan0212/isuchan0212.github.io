# Suchan Lee — personal website

POSTECH AI 연구자 Suchan Lee의 연구·프로젝트와 이력을 소개하는 정적 홈페이지입니다. HTML, CSS, 작은 JavaScript만 사용하며 설치나 빌드가 필요 없습니다.

## 로컬 미리보기

Node.js 18 이상에서:

```sh
npm run dev
```

브라우저에서 <http://127.0.0.1:4173>을 엽니다. 포트가 사용 중이면 `PORT=4174 npm run dev`로 변경할 수 있습니다. 미리보기 서버는 공개용 파일만 제공하며 원본 CV PDF는 제공하지 않습니다.

```sh
npm run check
```

## 파일과 콘텐츠 수정

- `index.html`: 소개(사진), Publications, Work experience, Awards, Off hours, 연락처
- `assets/cv.pdf`: CV 링크가 여는 공개용 PDF. 원본 `cv_2026.pdf`를 고치면 `cp cv_2026.pdf assets/cv.pdf`로 다시 복사하세요.
- `cv.html`: 이전의 웹 CV(현재 링크되지 않음)
- `styles.css`: 벽과 창, 색, 레이아웃, 모바일, 인쇄 스타일
- `script.js`: 창에서 쏟아지는 빛, CV 인쇄 버튼
- `assets/favicon.svg`, `assets/apple-touch-icon.png`: 사이트 아이콘(창, 해, 쏟아지는 빛)
- `assets/og.png`: 링크 미리보기 이미지(1200×630, 첫 화면 캡처). 첫 화면을 크게 바꾸면 다시 만들어 주세요.
- `assets/portrait.jpg`: 프로필 사진. 원본(`assets/suchan.jpeg`, Git 제외)을 960px로 줄이고 위치(GPS) 등 메타데이터를 지운 공개용 파일입니다. 사진을 바꿀 때도 같은 처리를 해주세요. 첫 화면 아래 소개 칸 왼쪽 벽감에 들어갑니다. 4:5 세로 비율, 가로 800px 이상 권장. 파일이 없으면 사람 실루엣 placeholder가 보입니다.
- `scripts/serve.mjs`: 외부 의존성 없는 로컬 미리보기 서버

### 디자인 메모

- 빛바랜 크림 베이지 벽 하나에 낙수장(Fallingwater)식 가로 창 하나. 창밖은 옅은 하늘색 하늘과 낮은 숲 능선, 그리고 해입니다. 벽에는 잎 그림자가 옅게 흔들립니다.
- 색은 베이지 위주에 창밖의 흐린 하늘색만 포인트로 씁니다. 전체에 옅은 필름 그레인을 덮어 오래된 사진처럼 바랜 느낌을 냅니다.
- 해가 뜨고 나면 창 전체에서 넓은 빛이 방 안으로 비스듬히 쏟아집니다. 그 안에 더 밝은 빛줄기 네 가닥이 천천히 일렁이고 먼지가 떠다닙니다(모양은 `script.js`가 창 위치로 계산). Contact에 오면 이메일 옆에 등이 켜지듯 주변이 밝아집니다.
- 창은 영화 스크린 비율(2.39:1)입니다. 페이지를 열면 극장처럼 방이 어둡고 창만 밝은 상태에서 해가 능선 위로 떠오르고, 빛이 쏟아지며 방에 불이 들어온 뒤 이름이 오프닝 타이틀처럼 나타납니다(약 5초, 한 번만). 창 유리에는 옅은 반사광이 있고, 구름 두 조각이 아주 느리게 흐릅니다.
- 섹션 이름이 화면 위쪽으로 올라오면 라벨이 따뜻한 색으로 물듭니다.
- 움직임 줄이기 설정을 켠 사용자에게는 모든 애니메이션이 꺼집니다.
- 글꼴은 Newsreader(이름과 본문)와 Pretendard(날짜, 섹션 이름 같은 작은 정보)를 CDN으로 불러옵니다.
- JavaScript가 없어도 모든 내용과 링크는 그대로 보입니다. 창에서 쏟아지는 빛은 그려지지 않습니다.

콘텐츠는 2026 CV를 바탕으로 정리했습니다. 내용이 바뀌면 `index.html`과 `assets/cv.pdf`를 함께 갱신하세요.

## CV 원본 제외

`cv_*.pdf`를 포함한 모든 PDF 파일(대소문자 확장자 포함, 공개용 `assets/cv.pdf`만 예외), `cv/`, `CV/`, `private/`는 `.gitignore`에 등록되어 있습니다. 원본 `cv_2026.pdf`는 로컬 참고용으로만 사용하고 Git에 포함하지 않습니다. 공개용 CV는 `assets/cv.pdf`입니다.

```sh
git check-ignore -v cv_2026.pdf
git ls-files '*.pdf' '*.PDF'
```

첫 명령은 제외 규칙을 보여주고, 두 번째 명령은 현재 추적 중인 PDF가 없으면 아무것도 출력하지 않습니다. `git add -f`로 원본을 강제 추가하지 마세요.

## GitHub Pages 배포

이 저장소 이름은 `isuchan0212.github.io`이므로 GitHub Pages 설정 후 주소는 `https://isuchan0212.github.io/`입니다.

1. 사이트 파일을 검토한 뒤 `main` 브랜치에 커밋·푸시합니다.
2. GitHub 저장소의 **Settings → Pages**에서 **Deploy from a branch**를 선택합니다.
3. 브랜치를 **main**, 폴더를 **/ (root)**로 설정합니다.

`.nojekyll`이 포함되어 있어 정적 파일을 그대로 제공합니다. 빌드나 패키지 설치가 필요 없습니다. 이 작업에서 원격 커밋, 푸시, Pages 설정 변경은 수행하지 않았습니다.

배포 설정 참고: [GitHub Pages 공식 문서](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
