document.addEventListener("DOMContentLoaded", function () {
  const diffElements = document.querySelectorAll('.diff-code');
  
  diffElements.forEach(function (el) {
    const diffString = el.textContent;
    
    // 숨겨두었던 컨테이너를 다시 보이게 설정
    el.style.display = 'block';
    el.textContent = ''; 
    
    const toggleWrapper = document.createElement('div');
    toggleWrapper.className = 'diff-toggle-wrapper';
    
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'diff-toggle-btn';
    toggleBtn.textContent = 'View Line-by-Line';
    
    toggleWrapper.appendChild(toggleBtn);
    el.appendChild(toggleWrapper);
    
    const diffContainer = document.createElement('div');
    el.appendChild(diffContainer);
    
    let currentFormat = 'side-by-side';
    
    function renderDiff() {
      diffContainer.innerHTML = '';
      
      const diff2htmlUi = new Diff2HtmlUI(diffContainer, diffString, {
        drawFileList: false,
        matching: 'words',
        outputFormat: currentFormat,
        synchronisedScroll: true
      });
      diff2htmlUi.draw();
      
      if (currentFormat === 'side-by-side') {
        attachCustomScrollbar(diffContainer);
      }
    }
    
    toggleBtn.addEventListener('click', () => {
      if (currentFormat === 'side-by-side') {
        currentFormat = 'line-by-line';
        toggleBtn.textContent = 'View Side-by-Side';
      } else {
        currentFormat = 'side-by-side';
        toggleBtn.textContent = 'View Line-by-Line';
      }
      renderDiff();
    });
    
    renderDiff();
  });
});

function attachCustomScrollbar(el) {
  // 1. 각 사이드(좌/우)의 가로 스크롤 동기화 및 통합 커스텀 스크롤바 생성
  const fileDiffs = el.querySelectorAll('.d2h-files-diff');
  fileDiffs.forEach(diff => {
    diff.style.position = 'relative'; // 커스텀 스크롤바 위치 기준

    const sides = diff.querySelectorAll('.d2h-file-side-diff');
    if (sides.length === 2) {
      const left = sides[0];
      const right = sides[1];
      
      // 통합 스크롤바 요소 생성
      const track = document.createElement('div');
      track.className = 'custom-diff-scrollbar-track';
      const thumb = document.createElement('div');
      thumb.className = 'custom-diff-scrollbar-thumb';
      track.appendChild(thumb);
      diff.appendChild(track);

      let isSyncingLeft = false;
      let isSyncingRight = false;
      let scrollTimeout = null;

      function updateScrollState() {
        const leftScrollWidth = left.scrollWidth;
        const leftClientWidth = left.clientWidth;
        
        if (leftScrollWidth <= leftClientWidth) {
          track.style.display = 'none';
          return;
        }
        track.style.display = 'block';

        const trackWidth = track.clientWidth;
        const thumbWidthRatio = leftClientWidth / leftScrollWidth;
        const thumbWidth = Math.max(thumbWidthRatio * trackWidth, 30);
        const maxThumbLeft = trackWidth - thumbWidth;
        const maxScrollLeft = leftScrollWidth - leftClientWidth;
        
        const thumbLeft = (left.scrollLeft / maxScrollLeft) * maxThumbLeft;

        thumb.style.width = thumbWidth + 'px';
        thumb.style.transform = `translateX(${thumbLeft}px)`;

        // 스크롤 중 애니메이션 클래스 제어
        track.classList.add('is-visible');
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {
          if (!track.classList.contains('is-dragging')) {
            track.classList.remove('is-visible');
          }
        }, 800);
      }

      left.addEventListener('scroll', function() {
        updateScrollState();
        if (!isSyncingLeft) {
          isSyncingRight = true;
          right.scrollLeft = left.scrollLeft;
        }
        isSyncingLeft = false;
      });

      right.addEventListener('scroll', function() {
        updateScrollState();
        if (!isSyncingRight) {
          isSyncingLeft = true;
          left.scrollLeft = right.scrollLeft;
        }
        isSyncingRight = false;
      });

      // 썸(Thumb) 드래그 기능 추가
      let isDragging = false;
      let startX, startScrollLeft;

      thumb.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startScrollLeft = left.scrollLeft;
        track.classList.add('is-dragging');
        e.preventDefault(); // 텍스트 선택 방지
      });

      window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const deltaX = e.clientX - startX;
        
        const trackWidth = track.clientWidth;
        const thumbWidth = parseFloat(thumb.style.width);
        const maxThumbLeft = trackWidth - thumbWidth;
        const maxScrollLeft = left.scrollWidth - left.clientWidth;
        
        const scrollRatio = maxScrollLeft / maxThumbLeft;
        left.scrollLeft = startScrollLeft + deltaX * scrollRatio;
      });

      window.addEventListener('mouseup', () => {
        if (isDragging) {
          isDragging = false;
          track.classList.remove('is-dragging');
          updateScrollState(); // 페이드 아웃 타이머 재시작
        }
      });

      // 초기화 및 창 크기 조절 대응
      setTimeout(updateScrollState, 100);
      window.addEventListener('resize', updateScrollState);
    }
  });
}
