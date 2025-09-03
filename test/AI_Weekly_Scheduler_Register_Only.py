"""
ai_weekly_scheduler_register_only.py

기능:
- 주어진 날짜/시간/제목으로 일정 등록
- 지난 시간 입력 차단
- Gradio 챗봇 UI에서 일정 등록만 수행
"""

from datetime import datetime, timedelta, date
import re
import pickle
from collections import defaultdict
import gradio as gr

# --------------------------
# 개발/테스트용 고정 현재 시간
CURRENT_TIME = datetime(2025, 9, 3, 6, 0, 0)
# --------------------------

WEEKDAY_KOR = ["월", "화", "수", "목", "금", "토", "일"]

class Scheduler:
    def __init__(self):
        self.schedule_db = defaultdict(list)
        self.pending_reset = False  # ✅ 초기화 대기 상태 플래그
        self._load()
    
    def reset(self):
        """일정 전체 초기화"""
        self.schedule_db.clear()
        self._persist()

    def _persist(self):
        with open("scheduler_state.pkl", "wb") as f:
            pickle.dump(dict(self.schedule_db), f)

    def _load(self):
        try:
            with open("scheduler_state.pkl", "rb") as f:
                self.schedule_db = defaultdict(list, pickle.load(f))
        except FileNotFoundError:
            self.schedule_db = defaultdict(list)

    # "YYYY.MM.DD HH시 제목" -> (datetime, title)
    def parse_schedule_entry(self, text):
        m = re.match(r"\s*(\d{4})[.\-](\d{1,2})[.\-](\d{1,2})\s+(\d{1,2})시\s+(.+)\s*$", text)
        if not m:
            raise ValueError("형식 오류: 'YYYY.MM.DD HH시 제목' 예: 2025.09.04 17시 농구약속")
        y, mo, d, h, title = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5).strip()
        dt = datetime(y, mo, d, h, 0, 0)
        return dt, title

    # 일정 추가
    def add_schedule(self, dt: datetime, title: str):
        if dt <= CURRENT_TIME:
            return False, "지난 시간에는 일정 등록이 불가합니다."
        ds = dt.date().isoformat()

        # ✅ 중복 일정 체크
        for e in self.schedule_db[ds]:
            if e["dt"].hour == dt.hour and e["title"] == title:
                return False, f"이미 등록한 일정입니다 : '{dt.strftime('%Y.%m.%d(%a) %H시')}' - '{title}'"
            
        # 중복이 아니면 새 일정 추가
        self.schedule_db[ds].append({
            "title": title,
            "dt": dt,
            "status": "pending",
            "auto_fixed": False
        })
        self._persist()
        return True, f"'{ds}' {dt.hour}시 '{title}' 일정이 등록되었습니다."
    
    # 일정 확인
    def view_schedules(self):
        if not self.schedule_db:
            return "등록된 일정이 없습니다."
        
        output_lines = []
        for ds in sorted(self.schedule_db.keys()):
            items = sorted(self.schedule_db[ds], key=lambda x: x["dt"])
            for item in items:
                dt = item["dt"]
                weekday = WEEKDAY_KOR[dt.weekday()]
                line = f"{dt.strftime('%Y.%m.%d')}({weekday}) - {dt.hour:02d}시 - {item['title']}"
                output_lines.append(line)
        return "\n".join(output_lines)

# --------------------------
# Gradio 챗봇 UI
# --------------------------
scheduler = Scheduler()

def bot_reply(user_text):
    user_text = user_text.strip()

    # ✅ 초기화 대기 상태일 때 (Y/N 응답 처리)
    if scheduler.pending_reset:
        if user_text.lower() == "y":
            scheduler.reset()
            scheduler.pending_reset = False
            return "✅ 모든 일정이 초기화되었습니다."
        elif user_text.lower() == "n":
            scheduler.pending_reset = False
            return "❎ 초기화를 취소했습니다."
        else:
            return "⚠️ 잘못된 입력입니다. 초기화를 진행하려면 Y 또는 N을 입력하세요."

    # ✅ 초기화 명령어 처리
    if user_text == "초기화":
        scheduler.pending_reset = True
        return "정말 초기화하시겠습니까? (Y/N)"

    # ✅ 일정 조회
    if user_text == "일정":
        return scheduler.view_schedules()

    # ✅ 일정 등록
    elif re.match(r"^\d{4}[.\-]\d{1,2}[.\-]\d{1,2}\s+\d{1,2}시\s+.+", user_text):
        dt, title = scheduler.parse_schedule_entry(user_text)
        ok, msg = scheduler.add_schedule(dt, title)
        return msg

    return "입력 형식: 'YYYY.MM.DD HH시 제목' (예: 2025.09.04 17시 농구약속)\n또는 '일정'을 입력해 확인하세요.\n'초기화'로 전체 삭제할 수 있습니다."

with gr.Blocks() as demo:
    gr.Markdown("## 간단 일정 등록 & 조회 챗봇")

    # user / bot 이름 제거 + 채팅 버블 스타일 적용
    chatbot = gr.Chatbot(
        show_label=False,        # user / bot 이름 제거
        bubble_full_width=False, # 버블 형태 (왼쪽=봇, 오른쪽=유저)
        height=400
    )

    txt = gr.Textbox(
        placeholder="예: 2025.09.04 17시 농구약속\n또는 '일정' 입력",
        lines=2
    )

    def submit(msg, history):
        reply = bot_reply(msg)
        # (user, bot) 튜플 형식으로 추가
        history = history + [(msg, reply)]
        return "", history

    txt.submit(submit, [txt, chatbot], [txt, chatbot])
    gr.Button("전송").click(submit, [txt, chatbot], [txt, chatbot])


if __name__ == "__main__":
    demo.launch(share=True)
