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

class Scheduler:
    def __init__(self):
        self.schedule_db = defaultdict(list)
        self._load()

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
        self.schedule_db[ds].append({
            "title": title,
            "dt": dt,
            "status": "pending"
        })
        self._persist()
        return True, f"'{ds}' {dt.hour}시 '{title}' 일정이 등록되었습니다."

# --------------------------
# Gradio 챗봇 UI
# --------------------------
scheduler = Scheduler()

def bot_reply(user_text):
    try:
        if re.match(r"^\d{4}[.\-]\d{1,2}[.\-]\d{1,2}\s+\d{1,2}시\s+.+", user_text):
            dt, title = scheduler.parse_schedule_entry(user_text)
            ok, msg = scheduler.add_schedule(dt, title)
            return msg
        return "입력 형식: 'YYYY.MM.DD HH시 제목' 예: 2025.09.04 17시 농구약속"
    except Exception as e:
        return f"오류 발생: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## 간단 일정 등록 챗봇")
    chatbot = gr.Chatbot()
    txt = gr.Textbox(placeholder="예: 2025.09.04 17시 농구약속", lines=2)

    def submit(msg, history):
        history = history + [["user", msg]]
        reply = bot_reply(msg)
        history = history + [["bot", reply]]
        return "", history

    txt.submit(submit, [txt, chatbot], [txt, chatbot])
    gr.Button("전송").click(submit, [txt, chatbot], [txt, chatbot])

if __name__ == "__main__":
    demo.launch()
