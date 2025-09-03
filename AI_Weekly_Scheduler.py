"""
ai_weekly_scheduler.py

요구사항:
- Gradio 챗봇 인터페이스로 주간(예: 2025.09.01 ~ 2025.09.07) 스케줄 입력/수정/취소/확인/달성률 기능 제공
- 과거 일정(지도학습)으로 '추천 시간대' 생성 (DecisionTreeClassifier 사용)
- Q-learning을 이용한 간단한 '블록(오전/오후/저녁) 스케줄 최적화'
- 반복되는 일정(2번 이상) 자동 고정 추천
- "지난 시간" 입력 차단 (기본 기준 시각은 Asia/Seoul 2025-09-03 06:00, 실제 배포시엔 datetime.now()로 교체)
"""

from datetime import datetime, timedelta, time, date
import re
import pickle
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Gradio
import gradio as gr

# --------------------------
# 설정: (개발/테스트 시 고정된 '현재' 시각)
# 실제 서비스에서는 datetime.now(시간대 적용) 사용 권장
CURRENT_TIME = datetime(2025, 9, 3, 6, 0, 0)  # Asia/Seoul 기준 (프로젝트 명세에서 고정)
# --------------------------

# 시간대 블록 정의
TIME_BLOCKS = ["오전", "오후", "저녁"]
# 오전: 05:00~11:59, 오후: 12:00~17:59, 저녁: 18:00~23:59
def hour_to_block(h):
    if 5 <= h <= 11:
        return "오전"
    if 12 <= h <= 17:
        return "오후"
    return "저녁"

# --------------------------
# 데이터 저장 구조
# schedule_db: dict -> { date_str: [ {title, dt(datetime), status('pending'/'done'/'fail'), auto_fixed(bool)} ] }
# past_records: DataFrame for ML (컬럼: weekday, importance, duration_hours, block)
# --------------------------
class Scheduler:
    def __init__(self):
        self.schedule_db = defaultdict(list)
        # 샘플 과거 데이터셋. 실제로는 저장/로드하여 누적 학습 가능.
        self.past_records = pd.DataFrame(columns=["weekday","importance","duration","block"])
        # 간단한 파일 저장
        self._load()

        # 지도학습 모델 (초기엔 None, 학습되면 사용)
        self.recommender = TimeRecommender()
        self._retrain_recommender_if_possible()

        # Q-learning 테이블 (상태: block index, 행동: block index)
        self.qtable = np.zeros((3,3))
        self.alpha = 0.1
        self.gamma = 0.9

        # 다단계 명령을 위한 대화 상태 저장
        self.pending_modify = {}  # session_id -> { 'title':..., 'candidates':[(date,title_index)], ... }

    def _persist(self):
        obj = {
            "schedule_db": dict(self.schedule_db),
            "past_records": self.past_records,
            "qtable": self.qtable
        }
        with open("scheduler_state.pkl","wb") as f:
            pickle.dump(obj, f)

    def _load(self):
        try:
            with open("scheduler_state.pkl","rb") as f:
                obj = pickle.load(f)
            # pickle에서 불러오기
            self.schedule_db = defaultdict(list, obj["schedule_db"])
            self.past_records = obj.get("past_records", pd.DataFrame(columns=["weekday","importance","duration","block"]))
            self.qtable = obj.get("qtable", np.zeros((3,3)))
        except FileNotFoundError:
            # 데모용 과거 데이터
            demo = [
                {"weekday":0,"importance":5,"duration":2,"block":"오전"},
                {"weekday":1,"importance":3,"duration":1,"block":"오후"},
                {"weekday":2,"importance":4,"duration":3,"block":"오전"},
                {"weekday":3,"importance":2,"duration":1,"block":"저녁"},
                {"weekday":4,"importance":5,"duration":2,"block":"오전"},
                {"weekday":5,"importance":1,"duration":1,"block":"오후"},
            ]
            self.past_records = pd.DataFrame(demo)

    def _retrain_recommender_if_possible(self):
        if len(self.past_records) >= 3:
            X = self.past_records[["weekday","importance","duration"]].values
            y = self.past_records["block"].values
            self.recommender.train(X, y)

    # 입력 파싱: "2025.09.04 17시 농구약속" -> (datetime, title) 반환, 실패 시 ValueError 발생
    def parse_schedule_entry(self, text):
        # 정규식: YYYY.MM.DD HH시 <제목>
        m = re.match(r"\s*(\d{4})[.\-](\d{1,2})[.\-](\d{1,2})\s+(\d{1,2})시\s+(.+)\s*$", text)
        if not m:
            raise ValueError("입력 형식이 'YYYY.MM.DD HH시 제목' 이어야 합니다. 예: 2025.09.04 17시 농구약속")
        y,mo,d,h,title = int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4)),m.group(5).strip()
        dt = datetime(y,mo,d,h,0,0)
        return dt, title

    def add_schedule(self, dt:datetime, title:str):
        # 지난 시간 입력 차단
        if dt <= CURRENT_TIME:
            return False, "지난 시간에 대해서는 일정을 입력할 수 없습니다."
        ds = dt.date().isoformat()
        self.schedule_db[ds].append({
            "title": title,
            "dt": dt,
            "status": "pending",
            "auto_fixed": False
        })
        self._persist()
        # 반복 일정 감지
        count = self._count_title_recent_days(title, days=7)
        msg = f"'{ds}' '{dt.hour}시'에 '{title}'이(가) 기록되었습니다."
        if count >= 2:
            msg += "\n같은 일정이 며칠 동안 반복되었습니다. 이 일정을 매일 자동등록하여 고정하시겠습니까? (예 / 아니오)"
        return True, msg

    def _count_title_recent_days(self, title, days=7):
        # 최근 n일 동안 특정 일정이 몇 번 있었는지 확인
        today = CURRENT_TIME.date()
        cnt = 0
        for i in range(days):
            d = (today + timedelta(days=i)).isoformat()
            for e in self.schedule_db.get(d, []):
                if e["title"] == title:
                    cnt += 1
                    break
        return cnt

    def list_schedules(self, week_start: date, week_end: date):
        # 주간 일정 목록 반환
        out = []
        d = week_start
        while d <= week_end:
            ds = d.isoformat()
            for e in sorted(self.schedule_db.get(ds, []), key=lambda x: x["dt"]):
                out.append(f"{d.strftime('%Y.%m.%d(%a)')} - {e['dt'].hour:02d}시 - {e['title']} - [{e['status']}]")
            d += timedelta(days=1)
        return out

    def find_schedule_by_title(self, title):
        # 일정 제목으로 검색
        res = []
        for ds, arr in self.schedule_db.items():
            for idx, e in enumerate(arr):
                if e["title"] == title:
                    res.append((ds, idx, e))
        return res

    def modify_schedule(self, title, new_dt=None, remove=False):
        found = self.find_schedule_by_title(title)
        if not found:
            return False, f"'{title}' 일정이 없습니다."
        # 여러 개 있을 경우 가장 가까운 일정 선택
        found_sorted = sorted(found, key=lambda x: datetime.fromisoformat(x[0]) if isinstance(x[0], str) else x[0])
        ds, idx, entry = found_sorted[0]
        if remove:
            self.schedule_db[ds].pop(idx)
            self._persist()
            return True, f"'{title}'이(가) 취소되었습니다."
        else:
            if new_dt <= CURRENT_TIME:
                return False, "지난 시간으로는 수정할 수 없습니다."
            # 일정 시간 변경
            self.schedule_db[ds].pop(idx)
            new_ds = new_dt.date().isoformat()
            entry["dt"] = new_dt
            self.schedule_db[new_ds].append(entry)
            self._persist()
            return True, f"'{title}'이(가) {new_dt.strftime('%Y.%m.%d %H시')}로 변경되었습니다."

    def mark_status_today(self, title, status):
        today = CURRENT_TIME.date().isoformat()
        for e in self.schedule_db.get(today, []):
            if e["title"] == title:
                e["status"] = status
                self._persist()
                return True, f"오늘의 '{title}' 일정이 '{status}' 처리되었습니다."
        return False, f"오늘의 '{title}' 일정이 없습니다."

    def get_achievement_rate(self):
        today = CURRENT_TIME.date().isoformat()
        arr = self.schedule_db.get(today, [])
        total = len(arr)
        done = sum(1 for e in arr if e["status"] == "done")
        if total == 0:
            return f"오늘({today}) 일정이 없습니다."
        rate = int((done/total)*100)
        summary = f"'{today}' : 일정 {total}개 중, {done}개 완료. 달성률: {rate}%."
        # 동기부여 메시지
        if rate == 100:
            summary += " 오늘의 일정을 모두 소화하셨네요. 👍 수고하셨습니다!"
        elif rate == 0:
            summary += " 혹시 수면캡슐 안에 갇혀계셨나요...? 🙄"
        elif rate >= 50:
            summary += " 벌써 절반이나 해결하셨네요. 😮 앞으로 얼마 남지 않았어요!"
        else:
            if done == 1:
                summary += " 성공적인 하루의 시작! 😎"
            else:
                summary += " 아직 하루가 끝난 건 아니니 앞으로 분발해주세요! 🤨"
        return summary

    # 간단한 Q-learning: 추천된 시간대 블록에 맞춰 최적화
    def q_learn_assign(self, recommended_block):
        rec_idx = TIME_BLOCKS.index(recommended_block)
        episodes = 50
        for _ in range(episodes):
            state = np.random.randint(0,3)
            # epsilon-greedy
            if np.random.rand() < 0.2:
                action = np.random.randint(0,3)
            else:
                action = np.argmax(self.qtable[state])
            r = 10 if action == rec_idx else -1
            next_state = action
            self.qtable[state,action] = (1 - self.alpha)*self.qtable[state,action] + self.alpha*(r + self.gamma*np.max(self.qtable[next_state]))
        # 시작 상태 0에서 최적 행동 선택
        best_idx = int(np.argmax(self.qtable[0]))
        return TIME_BLOCKS[best_idx]

    # 반복 일정 자동등록
    def auto_register_event(self, title, hour):
        today = CURRENT_TIME.date()
        for i in range(1, 30):
            d = today + timedelta(days=i)
            dt = datetime(d.year,d.month,d.day,hour,0,0)
            self.schedule_db[d.isoformat()].append({
                "title": title,
                "dt": dt,
                "status": "pending",
                "auto_fixed": True
            })
        self._persist()
        return True, f"'{title}'이(가) 앞으로 자동등록되도록 설정되었습니다."

# --------------------------
# 추천 모델 (지도학습)
# --------------------------
class TimeRecommender:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, weekday, importance, duration):
        X = np.array([[weekday, importance, duration]])
        return self.model.predict(X)[0]

# --------------------------
# Gradio 챗봇 백엔드
# --------------------------
scheduler = Scheduler()

def bot_reply(user_text, session_id="default"):
    # 한국어 명령 라우팅
    text = user_text.strip()
    try:
        # 1) 일정 추가
        if re.match(r"^\d{4}[.\-]\d{1,2}[.\-]\d{1,2}\s+\d{1,2}시\s+.+", text):
            try:
                dt, title = scheduler.parse_schedule_entry(text)
            except ValueError as e:
                return str(e)
            ok, msg = scheduler.add_schedule(dt, title)
            return msg

        # 2) 일정 수정
        if text.startswith("수정 "):
            title = text[len("수정 "):].strip()
            found = scheduler.find_schedule_by_title(title)
            if not found:
                return f"'{title}' 일정이 없습니다."
            scheduler.pending_modify[session_id] = {"title": title}
            return f"'{title}'을(를) 어떻게 수정하시겠습니까? (예: 2025.09.04 15시 또는 '일정제거')"

        # 수정 상태 처리
        if session_id in scheduler.pending_modify:
            pm = scheduler.pending_modify[session_id]
            if text == "일정제거":
                title = pm["title"]
                ok, msg = scheduler.modify_schedule(title, remove=True)
                del scheduler.pending_modify[session_id]
                return msg
            try:
                new_dt, _ = scheduler.parse_schedule_entry(text)
                title = pm["title"]
                ok, msg = scheduler.modify_schedule(title, new_dt=new_dt, remove=False)
                del scheduler.pending_modify[session_id]
                return msg
            except ValueError:
                return "수정 형식이 잘못되었습니다. 'YYYY.MM.DD HH시' 형식으로 입력하거나 '일정제거'를 입력하세요."

        # 3) 자동 등록 여부 확인
        if text in ("예", "아니오"):
            candidate = None
            for ds, arr in reversed(list(scheduler.schedule_db.items())):
                for e in arr:
                    c = scheduler._count_title_recent_days(e["title"], days=7)
                    if c >= 2 and not e.get("auto_query_answered", False):
                        candidate = e
                        cand_title = e["title"]
                        e["auto_query_answered"] = True
                        break
                if candidate: break
            if not candidate:
                return "자동 등록 추천할 일정이 없습니다."
            if text == "예":
                hour = candidate["dt"].hour
                ok, msg = scheduler.auto_register_event(cand_title, hour)
                return msg
            else:
                return "자동등록을 취소했습니다."

        # 4) 일정 확인
        if text == "일정":
            week_start = date(2025,9,1)
            week_end = date(2025,9,7)
            lines = scheduler.list_schedules(week_start, week_end)
            if not lines:
                return "해당 주에 등록된 일정이 없습니다."
            return "\n".join(lines)

        # 5) 완료/실패 처리
        m_done = re.match(r"^오늘\s+(.+)\s+완료$", text)
        m_fail = re.match(r"^오늘\s+(.+)\s+실패$", text)
        if m_done:
            title = m_done.group(1).strip()
            ok, msg = scheduler.mark_status_today(title, "done")
            return msg
        if m_fail:
            title = m_fail.group(1).strip()
            ok, msg = scheduler.mark_status_today(title, "fail")
            return msg

        # 6) 달성률
        if text == "달성률":
            return scheduler.get_achievement_rate()

        # 7) 추천 요청
        m_rec = re.match(r"^추천\s+(\d)\s+(\d+)$", text)
        if m_rec:
            importance = int(m_rec.group(1))
            duration = int(m_rec.group(2))
            weekday = CURRENT_TIME.weekday()
            if scheduler.recommender.model is None or not hasattr(scheduler.recommender.model, "predict"):
                return "추천 모델이 아직 학습되지 않았습니다."
            rec_block = scheduler.recommender.predict(weekday, importance, duration)
            opt = scheduler.q_learn_assign(rec_block)
            return f"과거 패턴 기반 추천 시간대: {rec_block}. 강화학습 기반 최적 추천: {opt}."
        
        # 8) 도움말
        help_msg = (
            "사용 가능 명령:\n"
            "1) 일정 추가: 'YYYY.MM.DD HH시 제목' (예: 2025.09.04 17시 농구약속)\n"
            "2) 일정 수정: '수정 제목' -> 이후 'YYYY.MM.DD HH시' 또는 '일정제거'\n"
            "3) 일정 확인: '일정'\n"
            "4) 오늘 일정 완료/실패: '오늘 제목 완료' / '오늘 제목 실패'\n"
            "5) 달성률 확인: '달성률'\n"
            "6) 추천(간단): '추천 [중요도(1-5)] [소요시간(시간)]' (예: 추천 5 2)\n"
        )
        return help_msg
    except Exception as e:
        return f"오류가 발생했습니다: {e}"

# --------------------------
# Gradio UI
# --------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 인공지능 주간 스케줄러 (챗봇)")
    chatbot = gr.Chatbot()
    txt = gr.Textbox(placeholder="명령을 입력하세요 (예: 2025.09.04 17시 농구약속)", lines=2)
    session_state = gr.State({})  # 세션 별 ID 저장
    def submit(msg, history, state):
        session_id = "user1"  # 단일 사용자 데모용; 다중 사용자면 고유 id 사용
        # append user message
        history = history + [[ "user", msg ]]
        reply = bot_reply(msg, session_id=session_id)
        history = history + [[ "bot", reply ]]
        return "", history, state
    txt.submit(submit, [txt, chatbot, session_state], [txt, chatbot, session_state])
    btn = gr.Button("전송")
    btn.click(submit, [txt, chatbot, session_state], [txt, chatbot, session_state])

if __name__ == "__main__":
    demo.launch()
