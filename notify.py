import os
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import socket

# ===============================
# 설정
# ===============================
EMAIL = "younghune135@gmail.com"
APP_PASSWORD = "zkbc acvb oydj sazv"

CHECK_INTERVAL = 60  # seconds

# pid : alias
WATCH_PIDS = {
    # 691260: "baseline experiments\nAda6000 server\nLATTICE clothing\nLATTICE office\nDAMRS office\nSMORE office\n",
    # 691529: "baseline experiments\nAda6000 server\nLGMRec office\nVBPR grocery\nLATTICE grocery\nFREEDOM grocery\n",
    # 691827: "baseline experiments\nAda6000 server\nBM3 grocery\nDAMRS grocery\nSMORE grocery\nLGMRec grocery\n",
    # 24978: "baseline experiments\nAda6000 server\nAlignRec clothing\n",
    # 139458: "baseline experiments\nAda6000 server\nAlignRec office\n",
    # 141021: "baseline experiments\nAda6000 server\nAnchorRec office\n",
    # 143022: "baseline experiments\nAda6000 server\nAlignRec game\n",
    # 144304: "baseline experiments\nAda6000 server\nAnchorRec game\n",
    # 474414: "baseline experiments\nAda6000 server\nVBPR game\n",
    # 1831143: "baseline experiments\nAda6000 server\nAlignRec grocery\n",
    # 1831616: "baseline experiments\nAda6000 server\nAnchorRec grocery\n",
    # 1979084: "RQ3 baby image gen complete"
    # 1979535: "RQ3 sports image gen complete"
    # 1979953: "RQ3 elec image gen complete"
    # 1980460: "RQ3 office image gen complete",
    # 1980874: "RQ3 clothing image gen complete",
    # 1981244: "RQ3 game image gen complete",
    # 1981604: "RQ3 grocery image gen complete",
    # 2032844: "hyper-0,1 AnchorRec baby",
    # 2033331: "hyper-0,1 AnchorRec sports",
    # 2033660: "hyper-0,1 AnchorRec elec",
    # 2041343: "hyper-0,1 AnchorRec clothing",
    # 2041531: "hyper-0,1 AnchorRec office",
    # 2041707: "hyper-0,1 AnchorRec game",
    # 2041889: "hyper-0,1 AnchorRec grocery",
    # 3517327: "hyper-0,1 AnchorRec elec",
    # 3520245: "hyper-tau AnchorRec baby",
    # 3523077: "hyper-tau AnchorRec sports",
    # 3523077: "hyper-tau AnchorRec elec",

    # 3895973: "hyper-tau AnchorRec clothing",
    # 3606268: "hyper-tau AnchorRec office",
    # 3896565: "hyper-tau AnchorRec game",
    # 3897813: "hyper-tau AnchorRec grocery",
    # 3915088: "hyper-cl sports append"
    # 3917465: "hyper-sim sports append"
    # 1346344: "hyper elec cl 0 sim 0.1 tau 0.1"
    # 1835942: "hyper grocery cl 0.0001 sim 0.1 tau 0.1"
    # 1835942: "hyper elec cl 0 sim 0.1 tau 0.1 gpu 2"
    # 2099764: "hyper elec cl 0.01 sim 0.1 tau 0.1 gpu 4"
    # 2123195: "hyper elec cl 0.1 sim 0.1 tau 0.1 gpu 5"
    # 2153668: "hyper elec cl 1 sim 0.1 tau 0.1"
    # 2364375: "hyper elec sim 0 to 0.01 cl 0.01 tau 0.1"
    # 2912085: "hyper elec cl 0.01 sim 0.1 tau all"
    # 3360195: "hyper elec sim 1"
    # 3488584: "ablation elec no cl no sim",
    # 61442: "proj sports",
    # 3929550: "uim sports",
    # 3966745: "proj office",
    # 3933778: "uim office",
    # 60112: "proj game",
    # 3934302: "uim game",
    # 3949727: "proj baby",
    # 3928953: "uim baby",
    # 62094: "anchor baby",
    # 62763: "anchor sports",
    # 63626: "anchor office",
    # 64091: "anchor game",
    92375: "baby anchor",
    92779: "game anchor"
}

# ===============================
# PID 감시
# ===============================
finished = {}

print("Monitoring processes...")
while len(finished) < len(WATCH_PIDS):
    for pid, alias in WATCH_PIDS.items():
        if pid in finished:
            continue
        try:
            os.kill(pid, 0)   # 살아있으면 OK
        except ProcessLookupError:
            finished[pid] = alias
            print(f"[DONE] {alias} (pid={pid})")

    time.sleep(CHECK_INTERVAL)

# ===============================
# 메일 내용 생성
# ===============================
finished_list = "\n".join(
    f"- {alias} (pid={pid})" for pid, alias in finished.items()
)

msg = MIMEText(f"""
Experiments finished successfully.

Host : {socket.gethostname()}
Time : {datetime.now()}

Finished jobs:
{finished_list}
""")

msg["Subject"] = "[AnchorRec] Experiments Finished"
msg["From"] = EMAIL
msg["To"] = EMAIL

# ===============================
# 메일 전송
# ===============================
server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
server.login(EMAIL, APP_PASSWORD)
server.send_message(msg)
server.quit()

print("Notification email sent.")