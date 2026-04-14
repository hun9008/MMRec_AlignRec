#!/bin/bash

LOGS_BABY=(
    "LATTICE-baby-Feb-12-2026-12-11-50.log"
    "FREEDOM-baby-Feb-12-2026-12-13-03.log"
    "LGMRec-baby-Feb-12-2026-12-35-05.log"
    "SMORE-baby-Feb-12-2026-12-37-19.log"
)

LOGS_SPORTS=(
    "LATTICE-sports-Feb-12-2026-12-28-45.log"
    "FREEDOM-sports-Feb-12-2026-12-31-10.log"
    "LGMRec-sports-Feb-12-2026-12-49-51.log"
    "SMORE-sports-Feb-12-2026-12-52-17.log"
)

LOGS_OFFICE=(
    "LATTICE-office-Feb-12-2026-12-41-11.log"
    "FREEDOM-office-Feb-12-2026-12-41-36.log"
    "LGMRec-office-Feb-12-2026-12-41-58.log"
    "SMORE-office-Feb-12-2026-12-42-25.log"
)

LOGS_GAME=(
    "LATTICE-game-Feb-12-2026-12-42-53.log"
    "FREEDOM-game-Feb-12-2026-12-44-06.log"
    "LGMRec-game-Feb-12-2026-12-45-33.log"
    "SMORE-game-Feb-12-2026-12-47-19.log"
)
# LOGS_BABY=(
#     "VBPR-baby-Nov-06-2025-23-12-10.log"
#     "LATTICE-baby-Nov-07-2025-16-36-57.log"
#     "FREEDOM-baby-Nov-07-2025-20-52-13.log"
#     "LGMRec-baby-Nov-12-2025-15-57-55.log"
#     "SMORE-baby-Nov-12-2025-13-56-09.log"
#     "BM3-baby-Nov-07-2025-22-38-26.log"
#     "DAMRS-baby-Nov-08-2025-03-59-42.log"
#     "ALIGNREC-baby-Oct-27-2025-20-34-20.log"
#     "ANCHORREC-baby-Jan-28-2026-16-47-33.log"
# )

# LOGS_SPORTS=(
#     "VBPR-sports-Nov-08-2025-22-35-21.log"
#     "LATTICE-sports-Nov-09-2025-03-03-55.log"
#     "FREEDOM-sports-Nov-09-2025-13-31-27.log"
#     "LGMRec-sports-Nov-12-2025-00-47-47.log"
#     "SMORE-sports-Nov-11-2025-20-00-28.log"
#     "BM3-sports-Nov-09-2025-16-49-13.log"
#     "DAMRS-sports-Nov-12-2025-13-11-17.log"
#     "ALIGNREC-sports-Oct-14-2025-02-55-36.log"
#     "ANCHORREC-sports-Feb-06-2026-14-59-47.log"
# )


# LOGS_ELEC=(
#     "VBPR-elec-Nov-09-2025-18-19-38.log"
#     "LATTICE-elec-Nov-12-2025-11-20-57.log"
#     "FREEDOM-elec-Nov-09-2025-23-54-29.log"
#     "LGMRec-elec-Nov-13-2025-10-52-57.log"
#     "SMORE-elec-Nov-14-2025-00-04-38.log"
#     "BM3-elec-Nov-10-2025-20-53-57.log"
#     "DAMRS-elec-Nov-13-2025-19-38-04.log"
#     "ALIGNREC-elec-Nov-14-2025-01-10-05.log"
#     "ANCHORREC-elec-Feb-10-2026-21-12-00.log"
# )

# LOGS_CLOTHING=(
#     "VBPR-clothing-Jan-30-2026-11-08-58.log"
#     "LATTICE-clothing-Jan-31-2026-14-53-57.log"
#     "FREEDOM-clothing-Jan-30-2026-14-03-38.log"
#     "LGMRec-clothing-Feb-02-2026-02-48-28.log"
#     "SMORE-clothing-Jan-31-2026-06-03-56.log"
#     "BM3-clothing-Jan-31-2026-06-03-52.log"
#     "DAMRS-clothing-Feb-01-2026-08-02-36.log"
#     "ALIGNREC-clothing-Feb-02-2026-12-05-27.log"
#     "ANCHORREC-clothing-Feb-06-2026-14-49-12.log"
# )

# LOGS_OFFICE=(
#     "VBPR-office-Jan-29-2026-12-40-09.log"
#     "LATTICE-office-Feb-01-2026-02-56-58.log"
#     "FREEDOM-office-Jan-30-2026-11-07-26.log"
#     "LGMRec-office-Jan-31-2026-14-54-03.log"
#     "SMORE-office-Feb-01-2026-07-37-19.log"
#     "BM3-office-Jan-29-2026-17-25-37.log"
#     "DAMRS-office-Feb-01-2026-04-42-59.log"
#     "ALIGNREC-office-Feb-02-2026-13-42-27.log"
#     "ANCHORREC-office-Feb-06-2026-11-58-18.log"
# )

# LOGS_GAME=(
#     "VBPR-game-Feb-02-2026-17-23-35.log"
#     "LATTICE-game-Feb-02-2026-02-51-59.log"
#     "FREEDOM-game-Jan-31-2026-06-04-06.log"
#     "LGMRec-game-Feb-03-2026-05-00-56.log"
#     "SMORE-game-Feb-02-2026-02-49-55.log"
#     "BM3-game-Feb-01-2026-03-48-50.log"
#     "DAMRS-game-Feb-01-2026-22-00-00.log"
#     "ALIGNREC-game-Feb-02-2026-13-45-17.log"
#     "ANCHORREC-game-Feb-06-2026-14-49-25.log"
# )

# LOGS_GROCERY=(
#     "VBPR-grocery-Jan-31-2026-17-55-33.log"
#     "LATTICE-grocery-Jan-31-2026-19-54-39.log"
#     "FREEDOM-grocery-Feb-01-2026-00-56-43.log"
#     "LGMRec-grocery-Feb-02-2026-11-59-47.log"
#     "SMORE-grocery-Feb-01-2026-07-54-24.log"
#     "BM3-grocery-Jan-31-2026-14-54-10.log"
#     "DAMRS-grocery-Jan-31-2026-22-15-29.log"
#     "ALIGNREC-grocery-Feb-03-2026-10-42-45.log"
#     "ANCHORREC-grocery-Feb-06-2026-14-49-56.log"
# )

for log in "${LOGS_BABY[@]}"; do
    python time_epoch.py --logs "$log"
done

for log in "${LOGS_SPORTS[@]}"; do
    python time_epoch.py --logs "$log"
done

for log in "${LOGS_ELEC[@]}"; do
    python time_epoch.py --logs "$log"
done

for log in "${LOGS_CLOTHING[@]}"; do
    python time_epoch.py --logs "$log"
done

for log in "${LOGS_OFFICE[@]}"; do
    python time_epoch.py --logs "$log"
done

for log in "${LOGS_GAME[@]}"; do
    python time_epoch.py --logs "$log"
done

for log in "${LOGS_GROCERY[@]}"; do
    python time_epoch.py --logs "$log"
done