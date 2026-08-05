# scripts/check_pireps.py
import asyncio
from app.sources.awc import fetch_pireps

async def main():
    # rough box over the northeastern US
    reports = await fetch_pireps(bbox=(38.0, -82.0, 45.0, -68.0), hours_back=6)
    print(f"{len(reports)} reports\n")
    for r in reports[:10]:
        print(f"{r.observation_time}  {r.latitude:.2f},{r.longitude:.2f}  "
              #f"FL{r.altitude_ft//100:03d}  {r.turbulence_severity}  {r.aircraft_type}")
              f"{f'FL{r.altitude_ft//100:03d}' if r.altitude_ft else 'FL---'}  "
              f"{r.turbulence_severity or '(uncoded)'}  {r.aircraft_type}")
        print(f"    raw: {r.raw_text[:80]}\n")

asyncio.run(main())
