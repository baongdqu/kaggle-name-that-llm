import zipfile
import pandas as pd

with zipfile.ZipFile('name-that-llm.zip') as z:
    for name in z.namelist():
        if name.endswith('.csv'):
            df = pd.read_csv(z.open(name))
            break

print('Total teams:', len(df))
print('\n=== TOP 25 LEADERBOARD ===')
for i, r in df.iloc[:25].iterrows():
    team_name = str(r['TeamName'])[:30].ljust(30)
    print(f"{r['Rank']:4d} | {team_name} | Score: {r['Score']:.3f} | Entries: {r['SubmissionCount']:2d} | Date: {r['LastSubmissionDate']}")

print('\n=== OUR TEAM ===')
my_team = df[df['TeamMemberUserNames'].str.contains('baonguyenduyquoc', case=False, na=False)]
for i, r in my_team.iterrows():
    print(f"Rank: {r['Rank']} | Team: {r['TeamName']} (TeamId: {r['TeamId']}) | Score: {r['Score']:.3f} | Entries: {r['SubmissionCount']} | Date: {r['LastSubmissionDate']}")

print('\n=== KEY RANK CUTOFFS ===')
ranks = [1, 5, 10, 15, 20, 50, 100, 200, 300, 500, 587, 1000, 1500, 2000, 2500]
for rk in ranks:
    if rk <= len(df):
        row = df.iloc[rk-1]
        print(f"Rank {rk:4d}: Score {row['Score']:.3f} ({row['TeamName']})")

print('\n=== TEAMS WITH SCORE 0.604 ===')
s_0604 = df[df['Score'] == 0.604]
print(f"Total teams with score 0.604: {len(s_0604)} (Ranks {s_0604['Rank'].min()} to {s_0604['Rank'].max()})")
