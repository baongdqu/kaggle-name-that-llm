import zipfile
import pandas as pd
import io

with zipfile.ZipFile('scratch/name-that-llm.zip') as z:
    for name in z.namelist():
        print(f"File in zip: {name}")
        with z.open(name) as f:
            df = pd.read_csv(f)
            print(f"Shape: {df.shape}")
            print(df.head())
            # Find our team
            my_team = df[df['TeamId'] == 17137218]
            if len(my_team) == 0:
                my_team = df[df['TeamName'].str.contains('Mohamed|Bao', case=False, na=False)]
            print("\n--- Our Team Rank & Score ---")
            print(my_team)
            
            # Find rank of 0.646
            df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
            df_sorted['Rank'] = df_sorted.index + 1
            team_entry = df_sorted[df_sorted['TeamId'] == 17137218]
            print("\n--- In Sorted LB ---")
            print(team_entry[['Rank', 'TeamId', 'TeamName', 'Score']])
            
            # Show surrounding teams around 0.646
            idx = df_sorted[df_sorted['Score'] <= 0.650].index
            if len(idx) > 0:
                start_i = max(0, idx[0] - 5)
                end_i = min(len(df_sorted), idx[0] + 15)
                print("\n--- Surrounding Teams ---")
                print(df_sorted.iloc[start_i:end_i][['Rank', 'TeamName', 'Score']])
            
            total_teams = len(df)
            print(f"\nTotal teams: {total_teams}")
