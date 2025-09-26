import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read only needed columns
    df = pd.read_csv(
        "data_for_technical_interview_Fall2025.csv",
        usecols=["DateTime_UTC", "Depth_m", "Temperature_degC", "LightLevel"],
        parse_dates=["DateTime_UTC"]
    )

    # Resample to hourly averages (downsample huge data)
    df_resampled = df.set_index("DateTime_UTC").resample("H").mean().reset_index()

    # -----------------------
    # Plot 1: Temperature vs Time
    # -----------------------
    plt.figure(figsize=(12,6))
    plt.plot(df_resampled["DateTime_UTC"], df_resampled["Temperature_degC"], 'b-')
    plt.xlabel("Time (Hourly)")
    plt.ylabel("Temperature (°C)")
    plt.title("Hourly Average Temperature vs Time")
    plt.grid(True)
    plt.show()

    # -----------------------
    # Plot 2: LightLevel vs Time
    # -----------------------
    plt.figure(figsize=(12,6))
    plt.plot(df_resampled["DateTime_UTC"], df_resampled["LightLevel"], 'orange')
    plt.xlabel("Time (Hourly)")
    plt.ylabel("Light Level")
    plt.title("Hourly Average Light Level vs Time")
    plt.grid(True)
    plt.show()

    # -----------------------
    # Plot 3: LightLevel vs Depth
    # -----------------------
    plt.figure(figsize=(8,6))
    plt.scatter(df["LightLevel"], df["Depth_m"], s=1, alpha=0.3, c='green')
    plt.xlabel("Light Level")
    plt.ylabel("Depth (m)")
    plt.title("Light Level vs Depth")
    plt.gca().invert_yaxis()   # depth increases downward
    plt.grid(True)
    plt.show()

    # -----------------------
    # Plot 4: Temperature vs Depth
    # -----------------------
    plt.figure(figsize=(8,6))
    plt.scatter(df["Temperature_degC"], df["Depth_m"], s=1, alpha=0.3, c='red')
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Depth (m)")
    plt.title("Temperature vs Depth")
    plt.gca().invert_yaxis()   # depth increases downward
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
