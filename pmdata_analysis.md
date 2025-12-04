# pmdata Directory Analysis

The `pmdata` directory appears to be a comprehensive multimodal dataset containing physiological and behavioral data for 16 participants.

## Directory Structure

The root directory contains:
- **Participant Folders**: `p01` through `p16` (16 participants).
- **Metadata**: `participant-overview.xlsx` (likely contains demographic or summary info).

## Participant Data (`p01` - `p16`)

Each participant's folder is structured with the following subdirectories:

### 1. `fitbit`
Contains high-resolution time-series data from Fitbit wearable devices in JSON format.
- **Key Files**:
  - `heart_rate.json`: Minute-level (or higher freq) heart rate data.
  - `steps.json`, `distance.json`, `calories.json`: Activity metrics.
  - `sleep.json`, `sleep_score.csv`: Sleep patterns and quality scores.
  - `sedentary_minutes.json`, `very_active_minutes.json`, etc.: Activity intensity breakdowns.
- **Format**: JSON arrays of objects with `dateTime` and `value` fields.

### 2. `pmsys`
Contains subjective wellness and training data in CSV format.
- **Key Files**:
  - `wellness.csv`: Daily self-reported metrics including:
    - `fatigue`, `mood`, `readiness`, `stress` (Likert scale 1-5 or similar).
    - `sleep_duration_h`, `sleep_quality`.
    - `soreness` (with area codes).
  - `injury.csv`: Injury reports.
  - `srpe.csv`: Session Rating of Perceived Exertion (training load).

### 3. `googledocs`
- **Key Files**:
  - `reporting.csv`: Likely contains additional self-reported logs or journal entries.

### 4. `food-images`
- Contains images, presumably of meals consumed by the participants, used for dietary analysis.

## Data Summary

| Data Source | Type | Format | Examples |
| :--- | :--- | :--- | :--- |
| **Fitbit** | Objective / Physiological | JSON | Heart rate, Steps, Sleep stages |
| **PMSYS** | Subjective / Self-report | CSV | Mood, Stress, Fatigue, Soreness |
| **Google Docs** | Self-report | CSV | Reporting logs |
| **Food Images** | Image | Images | Photos of meals |

This dataset is suitable for analyzing relationships between objective physiological metrics (Fitbit) and subjective wellness (PMSYS), as well as lifestyle factors like diet and activity.
