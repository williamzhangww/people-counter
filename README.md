# People Counter

This package provides a local web interface for privacy-preserving people usage analysis from video.

## Exported files per processed video
- `<video_stem>_usage_events.csv`
- `<video_stem>_visit_events.csv`
- `<video_stem>-annotated.mp4`

## Event definitions used in this final version
- **Visit event**: a person enters the analyzed ROI once.
- **Usage event**: a continuous stay in the ROI, allowing interruptions up to 5 seconds, whose cumulative visible duration exceeds 15 seconds.
- The same continuous stay is counted only once, no matter how long it lasts.
- Time is measured from the beginning of the video in seconds.
- In the current thesis setup, the whole frame is used as the ROI.

## CSV structure
### Usage Events
Columns:
- `event_id`
- `start_time`
- `end_time`
- `duration`

### Visit Events
Columns:
- `visit_id`
- `start_time`
- `end_time`
- `duration`

## First-time setup
1. Run `setup.bat`
2. Choose GPU or CPU mode
3. Wait for dependency installation
4. The setup automatically downloads `yolov8s.pt` into the `models/` folder if needed

## Run the application
1. Run `run.bat`
2. The browser opens automatically
3. Select one or more local videos and process them

## Notes
- Uploaded source files are stored in a temporary system folder during processing and are deleted automatically afterwards.
- Output files are saved in the `outputs/` folder.
- Engagement rate can be calculated as `usage events / visit events`.
- No personal data is stored or identifiable.
