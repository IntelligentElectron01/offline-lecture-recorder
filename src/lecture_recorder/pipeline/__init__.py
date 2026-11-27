"""Speech pipeline (after the lecture): transcribe -> free VRAM -> diarize -> merge -> plain files.

Run one session:   python -m lecture_recorder.pipeline <session-dir>
Queue + worker:    lecture_recorder.pipeline.queue (one GPU job at a time)
"""
