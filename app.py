ValueError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/cheer-app/app.py", line 107, in <module>
    out_frame = model.draw(frame)
File "/home/adminuser/venv/lib/python3.10/site-packages/easy_ViTPose/inference.py", line 297, in draw
    if self._yolo_res is not None and (show_raw_yolo or (self.tracker is None and show_yolo)):
