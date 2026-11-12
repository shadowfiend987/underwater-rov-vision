# Testing record

The release was checked on Windows with Python 3.12.14, OpenCV 4.12.0 and NumPy 2.2.6. Run the following commands from the repository root after installing the runtime requirements.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m tools.generate_demo_inputs
```

| Check | Result and scope |
|---|---|
| Regression suite | 28 test methods passed. The tests use generated pixels, explicit scan-state inputs and temporary files. Tests covering capture, keyboard and GUI handling use substitutes for those interfaces. |
| Release examples | The generated three-tag image decodes to IDs 2, 7 and 12. For the generated pole image and order R-B-Y, the red center is (120, 275), with a horizontal error of -240 pixels. |
| Earlier comparison with the original | The original color program and its extracted processor gave identical result dictionaries for seven generated frames under three target orders. |
| Earlier camera access check | One frame was read at 640 by 480 pixels. The device model was unconfirmed. |

The blue saturation minimum changed from 100 to 40 alongside the hue adjustment. The saved settings document these changes. The reported reduction in interference from a yellow background was a qualitative observation. No field accuracy, operating range, mission score or throughput estimate was established.

The examples and both README previews can be reproduced with the generator and the two single-image commands in the README. The program saves live captures locally under its output directory, which is excluded from version control.
