from fastapi import FastAPI
from fastapi.responses import FileResponse
from capture import run_capture
from utils import get_latest_result

app = FastAPI()


@app.post("/capture")
def capture():
    run_capture()
    return {"message": "촬영 완료"}


@app.get("/result")
def result():
    path = get_latest_result()

    if path:
        return FileResponse(path)
    else:
        return {"error": "결과 없음"}