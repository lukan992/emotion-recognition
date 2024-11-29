import uvicorn


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8031, reload=False)


def start(port: int):
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
