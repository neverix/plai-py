import numpy as np
from agent import MineRLAgent
from tqdm import tqdm
import os.path
import pickle
import torch
import time
import re
import pywinctl
import pyautogui
import requests
import appdirs


def choose(options, prompt, to_str=lambda x: x):
    assert len(options) >= 1
    if len(options) == 1:
        return 0
    for i, option in enumerate(options):
        print(f" [{i + 1}] {to_str(option)}")
    while True:
        i = input(f"{prompt}: ")
        try:
            i = int(i) - 1
            options[i]
            break
        except ValueError:
            pass
        except IndexError:
            pass
    return options[i]


def choose_window():
    must_contain = "Minecraft"
    windows = pywinctl.getWindowsWithTitle(must_contain, condition=pywinctl.Re.CONTAINS)
    if not windows:
        print("No Minecraft windows found")
        exit()
    i = choose(windows, "Main Minecraft window", lambda x: x.title)
    return windows[i]


def activate_window(window):
    window.activate(wait=True)
    time.sleep(1/20)
    pyautogui.press("esc")
    time.sleep(1/20)
    pyautogui.leftClick()
    time.sleep(1/20)


def screenshot_window(window):
    return pyautogui.screenshot().crop((window.left, window.top, window.right, window.bottom))


def download_url(url, output_file):
    if os.path.exists(output_file):
        print(url, "already downloaded")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("Downloading", url)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_file, "wb") as f:
            pbar = tqdm(total=int(r.headers.get("Content-Length", 0)) or None)
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_url_to_cache(url):
    filename = os.path.join(appdirs.user_cache_dir("plai"), "models", os.path.basename(url))
    download_url(url, filename)
    return filename


def load_model(device="cpu"):
    # I didn't want to make this so ugly, but also couldn't stand writing it all manually
    # Sort of https://xkcd.com/1319/ but on an aesthetic level
    url_prefix = "https://openaipublic.blob.core.windows.net/minecraft-rl/models/"
    options = [
        ("Foundational Model", "foundation-model-1x.weights", 1),
        ("Foundational Model", "foundation-model-2x.weights", 2),
        ("Foundational Model", "foundation-model-1x.weights", 3),
        ("Fine-Tuned from House", "bc-house-3x.weights", 3),
        ("Fine-Tuned from Early Game", "bc-early-game-2x.weights", 2),
        ("Fine-Tuned from Early Game", "bc-early-game-3x.weights", 3),
        ("RL from Foundation", "rl-from-foundation-2x.weights", 2),
        ("RL from House", "rl-from-house-2x.weights", 2),
        ("RL from Early Game", "rl-from-early-game-2x.weights", 2),
    ]
    options = [
        (name + f", {size}x weights",
         url_prefix + url,
         [url_prefix + "foundation-model-1x.model",
          url_prefix + "2x.model",
          url_prefix + "foundation-model-3x.model"][size - 1]
         ) for name, url, size in options]
    _, weights_url, model_url = choose(options, "Model", to_str=lambda x: x[0])
    weights_file = download_url_to_cache(weights_url)
    model_file = download_url_to_cache(model_url)

    print("Loading model...")
    config = pickle.load(open(model_file, "rb"))
    policy_kwargs = config["model"]["args"]["net"]["args"]
    pi_head_kwargs = config["model"]["args"]["pi_head_opts"]
    if "temperature" not in pi_head_kwargs and "temperature" in config:
        pi_head_kwargs["temperature"] = float(config["temperature"])

    agent = MineRLAgent(device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights_file)
    return agent


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    while True:
        try:
            fov = int(input("FOV? (70) ") or 70)  # TODO configure retrial
            break
        except ValueError:
            pass

    window = choose_window()
    print("Using window:", window.title)
    activate_window(window)
    time.sleep(1)

    fps = 30
    last = time.time()

    while True:
        if not window.isActive:
            print("Closing...")
            break
        img = screenshot_window(window)
        center = np.asarray([img.size[0] // 2, img.size[1] // 2])
        size = np.asarray([center[0] / 480 * 640, center[0]])
        img = img.crop(tuple(map(int, (center - size * 70 / fov).tolist() + (center + size * 70 / fov).tolist())))
        img.save("screen.png")
        action = model.get_action({"pov": np.asarray(img)})

        cam = action["camera"][0]
        cam = cam / 70 * img.height
        dx, dy = cam
        pyautogui.moveRel(dx, dy)
        mapping = {
            "attack": "lmb",
            "back": "s",
            "forward": "w",
            "jump": "space",
            "left": "a",
            "right": "d",
            "sneak": "shift",
            "sprint": "ctrl",
            "use": "rmb",
            "drop": "q",
            "inventory": "e",
            **{f"hotbar.{i}": str(i) for i in range(1, 10)}
        }
        for k, v in mapping.items():
            if action[k][0] > 0.5:
                if v == "lmb":
                    pyautogui.leftClick(_pause=False)
                elif v == "rmb":
                    pyautogui.rightClick(_pause=False)
                elif v in ("space", "q"):
                    pyautogui.press(v, _pause=False)
                elif v.startswith("hotbar."):
                    pyautogui.press(v, _pause=False)
                else:
                    pyautogui.keyDown(v, _pause=False)
            else:
                if v not in ("lmb", "rmb", "space"):
                    pyautogui.keyUp(v, _pause=False)

        remaining = 1 / fps - (time.time() - last)
        # time.sleep(max(0, remaining))
        last = time.time()


if __name__ == '__main__':
    main()
